"""Phase 1: 코호트 -> EHR 피처 -> 환자 단위 분할 -> 정형 베이스라인.

    python scripts/run_phase1.py --check
    python scripts/run_phase1.py --sensitivity
    python scripts/run_phase1.py --no-prescriptions   # 투약 피처 없이 같은 분할로 측정함
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# 필요한 원천 테이블. (파일명, 필수 여부, 용도)
REQUIRED_TABLES = [
    ("admissions.csv.gz", True, "코호트, 라벨"),
    ("patients.csv.gz", True, "나이, 성별"),
    ("diagnoses_icd.csv.gz", True, "Charlson, 진단 수"),
    ("prescriptions.csv.gz", False, "투약, 다약제"),
    ("procedures_icd.csv.gz", False, "시술 수"),
]


def check_data(raw_dir: Path) -> tuple[bool, list[str]]:
    """필수 테이블이 준비됐는지 확인. (준비완료, 메시지들)"""
    msgs, ready = [], True
    for name, required, purpose in REQUIRED_TABLES:
        hits = list(raw_dir.rglob(name))
        if hits:
            size_mb = hits[0].stat().st_size / 1024**2
            msgs.append(f"  [OK] {name:24s} {size_mb:8.1f} MB   {purpose}")
        elif required:
            msgs.append(f"  [오류] {name:24s} {'없음':>8s}      {purpose}  <- 필수")
            ready = False
        else:
            msgs.append(f"  ⬜ {name:24s} {'없음':>8s}      {purpose}  (선택, 건너뜀)")
    return ready, msgs


def main() -> int:
    from datapaths import RAW

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--check", action="store_true", help="데이터 준비 상태만 확인")
    ap.add_argument("--sensitivity", action="store_true", help="코호트 기준 민감도 분석")
    ap.add_argument("--model", default="xgboost", choices=["xgboost", "lightgbm", "logreg"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-prescriptions", action="store_true",
                    help="투약 테이블을 읽지 않는다. 투약 피처의 기여를 같은 분할에서 재고 "
                         "phase1_metrics_no_rx.json 에 따로 남긴다")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    print("=" * 74)
    print("  Phase 1: MIMIC-IV 코호트 + EHR 베이스라인")
    print("=" * 74)
    print(f"\n[0] 데이터 준비 상태  ({raw})")
    ready, msgs = check_data(raw)
    for m in msgs:
        print(m)
    if not ready:
        print("\n필수 테이블이 없습니다. 다운로드를 먼저 완료하세요:")
        print("  sh scripts/download_mimic.sh mimiciv")
        return 1
    if args.check:
        print("\n준비 완료. --check 없이 다시 실행하면 전체 파이프라인이 돕니다.")
        return 0

    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd

    from cohort import CohortConfig, build_cohort, load_admissions
    from features import ELECTIVE_ADMISSION_TYPES, build_features, feature_columns
    from metrics import classification_metrics, subgroup_metrics
    from models import build_tabular_model
    from splits import check_split, patient_split
    from tracking import ExperimentLogger

    print("\n[1] 코호트 구성")
    adm, pat = load_admissions(raw)
    cfg = CohortConfig(elective_admission_types=ELECTIVE_ADMISSION_TYPES)
    coh = build_cohort(adm, pat, cfg)
    print(coh.flow_table().to_string(index=False))
    summary = coh.summary()
    print()
    for k, v in summary.items():
        print(f"    {k:32s} {v}")

    artifact = {"phase": "1", "seed": args.seed, "model": args.model,
                "cohort": {k: v for k, v in summary.items()},
                "flow": coh.flow_table().to_dict(orient="records")}

    if args.sensitivity:
        print("\n[1b] 코호트 기준 민감도: 재입원률은 기준에 따라 크게 달라진다")
        rows = []
        for label, c in [
            ("절단 포함, 예정 미제외", CohortConfig()),
            ("절단 포함, 예정 제외", CohortConfig(elective_admission_types=ELECTIVE_ADMISSION_TYPES)),
            ("절단 제외, 예정 미제외", CohortConfig(exclude_last_admission=True)),
            ("절단 제외, 예정 제외",
             CohortConfig(exclude_last_admission=True,
                          elective_admission_types=ELECTIVE_ADMISSION_TYPES)),
        ]:
            s = build_cohort(adm, pat, c).summary()
            rows.append((label, s["n_admissions"], s["readmit_rate"]))
        print(f"    {'기준':30s} {'입원 수':>10s} {'재입원률':>10s}")
        for label, n, rate in rows:
            print(f"    {label:30s} {n:10,} {rate*100:9.2f}%")
        artifact["sensitivity"] = [{"criteria": label, "n_admissions": n,
                                    "readmit_rate": rate} for label, n, rate in rows]

    print("\n[2] EHR 피처 생성")
    dx = pd.read_csv(next(raw.rglob("diagnoses_icd.csv.gz")))
    rx_hits = list(raw.rglob("prescriptions.csv.gz"))
    pr_hits = list(raw.rglob("procedures_icd.csv.gz"))
    rx = (pd.read_csv(rx_hits[0], usecols=["hadm_id", "drug"])
          if rx_hits and not args.no_prescriptions else None)
    pr = pd.read_csv(pr_hits[0]) if pr_hits else None

    feat, missing = build_features(coh.df, pat, dx, rx, pr)
    cols = feature_columns(feat, verbose=True)
    print(f"    모델 입력 피처 {len(cols)}개")
    if missing:
        top = sorted(missing.items(), key=lambda kv: -kv[1])[:5]
        print("    결측률 상위:", ", ".join(f"{k} {v:.1%}" for k, v in top))

    print("\n[3] 환자 단위 분할 (V1)")
    y = feat["readmit_30d"].to_numpy()
    groups = feat["subject_id"].to_numpy()
    sp = patient_split(groups, test_size=0.2, val_size=0.1, seed=args.seed)
    split_info = check_split(y, groups, sp)
    for k, v in split_info.items():
        print(f"    {k:22s} {v}")

    X = feat[cols].to_numpy(dtype=np.float32)

    print(f"\n[4] EHR 단독 베이스라인 ({args.model})")
    n_pos = int(y[sp.train_idx].sum())
    n_neg = len(sp.train_idx) - n_pos
    params = {
        "phase": "1", "model": args.model, "seed": args.seed,
        "n_features": len(cols),
        "scale_pos_weight": round(n_neg / max(1, n_pos), 4),
        "elective_types": "|".join(ELECTIVE_ADMISSION_TYPES),
        "prescriptions": not args.no_prescriptions,
    }

    with ExperimentLogger("phase1-ehr-baseline", params=params,
                          run_name=f"{args.model}-seed{args.seed}") as run:
        run.note("Phase 1: MIMIC-IV EHR 단독 베이스라인. 예측 시점=퇴원.")
        run.log_split(
            n_train=len(sp.train_idx), n_val=len(sp.val_idx), n_test=len(sp.test_idx),
            method="patient",
            pos_rate={k.replace("pos_rate_", ""): v
                      for k, v in split_info.items() if k.startswith("pos_rate_")},
            n_subjects={k.replace("n_subjects_", ""): v
                        for k, v in split_info.items() if k.startswith("n_subjects_")},
        )
        run.log_metrics({"cohort_readmit_rate": summary["readmit_rate"],
                         "n_censored": summary["n_censored_last_admission"]})

        model = build_tabular_model(args.model, scale_pos_weight=n_neg / max(1, n_pos))
        model.fit(X[sp.train_idx], y[sp.train_idx])
        prob = model.predict_proba(X[sp.test_idx])[:, 1]
        m = classification_metrics(y[sp.test_idx], prob)

        print(f"    {'지표':16s} {'값':>10s}")
        for k, v in m.as_dict().items():
            print(f"    {k:16s} {v:10.4f}")
        if m.warnings:
            print(f"    warnings: {m.warnings}")
        print("\n    문헌 baseline: LACE 0.61 / 로지스틱 0.62 / ML 0.65~0.75")
        print("    AUROC 0.85 이상이면 누수를 먼저 의심할 것")
        run.log_metrics({f"test_{k}": v for k, v in m.as_dict().items()})

        artifact["split"] = dict(split_info)
        artifact["n_features"] = len(cols)
        artifact["scale_pos_weight"] = params["scale_pos_weight"]
        artifact["test"] = m.as_dict()
        artifact["test_warnings"] = list(m.warnings)

        # 서브그룹 (V4 배관)
        if "is_female" in feat.columns:
            g = np.where(feat["is_female"].to_numpy()[sp.test_idx] == 1, "F", "M")
            sub = {}
            for name, gm in subgroup_metrics(y[sp.test_idx], prob, g).items():
                print(f"    subgroup {name}: n={gm.n} auroc={gm.auroc:.4f} ece={gm.ece:.4f}")
                run.log_subgroup("sex", {name: {"auroc": gm.auroc, "ece": gm.ece}})
                sub[name] = {"n": gm.n, "auroc": gm.auroc, "ece": gm.ece}
            artifact["subgroup_sex"] = sub

    # mlflow.db 는 .gitignore 로 빠지므로 README 가 인용할 수치를 따로 남김.
    out = ROOT / "outputs" / ("phase1_metrics_no_rx.json" if args.no_prescriptions
                              else "phase1_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")
    print(f"\n  저장: {out}")

    print("\n" + "=" * 74)
    print("  Phase 1 완료. 다음: Phase 1.5 (M1-b 결측 모달리티 진단)")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
