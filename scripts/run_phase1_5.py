"""Phase 1.5: 결측 모달리티 진단.

흉부 X선이 있는 입원과 없는 입원이 어떻게 다른지 봄.
결측이 무작위가 아니면(MNAR) 영상 모델의 평가 대상이 편향됨.

    python scripts/run_phase1_5.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


RR_BOOT_N = 1000
RR_BOOT_SEED = 20260924   # 환자 단위 부트스트랩 시드


def main() -> int:
    from datapaths import RAW

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--with-model", action="store_true",
                    help="has_cxr 피처를 넣었을 때의 PR-AUC 변화 측정")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    # 위 한 줄이 모든 경고를 끄는데, 다운로드 대상 선택이 잘못됐다는 경고까지
    # 삼켜 버림. 이 경고만은 항상 보이게 되돌림.
    warnings.filterwarnings("always", category=RuntimeWarning)
    import numpy as np
    import pandas as pd

    from cohort import CohortConfig, build_cohort
    from features import ELECTIVE_ADMISSION_TYPES
    from modality import (
        diagnose_missingness,
        download_manifest,
        link_cxr_to_admissions,
        linked_studies,
    )

    raw = Path(args.raw_dir)
    print("=" * 74)
    print("  Phase 1.5: 결측 모달리티 진단 (M1-b)")
    print("=" * 74)

    adm_hits = list(raw.rglob("admissions.csv.gz"))
    meta_hits = list(raw.rglob("mimic-cxr-2.0.0-metadata.csv.gz"))
    if not adm_hits or not meta_hits:
        print("\n필요한 파일이 없습니다:")
        print(f"  admissions.csv.gz : {'있음' if adm_hits else '없음'}")
        print(f"  CXR metadata      : {'있음' if meta_hits else '없음  -> sh scripts/download_mimic.sh meta'}")
        return 1

    pat_hits = list(raw.rglob("patients.csv.gz"))
    if not pat_hits:
        print("\npatients.csv.gz 가 없어 **나이 필터 없이** 예비 분석으로 진행합니다.")
        print("   다운로드 완료 후 재실행해 확정할 것.\n")

    print("[1] 코호트 구성")
    adm = pd.read_csv(adm_hits[0], parse_dates=["admittime", "dischtime", "deathtime"])
    if pat_hits:
        pat = pd.read_csv(pat_hits[0])
    else:
        # 나이 필터를 건너뛰기 위한 최소 스텁 (anchor_age 없음 -> cohort 가 필터를 생략)
        pat = pd.DataFrame({"subject_id": adm.subject_id.unique()})
    coh = build_cohort(adm, pat, CohortConfig(elective_admission_types=ELECTIVE_ADMISSION_TYPES))
    print(coh.flow_table().to_string(index=False))
    s = coh.summary()
    print(f"\n    입원 {s['n_admissions']:,} / 환자 {s['n_subjects']:,} / "
          f"재입원률 {s['readmit_rate']:.4f}")

    print("\n[2] CXR 링킹 (재원 기간 중 촬영만)")
    # dicom_id 를 반드시 포함해야 함. 빠지면 linked_studies 가 "어느 장을 받을지"
    # 정하지 못해 download_manifest 의 정면 우선 선택이 경고 없이 꺼지고,
    # 알파벳순 첫 파일(측면일 수 있음)을 받게 됨.
    meta = pd.read_csv(
        meta_hits[0],
        usecols=["subject_id", "study_id", "dicom_id", "StudyDate", "StudyTime", "ViewPosition"],
    )
    print(f"    CXR 메타데이터 {len(meta):,}행 / 환자 {meta.subject_id.nunique():,}명")
    df = link_cxr_to_admissions(coh.df, meta)

    n = len(df)
    n_has = int(df.has_cxr.sum())
    print(f"    영상 보유 입원 {n_has:,} / {n:,} = {n_has / n * 100:.1f}%")
    print(f"    영상 없는 입원 {n - n_has:,} = {(n - n_has) / n * 100:.1f}%  <- M1 의 대상")
    with_cxr = df[df.has_cxr == 1]
    if len(with_cxr):
        print(f"    영상 보유 시 촬영 횟수: 중앙값 {with_cxr.n_cxr.median():.0f} / "
              f"평균 {with_cxr.n_cxr.mean():.2f} / 최대 {with_cxr.n_cxr.max()}")
        print(f"    마지막 촬영~퇴원 간격: 중앙값 {with_cxr.hours_cxr_before_discharge.median():.1f}시간")
        print(f"    AP 촬영 포함 비율: {with_cxr.cxr_view_ap.mean() * 100:.1f}%")

    print("\n[3] 영상 유무별 재입원률")
    for v, label in [(1, "영상 있음"), (0, "영상 없음")]:
        sub = df[df.has_cxr == v]
        if len(sub):
            print(f"    {label}: n={len(sub):>7,}  재입원률 {sub.readmit_30d.mean() * 100:5.2f}%")

    print("\n[4] MNAR 진단: 점이연 상관 (Bonferroni 보정)")
    diag = diagnose_missingness(df)
    corr = diag.correlations
    if not corr.empty:
        print(f"    {'변수':24s} {'r':>8s} {'|r|':>7s} {'p(보정)':>12s}")
        for _, row in corr.iterrows():
            print(f"    {row.variable:24s} {row.r:8.4f} {row.abs_r:7.4f} {row.p_bonferroni:12.2e}")
    print(f"\n    검정 수 {diag.n_tests}개")

    if diag.effect:
        e = diag.effect
        print("\n    상대위험도: 집단 불균형에서 r 이 축소되는 문제를 보완")
        print(f"      영상 있음 재입원률 {e['rate_exposed'] * 100:.2f}%  (n={e['n_exposed']:,})")
        print(f"      영상 없음 재입원률 {e['rate_unexposed'] * 100:.2f}%  (n={e['n_unexposed']:,})")
        print(f"      위험비 RR = {e['risk_ratio']:.3f}  "
              f"95% CI [{e['rr_ci_low']:.3f}, {e['rr_ci_high']:.3f}]")
        print(f"      오즈비 OR = {e['odds_ratio']:.3f}")

    # 교란 보정: 재원일수가 has_cxr 과 결과 양쪽에 연결되는 교란변수임
    # 보정 RR 은 README 의 주 수치라 결과 JSON 에도 남김.
    los_adjusted = None
    los_strata = None
    if "los_days" in df.columns:
        from modality import stratified_risk_ratio

        print("\n    재원일수 4분위 층화 (교란 보정)")
        q = pd.qcut(df.los_days, 4, labels=["Q1(단기)", "Q2", "Q3", "Q4(장기)"])
        tab, mh = stratified_risk_ratio(df.has_cxr.values, df.readmit_30d.values, q.values)
        los_strata = q.cat.codes.to_numpy()
        print(f"      {'층':10s} {'n':>9s} {'CXR률':>7s} {'RR':>7s} {'95% CI':>18s}")
        for _, r in tab.iterrows():
            print(f"      {r.stratum!s:10s} {r.n:9,} {r.exposure_rate * 100:6.1f}% "
                  f"{r.risk_ratio:7.3f}  [{r.rr_ci_low:.3f}, {r.rr_ci_high:.3f}]")
        crude = diag.effect.get("risk_ratio", float("nan"))
        print(f"\n      Mantel-Haenszel 요약 RR {mh:.3f}   (조 RR {crude:.3f}, 차이 {mh - crude:+.3f})")
        los_adjusted = {"stratifier": "los_days 4분위", "mh_risk_ratio": float(mh), "crude_risk_ratio": float(crude),
                        "strata": [{k: (str(v) if k == "stratum" else float(v)) for k, v in row.items()}
                                   for row in tab.to_dict(orient="records")]}
        if mh < crude:
            print("      -> 조 RR 은 재원일수 경로로 부풀려져 있다. 보정값을 함께 보고할 것.")

    # 위험비 구간을 환자 단위로 다시 측정함. 위 구간은 입원 415,231건을 독립으로 보는데 한 환자가 평균 2.29건이라
    # 같은 환자의 입원끼리 닮음. 환자마다 영상 있음, 없음 입원 수와 재입원 수를 모아 환자를 복원추출함.
    # 주 수치인 보정 RR(재원일수 4분위 MH)도 같은 복원추출에서 함께 측정함.
    rr_patient_boot = None
    if {"subject_id", "has_cxr", "readmit_30d"} <= set(df.columns):
        x = df.has_cxr.to_numpy().astype(np.int64)
        y = df.readmit_30d.to_numpy().astype(np.int64)
        codes, sid = np.unique(df.subject_id.to_numpy(), return_inverse=True)
        n_pat = len(codes)
        ne = np.bincount(sid, weights=x, minlength=n_pat)
        ee = np.bincount(sid, weights=x * y, minlength=n_pat)
        nu = np.bincount(sid, weights=1 - x, minlength=n_pat)
        eu = np.bincount(sid, weights=(1 - x) * y, minlength=n_pat)
        mh_parts = None
        if los_strata is not None:
            k = int(los_strata.max()) + 1
            cell = sid * k + los_strata

            def per_stratum(v):
                return np.bincount(cell, weights=v, minlength=n_pat * k).reshape(n_pat, k)

            # 층마다 환자별 (영상 있음 재입원, 영상 있음, 영상 없음 재입원, 영상 없음) 수
            mh_parts = [per_stratum(v) for v in (x * y, x, (1 - x) * y, 1 - x)]

        def mh_rr(w):
            a, n1, c, n0 = (w @ m for m in mh_parts)
            n = n1 + n0
            return float(np.sum(a * n0 / n) / np.sum(c * n1 / n))

        if mh_parts is not None and abs(mh_rr(np.ones(n_pat)) - los_adjusted["mh_risk_ratio"]) > 1e-9:
            raise SystemExit("환자별로 모은 수로 다시 낸 MH RR 이 층화 표의 값과 다르다.")
        brng = np.random.default_rng(RR_BOOT_SEED)
        rrs, mhs = [], []
        for _ in range(RR_BOOT_N):
            w = np.bincount(brng.integers(0, n_pat, n_pat), minlength=n_pat)
            rrs.append((w @ ee / (w @ ne)) / (w @ eu / (w @ nu)))
            if mh_parts is not None:
                mhs.append(mh_rr(w))
        lo, hi = np.percentile(rrs, [2.5, 97.5])
        if mhs:
            mlo, mhi = np.percentile(mhs, [2.5, 97.5])
            los_adjusted["patient_bootstrap"] = {"n_boot": RR_BOOT_N, "seed": RR_BOOT_SEED,
                                                 "ci_low": float(mlo), "ci_high": float(mhi)}
            print(f"\n    환자 단위 부트스트랩({RR_BOOT_N}회) 보정 RR 95% 구간 [{mlo:.3f}, {mhi:.3f}]")
        rr_patient_boot = {"n_boot": RR_BOOT_N, "seed": RR_BOOT_SEED, "n_patients": int(n_pat),
                           "admissions_per_patient": float(len(df) / n_pat), "ci_low": float(lo), "ci_high": float(hi)}
        print(f"\n    환자 단위 부트스트랩({RR_BOOT_N}회) 위험비 95% 구간 [{lo:.3f}, {hi:.3f}] "
              f"(환자 {n_pat:,}명, 환자당 입원 {len(df) / n_pat:.2f}건)")

    print(f"\n    판정: {diag.verdict()}")
    for note in diag.notes:
        print(f"    - {note}")

    print("\n    해석할 때 주의할 점")
    print("      has_cxr 이 성능을 올린다면 그것은 '영상의 내용'이 아니라")
    print("      '임상의가 영상을 찍기로 했다'는 진료 행위를 학습한 것이다.")
    print("      예측 시점(퇴원)에 알 수 있는 정보이긴 하나, 병원마다 오더 관행이")
    print("      다르면 일반화되지 않는다. 배포 시 정당성은 별도 논의가 필요하다.")

    if args.with_model:
        print("\n[5] has_cxr 피처를 EHR 모델에 넣었을 때의 변화")
        dx_hits = list(raw.rglob("diagnoses_icd.csv.gz"))
        if not dx_hits or not pat_hits:
            print("    diagnoses_icd.csv.gz / patients.csv.gz 가 필요합니다. 건너뜁니다.")
        else:
            from features import build_features, feature_columns
            from metrics import classification_metrics
            from models import build_tabular_model
            from splits import patient_split
            from tracking import ExperimentLogger

            dx = pd.read_csv(dx_hits[0])
            feat, _ = build_features(df, pat, dx, None, None)
            base_cols = [c for c in feature_columns(feat) if c not in ("has_cxr", "n_cxr",
                                                                       "cxr_view_ap")]
            y = feat["readmit_30d"].to_numpy()
            groups = feat["subject_id"].to_numpy()
            sp = patient_split(groups, test_size=0.2, val_size=0.1, seed=args.seed)
            n_pos = int(y[sp.train_idx].sum())
            spw = (len(sp.train_idx) - n_pos) / max(1, n_pos)

            results = {}
            for label, cols in [("EHR 단독", base_cols),
                                ("EHR + has_cxr", [*base_cols, "has_cxr"])]:
                X = feat[cols].to_numpy(dtype=np.float32)
                m = build_tabular_model("xgboost", scale_pos_weight=spw)
                m.fit(X[sp.train_idx], y[sp.train_idx])
                p = m.predict_proba(X[sp.test_idx])[:, 1]
                results[label] = classification_metrics(y[sp.test_idx], p)
                r = results[label]
                print(f"    {label:16s} AUROC {r.auroc:.4f}  PR-AUC {r.pr_auc:.4f}  ECE {r.ece:.4f}")

            d_auroc = results["EHR + has_cxr"].auroc - results["EHR 단독"].auroc
            d_pr = results["EHR + has_cxr"].pr_auc - results["EHR 단독"].pr_auc
            print(f"\n    변화량: AUROC {d_auroc:+.4f}  PR-AUC {d_pr:+.4f}")
            print("    -> 올랐다면 '영상 유무 자체가 신호'라는 선택 편향을 정량화한 것이고,")
            print("       안 올랐다면 그 편향이 이미 다른 EHR 피처(재원일수 등)에 담겨")
            print("       있었다는 뜻이다. 어느 쪽이든 영상의 '내용'과는 무관하다.")

            with ExperimentLogger("phase15-modality", params={"model": "xgboost",
                                                              "seed": args.seed}) as run:
                run.note("Phase 1.5: 결측 모달리티(M1-b): has_cxr 피처 효과")
                run.log_metrics({
                    "cxr_rate": diag.cxr_rate,
                    "outcome_r": diag.outcome_r,
                    "delta_auroc": d_auroc,
                    "delta_pr_auc": d_pr,
                })

    print("\n[6] Phase 2 다운로드 대상 산출")
    all_studies = linked_studies(coh.df, meta, one_per_admission=False)
    studies = linked_studies(coh.df, meta, one_per_admission=True)
    print(f"    재원 기간 중 촬영된 study {len(all_studies):,}개 "
          f"/ 환자 {all_studies.subject_id.nunique():,}명")
    print(f"    학습에 쓰는 것은 입원당 1건 -> {len(studies):,}개")
    print("    (전부 받으면 촬영 횟수가 많은 중환자가 학습에 여러 번 들어간다)")
    fn_hits = list(raw.rglob("IMAGE_FILENAMES"))
    manifest = {}
    if fn_hits:
        paths, manifest = download_manifest(studies, fn_hits[0])
        print(f"    IMAGE_FILENAMES 대조: study {manifest['studies_found']:,}"
              f"/{manifest['studies_requested']:,}  파일 {manifest['files']:,}장")
        print(f"    예상 용량 약 {manifest['est_gb']:,}GB  "
              f"(장당 {manifest['avg_kb']:,.0f}KB 실측)")
        out = ROOT / "outputs" / "phase2_download_list.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(paths) + "\n", encoding="utf-8")
        print(f"    저장: {out}")
    else:
        print("    IMAGE_FILENAMES 가 없어 건너뜁니다.")

    art = {
        "phase": "1.5",
        "cohort": {k: v for k, v in s.items()},
        "linkage": {
            "n_admissions": int(n),
            "n_with_cxr": int(n_has),
            "cxr_rate": float(n_has / n),
            "n_studies_all": len(all_studies),
            # 입원당 1건으로 줄인 뒤의 study 수.
            "n_studies_used": len(studies),
            "n_subjects_with_cxr": int(studies.subject_id.nunique()),
        },
        "readmit_by_cxr": {
            "with_cxr": float(df[df.has_cxr == 1].readmit_30d.mean()),
            "without_cxr": float(df[df.has_cxr == 0].readmit_30d.mean()),
        },
        "mnar": {
            "verdict": diag.verdict(),
            # verdict 는 진단 단계의 조 RR 로 쓴 문장임. 문서가 주 수치로 쓰는 것은 보정 RR 이라 따로 적음.
            "verdict_adjusted": (
                f"재원일수 4분위 보정 RR={los_adjusted['mh_risk_ratio']:.2f} "
                f"[{los_adjusted['patient_bootstrap']['ci_low']:.2f}, {los_adjusted['patient_bootstrap']['ci_high']:.2f}] "
                "(Mantel-Haenszel, 환자 단위 부트스트랩)"
                if los_adjusted and "patient_bootstrap" in los_adjusted else None),
            "cxr_rate": float(diag.cxr_rate),
            "outcome_r": float(diag.outcome_r),
            "n_tests": int(diag.n_tests),
            "effect": {k: float(v) for k, v in (diag.effect or {}).items()},
            "correlations": diag.correlations.to_dict(orient="records"),
            "notes": list(diag.notes),
            "los_adjusted": los_adjusted,
            "rr_patient_bootstrap": rr_patient_boot,
        },
        "download": manifest,
    }
    if len(with_cxr):
        art["linkage"]["n_cxr_median"] = float(with_cxr.n_cxr.median())
        art["linkage"]["ap_rate"] = float(with_cxr.cxr_view_ap.mean())
    p = ROOT / "outputs" / "phase1_5_metrics.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(art, ensure_ascii=False, indent=2, default=float),
                 encoding="utf-8")
    print(f"\n  저장: {p}")

    print("\n" + "=" * 74)
    print("  Phase 1.5 완료")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
