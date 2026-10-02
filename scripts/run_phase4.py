"""Phase 4: 신뢰성 감사: V4 서브그룹, 보정, SHAP, M3 DCA, 모델 비교.

Phase 3 에서 영상은 EHR 에 보태지 않음. 그래서 실제로 쓸 모델은
EHR 단독임. Phase 4 는 그 모델을 "숫자가 좋다"가 아니라 "이 숫자를 임상에서
믿고 쓸 수 있는가"의 관점에서 뜯음.

  V4   특정 집단에서 성능이 떨어지지 않는가. 정확도 차이보다 보정 차이를 봄.
  보정 예측확률을 확률로 읽어도 되는가. val 로 교정하고 test 로만 측정함.
  SHAP 무엇을 보고 판단하는가. 누수 의심 변수가 상위에 있으면 그 자체가 경보임.
  M3   임계값을 어디에 두든 아무것도 안 하는 것보다 나은 구간이 있는가(순편익).
  비교 모델, 하이퍼파라미터를 바꾸면 결론이 흔들리는가.

    python scripts/run_phase4.py            # 전체
    python scripts/run_phase4.py --check    # 입력 준비 상태만 확인

## 왜 보정을 val 로 맞추는가

test 로 교정하면 그 test 성능은 더 이상 미래 성능의 추정치가 아님. 교정기도
모델의 일부이므로 학습에 쓰인 데이터에서 평가하면 안 됨.

## DUA

지표 JSON 에는 환자 단위 정보가 들어가지 않음. SHAP 은 피처 이름과 집계된
기여도만 남김.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapaths import RAW  # noqa: E402

OUT_DIR = ROOT / "outputs"
METRICS_PATH = OUT_DIR / "phase4_metrics.json"

# 판정선: 결과를 보기 전에 고정함.
ECE_ACCEPTABLE = 0.05      # 이 아래면 확률로 읽어도 된다고 봄
SUBGROUP_GAP_BIG = 0.05    # 그룹 간 AUROC 격차가 이 이상이면 "집단 간 차이 있음"
DCA_MIN_RANGE = 0.02       # 순편익이 양수인 임계값 구간이 이만큼은 돼야 쓸모 있음


def _net_benefit(y, p, thr):
    """순편익 = TP/n − FP/n x (pt/(1−pt)).  (Vickers & Elkin 2006)"""
    pred = p >= thr
    n = len(y)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    w = thr / (1.0 - thr) if thr < 1 else float("inf")
    return tp / n - fp / n * w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shap-sample", type=int, default=3000)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd

    from cohort import CohortConfig, build_cohort, load_admissions
    from features import ELECTIVE_ADMISSION_TYPES, build_features, feature_columns
    from metrics import (
        MIN_SUBGROUP_N,
        classification_metrics,
        reliability_curve,
        subgroup_metrics,
    )
    from models import build_tabular_model
    from splits import patient_split
    from tracking import ExperimentLogger

    raw = Path(args.raw_dir)
    t0 = time.time()
    print("=" * 74)
    print("  Phase 4: 신뢰성 감사 (V4, 보정, SHAP, M3 DCA, 모델 비교)")
    print("=" * 74)

    print("\n[0] 입력 점검")
    dx_hits = list(raw.rglob("diagnoses_icd.csv.gz"))
    adm_hits = list(raw.rglob("admissions.csv.gz"))
    for name, hits in (("admissions", adm_hits), ("diagnoses_icd", dx_hits)):
        if not hits:
            print(f"  {name} 파일이 없다. download_mimic.sh 를 먼저 실행할 것.")
            return 1
        print(f"  [OK] {name}")
    if args.check:
        print("\n준비 완료.")
        return 0

    print("\n[1] 코호트 + 피처 + 분할")
    adm, pat = load_admissions(raw)
    coh = build_cohort(adm, pat,
                       CohortConfig(elective_admission_types=ELECTIVE_ADMISSION_TYPES))
    dx = pd.read_csv(dx_hits[0])
    # 투약, 시술 테이블은 읽지 않음(rx=None, pr=None). Phase 1 이 42개 피처로
    # 재고 여기는 32개로 재는 차이가 여기서 생김. 투약 9개를 빼면 AUROC 가
    # 0.7075 -> 0.7040 으로 0.0036 내려가는데(run_phase1.py --no-prescriptions),
    # 이 단계는 보정, 서브그룹, SHAP, 결정곡선 때문에 모델을 여러 번 적합하므로
    # 1,700만 행을 매번 읽는 값이 그만큼 되지 않음. 대신 이 절의 모든 수치는 32개 모델 기준임.
    # 영상 유무 관련 네 개는 Phase 3 에서 따로 다루므로 여기서 뺌.
    feat, _ = build_features(coh.df, pat, dx, None, None)
    cols = [c for c in feature_columns(feat)
            if c not in ("has_cxr", "n_cxr", "cxr_view_ap", "is_ap")]
    X = feat[cols].to_numpy(dtype=np.float32)
    y = feat["readmit_30d"].to_numpy().astype(int)
    g = feat["subject_id"].to_numpy()
    sp = patient_split(g, seed=args.seed)
    print(f"    {len(feat):,} 입원 / 피처 {len(cols)}개")
    print(f"    train {len(sp.train_idx):,} / val {len(sp.val_idx):,} "
          f"/ test {len(sp.test_idx):,}")

    n_pos = int(y[sp.train_idx].sum())
    pw = float((len(sp.train_idx) - n_pos) / max(n_pos, 1))

    results: dict = {"phase": "4", "seed": args.seed, "n_features": len(cols)}

    print("\n[2] 주 모델 (EHR 단독: Phase 3 에서 영상이 기여하지 않았으므로)")
    model = build_tabular_model("xgboost", scale_pos_weight=pw)
    model.fit(X[sp.train_idx], y[sp.train_idx])
    p_val = model.predict_proba(X[sp.val_idx])[:, 1]
    p_test = model.predict_proba(X[sp.test_idx])[:, 1]
    m_main = classification_metrics(y[sp.test_idx], p_test)
    print(f"    AUROC {m_main.auroc:.4f}  PR-AUC {m_main.pr_auc:.4f} "
          f"(무작위 {m_main.prevalence_baseline_pr_auc:.4f})  ECE {m_main.ece:.4f}")
    results["main_model"] = m_main.as_dict()

    # ------------------------------------------------------------ V4
    print("\n[3] V4: 서브그룹 (정확도보다 보정 차이를 본다)")
    te = sp.test_idx
    axes: dict[str, np.ndarray] = {}
    if "is_female" in feat.columns:
        axes["성별"] = np.where(feat["is_female"].to_numpy()[te] == 1, "여", "남")
    if "age" in feat.columns:
        a = feat["age"].to_numpy()[te]
        axes["연령대"] = np.select(
            [a < 45, a < 65, a < 80], ["~44", "45~64", "65~79"], default="80+")
    # 원본 범주형 컬럼은 결측(float NaN)과 문자열이 섞여 있을 수 있고, 병합으로
    # 같은 이름이 둘일 수도 있음. 그대로 np.unique 에 넣으면 str 과 float 을
    # 비교하다 죽음. 한 컬럼만 골라 전부 문자열로 강제함.
    def _labels(col_name: str) -> np.ndarray:
        obj = feat[col_name]
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:, 0]
        vals = np.asarray(obj)[te]
        return np.array(["결측" if pd.isna(v) else str(v) for v in vals], dtype=object)

    for raw_col, label in (("insurance", "보험"), ("admission_type", "입원유형")):
        if raw_col in feat.columns:
            axes[label] = _labels(raw_col)

    sub_out: dict[str, dict] = {}
    for name, grp in axes.items():
        n_uniq = len(set(map(str, grp)))
        if n_uniq < 2:
            print(f"    [{name}] 범주가 1개뿐이라 건너뜀")
            continue
        res = subgroup_metrics(y[te], p_test, grp)
        rows = {k: v.as_dict() for k, v in res.items()}
        small = [k for k, v in res.items() if v.n < MIN_SUBGROUP_N]
        big = {k: v for k, v in res.items() if v.n >= MIN_SUBGROUP_N}
        aurocs = [v.auroc for v in big.values()]
        gap = float(max(aurocs) - min(aurocs)) if len(aurocs) > 1 else 0.0
        eces = [v.ece for v in big.values()]
        ece_gap = float(max(eces) - min(eces)) if len(eces) > 1 else 0.0
        sub_out[name] = {
            "groups": rows, "auroc_gap": gap, "ece_gap": ece_gap,
            "small_groups": small,
            "verdict": ("집단 간 차이 있음" if gap >= SUBGROUP_GAP_BIG
                        else "집단 간 차이 없음"),
        }
        print(f"    [{name}] AUROC 격차 {gap:.4f}  ECE 격차 {ece_gap:.4f}"
              f"  -> {sub_out[name]['verdict']}")
        for k, v in sorted(res.items(), key=lambda kv: -kv[1].n):
            mark = "  (표본 부족)" if v.n < MIN_SUBGROUP_N else ""
            print(f"       {k:<12} n={v.n:>7,}  AUROC {v.auroc:.4f}  "
                  f"ECE {v.ece:.4f}{mark}")
    results["v4_subgroups"] = sub_out

    # ------------------------------------------------------- Calibration
    print("\n[4] 보정: val 로 교정하고 test 로만 잰다")
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, y[sp.val_idx])
    p_iso = iso.predict(p_test)

    platt = LogisticRegression(max_iter=1000)
    platt.fit(p_val.reshape(-1, 1), y[sp.val_idx])
    p_platt = platt.predict_proba(p_test.reshape(-1, 1))[:, 1]

    # 어느 교정을 쓸지는 val 안에서 고름. test ECE 로 고르면 선택이 test 에 기대고 보고 ECE 가
    # 낙관적이 됨. 교정기를 val 전체로 맞춘 뒤 val 로 재면 자기 데이터라 Isotonic 이 유리하므로,
    # val 을 환자 단위 5겹으로 나눠 밖 겹 예측으로 측정함.
    from metrics import val_oof_calibration_ece
    val_sel = val_oof_calibration_ece(p_val, y[sp.val_idx], g[sp.val_idx])
    print("    선택(val 5겹 밖 예측 ECE): "
          + "  ".join(f"{k} {v:.4f}" for k, v in val_sel.items()))

    cal = {}
    for name, pp in (("원본", p_test), ("Platt", p_platt), ("Isotonic", p_iso)):
        mm = classification_metrics(y[te], pp)
        cal[name] = {"ece": mm.ece, "brier": mm.brier, "auroc": mm.auroc}
        print(f"    {name:<9} ECE {mm.ece:.4f}  Brier {mm.brier:.4f}  "
              f"AUROC {mm.auroc:.4f}  (test)")
    best = min(val_sel, key=val_sel.get)
    results["calibration_selection"] = {
        "rule": "val 을 환자 단위 5겹으로 나눠 밖 겹 예측의 ECE 가 가장 낮은 방식",
        "val_oof_ece": val_sel,
    }
    results["calibration"] = cal
    results["calibration_best"] = best
    results["calibration_verdict"] = (
        "확률로 읽어도 된다" if cal[best]["ece"] < ECE_ACCEPTABLE
        else "확률로 읽으면 안 된다")
    print(f"    최선 {best} (ECE {cal[best]['ece']:.4f}) "
          f"-> {results['calibration_verdict']}")
    # 순위는 그대로인데 확률만 교정되는지 확인: AUROC 가 유지돼야 정상임.
    results["calibration_preserves_ranking"] = bool(
        abs(cal[best]["auroc"] - cal["원본"]["auroc"]) < 1e-6
        or best == "원본")

    p_for_curve = {"원본": p_test, "Platt": p_platt, "Isotonic": p_iso}[best]
    results["reliability_curve"] = reliability_curve(y[te], p_for_curve)

    # 배포되는 확률은 교정된 쪽임. 집단별 보정도 그 확률로 다시 재야
    # "이 집단에서도 확률을 믿어도 되는가"에 답이 됨. 원본으로만 재면
    # 전 집단이 똑같이 나쁜 상태에서의 격차라 임상적 의미가 약함.
    print(f"    집단별 보정 재검(교정 후: {best})")
    sub_cal: dict[str, dict] = {}
    for name, grp in axes.items():
        if len(set(map(str, grp))) < 2:
            continue
        res_c = subgroup_metrics(y[te], p_for_curve, grp)
        big_c = {k: v for k, v in res_c.items() if v.n >= MIN_SUBGROUP_N}
        eces_c = [v.ece for v in big_c.values()]
        gap_c = float(max(eces_c) - min(eces_c)) if len(eces_c) > 1 else 0.0
        sub_cal[name] = {"groups": {k: v.as_dict() for k, v in res_c.items()},
                         "ece_gap": gap_c}
        raw_gap = sub_out[name]["ece_gap"]
        print(f"       {name:<8} ECE 격차 {raw_gap:.4f}(원본) -> {gap_c:.4f}(교정)")
    results["v4_subgroups_calibrated"] = sub_cal
    worst_cal = max((v["ece_gap"] for v in sub_cal.values()), default=0.0)
    results["v4_calibrated_ece_gap_max"] = worst_cal
    results["v4_calibrated_verdict"] = (
        "모든 집단에서 확률을 믿어도 된다" if worst_cal < ECE_ACCEPTABLE
        else "일부 집단에서 확률이 어긋난다")
    print(f"       최대 격차 {worst_cal:.4f} -> {results['v4_calibrated_verdict']}")

    # -------------------------------------------------------------- SHAP
    print("\n[5] SHAP: 무엇을 보고 판단하는가")
    import shap
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(te, size=min(args.shap_sample, len(te)), replace=False)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X[idx])
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1]
    top = [{"feature": cols[i], "mean_abs_shap": float(imp[i])} for i in order[:15]]
    results["shap_top15"] = top
    results["shap_sample_n"] = len(idx)
    for r in top[:10]:
        print(f"    {r['feature']:<24} {r['mean_abs_shap']:.4f}")

    # 누수 의심 변수가 상위에 있으면 그 자체가 경보임.
    SUSPECT = ("los_days", "los_log", "n_diagnoses", "n_procedures")
    flagged = [r["feature"] for r in top[:5] if r["feature"] in SUSPECT]
    results["shap_suspect_in_top5"] = flagged
    if flagged:
        print(f"    [경고] 상위 5개에 재원기간, 건수 계열 {flagged}: "
              "결과와 가까운 변수라 해석에 주의")

    # --------------------------------------------------------- M3 DCA
    print("\n[6] M3: 결정곡선(순편익)")
    p_best = {"원본": p_test, "Platt": p_platt, "Isotonic": p_iso}[best]
    yt = y[te]
    prev = float(yt.mean())
    thrs = np.arange(0.05, 0.51, 0.01)
    curve = []
    positive_range = []
    for t in thrs:
        nb_m = _net_benefit(yt, p_best, float(t))
        nb_all = prev - (1 - prev) * (t / (1 - t))
        nb_none = 0.0
        curve.append({"threshold": float(t), "model": nb_m,
                      "treat_all": float(nb_all), "treat_none": nb_none})
        if nb_m > max(nb_all, nb_none):
            positive_range.append(float(t))
    width = (max(positive_range) - min(positive_range)) if positive_range else 0.0
    results["dca_curve"] = curve
    results["dca_useful_threshold_range"] = (
        [min(positive_range), max(positive_range)] if positive_range else [])
    results["dca_range_width"] = float(width)
    results["dca_verdict"] = ("쓸모 있는 임계값 구간이 있다" if width >= DCA_MIN_RANGE
                              else "아무것도 안 하는 것보다 나은 구간이 없다")
    if positive_range:
        print(f"    모델이 우세한 임계값 {min(positive_range):.2f}~"
              f"{max(positive_range):.2f} (폭 {width:.2f})")
    else:
        print("    모델이 우세한 임계값 구간 없음")
    print(f"    판정: {results['dca_verdict']}")

    # ---------------------------------------------------------- 모델 비교
    print("\n[7] 모델, 하이퍼파라미터 비교 (결론이 흔들리는가)")
    comp = {}
    failed: dict[str, str] = {}
    variants = [
        ("xgboost 기본", lambda: build_tabular_model("xgboost", scale_pos_weight=pw)),
        ("xgboost 얕게(depth3)",
         lambda: build_tabular_model("xgboost", scale_pos_weight=pw, max_depth=3)),
        ("xgboost 깊게(depth6)",
         lambda: build_tabular_model("xgboost", scale_pos_weight=pw, max_depth=6)),
        ("lightgbm", lambda: build_tabular_model("lightgbm", scale_pos_weight=pw)),
    ]
    for name, factory in variants:
        try:
            mdl = factory()
            mdl.fit(X[sp.train_idx], y[sp.train_idx])
            pp = mdl.predict_proba(X[te])[:, 1]
            mm = classification_metrics(yt, pp)
            comp[name] = mm.as_dict()
            print(f"    {name:<22} AUROC {mm.auroc:.4f}  PR-AUC {mm.pr_auc:.4f}"
                  f"  ECE {mm.ece:.4f}")
        except Exception as e:
            # 실패를 산출물에 남김. 화면에만 찍으면 JSON 에서 그 모델이 빠지고
            # AUROC 폭이 남은 모델로만 계산됨.
            failed[name] = f"{type(e).__name__}: {e}"
            print(f"    {name:<22} 실패: {type(e).__name__}")
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lr = make_pipeline(StandardScaler(),
                       LR(max_iter=2000, class_weight="balanced"))
    Xi = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    lr.fit(Xi[sp.train_idx], y[sp.train_idx])
    mm = classification_metrics(yt, lr.predict_proba(Xi[te])[:, 1])
    comp["로지스틱(참고)"] = mm.as_dict()
    print(f"    {'로지스틱(참고)':<22} AUROC {mm.auroc:.4f}  PR-AUC {mm.pr_auc:.4f}"
          f"  ECE {mm.ece:.4f}")

    aurocs = [v["auroc"] for v in comp.values()]
    spread = float(max(aurocs) - min(aurocs))
    results["model_comparison"] = comp
    results["model_comparison_failed"] = failed
    results["model_auroc_spread"] = spread
    results["model_comparison_verdict"] = (
        "모델 선택으로 결론이 바뀌지 않는다" if spread < SUBGROUP_GAP_BIG
        else "모델 선택이 결론을 바꾼다")
    print(f"    AUROC 폭 {spread:.4f} -> {results['model_comparison_verdict']}")

    # ------------------------------------------------------------- 저장
    print("\n[8] 저장")
    results["elapsed_sec"] = round(time.time() - t0, 1)
    METRICS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"    {METRICS_PATH}")

    try:
        with ExperimentLogger(
            "phase4-reliability",
            params={"phase": "4", "seed": args.seed},
            run_name=f"reliability-seed{args.seed}",
        ) as log:
            log.log_metrics({"auroc": m_main.auroc, "pr_auc": m_main.pr_auc,
                             "ece_raw": cal["원본"]["ece"],
                             "ece_best": cal[best]["ece"],
                             "dca_range_width": width,
                             "model_spread": spread})
    except Exception as e:
        print(f"    (실험 기록 건너뜀: {type(e).__name__})")

    print("\n" + "=" * 74)
    print("  Phase 4 완료")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
