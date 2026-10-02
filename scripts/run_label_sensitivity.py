"""CMS식 재입원 라벨 민감도. 라벨만 바꿔 Phase 1 EHR 모델과 Phase 3 융합을 다시 측정함.

주 라벨은 `admissions` 의 모든 다음 입원을 재입원으로 셈(관찰 입원 포함, README "CMS 재입원 정의와 다른 점").
CMS 병원 전체 재입원 측정에 가깝게 바꾼 라벨로도 두 결론(Phase 1 AUROC 가 문헌 범위 안, 영상이 EHR 에 보태지 않음)이
유지되는지 봄. 결과는 `outputs/label_sensitivity_cms.json` 에 남김.

라벨 변형(누적):
  main      주 라벨(검산용. Phase 1, Phase 3 결과와 같은 값이 나와야 함. 다르면 멈춤)
  obs_both  관찰 입원(EU OBSERVATION, OBSERVATION ADMIT, DIRECT OBSERVATION, AMBULATORY OBSERVATION)을 index 에서 빼고,
            재입원도 관찰 입원이 아닌 다음 입원만 셈
  cms       obs_both + index 에서 다른 급성기 병원 전원(ACUTE HOSPITAL), 자의 퇴원(AGAINST ADVICE) 제외
            + 다음 입원이 퇴원 당일이면 전원으로 보고 재입원으로 세지 않음
분할(환자 단위 seed 42), 피처, 모델(XGBoost), 예정 재입원 규칙은 주 분석과 같음. 융합은 Phase 3 과 같은 환자 배정을
쓰고 CMS 에서 빠진 행만 지움(인코더가 본 환자가 test 로 넘어가지 않게 분할을 다시 뽑지 않음).

    python scripts/run_label_sensitivity.py     # 결과: outputs/label_sensitivity_cms.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cohort import CohortConfig, build_cohort, load_admissions  # noqa: E402
from datapaths import RAW  # noqa: E402
from features import ELECTIVE_ADMISSION_TYPES, build_features, feature_columns  # noqa: E402
from metrics import classification_metrics  # noqa: E402
from models import build_tabular_model  # noqa: E402
from splits import patient_kfold, patient_split, split_with_fixed_folds, subject_folds  # noqa: E402

OUT = ROOT / "outputs" / "label_sensitivity_cms.json"
OBS = ("EU OBSERVATION", "OBSERVATION ADMIT", "DIRECT OBSERVATION", "AMBULATORY OBSERVATION")
EXCLUDED_DISCHARGE = ("ACUTE HOSPITAL", "AGAINST ADVICE")
SEED, PCA_DIM = 42, 32
IMAGE_DERIVED = ("has_cxr", "n_cxr", "cxr_view_ap", "is_ap")


def next_inpatient(adm: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """index 입원마다 퇴원 시각 이후 처음 시작한, 관찰 입원이 아닌 입원."""
    inp = adm[~adm.admission_type.isin(OBS)].sort_values(["subject_id", "admittime", "hadm_id"])
    cand = main[["hadm_id", "subject_id", "admittime", "dischtime", "admission_type", "discharge_location"]]
    return pd.merge_asof(
        cand.sort_values("dischtime").rename(columns={"hadm_id": "idx_hadm"}),
        inp[["subject_id", "admittime", "admission_type", "hadm_id"]].rename(
            columns={"admittime": "n_admit", "admission_type": "n_type", "hadm_id": "n_hadm"}).sort_values("n_admit"),
        left_on="dischtime", right_on="n_admit", by="subject_id", direction="forward", allow_exact_matches=True,
    ).set_index("idx_hadm")


def main_() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    from sklearn.decomposition import PCA

    from run_phase3 import EMBEDDINGS, LIFT_REAL

    raw = Path(RAW)
    adm, pat = load_admissions(raw)
    coh = build_cohort(adm, pat, CohortConfig(elective_admission_types=ELECTIVE_ADMISSION_TYPES))
    main = coh.df

    # 주 라벨과 CMS 정의의 차이를 주 코호트에서 셈(docs/report.md "라벨 정의. CMS 재입원 정의와 다른 점" 표의 크기 열)
    pos = main[main.readmit_30d == 1]
    diff = {"n_index": len(main), "n_positive": len(pos),
            "positive_next_is_observation": int(pos.next_admission_type.isin(OBS).sum()),
            "index_transfer_acute_hospital": int((main.discharge_location == "ACUTE HOSPITAL").sum()),
            "index_against_advice": int((main.discharge_location == "AGAINST ADVICE").sum())}
    # 퇴원처가 DIED 인데 사망 시각도 hospital_expire_flag 도 없어 코호트에 남은 입원. 그중 퇴원 뒤 30일 안에 다시
    # 입원한 것이 있어 기록 불일치로 보고, 원내 사망 판정은 사망 시각과 flag 로만 함(README, src/cohort.py).
    died = main[main.discharge_location == "DIED"]
    gap_next = died.days_to_next_admission
    diff["discharge_died_without_death_record"] = len(died)
    diff["discharge_died_with_deathtime_or_flag"] = int(died.deathtime.notna().sum()
                                                        + (died.hospital_expire_flag == 1).sum())
    diff["discharge_died_next_admission_within_30d"] = int(((gap_next >= 0) & (gap_next < 30)).sum())
    diff["discharge_died_readmit_30d_label"] = int(died.readmit_30d.sum())
    print("차이:", json.dumps(diff, ensure_ascii=False))

    nxt = next_inpatient(adm, main)
    gap = (nxt.n_admit - nxt.dischtime).dt.total_seconds() / 86400.0
    planned = nxt.n_type.isin(ELECTIVE_ADMISSION_TYPES)
    same_day = nxt.n_admit.dt.normalize() == nxt.dischtime.dt.normalize()
    lab_obs = (gap.notna() & (gap < 30) & ~planned).astype(int)
    lab_cms = (gap.notna() & (gap < 30) & ~planned & ~same_day).astype(int)
    keep_obs = ~main.admission_type.isin(OBS)
    keep_cms = keep_obs & ~main.discharge_location.isin(EXCLUDED_DISCHARGE)

    # 1. Phase 1 EHR 모델(주 분석과 같은 피처: 진단, 처방, 시술)
    dx = pd.read_csv(next(raw.rglob("diagnoses_icd.csv.gz")))
    rx = pd.read_csv(next(raw.rglob("prescriptions.csv.gz")), usecols=["hadm_id", "drug"])
    pr = pd.read_csv(next(raw.rglob("procedures_icd.csv.gz")))
    variants = {
        "main": main.assign(y=main.readmit_30d),
        "obs_both": main[keep_obs].assign(y=lambda d: lab_obs.reindex(d.hadm_id).to_numpy()),
        "cms": main[keep_cms].assign(y=lambda d: lab_cms.reindex(d.hadm_id).to_numpy()),
    }
    phase1 = {}
    for name, df in variants.items():
        feat, _ = build_features(df.drop(columns=["readmit_30d"]).rename(columns={"y": "readmit_30d"}), pat, dx, rx, pr)
        cols = feature_columns(feat)
        y = feat["readmit_30d"].to_numpy()
        sp = patient_split(feat["subject_id"].to_numpy(), test_size=0.2, val_size=0.1, seed=SEED)
        x = feat[cols].to_numpy(dtype=np.float32)
        n_pos = int(y[sp.train_idx].sum())
        model = build_tabular_model("xgboost", scale_pos_weight=(len(sp.train_idx) - n_pos) / max(1, n_pos))
        model.fit(x[sp.train_idx], y[sp.train_idx])
        m = classification_metrics(y[sp.test_idx], model.predict_proba(x[sp.test_idx])[:, 1])
        phase1[name] = {"n_index": len(feat), "readmit_rate": float(y.mean()), "n_pos": int(y.sum()),
                        "test_auroc": float(m.auroc), "test_pr_auc": float(m.pr_auc), "n_features": len(cols)}
        print("Phase 1", name, json.dumps(phase1[name], ensure_ascii=False), flush=True)
    counts = {"obs_excluded_index": int((~keep_obs).sum()),
              "transfer_ama_excluded_index": int((keep_obs & ~keep_cms).sum()),
              "same_day_not_counted": int((gap.notna() & (gap < 30) & ~planned & same_day)
                                          .reindex(main[keep_cms].hadm_id).sum())}

    ref1 = json.loads((ROOT / "outputs" / "phase1_metrics.json").read_text(encoding="utf-8"))["test"]["auroc"]
    if abs(phase1["main"]["test_auroc"] - ref1) > 1e-9:
        raise SystemExit(f"주 라벨 Phase 1 AUROC 가 phase1_metrics.json 과 다르다: {phase1['main']['test_auroc']} vs {ref1}")

    # 2. Phase 3 융합(Phase 3 과 같은 피처: 진단만, 영상에서 나온 피처는 뺌)
    feat, _ = build_features(main, pat, dx, None, None)
    ehr_cols = [c for c in feature_columns(feat) if c not in IMAGE_DERIVED]
    x_ehr = feat[ehr_cols].to_numpy(dtype=np.float32)
    g_full = feat["subject_id"].to_numpy()
    idx = main.set_index("hadm_id")
    elig_cms = ~idx.admission_type.isin(OBS) & ~idx.discharge_location.isin(EXCLUDED_DISCHARGE)
    labels = {
        "main": (feat["readmit_30d"].to_numpy().astype(int), np.ones(len(feat), dtype=bool)),
        "cms": (feat["hadm_id"].map(lab_cms).fillna(0).to_numpy().astype(int),
                feat["hadm_id"].map(elig_cms).fillna(False).to_numpy().astype(bool)),
    }
    fusion = {}
    for enc, path in EMBEDDINGS.items():
        z = np.load(path, allow_pickle=False)
        emb_all, emb_hadm = z["embedding"].astype(np.float32), z["hadm_id"]
        row = feat["hadm_id"].map(pd.Series(np.arange(len(emb_hadm)), index=emb_hadm))
        has_img = row.notna().to_numpy()
        emb = np.zeros((len(feat), emb_all.shape[1]), dtype=np.float32)
        emb[has_img] = emb_all[row[has_img].to_numpy().astype(int)]
        sub = np.where(has_img)[0]
        sp_sub = patient_split(g_full[sub], seed=SEED)
        split_with_fixed_folds(g_full, subject_folds(g_full[sub], sp_sub), seed=SEED)
        train0, test0 = sub[sp_sub.train_idx], sub[sp_sub.test_idx]
        for lab, (y, keep) in labels.items():
            train, test = train0[keep[train0]], test0[keep[test0]]

            def pw(i, y=y):
                n_pos = int(y[i].sum())
                return float((len(i) - n_pos) / max(n_pos, 1))

            pca = PCA(n_components=PCA_DIM, random_state=SEED)
            pca.fit(emb[train])
            emb_p = pca.transform(emb).astype(np.float32)

            def fit_eval(x, y=y, train=train, test=test):
                m = build_tabular_model("xgboost", scale_pos_weight=pw(train))
                m.fit(x[train], y[train])
                return float(classification_metrics(y[test], m.predict_proba(x[test])[:, 1]).auroc)

            a_ehr = fit_eval(x_ehr)
            a_pca = fit_eval(np.column_stack([x_ehr, emb_p]).astype(np.float32))
            a_raw = fit_eval(np.column_stack([x_ehr, emb]).astype(np.float32))
            oof = np.zeros(len(feat), dtype=np.float32)
            for tr_rel, va_rel in patient_kfold(g_full[train], n_splits=5):
                tr_i, va_i = train[tr_rel], train[va_rel]
                m_f = build_tabular_model("xgboost", scale_pos_weight=pw(tr_i))
                m_f.fit(emb_p[tr_i], y[tr_i])
                oof[va_i] = m_f.predict_proba(emb_p[va_i])[:, 1]
            m_all = build_tabular_model("xgboost", scale_pos_weight=pw(train))
            m_all.fit(emb_p[train], y[train])
            oof[test] = m_all.predict_proba(emb_p[test])[:, 1]
            a_late = fit_eval(np.column_stack([x_ehr, oof.reshape(-1, 1)]).astype(np.float32))
            best = max(a_pca - a_ehr, a_raw - a_ehr, a_late - a_ehr)
            fusion[f"{enc}_{lab}"] = {"n_train": len(train), "n_test": len(test),
                                      "pos_rate_test": float(y[test].mean()), "ehr_only": a_ehr, "fusion_pca": a_pca,
                                      "fusion_raw": a_raw, "fusion_late": a_late, "best_lift": float(best),
                                      "verdict": "영상이 보탠 것" if best >= LIFT_REAL else "영상이 보태지 않는다"}
            print("융합", enc, lab, json.dumps(fusion[f"{enc}_{lab}"], ensure_ascii=False), flush=True)

    # 주 라벨 융합은 Phase 3 결과와 같아야 함
    for enc, fname in (("cnn", "phase3_metrics.json"), ("raddino", "phase3_metrics_raddino.json")):
        key = f"{enc}_main"
        if key not in fusion:
            continue
        ref = json.loads((ROOT / "outputs" / fname).read_text(encoding="utf-8"))["fusion_best_lift"]
        if abs(fusion[key]["best_lift"] - ref) > 1e-9:
            raise SystemExit(f"{key} 융합 lift 가 {fname} 과 다르다: {fusion[key]['best_lift']} vs {ref}")

    out = {"definition_diff": diff, "phase1": phase1, "exclusions": counts, "fusion": fusion,
           "lift_threshold": LIFT_REAL, "seed": SEED}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n결과 저장: {OUT}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main_())
