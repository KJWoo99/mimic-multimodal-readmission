"""Phase 3: Fusion + 결측 모달리티 처리(M1a) + 검증 두 건(V1, V3).

Phase 2 의 영상 단독 결과(자세 기준선 대비 +0.03~+0.07)를 세 방향에서 다시 봄.

  V3  픽셀 없이 촬영 메타데이터만으로 얼마나 맞히는가.
      영상 모델이 이 선을 못 넘으면 "폐를 본 것"이 아니라 "누가 찍혔는지를 본 것"임.
  V1  분할 방식을 환자 단위에서 무작위로 바꾸면 성능이 얼마나 부풀려지는가.
      MIMIC-CXR 에서 가장 큰 누수 위험이라 숫자로 크기를 남김.
  M1a 영상이 없는 입원 91.4% 를 어떻게 다룰 것인가.
      드롭아웃 학습 + 영 벡터 fallback 으로 전체 코호트에서 재고, EHR 단독과 비교함.

Fusion 은 Phase 2-B 가 만든 임베딩 캐시를 읽으므로 GPU 가 필요 없음. 인코더가
고정되어 있어 "뒷단만 바뀌었다"는 비교 조건도 자동으로 보장됨.

    python scripts/run_phase3.py            # 전체
    python scripts/run_phase3.py --check    # 입력 준비 상태만 확인

## 왜 임베딩을 PCA 로 줄이는가

DenseNet121 임베딩은 1,024 차원이고 EHR 피처는 29 개임. 그대로 이어붙이면
정보량이 아니라 차원 수로 영상이 EHR 을 압도해 버려서, 영상이 쓸모없을 때조차
"Fusion 이 EHR 보다 나쁘다"는 결과가 나옴. 그건 영상의 무용함이 아니라 차원
불균형의 결과라 해석이 안 됨. 그래서 PCA 로 줄인 것을 주 결과로 삼고, 원본
이어붙이기도 함께 보고해 둘 다 같은 결론인지 확인함.

## DUA

임베딩, 모델은 MIMIC 파생물임. `outputs/models/` 아래 두어 `.gitignore` 로 막음.
저장하는 지표 JSON 에는 환자 단위 정보가 들어가지 않음.
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
# 인코더별 임베딩. Phase 2 의 파인튜닝 CNN 과 Phase 2-C 의 파운데이션 모델을
# 같은 뒷단으로 비교할 수 있어야 함: 뒷단이 다르면 차이가 인코더 때문인지
# 분류기 때문인지 갈라지지 않음.
EMBEDDINGS = {
    "cnn": OUT_DIR / "models" / "phase2_cxr_embeddings.npz",
    "raddino": OUT_DIR / "models" / "phase2_raddino_embeddings.npz",
}

# 판정선: 결과를 보기 전에 고정함.
LIFT_REAL = 0.02      # Fusion 이 EHR 을 이 이상 넘으면 "영상이 보탠 것"
INFLATION_BIG = 0.02  # V1 부풀림이 이 이상이면 "분할 방식이 결과를 만든다"


def _ap_auc(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.5,
                    help="M1a 학습 시 영상 블록을 지우는 비율")
    ap.add_argument("--encoder", default="cnn", choices=sorted(EMBEDDINGS),
                    help="어느 인코더의 임베딩을 쓸 것인가")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    embed_path = EMBEDDINGS[args.encoder]
    # 인코더가 다르면 산출물도 달라야 함. 덮어쓰면 이전 인코더로 낸 판정이
    # 소리 없이 다른 인코더의 것으로 바뀜.
    metrics_path = (OUT_DIR / "phase3_metrics.json" if args.encoder == "cnn"
                    else OUT_DIR / f"phase3_metrics_{args.encoder}.json")

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd

    from cohort import CohortConfig, build_cohort, load_admissions
    from cxr_dataset import build_cxr_index
    from features import ELECTIVE_ADMISSION_TYPES, build_features, feature_columns
    from metrics import classification_metrics
    from models import build_tabular_model
    from splits import assert_no_group_overlap, patient_split, split_with_fixed_folds, subject_folds
    from tracking import ExperimentLogger

    raw = Path(args.raw_dir)
    t0 = time.time()
    print("=" * 74)
    print("  Phase 3: Fusion + 결측 모달리티(M1a) + V1, V3")
    print("=" * 74)

    print("\n[0] 입력 점검")
    if not embed_path.exists():
        print(f"  임베딩 캐시가 없다: {embed_path}")
        print("  먼저 run_phase2_embed.py (cnn) 또는 run_phase2_raddino.py 를 실행할 것.")
        return 1
    z = np.load(embed_path, allow_pickle=False)
    emb_all = z["embedding"].astype(np.float32)
    emb_hadm = z["hadm_id"]
    # 파운데이션 모델 임베딩에는 체크포인트 지문이 없음(우리가 학습한 것이 아니라
    # 공개 가중치를 그대로 쓰므로). 대신 모델 id 를 출처로 남김.
    provenance = (str(z["checkpoint_sha256"]) if "checkpoint_sha256" in z
                  else str(z["model"]))
    print(f"  [OK] 임베딩 {emb_all.shape}  인코더 {args.encoder}  출처 {provenance[:24]}…")

    dx_hits = list(raw.rglob("diagnoses_icd.csv.gz"))
    adm_hits = list(raw.rglob("admissions.csv.gz"))
    meta_hits = list(raw.rglob("mimic-cxr-2.0.0-metadata.csv.gz"))
    for name, hits in (("admissions", adm_hits), ("diagnoses_icd", dx_hits),
                       ("cxr metadata", meta_hits)):
        if not hits:
            print(f"  {name} 파일이 없다. download_mimic.sh 를 먼저 실행할 것.")
            return 1
        print(f"  [OK] {name}")
    if args.check:
        print("\n준비 완료.")
        return 0

    # ---------------------------------------------------------------- 코호트
    print("\n[1] 코호트 + 피처")
    adm, pat = load_admissions(raw)
    coh = build_cohort(adm, pat,
                       CohortConfig(elective_admission_types=ELECTIVE_ADMISSION_TYPES))
    dx = pd.read_csv(dx_hits[0])
    feat, _ = build_features(coh.df, pat, dx, None, None)
    print(f"    전체 코호트 {len(feat):,} 입원")

    # "EHR 단독"이 영상 정보를 쥔 채 비교하는 일이 없어야 함. 여기서는 코호트
    # 테이블만 넘겼으므로 영상 파생 컬럼이 애초에 생기지 않지만, 피처 생성 규칙이
    # 바뀌면 경고 없이 섞일 수 있어 매 실행 검사함(Phase 2 는 CXR 인덱스를 넘겨서
    # has_cxr, n_cxr 이 생겼고 그때는 명시적으로 빼야 함).
    IMAGE_DERIVED = ("has_cxr", "n_cxr", "cxr_view_ap", "is_ap")
    ehr_cols = [c for c in feature_columns(feat) if c not in IMAGE_DERIVED]
    leaked = [c for c in feature_columns(feat) if c in IMAGE_DERIVED]
    print(f"    EHR 피처 {len(ehr_cols)}개"
          + (f"  (영상 파생 {leaked} 제외)" if leaked else "  (영상 파생 컬럼 없음)"))

    y_full = feat["readmit_30d"].to_numpy().astype(int)
    g_full = feat["subject_id"].to_numpy()
    X_ehr = feat[ehr_cols].to_numpy(dtype=np.float32)

    # 임베딩을 코호트 행 순서에 맞춰 정렬함. hadm_id 로 조인하고, 없는 입원은
    # 영 벡터 + has_image=0 으로 둠(M1a 의 fallback 이 바로 이 형태임).
    pos = pd.Series(np.arange(len(emb_hadm)), index=emb_hadm)
    row = feat["hadm_id"].map(pos)
    has_img = row.notna().to_numpy()
    emb_aligned = np.zeros((len(feat), emb_all.shape[1]), dtype=np.float32)
    emb_aligned[has_img] = emb_all[row[has_img].to_numpy().astype(int)]
    print(f"    영상 보유 {has_img.sum():,} / {len(feat):,} "
          f"({has_img.mean() * 100:.1f}%)")

    # -------------------------------------------------------------- 분할
    print("\n[2] 환자 단위 분할 (Phase 2 와 같은 규칙, seed)")
    # 영상 부분집합을 먼저 나눔. 이 분할이 Phase 2 인코더가 학습한 분할과
    # 같아야 하고(같은 환자 집합, 같은 seed 라 일치함), 전체 분할은 그것을
    # 물려받아야 함. 순서를 반대로 하면 인코더가 학습한 환자가 전체 분할의
    # test 로 넘어가, 임베딩이 그 환자의 라벨을 이미 본 상태로 평가됨.
    # 실측은 splits.py 의 split_with_fixed_folds 주석에 적음.
    sub = np.where(has_img)[0]
    sp_sub = patient_split(g_full[sub], seed=args.seed)
    sp_full = split_with_fixed_folds(
        g_full, subject_folds(g_full[sub], sp_sub), seed=args.seed)
    print(f"    train {len(sp_full.train_idx):,} / val {len(sp_full.val_idx):,} "
          f"/ test {len(sp_full.test_idx):,}"
          "   (영상 보유 환자의 소속은 인코더 분할을 그대로 따른다)")
    # val 은 여기서 쓰지 않음. 그래도 3분할로 자르는 이유는 Phase 2 와 분할
    # 규칙을 똑같이 두기 위해서임. 2분할로 바꾸면 같은 seed 라도 환자가 다르게
    # 떨어져, Phase 2 의 영상 모델과 여기 EHR 모델이 다른 test 위에서 비교됨.
    sub_train, _sub_val, sub_test = (sub[sp_sub.train_idx], sub[sp_sub.val_idx],
                                     sub[sp_sub.test_idx])
    assert_no_group_overlap(g_full[sub], sp_sub.train_idx, sp_sub.val_idx,
                            sp_sub.test_idx, names=("train", "val", "test"))
    print(f"    영상 부분집합 train {len(sub_train):,} / test {len(sub_test):,}")

    def pw(idx):
        yy = y_full[idx]
        n_pos = int(yy.sum())
        return float((len(yy) - n_pos) / max(n_pos, 1))

    results: dict = {"phase": "3", "seed": args.seed, "encoder": args.encoder}

    # ------------------------------------------------------- V3 메타데이터
    print("\n[3] V3: 픽셀 없이 촬영 메타데이터만")
    idx_res = build_cxr_index(coh.df, pd.read_csv(
        meta_hits[0], usecols=["subject_id", "study_id", "dicom_id",
                               "StudyDate", "StudyTime", "ViewPosition"]),
        raw / "mimic-cxr-jpg")
    idxdf = idx_res.df.set_index("hadm_id")
    # 픽셀을 제외한, 촬영이 남긴 모든 부수정보: 어떤 자세로 찍혔는가와
    # 언제 찍혔는가(퇴원까지 남은 시간). 둘 다 환자 상태의 대리지표임:
    # AP 는 못 일어나는 환자, 퇴원 직전 촬영은 확인 촬영일 가능성이 높음.
    hrs = feat["hadm_id"].map(idxdf["hours_before_discharge"]).to_numpy(dtype=np.float32)
    view = feat["hadm_id"].map(idxdf["ViewPosition"]).fillna("MISSING").astype(str)
    cats = sorted(view.unique())
    V = np.zeros((len(feat), len(cats)), dtype=np.float32)
    for i, c in enumerate(cats):
        V[:, i] = (view == c).to_numpy(dtype=np.float32)
    meta_block = np.column_stack(
        [np.nan_to_num(hrs, nan=-1.0), V]
    ).astype(np.float32)
    print(f"    메타데이터 특징: 촬영~퇴원 시간 1개 + 자세 {len(cats)}종 원핫")

    m3 = build_tabular_model("xgboost", scale_pos_weight=pw(sub_train))
    m3.fit(meta_block[sub_train], y_full[sub_train])
    p3 = m3.predict_proba(meta_block[sub_test])[:, 1]
    mm3 = classification_metrics(y_full[sub_test], p3)
    print(f"    메타데이터 단독  AUROC {mm3.auroc:.4f}  PR-AUC {mm3.pr_auc:.4f}  "
          f"({meta_block.shape[1]}개 특징)")
    results["v3_metadata_only"] = mm3.as_dict()

    # 영상 임베딩 단독(같은 부분집합, 같은 분할)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    pca.fit(emb_aligned[sub_train])
    emb_p = pca.transform(emb_aligned).astype(np.float32)
    print(f"    임베딩 PCA {emb_all.shape[1]} -> {args.pca_dim}차원 "
          f"(설명분산 {pca.explained_variance_ratio_.sum() * 100:.1f}%)")

    m_img = build_tabular_model("xgboost", scale_pos_weight=pw(sub_train))
    m_img.fit(emb_p[sub_train], y_full[sub_train])
    p_img = m_img.predict_proba(emb_p[sub_test])[:, 1]
    mm_img = classification_metrics(y_full[sub_test], p_img)
    print(f"    영상 임베딩 단독 AUROC {mm_img.auroc:.4f}  PR-AUC {mm_img.pr_auc:.4f}")
    results["image_embedding_only"] = mm_img.as_dict()
    results["v3_verdict"] = ("메타데이터를 넘지 못함"
                             if mm_img.auroc <= mm3.auroc + LIFT_REAL
                             else "픽셀이 메타데이터를 넘음")
    print(f"    판정: {results['v3_verdict']} "
          f"(영상 {mm_img.auroc:.4f} vs 메타 {mm3.auroc:.4f})")

    # --------------------------------------------------------------- V1
    print("\n[4] V1: 분할 방식 비교 (환자 단위 vs 무작위)")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(sub))
    n_te = len(sub_test)
    naive_test = sub[perm[:n_te]]
    naive_train = sub[perm[n_te:]]
    shared = len(set(g_full[naive_train]) & set(g_full[naive_test]))
    print(f"    무작위 분할에서 train, test 양쪽에 걸친 환자 {shared:,}명")

    # PCA 도 이 팔의 train 으로 다시 적합함. 환자 단위 train 으로 맞춘 것을
    # 그대로 쓰면 무작위 test 의 일부가 이미 PCA 적합에 들어가 있어, 재려는
    # 부풀림에 PCA 누수가 얹힘. 두 팔의 조건을 정확히 맞춰야 차이가 오직
    # 분할 방식에서만 옴.
    pca_nv = PCA(n_components=args.pca_dim, random_state=args.seed)
    pca_nv.fit(emb_aligned[naive_train])
    emb_nv = pca_nv.transform(emb_aligned).astype(np.float32)

    m_nv = build_tabular_model("xgboost", scale_pos_weight=pw(naive_train))
    m_nv.fit(emb_nv[naive_train], y_full[naive_train])
    p_nv = m_nv.predict_proba(emb_nv[naive_test])[:, 1]
    mm_nv = classification_metrics(y_full[naive_test], p_nv)
    infl = mm_nv.auroc - mm_img.auroc
    print(f"    환자 단위 AUROC {mm_img.auroc:.4f}  ->  무작위 {mm_nv.auroc:.4f}"
          f"   부풀림 {infl:+.4f}")
    results["v1_patient_split"] = mm_img.as_dict()
    results["v1_random_split"] = mm_nv.as_dict()
    results["v1_inflation_auroc"] = float(infl)
    results["v1_shared_subjects"] = int(shared)
    results["v1_verdict"] = ("분할 방식이 결과를 만든다" if infl >= INFLATION_BIG
                             else "분할 방식으로 설명되지 않는다")
    print(f"    판정: {results['v1_verdict']}")

    # 사전등록 판정선(절대 AUROC 0.02)은 여기서 거칠음. 이 모델의 실제 신호는
    # 무작위 대비 겨우 (AUROC-0.5) 이라, 부풀림이 판정선에 못 미쳐도 신호
    # 자체보다 클 수 있음. 기준을 사후에 바꾸지 않고 그 대비를 같이 남김.
    signal = mm_img.auroc - 0.5
    ratio = float(infl / signal) if signal > 0 else float("nan")
    results["v1_signal_above_chance"] = float(signal)
    results["v1_inflation_to_signal_ratio"] = ratio
    print(f"    참고: 실제 신호(무작위 대비) {signal:+.4f},  "
          f"부풀림/신호 = {ratio:.2f}배")
    if infl < INFLATION_BIG and ratio >= 1.0:
        print("      사전등록 판정선은 못 넘었으나 부풀림이 신호보다 크다. "
              "판정선은 그대로 두고 이 사실을 한계로 적는다.")

    # ------------------------------------------------------------ Fusion
    print("\n[5] Fusion: EHR + 영상 임베딩 (영상 보유 부분집합)")
    m_ehr_sub = build_tabular_model("xgboost", scale_pos_weight=pw(sub_train))
    m_ehr_sub.fit(X_ehr[sub_train], y_full[sub_train])
    p_ehr_sub = m_ehr_sub.predict_proba(X_ehr[sub_test])[:, 1]
    mm_ehr_sub = classification_metrics(y_full[sub_test], p_ehr_sub)
    print(f"    EHR 단독      AUROC {mm_ehr_sub.auroc:.4f}  PR-AUC {mm_ehr_sub.pr_auc:.4f}")

    fus = np.column_stack([X_ehr, emb_p]).astype(np.float32)
    m_fus = build_tabular_model("xgboost", scale_pos_weight=pw(sub_train))
    m_fus.fit(fus[sub_train], y_full[sub_train])
    p_fus = m_fus.predict_proba(fus[sub_test])[:, 1]
    mm_fus = classification_metrics(y_full[sub_test], p_fus)
    lift = mm_fus.auroc - mm_ehr_sub.auroc
    print(f"    Fusion(PCA)   AUROC {mm_fus.auroc:.4f}  PR-AUC {mm_fus.pr_auc:.4f}"
          f"   EHR 대비 {lift:+.4f}")

    # 차원 불균형이 결론을 만든 것이 아님을 확인하려고 원본도 같이 돌림.
    fus_raw = np.column_stack([X_ehr, emb_aligned]).astype(np.float32)
    m_fr = build_tabular_model("xgboost", scale_pos_weight=pw(sub_train))
    m_fr.fit(fus_raw[sub_train], y_full[sub_train])
    p_fr = m_fr.predict_proba(fus_raw[sub_test])[:, 1]
    mm_fr = classification_metrics(y_full[sub_test], p_fr)
    print(f"    Fusion(원본)  AUROC {mm_fr.auroc:.4f}  PR-AUC {mm_fr.pr_auc:.4f}"
          f"   EHR 대비 {mm_fr.auroc - mm_ehr_sub.auroc:+.4f}")

    results["ehr_only_subset"] = mm_ehr_sub.as_dict()
    results["fusion_pca"] = mm_fus.as_dict()
    results["fusion_raw"] = mm_fr.as_dict()
    # --- 후기 융합: 영상 예측확률 하나만 EHR 에 얹음 ---
    #
    # 앞의 이어붙이기(early fusion)는 EHR 29개 옆에 영상 32~768개를 붙이므로
    # 차원 수만으로 영상이 EHR 을 압도함. 후기 융합은 영상을 특징 한 개로
    # 줄여 그 불균형을 아예 없앰. 영상이 보탤 것이 있다면 가장 잘 드러나는 형태임.
    #
    # 단, train 구간의 영상 예측을 그대로 얹으면 안 됨. 자기 학습 데이터를 맞힌
    # 값이라 실제보다 훨씬 좋고, 뒷단 모델이 그 특징을 과신하게 배움. 환자 단위
    # k-fold 로 out-of-fold 예측을 만들어 씀.
    from splits import patient_kfold
    oof = np.zeros(len(feat), dtype=np.float32)
    n_folds = 5
    for tr_rel, va_rel in patient_kfold(g_full[sub_train], n_splits=n_folds):
        tr_i, va_i = sub_train[tr_rel], sub_train[va_rel]
        m_f = build_tabular_model("xgboost", scale_pos_weight=pw(tr_i))
        m_f.fit(emb_p[tr_i], y_full[tr_i])
        oof[va_i] = m_f.predict_proba(emb_p[va_i])[:, 1]
    # test 구간은 train 전체로 학습한 모델로 한 번에 예측함.
    m_full_img = build_tabular_model("xgboost", scale_pos_weight=pw(sub_train))
    m_full_img.fit(emb_p[sub_train], y_full[sub_train])
    oof[sub_test] = m_full_img.predict_proba(emb_p[sub_test])[:, 1]

    late = np.column_stack([X_ehr, oof.reshape(-1, 1)]).astype(np.float32)
    m_late = build_tabular_model("xgboost", scale_pos_weight=pw(sub_train))
    m_late.fit(late[sub_train], y_full[sub_train])
    p_late = m_late.predict_proba(late[sub_test])[:, 1]
    mm_late = classification_metrics(y_full[sub_test], p_late)
    lift_late = mm_late.auroc - mm_ehr_sub.auroc
    print(f"    Fusion(후기)  AUROC {mm_late.auroc:.4f}  PR-AUC {mm_late.pr_auc:.4f}"
          f"   EHR 대비 {lift_late:+.4f}   (영상을 특징 1개로, OOF {n_folds}겹)")
    results["fusion_late"] = mm_late.as_dict()
    results["fusion_late_lift"] = float(lift_late)
    results["fusion_late_n_folds"] = n_folds

    # 세 방식 중 가장 나은 것으로 판정함. 하나만 보고 "보태지 않는다"고 하면
    # 그 방식이 서툴렀을 뿐인 경우를 가려내지 못함.
    best_lift = max(lift, lift_late, mm_fr.auroc - mm_ehr_sub.auroc)
    results["fusion_best_lift"] = float(best_lift)
    results["fusion_lift_over_ehr"] = float(lift)
    results["fusion_verdict"] = ("영상이 보탠 것" if best_lift >= LIFT_REAL
                                 else "영상이 보태지 않는다")
    print(f"    판정: {results['fusion_verdict']} "
          f"(세 방식 중 최고 {best_lift:+.4f})")

    # -------------------------------------------------------------- M1a
    print(f"\n[6] M1a: 결측 모달리티 처리 (드롭아웃 {args.dropout:.0%} + 영벡터 fallback)")
    has_col = has_img.astype(np.float32).reshape(-1, 1)

    # M1a 는 전체 분할(sp_full)에서 학습, 평가하므로 여기 쓰는 PCA 도 전체 분할의
    # train 으로 적합함. 위쪽 `pca` 는 영상 부분집합 train 으로 맞춘 것이라
    # 범위가 다름(이제 두 train 은 포함 관계임).
    tr_img = sp_full.train_idx[has_img[sp_full.train_idx]]
    pca_m1 = PCA(n_components=args.pca_dim, random_state=args.seed)
    pca_m1.fit(emb_aligned[tr_img])
    emb_m1 = pca_m1.transform(emb_aligned).astype(np.float32)
    # 영상이 없는 행은 영벡터를 넣었는데, PCA 를 통과시키면 영벡터가 아니라
    # -평균, 성분 이 됨. 그대로 두면 "영상 없음" 이 두 가지 모양으로 존재하게 됨
    #   영상이 실제로 없는 행 : pca(0)
    #   드롭아웃한 행 : 정확히 0 (아래 Xd 에서 0 으로 덮음)
    # 학습은 후자로, 평가는 전자로 하게 되므로 결측 처리가 버티는지를 재려던
    # 실험 자체가 어긋남. 두 경우를 같은 모양(정확히 0)으로 맞춤.
    emb_m1[~has_img] = 0.0
    print(f"    M1a 용 PCA 재적합: 전체분할 train 중 영상 보유 {len(tr_img):,}행 "
          f"(설명분산 {pca_m1.explained_variance_ratio_.sum() * 100:.1f}%)")
    Xf = np.column_stack([X_ehr, has_col, emb_m1]).astype(np.float32)

    # 학습 시 영상 보유 행의 일부를 일부러 "없는 것"으로 만듦. 그래야 추론에서
    # 영상이 없을 때도 모델 성능이 떨어지지 않음.
    Xd = Xf.copy()
    tr = sp_full.train_idx
    cand = tr[has_img[tr]]
    drop = rng.choice(cand, size=int(len(cand) * args.dropout), replace=False)
    Xd[np.ix_(drop, np.arange(X_ehr.shape[1], Xf.shape[1]))] = 0.0
    print(f"    학습 {len(tr):,}건 중 영상 보유 {len(cand):,}건, 그중 {len(drop):,}건을 지움")

    # 기준선을 둘 둠. Fusion 에만 has_image 를 주고 EHR 과 비교하면 "픽셀이
    # 보탠 것"과 "영상이 있다는 사실을 안 것"이 섞임. Phase 1.5 에서 영상 유무는
    # 그 자체로 예측력이 있다고 이미 나옴(MNAR). 그래서 픽셀의 몫만 떼어내려면
    # has_image 까지 준 EHR 모델과 비교해야 함.
    X_ehr_has = np.column_stack([X_ehr, has_col]).astype(np.float32)

    m_ehr_full = build_tabular_model("xgboost", scale_pos_weight=pw(tr))
    m_ehr_full.fit(X_ehr[tr], y_full[tr])
    m_ehr_has = build_tabular_model("xgboost", scale_pos_weight=pw(tr))
    m_ehr_has.fit(X_ehr_has[tr], y_full[tr])
    m_m1 = build_tabular_model("xgboost", scale_pos_weight=pw(tr))
    m_m1.fit(Xd[tr], y_full[tr])

    te = sp_full.test_idx
    rows = []
    for name, mask in (("전체 코호트", np.ones(len(te), dtype=bool)),
                       ("영상 있음", has_img[te]),
                       ("영상 없음", ~has_img[te])):
        ii = te[mask]
        if len(ii) < 50 or len(np.unique(y_full[ii])) < 2:
            continue
        a_e, _ = _ap_auc(y_full[ii], m_ehr_full.predict_proba(X_ehr[ii])[:, 1])
        a_h, _ = _ap_auc(y_full[ii], m_ehr_has.predict_proba(X_ehr_has[ii])[:, 1])
        a_f, _ = _ap_auc(y_full[ii], m_m1.predict_proba(Xf[ii])[:, 1])
        rows.append({"subset": name, "n": len(ii),
                     "ehr_auroc": a_e, "ehr_plus_hasimage_auroc": a_h,
                     "fusion_m1_auroc": a_f,
                     "delta_vs_ehr": float(a_f - a_e),
                     # 픽셀의 몫: 영상 존재 정보를 양쪽에 똑같이 준 뒤의 차이
                     "delta_pixels": float(a_f - a_h)})
        print(f"    {name:<10} n={len(ii):>7,}  EHR {a_e:.4f}  "
              f"EHR+유무 {a_h:.4f}  ->  Fusion+M1a {a_f:.4f}"
              f"   픽셀 몫 {a_f - a_h:+.4f}")
    results["m1a_by_subset"] = rows
    results["m1a_pixel_contribution"] = {r["subset"]: r["delta_pixels"] for r in rows}
    by = {r["subset"]: r["delta_pixels"] for r in rows}

    # 판정을 둘로 나눔. 하나로 묶으면 서로 다른 두 질문의 답이 섞임.
    #   (1) 결측 처리가 버티는가  -> 영상 없는 입원에서 재는 것
    #   (2) 픽셀이 보태는가        -> 영상 있는 입원에서 재는 것
    d_missing = by.get("영상 없음", 0.0)
    d_present = by.get("영상 있음", 0.0)
    results["m1a_fallback_verdict"] = (
        "영상 없는 입원에서 무너지지 않는다" if d_missing > -LIFT_REAL
        else "결측 입원에서 성능이 깎인다")
    results["m1a_pixel_verdict"] = (
        "픽셀이 보탠다" if d_present >= LIFT_REAL
        else ("픽셀이 오히려 깎는다" if d_present <= -LIFT_REAL
              else "픽셀이 보태지도 깎지도 않는다"))
    print(f"    결측 처리 판정: {results['m1a_fallback_verdict']} "
          f"(영상 없음 {d_missing:+.4f})")
    print(f"    픽셀 기여 판정: {results['m1a_pixel_verdict']} "
          f"(영상 있음 {d_present:+.4f})")

    # ------------------------------------------------------------- 저장
    print("\n[7] 저장")
    results["elapsed_sec"] = round(time.time() - t0, 1)
    results["n_full"] = len(feat)
    results["n_image_subset"] = int(has_img.sum())
    results["embedding_dim"] = int(emb_all.shape[1])
    results["pca_dim"] = int(args.pca_dim)
    results["encoder_provenance"] = provenance
    if "checkpoint_sha256" in z:
        results["checkpoint_sha256"] = str(z["checkpoint_sha256"])
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"    {metrics_path}")

    try:
        with ExperimentLogger(
            "phase3-fusion",
            params={
                "phase": "3", "seed": args.seed, "pca_dim": args.pca_dim,
                "dropout": args.dropout, "encoder": args.encoder,
                "encoder_provenance": provenance,
            },
            run_name=f"fusion-{args.encoder}-pca{args.pca_dim}-seed{args.seed}",
        ) as log:
            log.log_metrics({"v3_metadata_auroc": mm3.auroc,
                             "image_embed_auroc": mm_img.auroc,
                             "v1_inflation": infl,
                             "fusion_lift": lift})
    except Exception as e:  # 기록 실패로 결과를 잃지 않음
        print(f"    (실험 기록 건너뜀: {type(e).__name__})")

    print("\n" + "=" * 74)
    print("  Phase 3 완료")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
