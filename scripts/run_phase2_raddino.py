"""Phase 2-C: RAD-DINO(흉부X선 파운데이션 모델) 동결 특징으로 재검증.

Phase 2 는 ImageNet 사전학습 CNN 을 224px 로 파인튜닝해 "영상은 자세만 본다"는
결론을 냄. 음성 결론에 대한 가장 자연스러운 반박이 "학습을 못 시킨 것 아니냐"이므로,
그 가능성을 따로 확인함(트러블슈팅 11).

여기서는 우리가 학습시키지 않음. 흉부X선 88만 장으로 자가지도 학습된
공개 파운데이션 모델(RAD-DINO)의 표현을 그대로 가져와, 그 위에 얕은 분류기만
얹음. 이 저장소의 학습 설정과 무관하게 "제대로 된 흉부X선 표현에 재입원 신호가
들어 있는가"를 직접 묻는 방식임.

  근거 1  RAD-DINO 는 MIMIC-CXR, CheXpert, NIH, PadChest, BRAX 882,775장으로
          DINOv2 방식(자가지도) 학습. MIT 라이선스, 공개 가중치.
  근거 2  입력 518px: MIMIC-CXR 해상도 실측 연구에서 512~1024px 가 정점이고
          그 이상은 오히려 떨어짐(DenseNet121 기준 256px 80.5% -> 512px 81.5%
          -> 1024px 81.5% -> 2048px 80.7%). 우리가 쓰던 224px 보다 유리한 구간임.

    python scripts/run_phase2_raddino.py            # 전체
    python scripts/run_phase2_raddino.py --check    # 준비 상태만 확인

## 라벨 누수는 없음: 다만 밝혀둘 것

RAD-DINO 의 사전학습 데이터에 MIMIC-CXR 이 포함되므로 우리 test 영상을 이미 본
적이 있음. 다만 그 학습은 라벨 없는 자가지도이고, 우리 라벨(30일 재입원)은
영상이 아니라 EHR 에서 옴. 따라서 재입원 라벨의 누수는 없음. 그래도 "표현이
이 데이터에 유리하게 맞춰져 있다"는 점은 결과에 유리한 방향의 편향이라
리포트에 함께 적음.

## DUA

임베딩은 MIMIC 파생물임. `outputs/models/` 아래 두어 `.gitignore` 로 막음.
모델 가중치는 HuggingFace 캐시에 받으며 저장소에 넣지 않음.
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
EMBED_PATH = OUT_DIR / "models" / "phase2_raddino_embeddings.npz"
METRICS_PATH = OUT_DIR / "phase2_raddino.json"

MODEL_ID = "microsoft/rad-dino"
# 결과를 만든 판(run_lowdose.py 와 같음). 판을 주지 않으면 그날의 최신판을 받음.
MODEL_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
LIFT_SAW_IMAGE = 0.05   # Phase 2 와 같은 판정선을 씀
LIFT_SAW_VIEW = 0.02


class RadDinoDataset:
    """전처리는 모델이 배포한 프로세서를 그대로 씀.

    우리 `build_transforms` 를 쓰면 안 됨: 사전학습 때와 다른 정규화, 크기로
    넣으면 표현이 어긋나고, 그러면 사전학습 표현으로 신호를 재는
    이 실험의 전제가 성립하지 않음.

    클래스를 모듈 최상위에 둠. DataLoader 워커가 spawn 으로 뜨면
    데이터셋을 피클링하는데, 함수 안에 정의하면 피클링이 안 돼 워커가 죽음.
    """

    def __init__(self, paths, processor):
        self.paths = list(paths)
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
            out = self.processor(images=img, return_tensors="pt")
        return out["pixel_values"][0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="앞 N장만 (점검용)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from cohort import CohortConfig, build_cohort, load_admissions
    from cxr_dataset import build_cxr_index, view_baseline
    from features import ELECTIVE_ADMISSION_TYPES, build_features, feature_columns
    from metrics import classification_metrics
    from models import build_tabular_model
    from splits import patient_split

    raw = Path(args.raw_dir)
    t0 = time.time()
    print("=" * 74)
    print("  Phase 2-C: RAD-DINO 동결 특징 재검증")
    print("=" * 74)

    print("\n[0] 입력 점검")
    meta_hits = list(raw.rglob("mimic-cxr-2.0.0-metadata.csv.gz"))
    dx_hits = list(raw.rglob("diagnoses_icd.csv.gz"))
    if not meta_hits or not dx_hits:
        print("  필요한 파일이 없다. download_mimic.sh 를 먼저 실행할 것.")
        return 1
    print("  [OK] cxr metadata / diagnoses_icd")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [OK] 장치 {device}")
    if args.check:
        print("\n준비 완료. (모델 가중치는 첫 실행 시 HuggingFace 에서 받는다)")
        return 0

    print("\n[1] 코호트 + 영상 인덱스 (Phase 2 와 동일 규칙, seed)")
    adm, pat = load_admissions(raw)
    coh = build_cohort(adm, pat,
                       CohortConfig(elective_admission_types=ELECTIVE_ADMISSION_TYPES))
    meta = pd.read_csv(meta_hits[0],
                       usecols=["subject_id", "study_id", "dicom_id",
                                "StudyDate", "StudyTime", "ViewPosition"])
    idx = build_cxr_index(coh.df, meta, raw / "mimic-cxr-jpg")
    df = idx.df.reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
    print(f"    영상 연결 {len(df):,} 입원")

    groups = df["subject_id"].to_numpy()
    sp = patient_split(groups, seed=args.seed)
    y = df["readmit_30d"].to_numpy().astype(int)
    print(f"    train {len(sp.train_idx):,} / val {len(sp.val_idx):,} "
          f"/ test {len(sp.test_idx):,}")

    print(f"\n[2] {MODEL_ID} 로드")
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(device).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"    파라미터 {n_par:,}  입력 처리 {type(processor).__name__}")

    loader = DataLoader(RadDinoDataset(df["path"].tolist(), processor),
                        batch_size=args.batch, shuffle=False,
                        num_workers=args.workers,
                        pin_memory=device.type == "cuda")

    print("\n[3] 임베딩 추출 (동결: 역전파 없음)")
    chunks = []
    t1 = time.time()
    with torch.no_grad():
        for bi, x in enumerate(loader):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                out = model(pixel_values=x)
            # CLS 토큰 = 영상 전체 표현
            cls = out.last_hidden_state[:, 0]
            chunks.append(cls.float().cpu().numpy())
            done = (bi + 1) * args.batch
            if done % (args.batch * 100) == 0:
                el = time.time() - t1
                print(f"    {min(done, len(df)):,}/{len(df):,}  ({el:.0f}s, "
                      f"남은 예상 {el / done * (len(df) - done):.0f}s)")
    emb = np.concatenate(chunks).astype(np.float32)
    print(f"    완료 {emb.shape}  ({time.time() - t1:.0f}s)")

    EMBED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMBED_PATH, embedding=emb,
        hadm_id=df["hadm_id"].to_numpy().astype(np.int64),
        subject_id=df["subject_id"].to_numpy().astype(np.int64),
        readmit_30d=y.astype(np.int8), model=np.array(MODEL_ID))
    print(f"    {EMBED_PATH}  ({EMBED_PATH.stat().st_size / 1024**2:.1f} MB)")

    # ------------------------------------------------------------ 평가
    print("\n[4] 자세 기준선 (픽셀 없이 ViewPosition 만)")
    vb = view_baseline(df, groups, seed=args.seed)
    print(f"    AUROC {vb['auroc']:.4f}  PR-AUC {vb['pr_auc']:.4f}")

    def pw(idx):
        n_pos = int(y[idx].sum())
        return float((len(idx) - n_pos) / max(n_pos, 1))

    print("\n[5] 동결 특징 위 분류기")
    res = {}
    from sklearn.decomposition import PCA
    pca = PCA(n_components=args.pca_dim, random_state=args.seed)
    pca.fit(emb[sp.train_idx])
    emb_p = pca.transform(emb).astype(np.float32)
    print(f"    PCA {emb.shape[1]} -> {args.pca_dim}차원 "
          f"(설명분산 {pca.explained_variance_ratio_.sum() * 100:.1f}%)")

    # 헤드는 val AUROC 로 고르고 그 헤드의 test 값을 보고함. test AUROC 최고를 고르면 낙관적이고
    # "영상을 봤다" 쪽으로 기움. 세 헤드의 test 값은 그대로 모두 남김.
    val_auroc: dict[str, float] = {}
    for name, X in ((f"PCA {args.pca_dim}차원", emb_p), ("원본 768차원", emb)):
        mdl = build_tabular_model("xgboost", scale_pos_weight=pw(sp.train_idx))
        mdl.fit(X[sp.train_idx], y[sp.train_idx])
        p = mdl.predict_proba(X[sp.test_idx])[:, 1]
        m = classification_metrics(y[sp.test_idx], p)
        res[name] = m.as_dict()
        val_auroc[name] = classification_metrics(
            y[sp.val_idx], mdl.predict_proba(X[sp.val_idx])[:, 1]).auroc
        print(f"    {name:<14} AUROC {m.auroc:.4f}  PR-AUC {m.pr_auc:.4f}")

    # 선형 프로브: 파운데이션 모델 평가의 표준. 얕은 분류기가 못 뽑으면
    # 표현 자체에 그 정보가 선형적으로 들어있지 않다는 뜻임.
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lp = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=3000, class_weight="balanced"))
    lp.fit(emb[sp.train_idx], y[sp.train_idx])
    m_lp = classification_metrics(y[sp.test_idx],
                                  lp.predict_proba(emb[sp.test_idx])[:, 1])
    res["선형 프로브"] = m_lp.as_dict()
    val_auroc["선형 프로브"] = classification_metrics(
        y[sp.val_idx], lp.predict_proba(emb[sp.val_idx])[:, 1]).auroc
    print(f"    {'선형 프로브':<14} AUROC {m_lp.auroc:.4f}  PR-AUC {m_lp.pr_auc:.4f}")

    print("\n[6] 같은 부분집합에서 EHR 단독")
    dx = pd.read_csv(dx_hits[0])
    # Phase 2, 3 과 같은 피처: 코호트 전체 행에서 만들고 이 부분집합의 행 순서로 가져옴.
    # build_cxr_index 의 df(5개 컬럼)에서 만들면 축소판 EHR 이 됨.
    feat_all, _ = build_features(coh.df, pat, dx, None, None)
    feat = df[["hadm_id"]].merge(feat_all, on="hadm_id", how="left", validate="1:1")
    cols = [c for c in feature_columns(feat)
            if c not in ("has_cxr", "n_cxr", "cxr_view_ap", "is_ap")]
    Xe = feat[cols].to_numpy(dtype=np.float32)
    gb = build_tabular_model("xgboost", scale_pos_weight=pw(sp.train_idx))
    gb.fit(Xe[sp.train_idx], y[sp.train_idx])
    m_ehr = classification_metrics(y[sp.test_idx],
                                   gb.predict_proba(Xe[sp.test_idx])[:, 1])
    print(f"    EHR 단독 AUROC {m_ehr.auroc:.4f}  PR-AUC {m_ehr.pr_auc:.4f}  "
          f"피처 {len(cols)}개")

    print("\n[7] 판정")
    best_name = max(val_auroc, key=val_auroc.get)
    best = res[best_name]["auroc"]
    lift = best - vb["auroc"]
    verdict = ("영상을 봤다" if lift >= LIFT_SAW_IMAGE
               else ("자세를 본 것" if lift < LIFT_SAW_VIEW else "판단 보류"))
    print(f"    최고 {best_name} {best:.4f}  -  자세 기준선 {vb['auroc']:.4f}"
          f"  =  추가분 {lift:+.4f}   [{verdict}]")
    print(f"    EHR 대비 {best - m_ehr.auroc:+.4f}")

    out = {
        "phase": "2-C", "model": MODEL_ID, "seed": args.seed,
        "n_admissions": len(df), "embedding_dim": int(emb.shape[1]),
        "pca_dim": args.pca_dim,
        "n_parameters": int(n_par),
        "split": sp.sizes(),
        "view_baseline": vb,
        "heads": res, "best_head": best_name,
        "head_selection": {"rule": "val AUROC 최고", "val_auroc": val_auroc},
        "ehr_same_subset": m_ehr.as_dict(),
        "lift_over_view": float(lift),
        "verdict": verdict,
        "pretrain_note": ("RAD-DINO 사전학습 데이터에 MIMIC-CXR 이 포함되어 "
                          "test 영상을 라벨 없이 본 적이 있다. 재입원 라벨은 EHR "
                          "에서 오므로 라벨 누수는 없으며, 이 편향은 영상에 "
                          "유리한 방향이다."),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    METRICS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                       default=float), encoding="utf-8")
    print(f"\n[8] 저장\n    {METRICS_PATH}")
    print("\n" + "=" * 74)
    print("  Phase 2-C 완료")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
