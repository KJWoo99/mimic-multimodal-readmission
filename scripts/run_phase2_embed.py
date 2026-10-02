"""Phase 2-B: CXR 임베딩 캐시 추출.

Phase 2 가 학습한 인코더로 영상 35,599장을 한 번만 통과시켜 특징 벡터를
저장함. 이후 Phase 3~5 는 원본 이미지 대신 이 벡터를 읽으므로 GPU 없이 돌아감.

    python scripts/run_phase2_embed.py            # 기본 체크포인트 사용
    python scripts/run_phase2_embed.py --check    # 준비 상태만 확인

## 왜 이렇게 하는가

Fusion(Phase 3), 분할 방식 비교(V1), 누수 A/B(V6), 시간분할(V7), 촬영시점 윈도우
실험(M2) 은 전부 같은 인코더를 그대로 두고 뒷단만 바꾸는 실험임. 매번
CNN 을 다시 돌리면 실험 하나에 50분씩 드는데, 임베딩을 고정해두면 수 초로 줄고
"인코더는 동일하다"는 비교 조건도 자동으로 보장됨.

Grad-CAM(V5) 은 예외임: 원본 픽셀에 역전파를 걸어야 하므로 이 캐시로는 못 하고
체크포인트를 직접 불러 GPU 에서 돌려야 함.

## DUA

임베딩은 MIMIC 영상의 파생물임. `.npz` 는 `.gitignore` 로 막혀 있고
`outputs/models/` 아래 두므로 저장소에 올라가지 않음. `subject_id`, `hadm_id`
같은 식별자를 함께 저장하지만 이는 로컬 조인용이며, 파일 자체가 공개 대상이 아님.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "outputs"
EMBED_PATH = OUT_DIR / "models" / "phase2_cxr_embeddings.npz"


def main() -> int:
    from datapaths import RAW

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--model", default="densenet121")
    # Phase 2 는 `--tag` 로 여러 설정을 나란히 남김. 여기서도 같은 꼬리표를
    # 받아 그 실행의 체크포인트를 씀.
    #
    # 꼬리표를 받지 않으면 새 설정으로 Phase 2 를 돌려도 옛 체크포인트로 임베딩을 뽑고,
    # 그 위에 얹힌 Phase 3, 4, 5 가 실패 없이 옛 모델의 결과가 됨(트러블슈팅 17).
    ap.add_argument("--tag", default="",
                    help="Phase 2 실행 꼬리표 (예: efficientnet_b0). 비우면 기본 실행")
    # 인코더는 학습할 때의 해상도로 써야 함. 384px 로 학습한 가중치에
    # 224px 를 넣으면 특징이 달라짐.
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("  Phase 2-B: CXR 임베딩 캐시 추출")
    print("=" * 74)

    suffix = f"_{args.tag}" if args.tag else ""
    metrics_path = OUT_DIR / f"phase2_metrics{suffix}.json"
    ckpt_name = f"phase2_cxr_{args.model}{suffix}"

    print("\n[0] 입력 점검")
    if not metrics_path.exists():
        print(f"  [오류] {metrics_path.name} 없음: Phase 2 를 먼저 실행하세요.")
        if args.tag:
            print(f"     (--tag {args.tag} 로 돌린 결과를 찾고 있습니다)")
        return 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ck = metrics.get("checkpoint")
    if not ck:
        print(f"  [오류] {metrics_path.name} 에 checkpoint 항목이 없습니다.")
        print("     체크포인트 저장 기능이 붙기 전의 실행 결과입니다: ")
        print("     python scripts/run_phase2.py 를 다시 돌려 가중치를 남기세요.")
        return 1
    # 학습 때와 다른 해상도로 뽑으면 특징이 달라지는데 오류는 안 남.
    trained_size = metrics.get("img_size")
    if trained_size and trained_size != args.size:
        print(f"  [오류] 이 체크포인트는 {trained_size}px 로 학습됐는데 "
              f"--size {args.size} 로 뽑으려 합니다.")
        print(f"     --size {trained_size} 로 다시 실행하세요.")
        return 1
    print(f"  [OK] 체크포인트 지문 {ck['sha256'][:16]}…  ({ck['path']})")
    print(f"     출처 {metrics_path.name}  모델 {args.model}  {args.size}px")

    if args.check:
        print("\n준비 완료.")
        return 0

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from checkpoint import load_checkpoint
    from cohort import CohortConfig, build_cohort, load_admissions
    from cxr_dataset import CXRDataset, build_cxr_index
    from features import ELECTIVE_ADMISSION_TYPES
    from models import _head_module_names, build_image_model

    raw = Path(args.raw_dir)
    img_root = raw / "mimic-cxr-jpg"

    print("\n[1] 코호트 + 영상 인덱스 (Phase 2 와 동일 규칙)")
    meta_hits = list(raw.rglob("mimic-cxr-2.0.0-metadata.csv.gz"))
    if not meta_hits:
        print("  [오류] CXR 메타데이터가 없습니다: sh scripts/download_mimic.sh meta 먼저 실행")
        return 1
    adm, pat = load_admissions(raw)
    coh = build_cohort(adm, pat, CohortConfig(
        elective_admission_types=ELECTIVE_ADMISSION_TYPES))
    # run_phase2.py 와 같은 인자, 같은 컬럼으로 인덱스를 만듦. 규칙이 조금이라도
    # 다르면 임베딩이 Phase 2 예측과 다른 행 집합을 가리켜 이후 Fusion 이 어긋남.
    meta = pd.read_csv(meta_hits[0],
                       usecols=["subject_id", "study_id", "dicom_id",
                                "StudyDate", "StudyTime", "ViewPosition"])
    idx = build_cxr_index(coh.df, meta, img_root)
    df = idx.df
    print(f"    영상 연결 {len(df):,} 입원")
    for k, v in idx.dropped.items():
        print(f"      제외 {k}: {v:,}")
    if df.empty:
        print("  [오류] 영상 파일이 없습니다: sh scripts/download_mimic.sh images 를 실행할 것")
        return 1

    print("\n[2] 체크포인트 로드")
    # 지문을 대조해 다른 실행의 가중치를 잘못 집어드는 것을 막음.
    state, info = load_checkpoint(ckpt_name, expect_sha256=ck["sha256"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_image_model(args.model, in_channels=3, num_classes=1)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"    장치 {device}  파라미터 {info.n_parameters:,}")

    # 분류 헤드를 떼고 그 앞의 특징 벡터를 뽑음.
    # 헤드 이름은 models._head_module_names 를 재사용함: 여기서 목록을 따로
    # 들고 있으면 모델이 추가될 때 한쪽만 갱신돼 어긋남.
    # (efficientnet_b0 의 classifier 는 Sequential(Dropout, Linear) 인데, eval 모드
    #  에서는 Dropout 이 항등이라 그대로 head 로 써도 원래 로짓과 같음.)
    head_attr = _head_module_names(args.model)[0]
    head = getattr(model, head_attr)
    setattr(model, head_attr, torch.nn.Identity())

    print("\n[3] 임베딩 추출")
    loader = DataLoader(
        CXRDataset(df, False, args.size), batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    t0 = time.time()
    chunks: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            feat = model(x)                    # (B, D): 헤드 제거 상태
            logit = head(feat).squeeze(-1)     # 원래 헤드로 예측도 함께 남김
            chunks.append(feat.float().cpu().numpy())
            logits.append(logit.float().cpu().numpy())
            if (i + 1) % 50 == 0:
                done = (i + 1) * args.batch
                print(f"    {min(done, len(df)):,}/{len(df):,}  "
                      f"({time.time() - t0:.0f}s)")
    emb = np.concatenate(chunks).astype(np.float32)
    logit_arr = np.concatenate(logits).astype(np.float32)
    print(f"    완료 {emb.shape}  ({time.time() - t0:.1f}s)")

    print("\n[4] 저장")
    EMBED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMBED_PATH,
        embedding=emb,
        logit=logit_arr,
        hadm_id=df["hadm_id"].to_numpy().astype(np.int64),
        subject_id=df["subject_id"].to_numpy().astype(np.int64),
        readmit_30d=df["readmit_30d"].to_numpy().astype(np.int8),
        checkpoint_sha256=np.array(ck["sha256"]),
        model=np.array(args.model),
        img_size=np.array(args.size),
    )
    size_mb = EMBED_PATH.stat().st_size / 1024**2
    print(f"    {EMBED_PATH}  ({size_mb:.1f} MB)")

    # 변수명을 meta 로 재사용하지 않음: 위에서 meta 는 CXR 메타데이터
    # DataFrame 임. 같은 이름을 덮어쓰면 이후 코드를 덧붙일 때 어긋남.
    summary = {
        "phase": "2-B",
        "n_admissions": len(df),
        "embedding_dim": int(emb.shape[1]),
        "model": args.model,
        "img_size": args.size,
        # 어느 Phase 2 실행에서 나온 임베딩인지 남김. 이것이 없으면 Phase 3~5 의
        # 수치가 어느 인코더의 것인지 나중에 확인할 방법이 없음.
        "tag": args.tag,
        "source_metrics": metrics_path.name,
        "checkpoint_sha256": ck["sha256"],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    meta_path = OUT_DIR / "phase2_embed_meta.json"
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    {meta_path}")

    print("\n" + "=" * 74)
    print("  Phase 2-B 완료. Phase 3(Fusion)부터는 이 캐시로 CPU 에서 돌아간다.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
