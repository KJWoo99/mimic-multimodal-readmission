"""절제실험 러너: 설정을 하나씩 바꿔가며 돌리고 그 경위를 기록함.

모델을 그냥 돌리는 것으로는 무엇이 결과를 좌우하는지 알 수 없음. 설정을 한 번에
하나씩 바꾸고 그 변화를 추적하는 스크립트임. 모델 여러 개를 한 번씩 돌린 것은
"여러 모델을 비교했다"일 뿐, 어떤 설정이 병목인지는 알려주지 않음.

    python scripts/run_ablation.py --id 1 --name baseline \\
        --hypothesis "기준선" --change "없음"
    python scripts/run_ablation.py --id 2 --name aug-strong \\
        --aug strong --hypothesis "6에폭에서 과적합 -> 증강을 세게 하면 늦춰질 것" \\
        --change "aug medium -> strong"

## 이 스크립트가 test 를 계산하지 않는 이유

탐색을 test 로 하면, 열 몇 번 돌려서 고른 설정에 그 시행들의 운이 섞임
(winner's curse). 그러면 test 는 더 이상 미래 성능의 추정치가 아님.

그래서 여기서는 test 를 아예 불러오지도 않음. 최종 조합이
정해지면 그때 `run_phase2.py` 로 딱 한 번 test 를 봄.

## 기록

실행마다 `outputs/ablation/<id>_<name>.json` 에 남김. 설정, 결과만이 아니라
가설, 바꾼 것, 관찰, 판단을 함께 넣음: 표만 있으면 "여러 번 돌렸다"에 그치고,
왜 그렇게 바꿨는지가 있어야 추적이 됨.
`scripts/render_experiment_log.py` 가 이것들을 읽어 `docs/EXPERIMENT_LOG.md` 를 만듦.
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

from cxr_dataset import AUG_LEVELS  # noqa: E402
from datapaths import RAW  # noqa: E402

OUT_DIR = ROOT / "outputs"
ABL_DIR = OUT_DIR / "ablation"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    # ── 기록용 (결과가 아니라 '왜'를 남김) ──────────────────────────
    ap.add_argument("--id", type=int, required=True, help="실험 번호")
    ap.add_argument("--name", required=True, help="짧은 이름 (파일명에 쓰임)")
    ap.add_argument("--hypothesis", default="", help="이번에 무엇을 기대하고 바꾸는가")
    ap.add_argument("--change", default="", help="기준선 대비 무엇을 바꿨는가")
    ap.add_argument("--parent", type=int, default=0, help="어느 실험을 보고 정했는가")
    # ── 바꿔가며 실험할 축 ────────────────────────────────────────────
    ap.add_argument("--model", default="efficientnet_b0")
    # 선택지를 여기 적어두면 증강 단계를 늘릴 때 이 파일을 같이 고쳐야 함.
    ap.add_argument("--aug", default="medium", choices=AUG_LEVELS)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--freeze-epochs", type=int, default=3)
    ap.add_argument("--scheduler", default="cosine", choices=("none", "cosine"))
    ap.add_argument("--pos-weight", default="auto",
                    help="auto | off | 실수값. 불균형 보정을 끄고 보는 실험용")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42, help="모델 초기화, 학습 시드")
    # 분할 시드를 따로 둠(run_phase2.py 와 같음). --seed 가 분할까지 정하면 시드 반복 회차가 다른 분할에서
    # 돌아 판정선(시드 표준편차의 2배)에 분할 변동이 섞이고, 다른 분할의 val 에 주 분할의 test 환자가 들어감.
    ap.add_argument("--split-seed", type=int, default=42,
                    help="환자 분할 시드. 시드 반복에서도 이 값은 고정할 것")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from cohort import CohortConfig, build_cohort, load_admissions
    from cxr_dataset import CXRDataset, build_cxr_index
    from features import ELECTIVE_ADMISSION_TYPES
    from models import build_image_model
    from splits import patient_split
    from train import TrainConfig, set_seed, train_image_model

    raw = Path(args.raw_dir)
    t0 = time.time()
    print("=" * 74)
    print(f"  절제실험 #{args.id}: {args.name}")
    print("=" * 74)
    if args.hypothesis:
        print(f"  가설    : {args.hypothesis}")
    if args.change:
        print(f"  바꾼 것 : {args.change}")
    if args.parent:
        print(f"  근거    : 실험 #{args.parent} 결과")

    meta_hits = list(raw.rglob("mimic-cxr-2.0.0-metadata.csv.gz"))
    if not meta_hits:
        print("  CXR 메타데이터가 없다.")
        return 1
    if args.check:
        print("\n준비 완료.")
        return 0

    print("\n[1] 코호트 + 영상 인덱스 (모든 실험에서 동일)")
    adm, pat = load_admissions(raw)
    coh = build_cohort(adm, pat,
                       CohortConfig(elective_admission_types=ELECTIVE_ADMISSION_TYPES))
    meta = pd.read_csv(meta_hits[0],
                       usecols=["subject_id", "study_id", "dicom_id",
                                "StudyDate", "StudyTime", "ViewPosition"])
    df = build_cxr_index(coh.df, meta, raw / "mimic-cxr-jpg").df.reset_index(drop=True)
    sp = patient_split(df["subject_id"].to_numpy(), seed=args.split_seed)
    tr_df, va_df = df.iloc[sp.train_idx], df.iloc[sp.val_idx]
    print(f"    train {len(tr_df):,} / val {len(va_df):,}"
          f"   (test {len(sp.test_idx):,} 건은 **불러오지 않는다**)")

    y_tr = tr_df["readmit_30d"].to_numpy()
    n_pos = int(y_tr.sum())
    if args.pos_weight == "auto":
        pw = float((len(y_tr) - n_pos) / max(n_pos, 1))
    elif args.pos_weight == "off":
        pw = None
    else:
        pw = float(args.pos_weight)

    kw = dict(batch_size=args.batch, num_workers=args.workers,
              pin_memory=torch.cuda.is_available(),
              persistent_workers=args.workers > 0)
    tr = DataLoader(CXRDataset(tr_df, True, args.size, args.aug), shuffle=True, **kw)
    va = DataLoader(CXRDataset(va_df, False, args.size, args.aug), shuffle=False, **kw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 모델을 만들기 직전에 시드를 설정함. 헤드 초기값이 이 시점의 전역 난수 상태로
    # 정해지므로, 여기서 고정하지 않으면 앞에서 난수를 얼마나 썼는지에 따라
    # 초기 가중치가 달라짐(트러블슈팅 16).
    set_seed(args.seed)
    model = build_image_model(args.model, in_channels=3, num_classes=1)
    cfg = TrainConfig(epochs=args.epochs, batch=args.batch, seed=args.seed,
                      early_stop_patience=args.patience,
                      freeze_epochs=args.freeze_epochs,
                      scheduler=args.scheduler, lr_head=args.lr_head,
                      lr_backbone=args.lr_backbone, weight_decay=args.weight_decay)
    print(f"\n[2] 학습  {args.model}  aug={args.aug}  lr={args.lr_head:g}/"
          f"{args.lr_backbone:g}  wd={args.weight_decay:g}  batch={args.batch}  "
          f"size={args.size}  sched={args.scheduler}  "
          f"pos_weight={'off' if pw is None else round(pw, 3)}")
    res = train_image_model(model, args.model, tr, va, cfg, device=device,
                            pos_weight=pw)

    best_ep, n_ep = res.best_epoch, len(res.history)
    stopped_early = n_ep < args.epochs
    print(f"\n[3] 결과  최고 val PR-AUC {res.best_metric:.4f} @ {best_ep}에폭"
          f"  (실행 {n_ep}/{args.epochs}, 조기종료 {stopped_early})")
    # 수렴 여부를 진단해 기록에 남김. best_epoch=1 인 모델의 점수를 "신호 없음"으로
    # 읽지 않게 함(트러블슈팅 11).
    if best_ep <= 2:
        diag = "학습 실패 의심: best 가 1~2에폭"
    elif not stopped_early:
        diag = "에폭 한도에 걸려 끊김: 수렴 아님"
    elif best_ep >= n_ep - 1:
        diag = "마지막까지 개선 중이었음"
    else:
        diag = "정상 수렴(정점 후 하락으로 조기종료)"
    print(f"    진단: {diag}")

    ABL_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "id": args.id, "name": args.name, "parent": args.parent,
        "hypothesis": args.hypothesis, "change": args.change,
        "config": {
            "model": args.model, "aug": args.aug, "lr_head": args.lr_head,
            "lr_backbone": args.lr_backbone, "weight_decay": args.weight_decay,
            "batch": args.batch, "size": args.size, "epochs": args.epochs,
            "patience": args.patience, "freeze_epochs": args.freeze_epochs,
            "scheduler": args.scheduler, "pos_weight": args.pos_weight,
            "seed": args.seed, "split_seed": args.split_seed,
        },
        "result": {
            "best_val_pr_auc": float(res.best_metric),
            "best_epoch": best_ep, "epochs_run": n_ep,
            "stopped_early": stopped_early,
            "final_train_loss": float(res.history[-1].train_loss),
            "first_train_loss": float(res.history[0].train_loss),
        },
        "diagnosis": diag,
        "history": [vars(h) for h in res.history],
        "elapsed_sec": round(time.time() - t0, 1),
        "note": "val 로만 평가. test 는 최종 조합 확인 때 run_phase2.py 로 1회만 본다.",
    }
    path = ABL_DIR / f"{args.id:02d}_{args.name}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=float),
                    encoding="utf-8")
    print(f"\n[4] 저장  {path}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
