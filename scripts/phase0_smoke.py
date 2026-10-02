"""Phase 0: 배관 검증.

성능이 아니라 파이프가 연결돼 있는지만 봄. 여기 숫자는 결과로 보고하지 않음.
개체 단위 분할, 누수 검사, 지표, 영상 학습 루프, 정형 GBDT, MLflow 기록을 훑음.

PneumoniaMNIST(MedMNIST v2, CC BY 4.0)를 씀. 흉부 X선이라 도메인이 같고
공개 데이터라 DUA 제약이 없음. MIMIC 은 여기서 쓰지 않음.

    python scripts/phase0_smoke.py --epochs 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from datapaths import PUBLIC

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--freeze-epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--model", default="densenet121")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--limit-train", type=int, default=2000, help="배관 검증이므로 일부만 사용")
    ap.add_argument("--data-root", default=str(PUBLIC))
    args = ap.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from metrics import classification_metrics, reliability_curve, subgroup_metrics
    from models import build_image_model, build_tabular_model, count_trainable
    from splits import LeakageError, assert_no_group_overlap, check_split, patient_split
    from tracking import ExperimentLogger
    from train import TrainConfig, predict_proba, train_image_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("  Phase 0: 배관 검증 (PneumoniaMNIST, 공개 데이터)")
    print("=" * 72)
    print(f"  device : {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else '-'})")

    import medmnist
    from medmnist import INFO

    info = INFO["pneumoniamnist"]
    DataClass = getattr(medmnist, info["python_class"])
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"\n[1] PneumoniaMNIST 로드 ({info['license']})")
    splits = {}
    for sp in ("train", "val", "test"):
        ds = DataClass(split=sp, download=True, root=str(root), size=28)
        splits[sp] = (ds.imgs, ds.labels.ravel().astype(int))
        print(f"    {sp:5s} {ds.imgs.shape}  양성률={splits[sp][1].mean():.3f}")

    # PneumoniaMNIST 에는 환자 ID 가 없음. 배관 검증이 목적이므로 가상의 개체 ID를
    # 부여해 "개체 단위 분할이 실제로 동작하는가"만 확인함.
    # (MIMIC 에서는 subject_id 를 그대로 씀)
    print("\n[2] 개체 단위 분할 + 누수 검사")
    X_tr, y_tr = splits["train"]
    rng = np.random.default_rng(42)
    n = len(y_tr)
    fake_subject = rng.integers(0, n // 4, n)  # 환자당 평균 4장 모사

    sp_res = patient_split(fake_subject, test_size=0.2, val_size=0.1, seed=42)
    split_info = check_split(y_tr, fake_subject, sp_res)
    for k, v in split_info.items():
        print(f"    {k:20s} {v}")

    # 반례 확인: 행 단위 무작위 분할이면 반드시 걸려야 함
    perm = rng.permutation(n)
    try:
        assert_no_group_overlap(fake_subject, perm[: int(n * 0.8)], perm[int(n * 0.8):],
                                names=("train", "test"))
        print("    [실패] 행 단위 분할이 누수 검사를 통과했다")
        return 1
    except LeakageError:
        print("    행 단위 무작위 분할 -> 누수 정상 탐지 (반례 확인)")

    print(f"\n[3] 영상 모델 학습 ({args.model}, 2단계 파인튜닝)")

    def to_tensor(imgs: np.ndarray, labels: np.ndarray, size: int):
        t = torch.from_numpy(imgs).float().div_(255.0).unsqueeze(1)      # [N,1,28,28]
        t = torch.nn.functional.interpolate(t, size=(size, size),
                                            mode="bilinear", align_corners=False)
        t = t.repeat(1, 3, 1, 1)                                          # 사전학습 백본용 3채널
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        t = (t - mean) / std
        return TensorDataset(t, torch.from_numpy(labels).long())

    lim = args.limit_train
    tr_ds = to_tensor(X_tr[:lim], y_tr[:lim], args.img_size)
    va_ds = to_tensor(*splits["val"], args.img_size)
    te_ds = to_tensor(*splits["test"], args.img_size)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch, shuffle=False)
    te_dl = DataLoader(te_ds, batch_size=args.batch, shuffle=False)
    print(f"    train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)} (배관 검증용 축소)")

    model = build_image_model(args.model, num_classes=1, pretrained=True)
    n_pos = int(y_tr[:lim].sum())
    n_neg = len(y_tr[:lim]) - n_pos
    tcfg = TrainConfig(epochs=args.epochs, freeze_epochs=args.freeze_epochs,
                       batch=args.batch)

    params = {
        "task": "phase0-plumbing", "data": "PneumoniaMNIST",
        "model": args.model, "img_size": args.img_size,
        "epochs": tcfg.epochs, "freeze_epochs": tcfg.freeze_epochs,
        "batch": tcfg.batch, "lr_head": tcfg.lr_head, "lr_backbone": tcfg.lr_backbone,
        "seed": tcfg.seed, "pos_weight": round(n_neg / max(1, n_pos), 4),
    }

    with ExperimentLogger("phase0-plumbing", params=params, run_name=f"{args.model}-smoke") as run:
        run.note("Phase 0 배관 검증. 성능 목적 아님: 이 숫자는 결과로 보고하지 않는다.")
        run.log_split(
            n_train=len(tr_ds), n_val=len(va_ds), n_test=len(te_ds),
            method="patient",
            pos_rate={"train": float(y_tr[:lim].mean()),
                      "val": float(splits["val"][1].mean()),
                      "test": float(splits["test"][1].mean())},
            n_subjects={"train": int(split_info.get("n_subjects_train", 0))},
        )

        res = train_image_model(model, args.model, tr_dl, va_dl, tcfg,
                                device=device, pos_weight=n_neg / max(1, n_pos))
        print(f"    best epoch={res.best_epoch} val PR-AUC={res.best_metric:.4f}"
              f"  (학습가능 파라미터 {count_trainable(model):,})")
        for log in res.history:
            run.log_metrics({"train_loss": log.train_loss, "val_pr_auc": log.val_metric},
                            step=log.epoch)

        print("\n[4] 평가 지표 (test)")
        prob, true = predict_proba(model, te_dl, device)
        m = classification_metrics(true, prob)
        for k, v in m.as_dict().items():
            print(f"    {k:16s} {v:.4f}" if isinstance(v, float) else f"    {k:16s} {v}")
        if m.warnings:
            print(f"    warnings: {m.warnings}")
        run.log_metrics({f"test_{k}": v for k, v in m.as_dict().items()})

        rc = reliability_curve(true, prob, n_bins=10)
        print(f"    reliability bins={len(rc['count'])} (보정 곡선 데이터 생성 확인)")

        # 서브그룹 분해 (V4 배관). 실제로는 성별, 연령대, 촬영자세를 씀.
        fake_group = rng.choice(["A", "B"], len(true))
        sg = subgroup_metrics(true, prob, fake_group)
        for gname, gm in sg.items():
            print(f"    subgroup {gname}: n={gm.n} auroc={gm.auroc:.4f} ece={gm.ece:.4f}")
            run.log_subgroup("fake_group", {gname: {"auroc": gm.auroc, "ece": gm.ece}})

        print("\n[5] 정형 GBDT 경로 (28x28 평탄화 = 배관 확인용 더미 피처)")
        # reshape(lim, -1) 로 쓰면 --limit-train 이 데이터 수보다 클 때
        # "cannot reshape array" 라는 엉뚱한 에러가 남. 실제 잘린 길이를 씀.
        sub_x, sub_y = X_tr[:lim], y_tr[:lim]
        flat_tr = sub_x.reshape(len(sub_x), -1).astype(np.float32) / 255.0
        flat_te = splits["test"][0].reshape(len(splits["test"][1]), -1).astype(np.float32) / 255.0
        gbdt = build_tabular_model("xgboost", scale_pos_weight=n_neg / max(1, n_pos),
                                   n_estimators=60, max_depth=3)
        gbdt.fit(flat_tr, sub_y)
        gprob = gbdt.predict_proba(flat_te)[:, 1]
        gm = classification_metrics(splits["test"][1], gprob)
        print(f"    XGBoost  AUROC={gm.auroc:.4f} PR-AUC={gm.pr_auc:.4f} "
              f"(baseline={gm.prevalence_baseline_pr_auc:.4f}) ECE={gm.ece:.4f}")
        run.log_metrics({"tabular_auroc": gm.auroc, "tabular_pr_auc": gm.pr_auc,
                         "tabular_ece": gm.ece})

        run_id = run.run_id

    print("\n[6] MLflow 기록 확인")
    import mlflow

    r = mlflow.get_run(run_id)
    print(f"    run_id       {run_id[:16]}")
    print(f"    git_commit   {r.data.tags.get('git_commit_short')} "
          f"(dirty={r.data.tags.get('git_dirty')})")
    print(f"    split_method {r.data.params.get('split_method')}")
    print(f"    기록된 지표  {len(r.data.metrics)}개 / 파라미터 {len(r.data.params)}개")

    print("\n" + "=" * 72)
    print("  Phase 0 배관 검증 완료: 6개 구간 전부 통과")
    print("  주의: 위 성능 숫자는 배관 확인용이며 결과로 보고하지 않는다.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
