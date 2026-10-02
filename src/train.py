"""영상 모델 학습 루프: AMP, 2단계 파인튜닝, EarlyStopping.

Phase 0 에서는 공개 데이터로 이 루프가 도는지만 확인하고,
Phase 2 이후 CXR subset 으로 데이터만 교체함.

메모리
----
학습은 AMP(fp16)로 돎. 배치, 입력 크기는 러너에서 인자로 받으며 여기서 고정하지
않음: 어느 쪽도 정확도상의 근거로 고른 값이 아니라 절제실험에서 바꿔볼 축임.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import freeze_backbone, unfreeze_all

__all__ = ["EpochLog", "TrainConfig", "predict_proba", "set_seed",
           "train_image_model"]


@dataclass
class TrainConfig:
    epochs: int = 20
    freeze_epochs: int = 3          # 1단계(헤드만) 에폭 수
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4       # 2단계에서 백본에 적용할 낮은 lr
    weight_decay: float = 1e-4
    batch: int = 32
    amp: bool = True
    early_stop_patience: int = 5
    seed: int = 42
    monitor: str = "pr_auc"         # 불균형이므로 PR-AUC 로 모델 선택
    # LR 스케줄. 기본은 "none" 이고, 보고한 Phase 2 실행은 전부 `--scheduler cosine` 으로 돌림.
    # 긴 학습(수십 에폭)에서는 고정 LR 이 후반에 진동만 하므로 "cosine" 을 씀.
    scheduler: str = "none"         # "none" | "cosine"
    min_lr_factor: float = 0.05     # cosine 의 하한 = 초기 lr x 이 값


@dataclass
class EpochLog:
    epoch: int
    phase: str                      # "head" | "full"
    train_loss: float
    val_metric: float
    lr: float
    seconds: float
    is_best: bool = False


@dataclass
class TrainResult:
    best_metric: float
    best_epoch: int
    best_state: dict = field(default_factory=dict)
    history: list[EpochLog] = field(default_factory=list)


def set_seed(seed: int) -> None:
    """난수를 고정함.

    모델을 만들기 전에 불러야 함. 학습 함수 안에서만 부르면 늦음. 분류기
    헤드는 `build_image_model()` 시점에 그때의 전역 난수 상태로 초기화되므로,
    그 앞에서 난수를 얼마나 썼는지에 따라 초기 가중치가 달라짐.

    그러면 같은 설정, 같은 분할, 같은 시드로 돌린 두 스크립트가 서로 다른 궤적을 그림(트러블슈팅 16).
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 예전 이름. 내부 호출부가 남아 있어 유지함.
_set_seed = set_seed


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """(확률, 정답) 반환. 확률이어야 보정(ECE/Brier)을 잴 수 있음."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=(device.type == "cuda")):
            logit = model(x).squeeze(1)
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        labels.append(y.numpy().ravel())
    return np.concatenate(probs), np.concatenate(labels)


def _make_sched(cfg, optim, n_epochs: int):
    """단계별 cosine 스케줄러. 설정이 "none" 이면 만들지 않음.

    T_max 를 그 단계의 에폭 수로 줌. 전체 에폭 수를 주면 1단계가 끝나기도 전에
    일정의 앞부분만 쓰고 버려져, 2단계가 엉뚱하게 높은 LR 에서 시작함.

    하한은 파라미터 그룹마다 자기 초기 LR 기준임
    -------------------------------------------------
    `CosineAnnealingLR` 의 `eta_min` 은 스칼라 하나라 모든 그룹에 같은 값이 걸림.
    2단계는 백본(lr_backbone)과 헤드(lr_head)가 10배 차이 나므로
    `eta_min=lr_head * min_lr_factor` 를 주면 헤드는 1e-3 -> 5e-5 로 20배
    줄지만 백본은 1e-4 -> 5e-5, 겨우 2배만 줄어 사실상 감쇠가 없음(트러블슈팅 21).

    같은 곡선을 배율로 쓰면 그룹마다 자기 초기 LR 에 곱해지므로 하한이
    자동으로 그룹별이 됨. lr_t = base * (f + (1-f) * (1 + cos(pi t / T)) / 2).
    """
    if cfg.scheduler != "cosine" or n_epochs <= 0:
        return None
    import math

    import torch as _t

    f = cfg.min_lr_factor

    def factor(epoch: int) -> float:
        t = min(epoch, n_epochs)
        return f + (1.0 - f) * (1.0 + math.cos(math.pi * t / n_epochs)) / 2.0

    return _t.optim.lr_scheduler.LambdaLR(optim, lr_lambda=factor)


def train_image_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    device: torch.device | str = "cuda",
    pos_weight: float | None = None,
    metric_fn=None,
    verbose: bool = True,
) -> TrainResult:
    """2단계 파인튜닝 학습.

    Parameters
    ----------
    metric_fn : (y_true, y_prob) -> float
        모델 선택 기준. 기본은 PR-AUC (불균형 대응).
    pos_weight : 양성 가중치. 보통 n_neg / n_pos.
    """
    device = torch.device(device)
    model = model.to(device)
    _set_seed(cfg.seed)

    if metric_fn is None:
        from sklearn.metrics import average_precision_score

        metric_fn = average_precision_score

    pw = torch.tensor([pos_weight], device=device) if pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    scaler = torch.amp.GradScaler(device.type, enabled=(cfg.amp and device.type == "cuda"))

    history: list[EpochLog] = []
    best_metric, best_epoch, best_state = -np.inf, -1, {}
    patience = 0
    sched = None

    for epoch in range(1, cfg.epochs + 1):
        phase = "head" if epoch <= cfg.freeze_epochs else "full"
        # 두 분기를 독립적으로 둠. elif 로 묶으면 freeze_epochs=0 일 때
        # (= 처음부터 전체 파인튜닝하려는 설정) epoch 1 에서 동결만 걸리고 해제
        # 분기가 실행되지 않아, 로그는 "full" 인데 백본은 끝까지 얼어붙음.
        # 에러 없이 헤드만 학습되므로 결과를 보기 전에는 알아채기 어려움.
        if epoch == 1 and cfg.freeze_epochs > 0:
            freeze_backbone(model, model_name)
            optim = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg.lr_head, weight_decay=cfg.weight_decay,
            )
            sched = _make_sched(cfg, optim, cfg.freeze_epochs)
        if epoch == cfg.freeze_epochs + 1:
            unfreeze_all(model)
            heads = {"densenet121": "classifier", "resnet50": "fc",
                     "efficientnet_b0": "classifier"}[model_name]
            optim = torch.optim.AdamW([
                {"params": [p for n, p in model.named_parameters() if not n.startswith(heads)],
                 "lr": cfg.lr_backbone},
                {"params": [p for n, p in model.named_parameters() if n.startswith(heads)],
                 "lr": cfg.lr_head},
            ], weight_decay=cfg.weight_decay)
            # 2단계에서 옵티마이저를 새로 만들므로 스케줄러도 다시 만듦.
            # 그러지 않으면 1단계용 스케줄러가 남아 엉뚱한 파라미터 그룹을 붙들고
            # 있거나, 이미 다 소진된 일정이 그대로 이어져 LR 이 하한에 붙음.
            sched = _make_sched(cfg, optim, cfg.epochs - cfg.freeze_epochs)

        t0 = time.time()
        model.train()
        total, nb = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().view(-1)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=(cfg.amp and device.type == "cuda")):
                loss = criterion(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            total += float(loss.item())
            nb += 1

        lr_now = optim.param_groups[0]["lr"]
        if sched is not None:
            sched.step()

        y_prob, y_true = predict_proba(model, val_loader, device)
        val_metric = float(metric_fn(y_true, y_prob))
        is_best = val_metric > best_metric
        if is_best:
            best_metric, best_epoch = val_metric, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        log = EpochLog(epoch, phase, total / max(1, nb), val_metric,
                       lr_now, time.time() - t0, is_best)
        history.append(log)
        if verbose:
            print(f"  ep{epoch:3d} [{phase:4s}] loss={log.train_loss:.4f} "
                  f"val_{cfg.monitor}={val_metric:.4f} ({log.seconds:.1f}s)"
                  + ("  *best" if is_best else ""))

        if patience >= cfg.early_stop_patience:
            if verbose:
                print(f"  [early-stop] {cfg.early_stop_patience} epoch 개선 없음 -> 중단")
            break

    if best_state:
        model.load_state_dict(best_state)
    return TrainResult(best_metric, best_epoch, best_state, history)
