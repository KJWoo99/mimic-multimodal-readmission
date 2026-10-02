"""2단계 파인튜닝 스케줄 검증.

`train_image_model` 은 앞 `freeze_epochs` 에폭 동안 백본을 얼려 헤드만 학습하고,
그 뒤 전체를 품. 이 전환이 어긋나면 에러 없이 학습 대상이 달라져,
"2단계 파인튜닝했다"는 기록과 실제가 따로 놈.

동결과 해제가 `if / elif` 로 묶이면 `freeze_epochs=0`(처음부터 전체 파인튜닝) 일 때 epoch 1 에서
동결만 걸리고 해제가 실행되지 않음: 로그는 "full" 인데 백본은 끝까지 얼어붙은 채 헤드만 학습됨.
두 분기가 독립인지 여기서 고정함.

GPU, 데이터가 필요 없도록 스케줄 로직만 떼어 검사함.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import build_image_model, freeze_backbone, unfreeze_all


def _trainable_ratio(model) -> float:
    total = sum(p.numel() for p in model.parameters())
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / total


def _simulate(freeze_epochs: int, n_epochs: int = 4) -> list[tuple[str, float]]:
    """train_image_model 의 에폭별 동결/해제 분기를 그대로 재현함."""
    model = build_image_model("densenet121", in_channels=3, num_classes=1)
    out = []
    for epoch in range(1, n_epochs + 1):
        phase = "head" if epoch <= freeze_epochs else "full"
        # --- train.py 와 동일한 분기 ---
        if epoch == 1 and freeze_epochs > 0:
            freeze_backbone(model, "densenet121")
        if epoch == freeze_epochs + 1:
            unfreeze_all(model)
        # ------------------------------
        out.append((phase, _trainable_ratio(model)))
    return out


@pytest.mark.parametrize("freeze_epochs", [0, 1, 3])
def test_phase_label_matches_actual_trainable_params(freeze_epochs):
    """'full' 로 기록된 에폭은 실제로 전체가 학습 가능해야 함.

    라벨과 실제가 어긋나면 실험 기록이 거짓이 됨.
    """
    for epoch, (phase, ratio) in enumerate(_simulate(freeze_epochs), start=1):
        if phase == "full":
            assert ratio == pytest.approx(1.0), (
                f"freeze_epochs={freeze_epochs} ep{epoch}: "
                f"'full' 인데 학습 가능 비율이 {ratio:.1%} 다"
            )
        else:
            assert ratio < 0.05, (
                f"freeze_epochs={freeze_epochs} ep{epoch}: "
                f"'head' 인데 백본이 안 얼었다({ratio:.1%})"
            )


def test_zero_freeze_epochs_trains_everything_from_start():
    """freeze_epochs=0 은 1에폭부터 전체 학습이어야 함."""
    schedule = _simulate(0)
    assert all(phase == "full" for phase, _ in schedule)
    assert all(ratio == pytest.approx(1.0) for _, ratio in schedule)


def test_default_three_epoch_freeze_is_unchanged():
    """기본값(3)의 동작은 바뀌면 안 됨: 이미 낸 Phase 2 결과의 전제임."""
    schedule = _simulate(3)
    phases = [p for p, _ in schedule]
    ratios = [r for _, r in schedule]
    assert phases == ["head", "head", "head", "full"]
    assert ratios[0] < 0.05 and ratios[1] < 0.05 and ratios[2] < 0.05
    assert ratios[3] == pytest.approx(1.0)


def test_unfreeze_happens_exactly_once_at_boundary():
    """해제는 freeze_epochs+1 에폭에 딱 한 번 일어남."""
    schedule = _simulate(2, n_epochs=5)
    ratios = [r for _, r in schedule]
    # ep1,2 는 동결 / ep3 부터 해제
    assert ratios[0] < 0.05 and ratios[1] < 0.05
    assert all(r == pytest.approx(1.0) for r in ratios[2:])


def test_set_seed_before_model_build_makes_init_reproducible():
    """모델 생성 앞에서 시드를 걸면, 그 앞에서 난수를 얼마나 썼든 초기값이 같음.

    이 성질이 없으면 같은 설정, 같은 분할, 같은 시드로 돌려도 두 스크립트가 다른
    궤적을 그림(트러블슈팅 16). 학습 함수 안에서만 시드를 걸면 이미 늦음: 헤드는 그 전에
    초기화된 뒤임.
    """
    import random

    import numpy as np
    import torch

    from train import set_seed

    def build_after_wasting(n: int) -> torch.Tensor:
        # 시드를 걸기 "전에" 난수를 n 번 씀 (다른 스크립트를 흉내냄)
        for _ in range(n):
            random.random()
            np.random.rand()
            torch.rand(1)
        set_seed(42)
        m = build_image_model("densenet121", in_channels=3, num_classes=1)
        return torch.cat([p.flatten() for p in m.parameters()])

    a = build_after_wasting(0)
    b = build_after_wasting(37)
    assert torch.equal(a, b), (
        "모델 생성 앞에서 시드를 걸었는데도 초기값이 다르다. "
        "set_seed 가 덮지 못하는 난수원이 있다")


def test_runners_seed_before_building_the_model():
    """두 러너 모두 build_image_model 앞에서 set_seed 를 불러야 함.

    수치로만 잡으면 GPU 로 몇 시간을 돌린 뒤에야 드러남. 호출 순서는 소스에서
    바로 읽을 수 있으므로 여기서 막음.
    """
    root = Path(__file__).resolve().parents[1]
    for name in ("run_ablation.py", "run_phase2.py"):
        src = (root / "scripts" / name).read_text(encoding="utf-8")
        i_seed = src.find("set_seed(args.seed)")
        i_build = src.find("build_image_model(")
        assert i_seed != -1, f"{name}: set_seed(args.seed) 호출이 없다"
        assert i_build != -1, f"{name}: build_image_model 호출이 없다"
        assert i_seed < i_build, (
            f"{name}: 모델을 만든 뒤에 시드를 걸고 있다. 헤드 초기값이 앞선 "
            "난수 소비량에 좌우된다")


# --------------------------------------------------------------------------
# cosine LR 하한(트러블슈팅 21)
#
# `CosineAnnealingLR` 의 eta_min 은 스칼라라 모든 파라미터 그룹에 같은 값이 걸림.
# 2단계는 백본 1e-4, 헤드 1e-3 으로 10배 차이가 나므로 eta_min 을 헤드 기준으로만
# 주면 백본은 1e-4 -> 5e-5 (2배)밖에 줄지 않음.
# --------------------------------------------------------------------------


def test_cosine_floor_is_per_group():
    import torch

    from train import TrainConfig, _make_sched

    cfg = TrainConfig(scheduler="cosine", lr_head=1e-3, lr_backbone=1e-4,
                      min_lr_factor=0.05)
    a = torch.nn.Parameter(torch.zeros(1))
    b = torch.nn.Parameter(torch.zeros(1))
    optim = torch.optim.AdamW([{"params": [a], "lr": cfg.lr_backbone},
                               {"params": [b], "lr": cfg.lr_head}])
    n = 17
    sched = _make_sched(cfg, optim, n)
    assert sched is not None

    assert optim.param_groups[0]["lr"] == pytest.approx(cfg.lr_backbone)
    assert optim.param_groups[1]["lr"] == pytest.approx(cfg.lr_head)
    for _ in range(n):
        optim.step()
        sched.step()

    assert optim.param_groups[0]["lr"] == pytest.approx(
        cfg.lr_backbone * cfg.min_lr_factor, rel=1e-6), "백본 하한이 자기 초기 LR 기준이 아니다"
    assert optim.param_groups[1]["lr"] == pytest.approx(
        cfg.lr_head * cfg.min_lr_factor, rel=1e-6), "헤드 하한이 자기 초기 LR 기준이 아니다"


def test_cosine_decays_both_groups_by_the_same_ratio():
    """두 그룹이 같은 곡선을 따라야 함: 감쇠 배율이 같아야 함."""
    import torch

    from train import TrainConfig, _make_sched

    cfg = TrainConfig(scheduler="cosine", lr_head=1e-3, lr_backbone=1e-4,
                      min_lr_factor=0.05)
    a = torch.nn.Parameter(torch.zeros(1))
    b = torch.nn.Parameter(torch.zeros(1))
    optim = torch.optim.AdamW([{"params": [a], "lr": cfg.lr_backbone},
                               {"params": [b], "lr": cfg.lr_head}])
    sched = _make_sched(cfg, optim, 10)
    for _ in range(10):
        optim.step()
        sched.step()
        r0 = optim.param_groups[0]["lr"] / cfg.lr_backbone
        r1 = optim.param_groups[1]["lr"] / cfg.lr_head
        assert r0 == pytest.approx(r1, rel=1e-9)


def test_scheduler_none_makes_nothing():
    import torch

    from train import TrainConfig, _make_sched

    cfg = TrainConfig(scheduler="none")
    a = torch.nn.Parameter(torch.zeros(1))
    optim = torch.optim.AdamW([{"params": [a], "lr": 1e-3}])
    assert _make_sched(cfg, optim, 10) is None
    assert _make_sched(TrainConfig(scheduler="cosine"), optim, 0) is None
