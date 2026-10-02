"""모델 팩토리: 정형(GBDT)과 영상(CNN) 단일 진입점.

Phase 0 에서는 공개 데이터로 배관만 검증하고, Phase 1 이후 MIMIC 으로 데이터만 갈아끼움.
그래서 모델 생성과 데이터는 처음부터 분리해 둠.

영상 모델 선택
-------------
흔히 쓰는 CXR 해상도는 224x224 이지만 보고한 결과는 384px 로 냄(절제 기준선 설정임. #2 에서 224 로 줄였을 때는 차이 없음). DenseNet121 은 CheXNet 이래 흉부 X선의 사실상 기본선임.
ResNet50 / EfficientNet 도 같은 인터페이스로 붙여 비교할 수 있게 함. CNN 하나로
결론을 내면 그 모델의 한계를 데이터의 한계로 착각할 수 있어, 같은 데이터로 여럿을 비교함.

2단계 파인튜닝
-------------
백본 동결 -> 헤드만 학습 -> 전체 미세조정.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "GBDT_SEED",
    "IMAGE_MODELS",
    "build_image_model",
    "build_tabular_model",
    "count_trainable",
    "freeze_backbone",
    "unfreeze_all",
]

IMAGE_MODELS = ("densenet121", "resnet50", "efficientnet_b0")


def build_image_model(
    name: str = "densenet121",
    num_classes: int = 1,
    pretrained: bool = True,
    in_channels: int = 3,
) -> nn.Module:
    """ImageNet 사전학습 백본 + 이진 분류 헤드(로짓 1개).

    출력이 1개인 이유: BCEWithLogitsLoss 로 pos_weight 를 걸어 불균형을 다루기 위함.
    (2클래스 softmax 보다 임계값 조정, 보정이 직관적임)
    """
    from torchvision import models as tvm

    name = name.lower()
    if name not in IMAGE_MODELS:
        raise ValueError(f"알 수 없는 모델 {name!r}. 지원: {IMAGE_MODELS}")

    weights = "DEFAULT" if pretrained else None
    if name == "densenet121":
        m = tvm.densenet121(weights=weights)
        in_f = m.classifier.in_features
        m.classifier = nn.Linear(in_f, num_classes)
    elif name == "resnet50":
        m = tvm.resnet50(weights=weights)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    else:  # efficientnet_b0
        m = tvm.efficientnet_b0(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)

    if in_channels != 3:
        m = _adapt_input_channels(m, name, in_channels)
    return m


def _adapt_input_channels(model: nn.Module, name: str, in_channels: int) -> nn.Module:
    """1채널 등 비 RGB 입력용 첫 conv 교체.

    사전학습 가중치를 채널 평균으로 축약해 승계함(무작위 초기화보다 수렴이 빠름).
    다만 CXR 은 3채널 복제로 쓰는 것이 관행이라 기본 경로는 아님.
    """
    if name == "densenet121":
        old = model.features.conv0
    elif name == "resnet50":
        old = model.conv1
    else:
        old = model.features[0][0]

    new = nn.Conv2d(in_channels, old.out_channels, old.kernel_size,
                    old.stride, old.padding, bias=old.bias is not None)
    with torch.no_grad():
        w = old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
        new.weight.copy_(w)
        if old.bias is not None:
            new.bias.copy_(old.bias)

    if name == "densenet121":
        model.features.conv0 = new
    elif name == "resnet50":
        model.conv1 = new
    else:
        model.features[0][0] = new
    return model


def _head_module_names(name: str) -> tuple[str, ...]:
    return {"densenet121": ("classifier",),
            "resnet50": ("fc",),
            "efficientnet_b0": ("classifier",)}[name.lower()]


def freeze_backbone(model: nn.Module, name: str) -> None:
    """헤드만 학습 (2단계 파인튜닝 1단계)."""
    heads = _head_module_names(name)
    for pname, p in model.named_parameters():
        p.requires_grad = any(pname.startswith(h) or f".{h}." in pname for h in heads)


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# GBDT 내부 난수(subsample, colsample)의 시드. 분할 시드와 다른 값임.
#
# 분할 시드(`--seed 42`)와 별개로, 기록된 수치가 어떤 모델 난수로 나왔는지 보이게 여기에 적음.
# 값은 xgboost, lightgbm 의 기본값(0)과 같음. 42 로 맞추면 Phase 1~5 의 GBDT 수치가 전부 조금씩 달라짐.
GBDT_SEED = 0


def build_tabular_model(name: str = "xgboost", scale_pos_weight: float | None = None,
                        seed: int = GBDT_SEED, **kw):
    """정형 데이터용 GBDT.

    `seed` 는 모델 내부 난수만 정함. 분할은 `splits.patient_split(seed=...)` 가
    따로 받음. 둘을 하나로 묶으면 시드를 바꿀 때 test 집합까지 바뀌어, 앞 실행의
    val 환자가 뒤 실행의 test 로 들어감.

    SMOTE 등 리샘플링을 쓸 때는 반드시 imblearn Pipeline 안에 넣어
    train fold 안에서만 적용되게 함. 분할 전에 적용하면 합성 샘플이
    val/test 로 새어 누수가 됨.
    """
    name = name.lower()
    if name == "xgboost":
        from xgboost import XGBClassifier

        params = dict(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr",  # 불균형이므로 PR 기준
            tree_method="hist", verbosity=0, random_state=seed,
        )
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
        params.update(kw)
        return XGBClassifier(**params)

    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        params = dict(n_estimators=400, max_depth=-1, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, verbose=-1,
                      random_state=seed)
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
        params.update(kw)
        return LGBMClassifier(**params)

    if name == "logreg":
        # 문헌 baseline 비교용 (LACE 0.61 / LR 0.62 수준, docs/report.md 성능 기대치 절)
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=seed, **kw)

    raise ValueError(f"알 수 없는 모델 {name!r}. 지원: xgboost/lightgbm/logreg")
