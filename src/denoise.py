"""저선량 흉부 X선 복원: 고전 셋(가우시안, 비국소 평균, BM3D)과 학습형 하나(DnCNN).

사전등록 `docs/prereg_lowdose.md` 4절. 입력과 출력은 모두 518x518 회색조 float(0~1)임.
고전 방법의 세기는 추정 잡음에 곱하는 계수로 두고, 계수는 val 의 PSNR 로 고름(스크립트가 함).

DnCNN(Zhang 등, IEEE TIP 2017)은 잡음 자체를 예측해 입력에서 빼는 잔차 학습 구조임. 세 선량비를 섞어
한 모델로 학습함(blind). 선량비마다 따로 두면 모델 셋 x 시드 셋이 되어 비교가 늘기만 하고, 실제
장비에서도 선량을 모르는 채로 복원하는 경우가 많음.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "DnCNN",
    "classical",
    "estimate_noise",
    "psnr",
    "ssim",
]


def estimate_noise(x: np.ndarray) -> float:
    """웨이블릿 MAD 로 잡음 표준편차(0~1 단위)를 추정함(Donoho 방식)."""
    from skimage.restoration import estimate_sigma

    return float(estimate_sigma(np.asarray(x, dtype=np.float64)))


def classical(method: str, x: np.ndarray, factor: float, sigma: float | None = None) -> np.ndarray:
    """고전 복원 하나. sigma 가 없으면 입력에서 추정함. 결과는 0~1 로 자름."""
    x = np.asarray(x, dtype=np.float32)
    s = estimate_noise(x) if sigma is None else sigma
    if method == "gaussian":
        # 가우시안은 잡음 크기와 무관하게 필터 폭만 정함. factor 가 곧 표준편차(픽셀)임.
        from scipy.ndimage import gaussian_filter
        y = gaussian_filter(x, sigma=factor)
    elif method == "nlm":
        from skimage.restoration import denoise_nl_means
        y = denoise_nl_means(x, h=factor * s, sigma=s, fast_mode=True, patch_size=5, patch_distance=6)
    elif method == "bm3d":
        import bm3d
        # 기본값(num_threads=0)은 CPU 수만큼 스레드를 띄워 병렬 작업자와 겹치면 과부하가 되고,
        # 라이브러리 설명대로 1 이 아니면 결과가 조금씩 달라짐. 재현을 위해 1 로 고정함.
        pro = bm3d.BM3DProfile()
        pro.num_threads = 1
        y = bm3d.bm3d(x, sigma_psd=max(factor * s, 1e-6), profile=pro)
    else:
        raise ValueError(f"모르는 방법: {method}")
    return np.clip(np.asarray(y, dtype=np.float32), 0.0, 1.0)


def psnr(x: np.ndarray, ref: np.ndarray) -> float:
    """0~1 영상의 PSNR(dB). 같으면 inf."""
    mse = float(np.mean((np.asarray(x, np.float64) - np.asarray(ref, np.float64)) ** 2))
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def ssim(x: np.ndarray, ref: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(np.asarray(x, np.float64), np.asarray(ref, np.float64), data_range=1.0))


def _dncnn_class():
    import torch.nn as nn

    class _DnCNN(nn.Module):
        """17층, 64채널. 첫 층 ReLU, 가운데 층 BN+ReLU, 마지막 층은 잡음을 내고 입력에서 뺌."""

        def __init__(self, depth: int = 17, width: int = 64):
            super().__init__()
            layers = [nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True)]
            for _ in range(depth - 2):
                layers += [nn.Conv2d(width, width, 3, padding=1, bias=False),
                           nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
            layers.append(nn.Conv2d(width, 1, 3, padding=1))
            self.body = nn.Sequential(*layers)

        def forward(self, x):
            return x - self.body(x)

    return _DnCNN


def DnCNN(depth: int = 17, width: int = 64):
    """torch 를 쓰는 곳에서만 불러오도록 클래스를 늦게 만듦(고전 방법만 쓸 때 torch 를 요구하지 않음)."""
    return _dncnn_class()(depth, width)
