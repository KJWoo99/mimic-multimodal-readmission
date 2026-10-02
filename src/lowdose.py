"""흉부 X선 저선량 모사, RAD-DINO 입력 격자, 소견 라벨.

사전등록은 `docs/prereg_lowdose.md`. 이 모듈은 그 3절(모사)과 2절(라벨)을 코드로 옮긴 것임.

왜 픽셀값을 광자 수로 바로 쓰지 않는가
------------------------------------
표시 영상에서 밝은 곳(뼈, 종격동)은 많이 흡수되어 검출기에 광자가 적게 닿은 곳임. 광자가 적을수록
상대 잡음이 큼. 픽셀값을 그대로 광자 수로 보고 포아송 잡음을 넣으면 밝은 곳의 상대 잡음이 가장 작아져
실제와 반대가 됨. 그래서 픽셀을 감쇠 선적분 p 로 되돌리고, 광자 수 N0 exp(-p) 에서 잡음을 넣음.

왜 모자란 만큼만 더하는가
------------------------
원본에도 이미 원 선량의 잡음이 있음. 선량비 r 의 영상은 광자 수가 r N 이고 분산도 r N 임. 원본을 r 배 하면
분산이 r^2 N 이므로 r N (1 - r) 만 더하면 됨(Veldkamp 등 2009). 광자 수가 충분히 커서 가우시안으로 근사함.

한계: JPG 는 검출기 원시 자료가 아니라 로그 변환, 창 조절, 윤곽 강조, 8비트 양자화를 거친 영상임. 여기의
k, N0 는 그 처리를 거친 영상에 맞춘 유효값이고, 전자 잡음은 잴 수 없어 넣지 않음.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

__all__ = [
    "DOSES",
    "FINDINGS",
    "K_MAIN",
    "RADDINO_MEAN",
    "RADDINO_SIZE",
    "RADDINO_STD",
    "finding_labels",
    "grid518",
    "insert_lowdose_noise",
    "n0_from_sigma",
    "noise_seed",
    "to_model_input",
]

# 사전등록 2절: 소견 다섯. 키는 CheXpert 열 이름, 값은 문서에 쓰는 이름.
FINDINGS = {
    "Pleural Effusion": "흉수",
    "Cardiomegaly": "심비대",
    "Edema": "폐부종",
    "Atelectasis": "무기폐",
    "Pneumothorax": "기흉",
}
# 사전등록 3절 5번: 선량비. 주 조건은 1/4.
DOSES = (0.5, 0.25, 0.125)
# 사전등록 3절 1번: 가장 밝은 곳의 투과율이 가장 어두운 곳의 1/100.
K_MAIN = math.log(100.0)

# RAD-DINO 배포 프로세서(preprocessor_config.json)와 같은 값. tests 에서 프로세서 출력과 대조함.
RADDINO_SIZE = 518
RADDINO_MEAN = 0.5307
RADDINO_STD = 0.2583


def finding_labels(chexpert: np.ndarray, uncertain: str = "ignore") -> np.ndarray:
    """CheXpert 값(1, 0, -1, NaN)을 이진 라벨로. 분석에서 뺄 행은 NaN.

    빈칸(NaN)은 "보고서에 언급 없음"이라 음성임. 불확실(-1)은 주 분석에서 뺌(U-Ignore).
    `uncertain="zeros"` 면 불확실을 음성으로 둠(U-Zeros, 민감도).
    """
    v = np.asarray(chexpert, dtype=float)
    out = np.where(v == 1, 1.0, 0.0)
    if uncertain == "ignore":
        out[v == -1] = np.nan
    elif uncertain != "zeros":
        raise ValueError(f"uncertain 은 ignore 또는 zeros: {uncertain}")
    return out


def noise_seed(dicom_id: str, dose: float, tag: str = "main") -> int:
    """영상, 선량비, 조건마다 고정된 시드. 같은 입력이면 어느 장비에서든 같은 잡음이 나옴."""
    h = hashlib.sha256(f"{dicom_id}|{dose:.6f}|{tag}".encode()).digest()
    return int.from_bytes(h[:8], "little")


def insert_lowdose_noise(v: np.ndarray, dose: float, n0: float, k: float = K_MAIN,
                         rng: np.random.Generator | None = None) -> np.ndarray:
    """원 선량 8비트 영상 v 에 선량비 dose 의 저선량 잡음을 더한 8비트 영상.

    dose = 1 이면 원본을 그대로 돌려줌.
    """
    if not 0 < dose <= 1:
        raise ValueError(f"선량비는 (0, 1]: {dose}")
    v = np.asarray(v)
    if dose == 1:
        return v.astype(np.uint8, copy=True)
    rng = rng or np.random.default_rng(0)
    p = v.astype(np.float64) * (k / 255.0)
    counts = n0 * np.exp(-p)                       # 원 선량의 기대 광자 수
    extra = np.sqrt(dose * counts * (1.0 - dose))  # 모자란 분산만큼
    low = (dose * counts + extra * rng.standard_normal(v.shape)) / dose
    low = np.maximum(low, 1.0)                     # 광자 수는 1 아래로 내려가지 않음
    p_low = -np.log(low / n0)
    return np.clip(np.rint(p_low * (255.0 / k)), 0, 255).astype(np.uint8)


def n0_from_sigma(sigma01: float, mean_v: float, k: float = K_MAIN) -> float:
    """잰 잡음 표준편차(0~1 단위)가 평균 밝기 mean_v(0~255)에서 나오려면 필요한 N0.

    로그 영역 잡음은 1/sqrt(N) 이고 표시 단위로는 (255/k)/sqrt(N) 임. N = N0 exp(-p) 이므로
    sigma01 * 255 = (255/k) exp(p/2) / sqrt(N0), 곧 N0 = exp(p) / (k sigma01)^2.
    """
    if sigma01 <= 0:
        raise ValueError("잡음 표준편차가 0 이하다")
    p = mean_v * k / 255.0
    return float(math.exp(p) / (k * sigma01) ** 2)


def grid518(v: np.ndarray) -> np.ndarray:
    """8비트 흑백 영상을 RAD-DINO 프로세서와 같은 방식으로 518x518 8비트 격자로.

    프로세서(BitImageProcessor)는 RGB 로 바꾼 뒤 짧은 변을 518 로 bicubic 축소하고(긴 변은 내림),
    8비트로 되돌린 다음 가운데 518 을 자름. 흑백이라 세 채널이 같으므로 한 채널만 둠.
    """
    from PIL import Image

    v = np.asarray(v, dtype=np.uint8)
    h, w = v.shape
    if h <= w:
        nh, nw = RADDINO_SIZE, int(RADDINO_SIZE * w / h)
    else:
        nh, nw = int(RADDINO_SIZE * h / w), RADDINO_SIZE
    img = Image.fromarray(v).convert("RGB").resize((nw, nh), Image.Resampling.BICUBIC)
    a = np.asarray(img)[:, :, 0]
    top = (nh - RADDINO_SIZE) // 2
    left = (nw - RADDINO_SIZE) // 2
    return np.ascontiguousarray(a[top:top + RADDINO_SIZE, left:left + RADDINO_SIZE])


def to_model_input(g: np.ndarray) -> np.ndarray:
    """518 격자(8비트, (N, 518, 518) 또는 (518, 518))를 RAD-DINO 입력(N, 3, 518, 518) float32 로."""
    g = np.asarray(g)
    if g.ndim == 2:
        g = g[None]
    x = (g.astype(np.float32) / 255.0 - RADDINO_MEAN) / RADDINO_STD
    return np.repeat(x[:, None], 3, axis=1)
