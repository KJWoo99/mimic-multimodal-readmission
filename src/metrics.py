"""평가 지표: 판별력 + 보정 + 서브그룹 분해.

이 프로젝트는 "성능이 얼마나 좋은가"가 아니라 "이 숫자를 믿어도 되는가"에 답하는 것이
목적이므로, 판별력(AUROC/PR-AUC)만이 아니라 보정(calibration)을 같은 비중으로 다룸.

왜 PR-AUC 를 우선하는가
----------------------
30일 재입원은 양성률이 대략 15~20% 인 불균형 문제임. AUROC 는 음성이 많을 때
낙관적으로 보이는 성질이 있어, 불균형 상황에서는 PR-AUC 가 성능을 덜 부풀림.

왜 보정을 따로 보는가
--------------------
"위험 점수 30%"가 실제로 30% 확률을 뜻하지 않으면 임계값 운영이 불가능함.
AUROC 가 같아도 보정이 틀어진 모델은 임상에서 못 씀. 서브그룹별로는 전체 AUROC 에
가려져 특정 집단에서만 보정이 틀어질 수 있음(V4 가 겨냥하는 지점).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

__all__ = [
    "MIN_SUBGROUP_N",
    "classification_metrics",
    "expected_calibration_error",
    "reliability_curve",
    "subgroup_metrics",
    "val_oof_calibration_ece",
]

# 서브그룹 최소 표본수. 표본이 작으면 숫자가 사실상 무의미하므로(31건이면 정확도의 95% CI 가
# +-13%p), 미달 그룹은 계산하되 명시적으로 표시함.
MIN_SUBGROUP_N = 50


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """ECE: 예측확률과 실제 빈도의 가중 평균 절대차.

    sklearn 에 없어 직접 구현함. 0에 가까울수록 예측 확률이 실제 비율과 맞는다는 뜻.

    strategy
    --------
    "uniform"  : [0,1] 을 균등 폭으로 분할 (표준)
    "quantile" : 각 bin 의 표본수가 같도록 분위수로 분할.
                 예측이 좁은 구간에 몰릴 때(불균형 문제에서 흔함) uniform 은
                 빈 bin 이 많아져 왜곡되므로 이쪽이 안정적임.

    ECE 단독으로는 모델의 유용성을 판단할 수 없음
    -----------------------------------------------
    "모든 환자에게 양성률과 같은 값을 출력"하는 무용한 모델도 ECE 는 0에 가까움.
    집계 수준의 보정만 재기 때문임.
    따라서 ECE 는 반드시 AUROC/PR-AUC 와 함께 보고함.
    판별력이 없으면 보정이 좋아도 의미가 없음.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true 와 y_prob 의 길이가 다릅니다")
    if len(y_true) == 0:
        return float("nan")

    if strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            return float(abs(y_prob.mean() - y_true.mean()))
    elif strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(f"strategy 는 uniform/quantile 중 하나여야 함: {strategy!r}")

    # 마지막 bin 이 오른쪽 경계를 포함하도록 처리
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, len(edges) - 2)

    ece = 0.0
    n = len(y_true)
    for b in range(len(edges) - 1):
        mask = idx == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> dict[str, list[float]]:
    """reliability diagram 용 (평균 예측확률, 실제 양성률, 표본수) 데이터."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if strategy == "quantile":
        edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, len(edges) - 2)

    mean_pred, frac_pos, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        mean_pred.append(float(y_prob[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return {"mean_predicted": mean_pred, "fraction_positive": frac_pos, "count": counts}


def equal_error_rate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """민감도와 특이도가 같아지는 지점의 오류율과 그때의 임계값.

    반환은 (eer, threshold) 다.

    ## 왜 이 지표를 같이 두는가

    AUROC 는 임계값 전체에 걸친 요약이라 "그래서 어디서 자를 것인가"에 답하지
    않음. 반대로 정확도, F1 은 임계값 하나를 고른 뒤의 값이라 그 선택이 결과를
    좌우함. EER 은 그 사이에 있음. 놓치는 비율과 잘못 부르는 비율이 같아지는
    한 점이라 임의 선택이 없고, 그 지점의 오류율이 곧 숫자가 됨.

    읽을 때 주의할 것이 있음. EER 은 양쪽 오류에 같은 비중을 둠. 재입원 예측이나
    패혈증 조기경보처럼 놓치는 쪽이 훨씬 비싼 문제에서는 그 균형점이 실제 운영점이
    아님. 그래서 EER 은 모델끼리 비교하는 자로 쓰고, 운영점은 용량이나 비용으로
    따로 정함.

    드문 사건에서는 특히 조심해야 함. 양성률이 1% 면 EER 이 좋아 보여도 그
    임계값에서의 정밀도는 매우 낮을 수 있음. PR-AUC 와 함께 봐야 하는 이유임.
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).astype(int)
    if y_true.sum() in (0, len(y_true)):
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y_true, np.asarray(y_prob, dtype=float))
    fnr = 1.0 - tpr
    # 두 곡선이 교차하는 점. 격자가 이산적이라 정확히 만나는 점이 없을 수 있으므로
    # 차이가 가장 작은 지점을 쓰고, 그 지점의 두 값을 평균함.
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2), float(thr[i])


@dataclass
class MetricResult:
    n: int
    n_pos: int
    pos_rate: float
    auroc: float
    pr_auc: float
    brier: float
    ece: float
    prevalence_baseline_pr_auc: float  # 무작위 모델의 PR-AUC = 양성률
    eer: float = float("nan")
    eer_threshold: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        """MLflow 로깅용 평면 dict (warnings 제외)."""
        return {
            "n": self.n,
            "n_pos": self.n_pos,
            "pos_rate": self.pos_rate,
            "auroc": self.auroc,
            "pr_auc": self.pr_auc,
            "brier": self.brier,
            "ece": self.ece,
            "pr_auc_baseline": self.prevalence_baseline_pr_auc,
            "eer": self.eer,
            "eer_threshold": self.eer_threshold,
        }


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> MetricResult:
    """이진 분류 표준 지표 묶음.

    PR-AUC 는 항상 양성률(무작위 baseline)과 함께 봐야 함.
    양성률 0.18 인 문제에서 PR-AUC 0.20 은 사실상 무작위와 같음.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    n_pos = int(y_true.sum())
    pos_rate = float(n_pos / n) if n else float("nan")

    warns: list[str] = []
    if n < MIN_SUBGROUP_N:
        warns.append(f"표본 {n}건: {MIN_SUBGROUP_N}건 미만이라 신뢰구간이 넓다")
    if n_pos == 0 or n_pos == n:
        warns.append("한쪽 클래스만 존재: AUROC/PR-AUC 계산 불가")
        # 키워드로 넘김. 위치 인자로 쓰면 필드를 하나 끼워 넣는 순간 값이
        # 한 칸씩 밀리는데 오류는 나지 않음.
        return MetricResult(
            n=n, n_pos=n_pos, pos_rate=pos_rate,
            auroc=float("nan"), pr_auc=float("nan"), brier=float("nan"),
            ece=float("nan"), prevalence_baseline_pr_auc=pos_rate, warnings=warns)

    eer, eer_thr = equal_error_rate(y_true, y_prob)
    return MetricResult(
        n=n,
        n_pos=n_pos,
        pos_rate=pos_rate,
        auroc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
        ece=expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy="quantile"),
        prevalence_baseline_pr_auc=pos_rate,
        eer=eer,
        eer_threshold=eer_thr,
        warnings=warns,
    )


def subgroup_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    n_bins: int = 10,
) -> dict[str, MetricResult]:
    """서브그룹별 지표 (V4).

    전체 AUROC 에 가려진 특정 그룹의 성능 저하를 찾는 것이 목적임.
    단순 정확도 차이보다 보정(ECE) 차이가 임상적으로 더 의미 있음.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    groups = np.asarray(groups)
    out: dict[str, MetricResult] = {}
    for g in np.unique(groups):
        mask = groups == g
        out[str(g)] = classification_metrics(y_true[mask], y_prob[mask], n_bins=n_bins)
    return out


def val_oof_calibration_ece(
    p_val: np.ndarray, y_val: np.ndarray, groups_val: np.ndarray,
    n_splits: int = 5, n_bins: int = 10,
) -> dict[str, float]:
    """교정 방식(원본, Platt, Isotonic)을 val 안에서만 비교하는 ECE.

    교정기를 val 전체로 맞추고 같은 val 로 재면 자기 데이터라 Isotonic 이 유리함.
    그래서 val 을 환자 단위 k겹으로 나눠, 각 겹을 나머지 겹으로 맞춘 교정기로 예측하고
    그 밖 겹 예측을 모아 ECE 를 측정함. test 는 쓰지 않음. ECE 는 classification_metrics 와
    같은 분위수 구간임.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    p = np.asarray(p_val, dtype=float)
    y = np.asarray(y_val).astype(int)
    g = np.asarray(groups_val)
    oof_iso = np.zeros_like(p)
    oof_platt = np.zeros_like(p)
    for tr, va in GroupKFold(n_splits=n_splits).split(p, y, groups=g):
        iso = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr])
        oof_iso[va] = iso.predict(p[va])
        pl = LogisticRegression(max_iter=1000).fit(p[tr].reshape(-1, 1), y[tr])
        oof_platt[va] = pl.predict_proba(p[va].reshape(-1, 1))[:, 1]
    return {
        name: expected_calibration_error(y, q, n_bins=n_bins, strategy="quantile")
        for name, q in (("원본", p), ("Platt", oof_platt), ("Isotonic", oof_iso))
    }
