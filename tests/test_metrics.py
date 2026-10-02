"""평가 지표 회귀 테스트.

ECE 는 sklearn 에 없어 직접 구현했으므로, 독립 계산과의 일치를 테스트로 고정함.
ECE 를 쓰는 보정 판정이 모두 이 구현에 기대므로 여기서 먼저 확인함.
"""
from __future__ import annotations

import numpy as np
import pytest

from metrics import (
    MIN_SUBGROUP_N,
    classification_metrics,
    equal_error_rate,
    expected_calibration_error,
    reliability_curve,
    subgroup_metrics,
)
from models import build_tabular_model


def test_eer_of_perfect_model_is_zero():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    eer, thr = equal_error_rate(y, p)
    assert eer == pytest.approx(0.0, abs=1e-9)
    assert 0.4 <= thr <= 0.6001


def test_eer_of_random_scores_is_near_half():
    rng = np.random.default_rng(0)
    y = (rng.uniform(0, 1, 20000) < 0.3).astype(int)
    p = rng.uniform(0, 1, 20000)
    eer, _ = equal_error_rate(y, p)
    assert 0.45 <= eer <= 0.55


def test_eer_threshold_actually_balances_the_two_errors():
    """EER 임계값에서 놓친 비율과 잘못 부른 비율이 실제로 비슷해야 함.

    곡선에서 교차점을 찾는 것과, 그 임계값으로 실제로 잘라봤을 때 두 오류가
    맞아떨어지는 것은 다른 이야기임. 임계값을 잘못 짚어도 EER 값 자체는
    그럴듯하게 나오므로 여기서 함께 봄.
    """
    rng = np.random.default_rng(3)
    y = (rng.uniform(0, 1, 4000) < 0.2).astype(int)
    p = np.clip(rng.normal(0.3 + 0.3 * y, 0.2), 0, 1)
    eer, thr = equal_error_rate(y, p)
    pred = p >= thr
    fnr = float((~pred[y == 1]).mean())
    fpr = float(pred[y == 0].mean())
    assert abs(fnr - fpr) < 0.02, f"fnr {fnr:.3f} vs fpr {fpr:.3f}"
    assert abs(eer - (fnr + fpr) / 2) < 0.02


def test_eer_is_nan_when_one_class_missing():
    y = np.zeros(10, dtype=int)
    eer, thr = equal_error_rate(y, np.linspace(0, 1, 10))
    assert np.isnan(eer) and np.isnan(thr)


def test_classification_metrics_reports_eer():
    rng = np.random.default_rng(11)
    y = (rng.uniform(0, 1, 2000) < 0.25).astype(int)
    p = np.clip(rng.normal(0.3 + 0.2 * y, 0.2), 0, 1)
    m = classification_metrics(y, p)
    assert 0.0 <= m.eer <= 1.0
    assert m.as_dict()["eer"] == m.eer


def test_ece_matches_independent_calculation():
    """sklearn calibration_curve 기반 수동계산과 일치해야 함."""
    from sklearn.calibration import calibration_curve

    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)

    n_bins = 10
    frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    w = np.array([(idx == b).sum() for b in range(n_bins)]) / len(p)
    expected = float(np.sum(w[w > 0] * np.abs(np.array(mean_pred) - np.array(frac))))

    got = expected_calibration_error(y, p, n_bins=n_bins, strategy="uniform")
    assert abs(got - expected) < 1e-12


def test_ece_is_small_for_calibrated_model():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 100000)
    y = (rng.uniform(0, 1, 100000) < p).astype(int)
    assert expected_calibration_error(y, p, strategy="uniform") < 0.01


def test_ece_is_large_for_overconfident_model():
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 50000)
    y = (rng.uniform(0, 1, 50000) < p).astype(int)
    inflated = np.clip(p * 1.5, 0, 1)
    assert expected_calibration_error(y, inflated, strategy="uniform") > 0.1


def test_pr_auc_baseline_equals_prevalence():
    """무작위 예측의 PR-AUC 는 양성률과 같아야 함.

    이 baseline 없이 PR-AUC 단독으로 보고하면 성능을 오독함.
    """
    rng = np.random.default_rng(3)
    y = (rng.uniform(0, 1, 20000) < 0.18).astype(int)
    p = rng.uniform(0, 1, 20000)
    m = classification_metrics(y, p)
    assert abs(m.pr_auc - m.prevalence_baseline_pr_auc) < 0.02
    assert abs(m.auroc - 0.5) < 0.02


def test_small_sample_warning():
    rng = np.random.default_rng(4)
    n = MIN_SUBGROUP_N - 1
    y = (rng.uniform(0, 1, n) < 0.5).astype(int)
    m = classification_metrics(y, rng.uniform(0, 1, n))
    assert any("건" in w for w in m.warnings)


def test_single_class_returns_nan_with_warning():
    m = classification_metrics(np.zeros(100, dtype=int), np.random.default_rng(5).uniform(0, 1, 100))
    assert np.isnan(m.auroc)
    assert m.warnings


def test_subgroup_metrics_partitions_data():
    rng = np.random.default_rng(6)
    n = 2000
    y = (rng.uniform(0, 1, n) < 0.3).astype(int)
    p = rng.uniform(0, 1, n)
    g = rng.choice(["A", "B", "C"], n)
    out = subgroup_metrics(y, p, g)
    assert set(out) == {"A", "B", "C"}
    assert sum(v.n for v in out.values()) == n


def test_reliability_curve_shapes():
    rng = np.random.default_rng(7)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    rc = reliability_curve(y, p, n_bins=10)
    assert len(rc["mean_predicted"]) == len(rc["fraction_positive"]) == len(rc["count"])
    assert sum(rc["count"]) == len(y)


def test_ece_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([0, 1]), np.array([0.5]))


# ── GBDT 시드 ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "logreg"])
def test_tabular_model_fixes_its_own_seed(name):
    """모델 내부 난수 시드가 명시돼 있어야 함.

    `random_state` 를 비워 두면 LightGBM 은 fit 시점의 전역 난수에서 시드를
    뽑음. 앞 코드가 난수를 얼마나 썼는지에 따라 결과가 달라지므로, 같은
    스크립트를 같은 기계에서 돌려도 값이 흔들릴 수 있음. XGBoost 는 기본값이
    0 이라 이 구멍이 드러나지 않음.
    """
    pytest.importorskip({"xgboost": "xgboost", "lightgbm": "lightgbm",
                         "logreg": "sklearn"}[name])
    m = build_tabular_model(name)
    assert m.get_params().get("random_state") is not None, (
        f"{name} 의 random_state 가 비어 있다")


def test_tabular_model_seed_is_passed_through():
    """호출자가 시드를 바꾸면 모델에 그대로 닿아야 함."""
    pytest.importorskip("xgboost")
    assert build_tabular_model("xgboost", seed=7).get_params()["random_state"] == 7


def test_val_oof_calibration_ece_uses_held_out_folds():
    """교정 방식 선택용 ECE 는 val 안의 밖 겹 예측으로 측정함. 교정기를 자기 데이터로 재면
    Isotonic 이 0 에 가깝게 나와 늘 이김. 밖 겹이면 그렇지 않음."""
    import numpy as np

    from metrics import expected_calibration_error, val_oof_calibration_ece

    rng = np.random.default_rng(0)
    n = 4000
    p = rng.uniform(0, 1, n)
    y = (rng.uniform(0, 1, n) < p ** 2).astype(int)   # 원본 확률이 체계적으로 높음
    g = np.arange(n) // 4                              # 환자당 4행
    got = val_oof_calibration_ece(p, y, g)
    assert set(got) == {"원본", "Platt", "Isotonic"}
    assert got["원본"] > got["Isotonic"] and got["원본"] > got["Platt"]
    from sklearn.isotonic import IsotonicRegression
    in_sample = expected_calibration_error(
        y, IsotonicRegression(out_of_bounds="clip").fit(p, y).predict(p), strategy="quantile")
    assert got["Isotonic"] > in_sample                 # 밖 겹이라 자기 데이터 값보다 큼
