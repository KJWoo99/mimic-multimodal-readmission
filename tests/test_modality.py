"""결측 모달리티 진단 회귀 테스트.

특히 퇴원 이후 촬영을 절대 세지 않는지(미래 정보 누수)를 고정함.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modality import (
    diagnose_missingness,
    link_cxr_to_admissions,
    parse_study_datetime,
    risk_ratio,
)


def test_parse_study_datetime():
    """StudyDate(int) + StudyTime(float HHMMSS.fff) 파싱."""
    d = pd.Series([21800506, 21800101])
    t = pd.Series([213014.531, 0.187])       # 21:30:14 / 00:00:00
    out = parse_study_datetime(d, t)
    assert out.iloc[0].year == 2180
    assert out.iloc[0].hour == 21 and out.iloc[0].minute == 30
    assert out.iloc[1].hour == 0 and out.iloc[1].minute == 0


def test_parse_study_datetime_handles_missing_time():
    out = parse_study_datetime(pd.Series([21800506]), pd.Series([np.nan]))
    assert out.notna().all()
    assert out.iloc[0].hour == 0


@pytest.fixture
def linking_tables():
    cohort = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "hadm_id": [10, 20, 30],
        "admittime": pd.to_datetime(["2180-01-01 08:00", "2180-02-01 08:00", "2180-03-01 08:00"]),
        "dischtime": pd.to_datetime(["2180-01-10 12:00", "2180-02-10 12:00", "2180-03-10 12:00"]),
    })
    cxr = pd.DataFrame({
        "subject_id": [1, 1, 2, 3, 3],
        "study_id": [101, 102, 201, 301, 302],
        # subject 1: 재원 중 2건
        # subject 2: 재원 중 1건
        # subject 3: 입원 전 1건 + 퇴원 후 1건  -> 둘 다 세면 안 됨
        "StudyDate": [21800102, 21800109, 21800205, 21800220, 21800315],
        "StudyTime": [100000.0, 90000.0, 120000.0, 100000.0, 100000.0],
        "ViewPosition": ["AP", "PA", "PA", "AP", "AP"],
    })
    return cohort, cxr


def test_link_counts_only_studies_within_stay(linking_tables):
    cohort, cxr = linking_tables
    out = link_cxr_to_admissions(cohort, cxr).set_index("hadm_id")
    assert out.loc[10, "n_cxr"] == 2
    assert out.loc[20, "n_cxr"] == 1
    assert out.loc[30, "n_cxr"] == 0      # 입원 전, 퇴원 후만 있으므로 0


def test_link_excludes_post_discharge_studies(linking_tables):
    """퇴원 이후 촬영을 세면 미래 정보 누수임."""
    cohort, cxr = linking_tables
    out = link_cxr_to_admissions(cohort, cxr).set_index("hadm_id")
    assert out.loc[30, "has_cxr"] == 0


def test_has_cxr_is_binary(linking_tables):
    cohort, cxr = linking_tables
    out = link_cxr_to_admissions(cohort, cxr)
    assert set(out.has_cxr.unique()) <= {0, 1}
    assert (out.has_cxr == (out.n_cxr > 0).astype(int)).all()


def test_hours_before_discharge_is_nan_when_no_cxr(linking_tables):
    """영상이 없으면 '퇴원 몇 시간 전'이 정의되지 않음. 0 으로 채우면 안 됨."""
    cohort, cxr = linking_tables
    out = link_cxr_to_admissions(cohort, cxr).set_index("hadm_id")
    assert np.isnan(out.loc[30, "hours_cxr_before_discharge"])
    assert out.loc[10, "hours_cxr_before_discharge"] >= 0


def test_hours_before_discharge_uses_most_recent(linking_tables):
    """여러 장이 있으면 가장 최근(퇴원에 가장 가까운) 촬영 기준이어야 함."""
    cohort, cxr = linking_tables
    out = link_cxr_to_admissions(cohort, cxr).set_index("hadm_id")
    # subject 1: 01-02 10:00 과 01-09 09:00, 퇴원 01-10 12:00 -> 후자가 더 최근
    expected = (pd.Timestamp("2180-01-10 12:00") - pd.Timestamp("2180-01-09 09:00")).total_seconds() / 3600
    assert out.loc[10, "hours_cxr_before_discharge"] == pytest.approx(expected)


def test_link_requires_expected_columns():
    with pytest.raises(ValueError):
        link_cxr_to_admissions(pd.DataFrame({"subject_id": [1]}), pd.DataFrame())


def test_diagnose_detects_strong_association():
    """has_cxr 이 결과와 강하게 연관되면 MNAR 로 판정되어야 함."""
    rng = np.random.default_rng(0)
    n = 4000
    has = rng.integers(0, 2, n)
    # 영상을 찍은 환자가 더 자주 재입원하도록 구성
    y = (rng.uniform(0, 1, n) < np.where(has == 1, 0.45, 0.10)).astype(int)
    df = pd.DataFrame({"has_cxr": has, "readmit_30d": y,
                       "charlson_score": has * 3 + rng.integers(0, 3, n)})
    d = diagnose_missingness(df)
    assert d.outcome_r > 0.1
    assert d.outcome_p_bonferroni < 0.05
    assert "MNAR 강함" in d.verdict()


def test_diagnose_detects_no_association():
    rng = np.random.default_rng(1)
    n = 4000
    df = pd.DataFrame({
        "has_cxr": rng.integers(0, 2, n),
        "readmit_30d": rng.integers(0, 2, n),
        "charlson_score": rng.integers(0, 5, n),
    })
    d = diagnose_missingness(df)
    assert abs(d.outcome_r) < 0.1
    assert "근거 부족" in d.verdict()


def test_diagnose_applies_bonferroni():
    rng = np.random.default_rng(2)
    n = 2000
    df = pd.DataFrame({
        "has_cxr": rng.integers(0, 2, n),
        "readmit_30d": rng.integers(0, 2, n),
        "charlson_score": rng.integers(0, 5, n),
        "los_days": rng.uniform(1, 20, n),
        "age": rng.integers(18, 90, n),
    })
    d = diagnose_missingness(df)
    assert d.n_tests == len(d.correlations)
    assert (d.correlations.p_bonferroni >= d.correlations.p).all()
    assert (d.correlations.p_bonferroni <= 1.0).all()


def test_diagnose_requires_indicator_column():
    with pytest.raises(ValueError):
        diagnose_missingness(pd.DataFrame({"readmit_30d": [0, 1]}))


def test_risk_ratio_basic():
    """노출군 50% vs 비노출군 25% -> RR = 2.0"""
    ind = np.array([1] * 100 + [0] * 100)
    out = np.array([1] * 50 + [0] * 50 + [1] * 25 + [0] * 75)
    e = risk_ratio(ind, out)
    assert e["rate_exposed"] == pytest.approx(0.5)
    assert e["rate_unexposed"] == pytest.approx(0.25)
    assert e["risk_ratio"] == pytest.approx(2.0)
    assert e["rr_ci_low"] < 2.0 < e["rr_ci_high"]


def test_risk_ratio_no_association():
    ind = np.array([1] * 500 + [0] * 500)
    out = np.array(([1] * 100 + [0] * 400) * 2)
    e = risk_ratio(ind, out)
    assert e["risk_ratio"] == pytest.approx(1.0)
    assert e["rr_ci_low"] < 1.0 < e["rr_ci_high"]


def test_risk_ratio_catches_what_correlation_misses():
    """핵심 회귀 테스트: 집단 불균형에서 점이연 상관은 축소됨.

    실데이터에서 영상 보유 8.6% vs 미보유 91.4% 일 때
    재입원률 21.67% vs 15.97%(RR 1.36) 였는데도 r 은 0.043 에 그침.
    r 만 보고 판정하면 'MCAR' 로 잘못 결론내게 됨.
    """
    rng = np.random.default_rng(7)
    n_exp, n_unexp = 3500, 37000          # 실제 비율(8.6%)을 모사
    ind = np.array([1] * n_exp + [0] * n_unexp)
    out = np.concatenate([
        (rng.uniform(0, 1, n_exp) < 0.2167).astype(int),
        (rng.uniform(0, 1, n_unexp) < 0.1597).astype(int),
    ])
    df = pd.DataFrame({"has_cxr": ind, "readmit_30d": out})
    d = diagnose_missingness(df)

    # 상관은 작게 나오지만
    assert abs(d.outcome_r) < 0.1
    # 위험비는 실질적 연관을 잡아냄
    assert d.effect["risk_ratio"] > 1.2
    assert d.effect["rr_ci_low"] > 1.0
    # 최종 판정은 MNAR 이어야 함
    assert "MNAR 강함" in d.verdict()


def test_diagnosis_includes_effect_measures():
    rng = np.random.default_rng(8)
    n = 2000
    df = pd.DataFrame({
        "has_cxr": rng.integers(0, 2, n),
        "readmit_30d": rng.integers(0, 2, n),
    })
    d = diagnose_missingness(df)
    for key in ("risk_ratio", "odds_ratio", "rr_ci_low", "rr_ci_high",
                "rate_exposed", "rate_unexposed"):
        assert key in d.effect


def test_stratified_risk_ratio_detects_confounding():
    """교란이 있으면 조 RR 이 부풀려지고 MH RR 이 이를 보정해야 함.

    실데이터에서 재원일수가 교란변수임이 확인됨.
    조 RR 1.340 -> LOS 4분위 보정 후 MH RR 1.202.
    조 RR 만 보고하면 연관을 과장하게 됨.
    """
    from modality import stratified_risk_ratio

    rng = np.random.default_rng(11)
    rows = []
    # 층이 올라갈수록 노출률과 기저 위험이 함께 오름 = 교란 구조
    for s, (exp_rate, base_risk) in enumerate(
        [(0.05, 0.10), (0.10, 0.15), (0.20, 0.20), (0.35, 0.25)]
    ):
        n = 20000
        ind = (rng.uniform(0, 1, n) < exp_rate).astype(int)
        # 층 내 실제 효과는 1.2 배로 고정
        p = np.where(ind == 1, base_risk * 1.2, base_risk)
        out = (rng.uniform(0, 1, n) < p).astype(int)
        rows.append(pd.DataFrame({"has_cxr": ind, "readmit_30d": out, "stratum": s}))
    df = pd.concat(rows, ignore_index=True)

    crude = risk_ratio(df.has_cxr.values, df.readmit_30d.values)["risk_ratio"]
    tab, mh = stratified_risk_ratio(df.has_cxr.values, df.readmit_30d.values,
                                    df.stratum.values)

    assert len(tab) == 4
    assert crude > mh, "교란이 있으면 조 RR 이 MH RR 보다 커야 한다"
    assert 1.1 < mh < 1.3, f"MH RR 이 진짜 효과(1.2) 근처여야 함: {mh}"


def test_stratified_risk_ratio_no_confounding():
    """교란이 없으면 조 RR 과 MH RR 이 비슷해야 함."""
    from modality import stratified_risk_ratio

    rng = np.random.default_rng(12)
    n = 40000
    ind = rng.integers(0, 2, n)
    st = rng.integers(0, 4, n)          # 층과 노출이 독립
    out = (rng.uniform(0, 1, n) < np.where(ind == 1, 0.24, 0.20)).astype(int)
    crude = risk_ratio(ind, out)["risk_ratio"]
    _, mh = stratified_risk_ratio(ind, out, st)
    assert abs(crude - mh) < 0.05


def test_stratified_skips_strata_without_variation():
    from modality import stratified_risk_ratio

    ind = np.array([1, 1, 1, 0, 1, 0, 1, 0])
    out = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    st = np.array(["A", "A", "A", "B", "B", "B", "B", "B"])
    tab, _mh = stratified_risk_ratio(ind, out, st)
    # A 층은 노출군만 있어 계산 불가 -> 제외
    assert set(tab.stratum) == {"B"}
