"""코호트 정의와 피처 회귀 테스트.

특히 어떤 컬럼이 모델 입력에 들어가면 안 되는가를 고정함.
누수 컬럼은 한 번 새면 성능으로 위장되어 발견이 매우 어려움.
"""
from __future__ import annotations

import pandas as pd
import pytest

from cohort import CohortConfig, build_cohort
from features import (
    CHARLSON_WEIGHTS,
    add_charlson,
    add_medication_features,
    build_features,
    feature_columns,
)


@pytest.fixture
def toy_tables():
    adm = pd.DataFrame({
        "subject_id": [1, 1, 1, 2, 2, 3, 4, 4, 5],
        "hadm_id": [10, 11, 12, 20, 21, 30, 40, 41, 50],
        "admittime": pd.to_datetime([
            "2180-01-01 08:00", "2180-01-20 09:00", "2180-06-01 10:00",
            "2181-02-01 08:00", "2181-02-15 08:00",
            "2182-03-01 08:00",
            "2183-04-01 08:00", "2183-04-03 08:00",
            "2184-05-01 08:00",
        ], format="ISO8601"),
        "dischtime": pd.to_datetime([
            "2180-01-05 12:00", "2180-01-25 12:00", "2180-06-10 12:00",
            "2181-02-10 12:00", "2181-02-20 12:00",
            "2182-03-10 12:00",
            "2183-04-01 20:00", "2183-04-05 12:00",
            "2184-05-06 12:00",
        ], format="ISO8601"),
        "deathtime": [pd.NaT] * 4 + [pd.Timestamp("2181-02-20 12:00")] + [pd.NaT] * 4,
        "hospital_expire_flag": [0, 0, 0, 0, 1, 0, 0, 0, 0],
        "admission_type": ["EW EMER."] * 8 + ["ELECTIVE"],
        "discharge_location": ["HOME"] * 9,
    })
    pat = pd.DataFrame({
        "subject_id": [1, 2, 3, 4, 5],
        "anchor_age": [65, 70, 55, 12, 60],       # subject 4 는 미성년
        "anchor_year": [2180, 2181, 2182, 2183, 2184],
        "gender": ["M", "F", "M", "F", "M"],
    })
    dx = pd.DataFrame({
        "hadm_id": [10, 10, 11, 20, 30, 50],
        "icd_code": ["I21", "E119", "I50", "N18", "C78", "K700"],
        "icd_version": [10] * 6,
    })
    rx = pd.DataFrame({
        "hadm_id": [10] * 6 + [20] * 2,
        "drug": ["Warfarin Sodium", "Insulin Glargine", "Morphine Sulfate",
                 "Furosemide", "Aspirin", "Metoprolol", "Aspirin", "Tylenol"],
    })
    return adm, pat, dx, rx


def test_minor_excluded(toy_tables):
    adm, pat, _, _ = toy_tables
    r = build_cohort(adm, pat, CohortConfig())
    assert 4 not in set(r.df.subject_id)


def test_in_hospital_death_excluded(toy_tables):
    """사망 건을 남기면 '재입원 안 함'으로 잡혀 음성 라벨이 오염됨."""
    adm, pat, _, _ = toy_tables
    r = build_cohort(adm, pat, CohortConfig())
    assert 21 not in set(r.df.hadm_id)


def test_short_stay_excluded(toy_tables):
    adm, pat, _, _ = toy_tables
    r = build_cohort(adm, pat, CohortConfig(min_los_days=1.0))
    assert 40 not in set(r.df.hadm_id)


def test_discharge_before_admission_is_dropped_even_without_los_floor(toy_tables):
    """퇴원이 입원보다 앞서거나 같은 기록(MIMIC-IV v3.1 실측 180건)은 min_los_days=0 이어도 빠짐
    (트러블슈팅 6번)."""
    adm, pat, _, _ = toy_tables
    bad = adm.iloc[[5]].copy()
    bad["hadm_id"] = 31
    bad["admittime"] = pd.Timestamp("2182-06-01 08:00")
    bad["dischtime"] = pd.Timestamp("2182-06-01 07:00")
    r = build_cohort(pd.concat([adm, bad], ignore_index=True), pat, CohortConfig(min_los_days=0.0))
    assert 31 not in set(r.df.hadm_id)
    assert any("퇴원" in str(s) and "1건" in str(s) for s in r.flow_table().iloc[:, 0])


def test_readmission_label(toy_tables):
    """퇴원 2180-01-05 -> 다음 입원 2180-01-20 (15일) 이면 양성."""
    adm, pat, _, _ = toy_tables
    r = build_cohort(adm, pat, CohortConfig())
    assert int(r.df.loc[r.df.hadm_id == 10, "readmit_30d"].iloc[0]) == 1
    # 2180-01-25 -> 2180-06-01 은 30일 초과이므로 음성
    assert int(r.df.loc[r.df.hadm_id == 11, "readmit_30d"].iloc[0]) == 0


def test_elective_readmission_excluded_from_outcome(toy_tables):
    """예정 재입원을 제외하면 재입원률이 달라져야 함(민감도 분석의 근거)."""
    adm, pat, _, _ = toy_tables
    base = build_cohort(adm, pat, CohortConfig()).summary()["readmit_rate"]
    excl = build_cohort(
        adm, pat, CohortConfig(elective_admission_types=("ELECTIVE",))
    ).summary()["readmit_rate"]
    assert excl <= base


def test_readmission_that_ends_in_death_counts(toy_tables):
    """재입원 중 사망해도 재입원임. 사망한 입원은 index 에서만 빠짐.

    subject 2: 입원 20 퇴원 2181-02-10 -> 입원 21 입원 2181-02-15(5일 뒤), 21 에서 원내 사망.
    필터를 건 코호트 안에서 다음 입원을 찾으면 21 이 사라져 20 이 음성이 됨(트러블슈팅 24).
    """
    adm, pat, _, _ = toy_tables
    r = build_cohort(adm, pat, CohortConfig())
    assert 21 not in set(r.df.hadm_id)                                   # index 로는 제외
    assert int(r.df.loc[r.df.hadm_id == 20, "readmit_30d"].iloc[0]) == 1  # 재입원으로는 셈


def test_short_readmission_counts_and_prior_count_uses_all_admissions(toy_tables):
    """재원 1일 미만 입원도 재입원이고 과거 입원임. 재원 1일 기준은 index 에만 적용함."""
    adm, pat, _, _ = toy_tables
    extra = pd.DataFrame({
        "subject_id": [3, 3], "hadm_id": [31, 32],
        "admittime": pd.to_datetime(["2182-03-20 08:00", "2182-04-05 08:00"]),
        "dischtime": pd.to_datetime(["2182-03-20 20:00", "2182-04-12 08:00"]),  # 31 은 12시간
        "deathtime": [pd.NaT, pd.NaT], "hospital_expire_flag": [0, 0],
        "admission_type": ["EU OBSERVATION", "EW EMER."], "discharge_location": ["HOME"] * 2,
    })
    r = build_cohort(pd.concat([adm, extra], ignore_index=True), pat, CohortConfig())
    assert 31 not in set(r.df.hadm_id)                                   # 짧아서 index 로는 제외
    assert int(r.df.loc[r.df.hadm_id == 30, "readmit_30d"].iloc[0]) == 1  # 10일 뒤 짧은 재입원
    # 32 의 과거 12개월 입원: 30, 31 두 건(31 은 코호트에 없지만 입원임)
    assert int(r.df.loc[r.df.hadm_id == 32, "prior_admissions_12m"].iloc[0]) == 2


def test_prior_admissions_uses_only_past(toy_tables):
    """미래 입원을 세면 누수임."""
    adm, pat, _, _ = toy_tables
    r = build_cohort(adm, pat, CohortConfig())
    s1 = r.df[r.df.subject_id == 1].sort_values("admittime")
    assert s1.prior_admissions_12m.tolist() == [0, 1, 2]


def test_last_admission_flagged_as_censored(toy_tables):
    adm, pat, _, _ = toy_tables
    r = build_cohort(adm, pat, CohortConfig())
    assert "is_last_admission" in r.df.columns
    assert r.n_censored > 0
    # 절단 건은 항상 음성이어야 함(다음 입원이 없으므로)
    assert (r.df.loc[r.df.is_last_admission == 1, "readmit_30d"] == 0).all()


def test_flow_is_monotonically_decreasing(toy_tables):
    adm, pat, _, _ = toy_tables
    r = build_cohort(adm, pat, CohortConfig())
    counts = r.flow_table().n_admissions.tolist()
    assert counts == sorted(counts, reverse=True)


@pytest.mark.parametrize(
    "code,version,condition,weight",
    [
        ("I21", 10, "myocardial_infarction", 1),
        ("I50", 10, "congestive_heart_failure", 1),
        ("E119", 10, "diabetes_uncomplicated", 1),
        ("E112", 10, "diabetes_complicated", 2),
        ("N18", 10, "renal", 2),
        ("C78", 10, "metastatic_tumor", 6),
        ("B20", 10, "aids_hiv", 6),
        ("410", 9, "myocardial_infarction", 1),
        ("428", 9, "congestive_heart_failure", 1),
        ("1970", 9, "metastatic_tumor", 6),
    ],
)
def test_charlson_mapping(code, version, condition, weight):
    dx = pd.DataFrame({"hadm_id": [1], "icd_code": [code], "icd_version": [version]})
    cohort = pd.DataFrame({"hadm_id": [1], "subject_id": [1]})
    out = add_charlson(cohort, dx)
    assert out[f"cci_{condition}"].iloc[0] == 1
    assert out["charlson_score"].iloc[0] == weight


@pytest.mark.parametrize("code,expected", [("5880", 1), ("5859", 1), ("586", 1), ("58881", 0), ("5889", 0), ("587", 0)])
def test_charlson_icd9_renal_is_585_586_5880_only(code, expected):
    """Quan 2005, mimic-code charlson.sql 과 같게 588 은 588.0 만 신장질환임(공식 charlson 과 대조할 때 160입원 차이)."""
    dx = pd.DataFrame({"hadm_id": [1], "icd_code": [code], "icd_version": [9]})
    out = add_charlson(pd.DataFrame({"hadm_id": [1], "subject_id": [1]}), dx)
    assert out["cci_renal"].iloc[0] == expected


def test_charlson_hierarchy_avoids_double_counting():
    """중증이 있으면 경증은 세지 않음(당뇨/간질환/악성종양)."""
    dx = pd.DataFrame({
        "hadm_id": [1, 1, 2, 2, 3, 3],
        "icd_code": ["E119", "E112", "K700", "K704", "C50", "C78"],
        "icd_version": [10] * 6,
    })
    cohort = pd.DataFrame({"hadm_id": [1, 2, 3], "subject_id": [1, 2, 3]})
    out = add_charlson(cohort, dx).set_index("hadm_id")
    assert out.loc[1, "charlson_score"] == 2   # 중증 당뇨만
    assert out.loc[2, "charlson_score"] == 3   # 중증 간질환만
    assert out.loc[3, "charlson_score"] == 6   # 전이만


def test_charlson_weights_are_complete():
    from features import _CHARLSON_ICD9, _CHARLSON_ICD10

    assert set(CHARLSON_WEIGHTS) == set(_CHARLSON_ICD10) == set(_CHARLSON_ICD9)


def test_medication_flags():
    rx = pd.DataFrame({
        "hadm_id": [1] * 6 + [2] * 2,
        "drug": ["Warfarin", "Insulin", "Morphine", "Furosemide", "Aspirin", "Metoprolol",
                 "Aspirin", "Tylenol"],
    })
    cohort = pd.DataFrame({"hadm_id": [1, 2], "subject_id": [1, 2]})
    out = add_medication_features(cohort, rx).set_index("hadm_id")
    assert out.loc[1, "polypharmacy"] == 1
    assert out.loc[2, "polypharmacy"] == 0
    assert out.loc[1, "drug_anticoagulant"] == 1
    assert out.loc[2, "drug_anticoagulant"] == 0
    assert out.loc[1, "n_high_risk_drug_classes"] == 4


LEAKY_COLUMNS = [
    "readmit_30d",              # 라벨 그 자체
    "is_last_admission",        # 1이면 라벨이 항상 0
    "days_to_next_admission",   # 결과 시점
    "next_admittime",
    "next_admission_type",
    "excluded_planned_readmit",
    "discharge_location",       # '사망', '호스피스'가 결과를 알려줌
    "hospital_expire_flag",
    "deathtime",
    "admit_year_shifted",       # 시프트된 연도: 의미 없음
    "subject_id",
    "hadm_id",
]


def test_no_leaky_columns_in_features(toy_tables):
    """가장 중요한 회귀 테스트: 누수 컬럼이 하나라도 모델 입력에 들어가면 실패."""
    adm, pat, dx, rx = toy_tables
    coh = build_cohort(adm, pat, CohortConfig())
    feat, _ = build_features(coh.df, pat, dx, rx, None)
    cols = set(feature_columns(feat))
    leaked = [c for c in LEAKY_COLUMNS if c in cols]
    assert not leaked, f"누수 컬럼이 모델 입력에 포함됨: {leaked}"


def test_leaky_columns_stay_out_even_when_numeric(toy_tables):
    """문자열 누수 열(discharge_location 등)은 숫자 특징만 고르는 단계에서 저절로 빠져, 제외 목록에서
    지워도 위 테스트가 통과함. 누가 숫자로 부호화해도 빠지는지 봄."""
    adm, pat, dx, rx = toy_tables
    coh = build_cohort(adm, pat, CohortConfig())
    feat, _ = build_features(coh.df, pat, dx, rx, None)
    for c in LEAKY_COLUMNS:
        feat[c] = 1.0
    cols = set(feature_columns(feat))
    leaked = [c for c in LEAKY_COLUMNS if c in cols]
    assert not leaked, f"숫자로 바꾼 누수 컬럼이 모델 입력에 포함됨: {leaked}"


def test_no_duplicate_merge_columns(toy_tables):
    """중복 병합(_x/_y/_pat)으로 같은 변수가 두 번 들어가면 안 됨."""
    adm, pat, dx, rx = toy_tables
    coh = build_cohort(adm, pat, CohortConfig())
    feat, _ = build_features(coh.df, pat, dx, rx, None)
    dups = [c for c in feat.columns if c.endswith(("_x", "_y", "_pat"))]
    assert not dups, f"중복 병합 컬럼: {dups}"


def test_all_nan_columns_excluded(toy_tables):
    """원천 테이블이 없어 전부 결측인 피처는 입력에서 제외함."""
    adm, pat, dx, rx = toy_tables
    coh = build_cohort(adm, pat, CohortConfig())
    feat, _ = build_features(coh.df, pat, dx, rx, procedures=None)
    cols = feature_columns(feat)
    assert "n_procedures" not in cols
    for c in cols:
        assert not feat[c].isna().all(), f"{c} 가 전부 결측인데 입력에 포함됨"


# --- _prior_admissions 의 정렬 전제 ---

def test_prior_admissions_rejects_unsorted_input():
    """정렬이 깨진 입력은 거부해야 함.

    거부하지 않으면 과거 입원을 빠뜨려 값이 과소계산됨(에러 없이 틀린 값).
    """
    from cohort import _prior_admissions

    df = pd.DataFrame({
        "subject_id": [1, 1, 1],
        # 일부러 시간 역순으로 둠
        "admittime": pd.to_datetime(["2150-06-01", "2150-01-01", "2150-03-01"]),
    })
    with pytest.raises(ValueError, match="오름차순 정렬"):
        _prior_admissions(df)


def test_prior_admissions_sorted_input_is_accepted_and_correct():
    """정렬된 입력은 통과하고, 12개월 창 안의 과거 입원만 셈."""
    from cohort import _prior_admissions

    df = pd.DataFrame({
        "subject_id": [1, 1, 1, 2],
        "admittime": pd.to_datetime(
            ["2150-01-01", "2150-03-01", "2152-01-01", "2150-05-01"]
        ),
    })
    got = _prior_admissions(df, months=12)
    # 환자1: 1번째 0건 / 2번째 1건(2개월 전) / 3번째 0건(둘 다 12개월 밖)
    # 환자2: 0건
    assert got.tolist() == [0, 1, 0, 0]


def test_prior_admissions_never_counts_future():
    """미래 입원은 어떤 경우에도 세지 않음: 누수 방지의 핵심."""
    from cohort import _prior_admissions

    df = pd.DataFrame({
        "subject_id": [1, 1, 1],
        "admittime": pd.to_datetime(["2150-01-01", "2150-02-01", "2150-03-01"]),
    })
    got = _prior_admissions(df, months=12)
    assert got[0] == 0, "첫 입원은 과거가 없으므로 0 이어야 한다"
    assert got.tolist() == [0, 1, 2]


# --- 나이 상한(top-coding) 처리 ---

def test_age_top_coded_marks_both_sources():
    """상한에 걸린 나이는 출처와 무관하게 전부 표시돼야 함.

    MIMIC-IV 는 89세 초과를 91 로 일괄 표기함(원본 상한). 그런데
    age = anchor_age + 경과연수 이므로, anchor_age 가 91 미만이어도 계산 결과가
    91 을 넘어 잘리는 경우가 생김(실측 546,028건 중 3,773건).
    플래그를 anchor_age 로만 세우면 그 행들을 놓쳐, "이 91 이 실제 나이인지
    잘린 값인지" 를 구분하려던 목적을 이루지 못함.
    """
    from features import AGE_TOP_CODE, add_demographics

    cohort = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "hadm_id": [10, 20, 30],
        "admittime": pd.to_datetime(["2130-01-01", "2135-01-01", "2130-01-01"]),
    })
    patients = pd.DataFrame({
        "subject_id": [1, 2, 3],
        # 1: 원본 상한 / 2: anchor 는 89 지만 5년 경과로 94 -> clip / 3: 평범
        "anchor_age": [AGE_TOP_CODE, 89, 60],
        "anchor_year": [2130, 2130, 2130],
        "gender": ["F", "M", "F"],
    })
    out = add_demographics(cohort, patients)

    assert out["age"].tolist() == [AGE_TOP_CODE, AGE_TOP_CODE, 60]
    # 세 번째만 상한이 아님
    assert out["age_top_coded"].tolist() == [1, 1, 0]
    # 출처 구분: 원본 상한은 1번뿐
    assert out["age_top_coded_source"].tolist() == [1, 0, 0]


def test_age_top_coded_source_is_not_a_model_feature():
    """출처 컬럼은 진단용이므로 모델 입력에서 빠져야 함."""
    from features import feature_columns

    df = pd.DataFrame({
        "age": [70.0, 91.0],
        "age_top_coded": [0, 1],
        "age_top_coded_source": [0, 1],
        "readmit_30d": [0, 1],
    })
    cols = feature_columns(df)
    assert "age_top_coded" in cols
    assert "age_top_coded_source" not in cols


def test_age_uses_anchor_year_offset():
    """나이는 anchor_age 에 (입원연도 - anchor_year) 를 더해 보정해야 함."""
    from features import add_demographics

    cohort = pd.DataFrame({
        "subject_id": [1],
        "hadm_id": [10],
        "admittime": pd.to_datetime(["2137-06-01"]),
    })
    patients = pd.DataFrame({
        "subject_id": [1], "anchor_age": [50], "anchor_year": [2130], "gender": ["M"],
    })
    out = add_demographics(cohort, patients)
    assert out["age"].iloc[0] == 57      # 50 + (2137 - 2130)
    assert out["age_top_coded"].iloc[0] == 0
