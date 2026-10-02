"""EHR 피처 엔지니어링 (Phase 1).

예측 시점 = 퇴원 시점 (discharge)
------------------------------------
이 정의가 무엇이 누수인지를 결정함.
 - 퇴원 시점 예측이므로 재원일수(LOS), 재원 중 진단, 재원 중 투약은 사용 가능함.
 - 반대로 입원 시점 예측 모델로는 쓸 수 없음(그 시점엔 LOS 가 미확정).
   이 금지 사용례는 MODEL_CARD 2절(금지 사용례)에 적음.
 - 다음 입원 이후의 정보는 어떤 것도 쓰지 않음.

피처군(MODEL_CARD 4절, TRIPOD 9절과 같은 구성)
 - 인구학 : 나이, 성별, 인종/민족, 보험, 입원 경로
 - 임상   : Charlson Comorbidity Index, LOS, 진단 수, 시술 수,
            최근 12개월 입원 횟수, 응급 여부
 - 투약   : 총 투약 수, 고위험 약물, 다약제 복용(polypharmacy)

인코딩, 스케일링은 여기서 하지 않음.
   train 에만 fit 해야 하므로 sklearn Pipeline 안에서 처리함(누수 차단).
   이 모듈은 행 단위로 결정되는 원시 피처만 만듦.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "CHARLSON_WEIGHTS",
    "ELECTIVE_ADMISSION_TYPES",
    "EMERGENCY_ADMISSION_TYPES",
    "FeatureConfig",
    "add_admission_features",
    "add_charlson",
    "add_demographics",
    "add_diagnosis_counts",
    "add_medication_features",
    "add_procedure_counts",
    "build_features",
]

#
#   EW EMER.                    177,459   비예정
#   EU OBSERVATION              119,456   비예정 (응급실 경유 관찰)
#   OBSERVATION ADMIT            84,437   애매
#   URGENT                       54,929   비예정
#   SURGICAL SAME DAY ADMISSION  42,898   예정 (당일 수술 입원)
#   DIRECT OBSERVATION           24,551   애매
#   DIRECT EMER.                 21,973   비예정
#   ELECTIVE                     13,130   예정
#   AMBULATORY OBSERVATION        7,195   애매
#
# CMS HRRP 는 unplanned readmission 만 outcome 으로 셈.
# 애매한 OBSERVATION 계열은 보수적으로 '예정 아님'으로 둠.
# 이 선택이 재입원률을 크게 흔들므로(합성 검증에서 25% -> 12.5%),
# 민감도 분석으로 대안 기준도 함께 보고할 것.
ELECTIVE_ADMISSION_TYPES: tuple[str, ...] = (
    "ELECTIVE",
    "SURGICAL SAME DAY ADMISSION",
)

EMERGENCY_ADMISSION_TYPES: tuple[str, ...] = (
    "EW EMER.",
    "EU OBSERVATION",
    "URGENT",
    "DIRECT EMER.",
)

# 재입원 예측에서 표준으로 쓰이는 동반질환 지수.
# 각 항목의 가중치는 원 논문(Charlson 1987) 값임. ICD 매핑만 Quan 2005 를 따름.
# (Quan 2011 은 가중치를 새로 매긴 판이라 여기 값과 다름.)
CHARLSON_WEIGHTS: dict[str, int] = {
    "myocardial_infarction": 1,
    "congestive_heart_failure": 1,
    "peripheral_vascular": 1,
    "cerebrovascular": 1,
    "dementia": 1,
    "chronic_pulmonary": 1,
    "rheumatic": 1,
    "peptic_ulcer": 1,
    "mild_liver": 1,
    "diabetes_uncomplicated": 1,
    "diabetes_complicated": 2,
    "hemiplegia": 2,
    "renal": 2,
    "malignancy": 2,
    "severe_liver": 3,
    "metastatic_tumor": 6,
    "aids_hiv": 6,
}

# ICD-10 접두사 패턴 (Quan 2005). ICD-9 는 별도 매핑.
_CHARLSON_ICD10 = {
    "myocardial_infarction": r"^(I21|I22|I252)",
    "congestive_heart_failure": r"^(I099|I110|I130|I132|I255|I420|I42[5-9]|I43|I50|P290)",
    "peripheral_vascular": r"^(I70|I71|I731|I738|I739|I771|I790|I792|K551|K558|K559|Z958|Z959)",
    "cerebrovascular": r"^(G45|G46|H340|I6[0-9])",
    "dementia": r"^(F00|F01|F02|F03|F051|G30|G311)",
    "chronic_pulmonary": r"^(I278|I279|J4[0-7]|J6[0-7]|J684|J701|J703)",
    "rheumatic": r"^(M05|M06|M315|M3[2-4]|M351|M353|M360)",
    "peptic_ulcer": r"^(K2[5-8])",
    "mild_liver": r"^(B18|K700|K70[1-3]|K709|K71[3-5]|K717|K7[3-4]|K760|K76[2-4]|K768|K769|Z944)",
    "diabetes_uncomplicated": r"^(E100|E101|E106|E108|E109|E110|E111|E116|E118|E119|"
                              r"E120|E121|E126|E128|E129|E130|E131|E136|E138|E139|"
                              r"E140|E141|E146|E148|E149)",
    "diabetes_complicated": r"^(E10[2-5]|E107|E11[2-5]|E117|E12[2-5]|E127|E13[2-5]|E137|E14[2-5]|E147)",
    "hemiplegia": r"^(G041|G114|G80[1-2]|G81|G82|G83[0-4]|G839)",
    "renal": r"^(I120|I131|N03[2-7]|N05[2-7]|N18|N19|N250|Z49[0-2]|Z940|Z992)",
    "malignancy": r"^(C[0-1][0-9]|C2[0-6]|C3[0-4]|C3[7-9]|C4[0-1]|C43|C4[5-9]|"
                  r"C5[0-8]|C6[0-9]|C7[0-6]|C8[1-5]|C88|C9[0-7])",
    "severe_liver": r"^(I850|I859|I864|I982|K704|K711|K721|K729|K765|K766|K767)",
    "metastatic_tumor": r"^(C7[7-9]|C80)",
    "aids_hiv": r"^(B2[0-2]|B24)",
}

_CHARLSON_ICD9 = {
    "myocardial_infarction": r"^(410|412)",
    "congestive_heart_failure": r"^(39891|40201|40211|40291|40401|40403|40411|40413|"
                                r"40491|40493|425[4-9]|428)",
    "peripheral_vascular": r"^(0930|4373|44[0-1]|4431|4432|4433|4434|4435|4436|4437|4438|4439|"
                           r"4471|5571|5579|V434)",
    "cerebrovascular": r"^(36234|43[0-8])",
    "dementia": r"^(290|2941|3312)",
    "chronic_pulmonary": r"^(4168|4169|49[0-5]|49[6-9]|50[0-5]|5064|5081|5088)",
    "rheumatic": r"^(4465|710[0-4]|714[0-2]|7148|725)",
    "peptic_ulcer": r"^(53[1-4])",
    "mild_liver": r"^(07022|07023|07032|07033|07044|07054|0706|0709|570|571|"
                  r"5733|5734|5738|5739|V427)",
    "diabetes_uncomplicated": r"^(250[0-3]|2508|2509)",
    "diabetes_complicated": r"^(250[4-7])",
    "hemiplegia": r"^(3341|342|343|344[0-6]|3449)",
    # Quan 2005 와 mimic-code charlson.sql 은 585, 586 과 588.0(신성 골이영양증)만 신장질환으로 봄.
    # 58[5-8] 로 넓게 잡으면 587, 588.1~588.9(신성 부갑상선 기능항진 등)까지 들어감(트러블슈팅 25).
    "renal": r"^(40301|40311|40391|40402|40403|40412|40413|40492|40493|582|583[0-7]|"
             r"585|586|5880|V420|V451|V56)",
    "malignancy": r"^(1[4-6][0-9]|17[0-2]|17[4-6]|17[9]|18[0-9]|19[0-5]|20[0-8]|2386)",
    "severe_liver": r"^(4560|4561|4562|572[2-8])",
    "metastatic_tumor": r"^(19[6-9])",
    "aids_hiv": r"^(04[2-4])",
}

# 고위험 약물 (재입원과 연관이 보고된 계열).
# 이름 기반 매칭이라 완벽하지 않음: 한계로 기록할 것.
HIGH_RISK_DRUG_PATTERNS: dict[str, str] = {
    "anticoagulant": r"warfarin|heparin|enoxaparin|apixaban|rivaroxaban|dabigatran",
    "insulin": r"insulin",
    "opioid": r"morphine|oxycodone|hydromorphone|fentanyl|methadone|hydrocodone",
    "antiarrhythmic": r"amiodarone|digoxin|sotalol|flecainide",
    "diuretic": r"furosemide|bumetanide|torsemide|spironolactone|hydrochlorothiazide",
    "immunosuppressant": r"tacrolimus|cyclosporine|mycophenolate|azathioprine",
}

POLYPHARMACY_THRESHOLD = 5

# MIMIC-IV 는 89세 초과 환자의 anchor_age 를 일괄 91 로 상한 처리(top-coding)함.
# MIMIC-IV 문서의 정의("anchor_year 에 89세를 넘으면 anchor_age 를 91 로 둔다")와 실제 분포가
# 같음. patients 에서 90 은 0명이고 91 만 상한값으로 있음(따로 셈).
# 상한 처리를 무시하면 고령군의 나이-위험 관계 해석이 왜곡됨.
AGE_TOP_CODE = 91


@dataclass
class FeatureConfig:
    elective_types: tuple[str, ...] = ELECTIVE_ADMISSION_TYPES
    emergency_types: tuple[str, ...] = EMERGENCY_ADMISSION_TYPES
    polypharmacy_threshold: int = POLYPHARMACY_THRESHOLD
    charlson_use_icd9: bool = True
    missing_report: dict = field(default_factory=dict)


# 인구학
def add_demographics(cohort: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """나이, 성별. MIMIC-IV 는 anchor_age / anchor_year 체계를 씀.

    anchor_age 는 anchor_year 시점의 나이이며, 89세 초과는 비식별화를 위해
    일괄 91 로 표기됨. 이 상한 처리(top-coding)를 그대로 두면 고령군 해석이
    왜곡되므로 플래그로 남김.

    build_cohort() 가 나이 필터를 위해 이미 anchor_age/gender 를 병합해 둘 수 있음.
       그대로 다시 merge 하면 anchor_age_pat 같은 중복 컬럼이 생기고, 그것이
       feature_columns() 를 통과해 사실상 같은 변수가 두 번 입력됨.
       이미 있는 컬럼은 병합 대상에서 제외함.
    """
    df = cohort.copy()
    wanted = ("anchor_age", "anchor_year", "gender")
    cols = ["subject_id"] + [
        c for c in wanted if c in patients.columns and c not in df.columns
    ]
    if len(cols) > 1:
        df = df.merge(patients[cols], on="subject_id", how="left")

    if "anchor_age" in df.columns:
        if "anchor_year" in df.columns and "admittime" in df.columns:
            # 입원 연도 - anchor_year 만큼 보정
            df["age"] = df["anchor_age"] + (df["admittime"].dt.year - df["anchor_year"])
        else:
            df["age"] = df["anchor_age"]
        # 상한에 걸린 나이는 두 갈래로 생김.
        #   (a) 원본 상한: MIMIC-IV 가 89세 초과를 일괄 91 로 표기
        #   (b) 우리 clip: anchor_age + 경과연수가 91 을 넘어 잘린 경우
        # 실측(546,028 입원): age==91 인 20,313건 중 5,536건이 (b) 이고, 그중
        # 3,773건은 anchor_age 가 91 미만이라 원본 기준으로는 상한이 아니었음.
        # 플래그를 anchor_age 로만 세우면 그 27% 를 놓쳐, "이 91 은 실제 나이인가
        # 잘린 값인가" 를 구분하려던 목적을 이루지 못함. 잘린 사실 자체를 표시함.
        df["age_top_coded"] = (
            (df["anchor_age"] >= AGE_TOP_CODE) | (df["age"] >= AGE_TOP_CODE)
        ).astype(int)
        # 원본 상한만 따로 보고 싶을 때를 위해 출처도 남김(진단, 감사용).
        df["age_top_coded_source"] = (df["anchor_age"] >= AGE_TOP_CODE).astype(int)
        df["age"] = df["age"].clip(lower=0, upper=AGE_TOP_CODE)

    if "gender" in df.columns:
        df["is_female"] = (df["gender"] == "F").astype(int)
    return df


# 입원 관련
def add_admission_features(cohort: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """입원 유형, 경로, 보험, 응급실 체류 등.

    discharge_location 은 쓰지 않음.
       '사망', '호스피스' 등이 결과를 사실상 알려주는 값이라 누수가 됨.
    """
    cfg = cfg or FeatureConfig()
    df = cohort.copy()

    if "los_days" not in df.columns and {"admittime", "dischtime"} <= set(df.columns):
        df["los_days"] = (df.dischtime - df.admittime).dt.total_seconds() / 86400.0
    df["los_log"] = np.log1p(df["los_days"].clip(lower=0))

    if "admission_type" in df.columns:
        df["is_emergency"] = df.admission_type.isin(cfg.emergency_types).astype(int)
        df["is_elective"] = df.admission_type.isin(cfg.elective_types).astype(int)

    # 응급실 체류시간 (edregtime ~ edouttime). 응급실 미경유는 결측이 정상임.
    if {"edregtime", "edouttime"} <= set(df.columns):
        ed_in = pd.to_datetime(df.edregtime, errors="coerce")
        ed_out = pd.to_datetime(df.edouttime, errors="coerce")
        df["ed_hours"] = (ed_out - ed_in).dt.total_seconds() / 3600.0
        df["via_ed"] = ed_in.notna().astype(int)
        df["ed_hours"] = df["ed_hours"].fillna(0.0)

    if "admittime" in df.columns:
        # admit_year 를 피처로 쓰지 않음.
        #    MIMIC-IV 는 환자별로 날짜를 무작위 시프트해 비식별화하므로 연도의 절대값에
        #    의미가 없음(환자마다 오프셋이 다름). 실제 기간은 patients.anchor_year_group
        #    (예: "2008 - 2010")에만 담겨 있음.
        #    V7(시간 분할 검증)도 admittime 의 연도가 아니라 anchor_year_group 으로
        #      수행해야 함. 시프트된 연도로 나누면 시간 분할이 아니라 무작위 분할이 됨.
        df["admit_year_shifted"] = df.admittime.dt.year   # 진단용으로만 보관, 피처 아님
        # 월, 요일은 시프트가 연 단위라 상대적 의미가 보존됨(계절성, 주말 효과).
        df["admit_month"] = df.admittime.dt.month
        df["admit_weekday"] = df.admittime.dt.weekday
    if "dischtime" in df.columns:
        # 주말 퇴원은 후속 관리가 어려워 재입원과 연관이 보고됨
        df["discharge_weekend"] = (df.dischtime.dt.weekday >= 5).astype(int)
    return df


# Charlson Comorbidity Index
def _normalize_icd(code: pd.Series) -> pd.Series:
    return code.astype(str).str.upper().str.replace(".", "", regex=False).str.strip()


def add_charlson(
    cohort: pd.DataFrame,
    diagnoses: pd.DataFrame,
    cfg: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Charlson Comorbidity Index (Quan et al. 2005 매핑).

    diagnoses_icd 는 해당 입원에서 기록된 진단이므로 퇴원 시점에 사용 가능함.
    각 동반질환 이진 플래그와 가중합 지수를 함께 만듦
    (SHAP 에서 어떤 질환이 기여했는지 보려면 개별 플래그가 필요함).
    """
    cfg = cfg or FeatureConfig()
    dx = diagnoses.copy()
    dx["code"] = _normalize_icd(dx["icd_code"])

    ver = dx["icd_version"].astype(int) if "icd_version" in dx.columns else pd.Series(10, index=dx.index)
    flags = pd.DataFrame({"hadm_id": dx["hadm_id"]})

    for cond in CHARLSON_WEIGHTS:
        m10 = (ver == 10) & dx["code"].str.match(_CHARLSON_ICD10[cond], na=False)
        if cfg.charlson_use_icd9:
            m9 = (ver == 9) & dx["code"].str.match(_CHARLSON_ICD9[cond], na=False)
            flags[cond] = (m10 | m9).astype(int)
        else:
            flags[cond] = m10.astype(int)

    agg = flags.groupby("hadm_id", sort=False).max()

    # 계층 규칙: 중증이 있으면 경증은 세지 않음(이중 계산 방지)
    if {"diabetes_complicated", "diabetes_uncomplicated"} <= set(agg.columns):
        agg.loc[agg.diabetes_complicated == 1, "diabetes_uncomplicated"] = 0
    if {"severe_liver", "mild_liver"} <= set(agg.columns):
        agg.loc[agg.severe_liver == 1, "mild_liver"] = 0
    if {"metastatic_tumor", "malignancy"} <= set(agg.columns):
        agg.loc[agg.metastatic_tumor == 1, "malignancy"] = 0

    agg["charlson_score"] = sum(agg[c] * w for c, w in CHARLSON_WEIGHTS.items())
    agg = agg.add_prefix("cci_").rename(columns={"cci_charlson_score": "charlson_score"})

    out = cohort.merge(agg, left_on="hadm_id", right_index=True, how="left")
    fill_cols = [c for c in out.columns if c.startswith("cci_")] + ["charlson_score"]
    out[fill_cols] = out[fill_cols].fillna(0).astype(int)
    return out


# 진단 / 시술 수
def add_diagnosis_counts(cohort: pd.DataFrame, diagnoses: pd.DataFrame) -> pd.DataFrame:
    """진단 개수: 질병 부담의 대리 지표."""
    cnt = diagnoses.groupby("hadm_id", sort=False).size().rename("n_diagnoses")
    out = cohort.merge(cnt, left_on="hadm_id", right_index=True, how="left")
    out["n_diagnoses"] = out["n_diagnoses"].fillna(0).astype(int)
    return out


def add_procedure_counts(cohort: pd.DataFrame, procedures: pd.DataFrame | None) -> pd.DataFrame:
    """시술 개수. 테이블이 없으면 건너뜀(다운로드 진행 중일 수 있음)."""
    out = cohort.copy()
    if procedures is None or len(procedures) == 0:
        out["n_procedures"] = np.nan
        return out
    cnt = procedures.groupby("hadm_id", sort=False).size().rename("n_procedures")
    out = out.merge(cnt, left_on="hadm_id", right_index=True, how="left")
    out["n_procedures"] = out["n_procedures"].fillna(0).astype(int)
    return out


# 투약
def add_medication_features(
    cohort: pd.DataFrame,
    prescriptions: pd.DataFrame | None,
    cfg: FeatureConfig | None = None,
) -> pd.DataFrame:
    """투약 수, 다약제 복용, 고위험 약물 플래그.

    약물명 문자열 매칭이라 완전하지 않음(오탈자, 상품명, 복합제 미포함).
       RxNorm 등 표준 사전 매핑이 정석이며, 이 한계는 MODEL_CARD 에 남김.
    """
    cfg = cfg or FeatureConfig()
    out = cohort.copy()
    if prescriptions is None or len(prescriptions) == 0:
        out["n_medications"] = np.nan
        out["polypharmacy"] = np.nan
        for k in HIGH_RISK_DRUG_PATTERNS:
            out[f"drug_{k}"] = np.nan
        return out

    rx = prescriptions.copy()
    name_col = "drug" if "drug" in rx.columns else rx.columns[-1]
    rx["_name"] = rx[name_col].astype(str).str.lower()

    n_uniq = rx.groupby("hadm_id", sort=False)["_name"].nunique().rename("n_medications")
    out = out.merge(n_uniq, left_on="hadm_id", right_index=True, how="left")
    out["n_medications"] = out["n_medications"].fillna(0).astype(int)
    out["polypharmacy"] = (out["n_medications"] >= cfg.polypharmacy_threshold).astype(int)

    for key, pat in HIGH_RISK_DRUG_PATTERNS.items():
        hit = rx.loc[rx["_name"].str.contains(pat, regex=True, na=False), "hadm_id"].unique()
        out[f"drug_{key}"] = out["hadm_id"].isin(hit).astype(int)

    drug_cols = [f"drug_{k}" for k in HIGH_RISK_DRUG_PATTERNS]
    out["n_high_risk_drug_classes"] = out[drug_cols].sum(axis=1)
    return out


# 통합
def build_features(
    cohort: pd.DataFrame,
    patients: pd.DataFrame,
    diagnoses: pd.DataFrame,
    prescriptions: pd.DataFrame | None = None,
    procedures: pd.DataFrame | None = None,
    cfg: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """전체 피처 생성. (피처 DataFrame, 결측 리포트) 반환.

    결측 리포트를 함께 내놓는 이유: TRIPOD+AI 는 결측 처리 방법의 보고를 요구하고,
    결측 자체가 정보인 경우(예: 영상 유무 = MNAR)가 이 프로젝트의 핵심 주제이기 때문임.
    """
    cfg = cfg or FeatureConfig()
    df = add_demographics(cohort, patients)
    df = add_admission_features(df, cfg)
    df = add_charlson(df, diagnoses, cfg)
    df = add_diagnosis_counts(df, diagnoses)
    df = add_procedure_counts(df, procedures)
    df = add_medication_features(df, prescriptions, cfg)

    missing = {c: float(df[c].isna().mean()) for c in df.columns if df[c].isna().any()}
    return df, missing


def feature_columns(df: pd.DataFrame, verbose: bool = False) -> list[str]:
    """모델 입력으로 쓸 수치형 피처 목록.

    식별자, 시각, 라벨, 누수 위험 컬럼을 제외하고, 아래 두 가지를 방어적으로 걸러냄.
      - 병합 접미사가 붙은 중복 컬럼(_x/_y/_pat) : 같은 변수가 두 번 입력되는 것을 막음
      - 전부 결측인 컬럼                         : 원천 테이블이 없을 때 생긴 빈 피처
    """
    drop = {
        "subject_id", "hadm_id", "admit_provider_id",
        "admittime", "dischtime", "deathtime", "edregtime", "edouttime",
        "next_admittime", "next_admission_type", "days_to_next_admission",
        "readmit_30d", "excluded_planned_readmit",
        # 누수: is_last_admission=1 이면 라벨이 항상 0 임.
        #   (다음 입원 기록이 없어 음성이 된 것이므로 결과를 그대로 알려주는 변수)
        "is_last_admission",
        "hospital_expire_flag",          # 코호트에서 이미 제외됨
        "discharge_location",            # 누수 (사망, 호스피스가 결과를 알려줌)
        "anchor_age", "anchor_year", "gender",
        # 시프트된 연도: 절대값에 의미가 없음(환자별 오프셋). V7 은 anchor_year_group 사용.
        "admit_year_shifted",
        # 상한의 '출처'는 진단, 감사용 기록임. age_top_coded 와 거의 같은 정보라
        # 둘 다 넣으면 사실상 같은 변수를 두 번 주는 셈이 됨.
        "age_top_coded_source",
    }
    suffixes = ("_x", "_y", "_pat")
    cols, dropped = [], {}
    for c in df.columns:
        if c in drop:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c.endswith(suffixes):
            dropped[c] = "중복 병합 컬럼"
            continue
        if df[c].isna().all():
            dropped[c] = "전부 결측"
            continue
        cols.append(c)

    if verbose and dropped:
        for c, why in dropped.items():
            print(f"  [제외] {c}: {why}")
    return cols
