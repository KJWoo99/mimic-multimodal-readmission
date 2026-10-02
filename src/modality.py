"""결측 모달리티 진단 (Phase 1.5 / M1-b).

핵심 질문
--------
MIMIC-IV 환자 전부가 흉부 X선을 찍지는 않음. 누가 찍었는가는 무작위인가?

의사가 흉부 X선을 오더했다는 것은 그 환자가 더 아프거나 특정 증상이 있었다는 뜻임.
즉 영상 결측은 MCAR(완전 무작위 결측)이 아니라 MNAR(비무작위 결측) 일 가능성이 높고,
결측 여부 자체가 결과와 직접 연관될 수 있음.

왜 CXR 다운로드 전에 하는가
-----------------------------
MIMIC-CXR-JPG 는 570GB 다. 전체를 받을 수 없으므로 어떤 코호트의 영상을 받을지
먼저 정해야 함. 이 분석이 그 근거가 되며, 메타데이터(58MB)만으로 수행 가능함.

해석할 때 주의할 점: 반드시 함께 보고할 것
------------------------------------
`has_cxr` 가 성능을 올린다면 그것은 "영상의 내용"이 아니라 "의사의 오더 행위" 를
학습한 것임. 배포 시 이것이 정당한가는 별개 문제임.
  - 예측 시점에 오더 정보를 알 수 있는가?
  - 병원마다 오더 관행이 다르면 일반화되는가?
이 프로젝트의 예측 시점은 퇴원이므로 재원 중 촬영 여부는 알 수 있음.
그러나 그 정보가 임상적으로 무엇을 뜻하는지는 별도로 논의해야 함.

시각 링킹의 전제
---------------
MIMIC-CXR 과 MIMIC-IV 는 환자별로 동일한 날짜 시프트를 공유함.
따라서 같은 subject_id 안에서는 두 데이터셋의 시각을 직접 비교할 수 있음.
(CXR StudyDate 연도 범위 2110~2208, admittime 2105~2214 로 겹침)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

__all__ = [
    "CXR_LINK_COLUMNS",
    "ModalityDiagnosis",
    "diagnose_missingness",
    "download_manifest",
    "link_cxr_to_admissions",
    "linked_studies",
    "parse_study_datetime",
    "risk_ratio",
    "stratified_risk_ratio",
]

# 링킹 결과로 코호트에 붙는 컬럼
CXR_LINK_COLUMNS = ("has_cxr", "n_cxr", "hours_cxr_before_discharge", "cxr_view_ap")


def parse_study_datetime(date_col: pd.Series, time_col: pd.Series) -> pd.Series:
    """MIMIC-CXR 의 StudyDate(int YYYYMMDD) + StudyTime(float HHMMSS.fff) -> datetime.

    StudyTime 은 `213014.531` = 21:30:14.531 형식임.
    자정 직후는 `0.187` 처럼 앞자리가 없으므로 6자리로 zero-pad 해야 함.
    """
    d = pd.to_datetime(date_col.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    t = time_col.fillna(0).astype(float)
    secs = (
        (t // 10000) * 3600           # 시
        + ((t % 10000) // 100) * 60   # 분
        + (t % 100)                   # 초
    )
    return d + pd.to_timedelta(secs, unit="s")


def link_cxr_to_admissions(
    cohort: pd.DataFrame,
    cxr_meta: pd.DataFrame,
    within_stay_only: bool = True,
) -> pd.DataFrame:
    """입원별로 CXR 촬영 여부, 횟수, 마지막 촬영 시점을 붙임.

    Parameters
    ----------
    within_stay_only : True 면 admittime ~ dischtime 사이 촬영만 셈.
        예측 시점이 퇴원이므로 퇴원 이후 촬영은 절대 쓰면 안 됨(미래 정보).

    Notes
    -----
    `hours_cxr_before_discharge` 는 M2(영상 시점 정렬)에서 씀.
    퇴원 직전 영상과 입원 첫날 영상은 완전히 다른 정보이므로,
    "예측 시점 이전 N시간 내 영상만 사용" 제약을 걸어 성능 변화를 볼 때 필요함.
    """
    need = {"subject_id", "hadm_id", "admittime", "dischtime"}
    missing = need - set(cohort.columns)
    if missing:
        raise ValueError(f"cohort 에 필요한 컬럼이 없습니다: {sorted(missing)}")

    cxr = cxr_meta.copy()
    if "study_datetime" not in cxr.columns:
        cxr["study_datetime"] = parse_study_datetime(cxr["StudyDate"], cxr["StudyTime"])
    cxr = cxr.dropna(subset=["study_datetime"])

    # study 단위로 축약(한 study 에 여러 장이 있으므로): AP 여부는 study 내 any 로 봄
    if "ViewPosition" in cxr.columns:
        cxr["_is_ap"] = (cxr["ViewPosition"] == "AP").astype(int)
    else:
        cxr["_is_ap"] = 0
    studies = (
        cxr.groupby(["subject_id", "study_id"], sort=False)
        .agg(study_datetime=("study_datetime", "min"), is_ap=("_is_ap", "max"))
        .reset_index()
    )

    # subject_id 로 조인한 뒤 시간 조건으로 거름.
    # 전체 교차조인이 아니라 환자 단위 조인이라 규모가 감당 가능함.
    merged = cohort[["subject_id", "hadm_id", "admittime", "dischtime"]].merge(
        studies, on="subject_id", how="left"
    )
    if within_stay_only:
        in_window = (
            merged.study_datetime.notna()
            & (merged.study_datetime >= merged.admittime)
            & (merged.study_datetime <= merged.dischtime)
        )
    else:
        in_window = merged.study_datetime.notna() & (merged.study_datetime <= merged.dischtime)
    merged = merged[in_window]

    merged["_hours_before_disch"] = (
        merged.dischtime - merged.study_datetime
    ).dt.total_seconds() / 3600.0

    agg = merged.groupby("hadm_id", sort=False).agg(
        n_cxr=("study_id", "nunique"),
        hours_cxr_before_discharge=("_hours_before_disch", "min"),  # 가장 최근 촬영
        cxr_view_ap=("is_ap", "max"),
    )

    out = cohort.merge(agg, left_on="hadm_id", right_index=True, how="left")
    out["n_cxr"] = out["n_cxr"].fillna(0).astype(int)
    out["cxr_view_ap"] = out["cxr_view_ap"].fillna(0).astype(int)
    out["has_cxr"] = (out["n_cxr"] > 0).astype(int)
    # 영상이 없으면 '퇴원 몇 시간 전' 이 정의되지 않음. 0 으로 채우면 안 됨.
    return out


def linked_studies(cohort: pd.DataFrame, cxr_meta: pd.DataFrame,
                   one_per_admission: bool = True) -> pd.DataFrame:
    """코호트 재원 기간에 촬영된 study 목록. Phase 2 다운로드 대상 산출에 씀.

    one_per_admission 은 cxr_dataset.build_cxr_index 와 같은 규칙이어야 함.
    다르면 안 쓸 파일을 받게 됨. 촬영 횟수는 중앙값 1, 최대 97 로 편차가 커서
    전부 받아 쓰면 중환자가 학습에 수십 번 들어감.
    """
    from cxr_dataset import FRONTAL_VIEWS

    cxr = cxr_meta.copy()
    if "study_datetime" not in cxr.columns:
        cxr["study_datetime"] = parse_study_datetime(cxr["StudyDate"], cxr["StudyTime"])
    cxr = cxr.dropna(subset=["study_datetime"])
    cxr["_is_ap"] = (cxr.get("ViewPosition") == "AP").astype(int)

    # study 안에서 어느 장을 받을지까지 여기서 정함.
    # 학습 인덱스(build_cxr_index)와 같은 규칙(정면 우선 -> dicom_id)으로 대표 1장을 여기서 확정함.
    # 두 곳의 규칙이 다르면 받은 파일과 학습이 찾는 파일이 어긋남(트러블슈팅 7).
    if "dicom_id" in cxr.columns and "ViewPosition" in cxr.columns:
        cxr["_not_frontal"] = (~cxr["ViewPosition"].isin(FRONTAL_VIEWS)).astype(int)
        rep = (
            cxr.sort_values(["study_id", "_not_frontal", "dicom_id"])
            .drop_duplicates("study_id", keep="first")[["study_id", "dicom_id"]]
            .rename(columns={"dicom_id": "dicom_view"})
        )
    else:
        rep = None

    studies = (
        cxr.groupby(["subject_id", "study_id"], sort=False)
        .agg(study_datetime=("study_datetime", "min"), is_ap=("_is_ap", "max"))
        .reset_index()
    )
    if rep is not None:
        studies = studies.merge(rep, on="study_id", how="left")
    merged = cohort[["subject_id", "hadm_id", "admittime", "dischtime"]].merge(
        studies, on="subject_id", how="inner"
    )
    keep = (merged.study_datetime >= merged.admittime) & (
        merged.study_datetime <= merged.dischtime
    )
    out = merged[keep].drop_duplicates(["subject_id", "study_id"])
    if one_per_admission:
        out = out.assign(
            _h=(out.dischtime - out.study_datetime).dt.total_seconds() / 3600.0
        ).sort_values(["hadm_id", "_h", "study_id"]).drop_duplicates("hadm_id")
        out = out.drop(columns="_h")
    return out.reset_index(drop=True)


# CXR JPG 한 장의 평균 크기(KB). 216장 표본 실측.
CXR_AVG_KB = 1434.0


def download_manifest(studies: pd.DataFrame, image_filenames: Path,
                      avg_kb: float = CXR_AVG_KB,
                      one_file_per_study: bool = True,
                      frontal_first: bool = True) -> tuple[list[str], dict]:
    """IMAGE_FILENAMES 에서 대상 study 의 파일 경로만 추림.

    한 study 에 정면(PA/AP)과 측면이 함께 들어 있음. 학습은 1장만 쓰므로
    정면을 우선 고름. 측면이 뽑히면 표준 판독 화면이 아니게 됨.

    `studies` 에 `dicom_view`(받을 dicom_id) 컬럼이 있어야 이 선택이 동작함.
    `linked_studies()` 가 그 컬럼을 만들어 줌. 호출부가 그 컬럼 없이 넘기면 알파벳순 첫 파일을 받게
    되므로 경고를 남김(트러블슈팅 7).
    """
    import warnings as _warnings
    want = {f"s{int(s)}": int(s) for s in studies["study_id"].unique()}
    view_of: dict[int, str] = {}
    if frontal_first:
        if "dicom_view" in studies.columns:
            view_of = {
                int(sid): str(dv)
                for sid, dv in zip(studies.study_id, studies.dicom_view, strict=False)
                if isinstance(dv, str) or (dv == dv)      # NaN 제외
            }
        else:
            _warnings.warn(
                "studies 에 'dicom_view' 가 없어 정면 우선 선택이 비활성화된다. "
                "알파벳순 첫 파일을 받게 되어 학습 인덱스와 어긋날 수 있다.",
                RuntimeWarning, stacklevel=2,
            )

    by_study: dict[str, list[str]] = {}
    with image_filenames.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("/")
            if len(parts) >= 4 and parts[3] in want:
                by_study.setdefault(parts[3], []).append(line)

    paths = []
    for s, files in by_study.items():
        files.sort()
        if one_file_per_study:
            pick = files[0]
            if frontal_first and view_of:
                sid = want[s]
                pref = view_of.get(sid)
                if pref:
                    for f in files:
                        if Path(f).stem == pref:
                            pick = f
                            break
            paths.append(pick)
        else:
            paths.extend(files)

    return paths, {
        "studies_requested": len(want),
        "studies_found": len(by_study),
        "files": len(paths),
        # 정면 우선 선택이 실제로 동작했는지. False 면 알파벳순 첫 파일을 받은
        # 것이므로 학습 인덱스와 어긋남: 산출물에 남겨 사후 확인이 가능하게 함.
        "frontal_first_applied": bool(frontal_first and view_of),
        "avg_kb": avg_kb,
        "est_gb": round(len(paths) * avg_kb / 1024 / 1024, 1),
        "one_file_per_study": one_file_per_study,
    }


def risk_ratio(indicator: np.ndarray, outcome: np.ndarray) -> dict[str, float]:
    """이진 지표 대비 결과의 위험비(RR), 오즈비(OR)와 95% 신뢰구간.

    왜 점이연 상관만으로는 부족한가
    --------------------------------
    점이연 상관 r 은 두 집단의 크기가 크게 불균형하면 구조적으로 축소됨.
    영상 보유 8.6% vs 미보유 91.4% 인 상황에서
    재입원률이 21.67% vs 15.97%(1.36배) 였는데도 r 은 0.043 에 그침.
    r 만 보고 "연관 없음"으로 결론내면 틀린 판단이 됨.
    따라서 상대위험도를 함께 보고함.
    """
    ind = np.asarray(indicator).astype(int)
    out = np.asarray(outcome).astype(int)
    a = int(((ind == 1) & (out == 1)).sum())   # 노출+, 결과+
    b = int(((ind == 1) & (out == 0)).sum())
    c = int(((ind == 0) & (out == 1)).sum())   # 노출-, 결과+
    d = int(((ind == 0) & (out == 0)).sum())

    p1 = a / (a + b) if (a + b) else float("nan")
    p0 = c / (c + d) if (c + d) else float("nan")
    rr = p1 / p0 if p0 else float("nan")

    # log(RR) 의 표준오차로 95% CI
    if a and c and (a + b) and (c + d):
        se = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
        lo, hi = float(np.exp(np.log(rr) - 1.96 * se)), float(np.exp(np.log(rr) + 1.96 * se))
    else:
        lo = hi = float("nan")

    orr = (a * d) / (b * c) if (b and c) else float("nan")
    return {
        "rate_exposed": p1, "rate_unexposed": p0,
        "risk_ratio": float(rr), "rr_ci_low": lo, "rr_ci_high": hi,
        "odds_ratio": float(orr),
        "n_exposed": a + b, "n_unexposed": c + d,
    }


def stratified_risk_ratio(
    indicator: np.ndarray,
    outcome: np.ndarray,
    strata: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    """층화 위험비 + Mantel-Haenszel 요약 RR.

    왜 조 RR(crude) 만으로는 부족한가
    ----------------------------------
    `has_cxr` 는 재원일수와 강하게 연관됨(r=0.127).
    재원이 길수록 영상을 찍을 확률이 오르고(3.7% -> 16.1%), 동시에 더 아파서
    재입원 위험도 오름. 즉 재원일수가 교란변수임.

    보정하지 않은 조 RR 은 이 경로를 통해 부풀려짐.
    실측에서 조 RR 1.340 -> LOS 4분위 보정 후 MH RR 1.202 로 0.138 감소함.
    조 RR 만 보고하면 위험 증가를 20% 가 아니라 34% 로 과장하게 됨.

    다만 층별 RR 이 1.177 / 1.190 / 1.207 / 1.206 으로 일관되게 1 을 넘었고
    모든 층의 신뢰구간이 1 을 제외해, 보정 후에도 연관 자체는 유지됨
    (outputs/phase1_5_metrics.json 의 mnar.los_adjusted).

    Returns
    -------
    (층별 표, Mantel-Haenszel 요약 RR)
    """
    ind = np.asarray(indicator).astype(int)
    out = np.asarray(outcome).astype(int)
    st = np.asarray(strata)

    rows, mh_num, mh_den = [], 0.0, 0.0
    for s in pd.unique(st):
        m = st == s
        if np.unique(ind[m]).size < 2:
            continue
        e = risk_ratio(ind[m], out[m])
        rows.append({
            "stratum": s, "n": int(m.sum()),
            "exposure_rate": float(ind[m].mean()),
            "rate_exposed": e["rate_exposed"], "rate_unexposed": e["rate_unexposed"],
            "risk_ratio": e["risk_ratio"],
            "rr_ci_low": e["rr_ci_low"], "rr_ci_high": e["rr_ci_high"],
        })
        a = int(((ind[m] == 1) & (out[m] == 1)).sum())
        b = int(((ind[m] == 1) & (out[m] == 0)).sum())
        c = int(((ind[m] == 0) & (out[m] == 1)).sum())
        d = int(((ind[m] == 0) & (out[m] == 0)).sum())
        n = a + b + c + d
        if n:
            mh_num += a * (c + d) / n
            mh_den += c * (a + b) / n

    mh = float(mh_num / mh_den) if mh_den else float("nan")
    return pd.DataFrame(rows), mh


@dataclass
class ModalityDiagnosis:
    """결측 모달리티 진단 결과."""

    n_admissions: int
    n_with_cxr: int
    cxr_rate: float
    correlations: pd.DataFrame          # 변수별 점이연 상관
    outcome_r: float                    # has_cxr 와 결과의 상관
    outcome_p: float
    outcome_p_bonferroni: float
    n_tests: int
    effect: dict[str, float] = field(default_factory=dict)   # RR / OR
    notes: list[str] = field(default_factory=list)

    def verdict(
        self, effect_threshold: float = 0.1, alpha: float = 0.05, rr_threshold: float = 1.2
    ) -> str:
        """MNAR 판정.

        두 축을 함께 봄.
          - 점이연 상관 |r| : 선형 연관의 크기
          - 위험비 RR       : 집단 불균형에 강건한 상대위험도

        대규모 표본에서는 p값이 과민해 사소한 연관도 유의하게 나오므로,
        p값은 보조로만 씀. 둘 중 하나라도 실질적이면 MNAR 로 봄.
        RR 은 신뢰구간이 1을 포함하지 않을 때만 실질적인 것으로 취급함.
        """
        sig = self.outcome_p_bonferroni < alpha
        strong_r = abs(self.outcome_r) >= effect_threshold

        rr = self.effect.get("risk_ratio", float("nan"))
        lo = self.effect.get("rr_ci_low", float("nan"))
        hi = self.effect.get("rr_ci_high", float("nan"))
        rr_excludes_one = not (np.isnan(lo) or np.isnan(hi)) and (lo > 1.0 or hi < 1.0)
        strong_rr = (not np.isnan(rr)) and (rr >= rr_threshold or rr <= 1 / rr_threshold)

        if (strong_r or (strong_rr and rr_excludes_one)) and sig:
            why = []
            if strong_r:
                why.append(f"|r|={abs(self.outcome_r):.3f}")
            if strong_rr:
                why.append(f"RR={rr:.2f} [{lo:.2f}, {hi:.2f}]")
            return f"MNAR 강함: 영상 유무가 결과와 실질적으로 연관됨 ({', '.join(why)})"
        if sig:
            return (f"유의하나 효과크기 작음 (|r|={abs(self.outcome_r):.3f}, RR={rr:.2f}) "
                    ": 표본이 커서 검출된 것일 수 있음")
        return "결과와의 연관 근거 부족: 이 코호트에서는 MCAR 로 볼 여지"


def diagnose_missingness(
    df: pd.DataFrame,
    outcome_col: str = "readmit_30d",
    indicator_col: str = "has_cxr",
    covariates: list[str] | None = None,
    alpha: float = 0.05,
) -> ModalityDiagnosis:
    """영상 결측이 MNAR 인지 진단함.

    점이연 상관(point-biserial correlation)으로 이진 지표와 연속/이진 변수의 연관을 계량함.
    다중검정이므로 Bonferroni 보정을 적용함.

    상관은 인과가 아님. "영상을 찍은 환자가 더 아팠다" 와
       "영상이 재입원을 유발했다" 는 전혀 다른 이야기임.
    """
    if indicator_col not in df.columns:
        raise ValueError(f"{indicator_col} 컬럼이 없습니다. link_cxr_to_admissions 를 먼저 실행하세요.")

    ind = df[indicator_col].to_numpy().astype(float)
    notes: list[str] = []

    if covariates is None:
        # 기본: 관측 가능한 임상 변수들. 존재하는 것만 씀.
        candidates = [
            "charlson_score", "los_days", "age", "n_diagnoses",
            "prior_admissions_12m", "is_emergency", "is_female",
            "n_medications", "polypharmacy",
        ]
        covariates = [c for c in candidates if c in df.columns and df[c].notna().any()]

    targets = [outcome_col, *covariates]
    rows = []
    for col in targets:
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(v) & ~np.isnan(ind)
        if mask.sum() < 3 or np.unique(ind[mask]).size < 2 or np.unique(v[mask]).size < 2:
            notes.append(f"{col}: 분산 부족으로 건너뜀")
            continue
        r, p = pointbiserialr(ind[mask], v[mask])
        rows.append({"variable": col, "r": float(r), "p": float(p), "n": int(mask.sum())})

    corr = pd.DataFrame(rows)
    n_tests = max(1, len(corr))
    if not corr.empty:
        corr["p_bonferroni"] = (corr["p"] * n_tests).clip(upper=1.0)
        corr["abs_r"] = corr["r"].abs()
        corr = corr.sort_values("abs_r", ascending=False).reset_index(drop=True)

    out_row = corr[corr.variable == outcome_col] if not corr.empty else pd.DataFrame()
    o_r = float(out_row.r.iloc[0]) if len(out_row) else float("nan")
    o_p = float(out_row.p.iloc[0]) if len(out_row) else float("nan")
    o_pb = float(out_row.p_bonferroni.iloc[0]) if len(out_row) else float("nan")

    # 위험비: 집단 불균형에서 r 이 축소되는 문제를 보완함
    eff: dict[str, float] = {}
    if outcome_col in df.columns:
        y = pd.to_numeric(df[outcome_col], errors="coerce").to_numpy()
        m = ~np.isnan(y) & ~np.isnan(ind)
        if m.sum() and np.unique(ind[m]).size == 2:
            eff = risk_ratio(ind[m], y[m])

    notes.append(
        "대규모 표본에서는 p값이 과민하므로 효과크기를 우선 본다. "
        "다중검정에는 Bonferroni 보정을 적용했다."
    )
    notes.append(
        "점이연 상관 r 은 두 집단 크기가 크게 불균형하면 구조적으로 축소되므로 "
        "위험비(RR)를 함께 본다."
    )
    return ModalityDiagnosis(
        n_admissions=len(df),
        n_with_cxr=int(df[indicator_col].sum()),
        cxr_rate=float(df[indicator_col].mean()),
        correlations=corr,
        outcome_r=o_r,
        outcome_p=o_p,
        outcome_p_bonferroni=o_pb,
        n_tests=n_tests,
        effect=eff,
        notes=notes,
    )
