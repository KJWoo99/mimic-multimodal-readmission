"""코호트 정의 및 30일 재입원 라벨 계산.

MIMIC-IV 에 30일 재입원 라벨은 없음. 직접 계산함.
환자별로 입원을 시간순 정렬해 (다음 admittime) − (현재 dischtime) < 30일 이면 양성.
다음 입원은 원본 admissions 전체에서 찾음. 아래 포함, 제외 기준은 index 입원에만 적용함.

포함  만 18세 이상, index 재원 1일 이상
제외  index 가 원내 사망 (사망하면 재입원이 불가능해 음성 라벨이 오염됨)
제외  예정 재입원은 outcome 에서만 제외. index 로는 씀

재입원률은 이 기준에 따라 크게 달라짐. 문헌의 18% 를 목표로 삼지 않고
직접 계산한 값을 쓰되, 다르면 어느 기준이 달랐는지 적음.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["CohortConfig", "CohortResult", "build_cohort", "load_admissions"]

READMISSION_DAYS = 30
MIN_AGE = 18
MIN_LOS_DAYS = 1.0


@dataclass
class CohortConfig:
    readmission_days: int = READMISSION_DAYS
    min_age: int = MIN_AGE
    min_los_days: float = MIN_LOS_DAYS
    exclude_in_hospital_death: bool = True
    # 예정 재입원으로 간주할 admission_type 값.
    # 실제 값 분포를 확인한 뒤 채움. 빈 집합이면 '예정 재입원 미제외' 상태임.
    elective_admission_types: tuple[str, ...] = ()

    # 각 환자의 마지막 입원은 다음 입원 기록이 없어 무조건 음성이 됨.
    # 그러나 "재입원하지 않았다"와 "관찰 기간이 끝나 보이지 않는다"는 다른 사건임.
    # MIMIC-IV 는 환자별로 날짜를 무작위 시프트해 전역 관찰 종료일을 알 수 없으므로
    # 절단 시점을 특정할 수 없음. 그래서 선택지를 열어두고 결정을 명시적으로 남김.
    #
    #   False : 마지막 입원도 음성으로 포함 (표준 관행, 음성이 다소 과대)
    #   True  : 마지막 입원 제외 (편향은 줄지만 표본이 크게 감소하고,
    #           다빈도 입원 환자만 남아 선택 편향이 새로 생김)
    #
    # 둘 다 완벽하지 않음. 기본값은 관행을 따르되 is_last_admission 플래그를 남겨
    # 민감도 분석으로 양쪽을 모두 보고할 수 있게 함.
    exclude_last_admission: bool = False


@dataclass
class CohortResult:
    df: pd.DataFrame
    flow: list[tuple[str, int, int]] = field(default_factory=list)  # (단계, 입원수, 환자수)
    n_censored: int = 0  # 마지막 입원(우측 절단) 건수

    def flow_table(self) -> pd.DataFrame:
        """TRIPOD+AI 가 요구하는 참여자 흐름(flow diagram) 데이터."""
        return pd.DataFrame(self.flow, columns=["step", "n_admissions", "n_subjects"])

    def summary(self) -> dict[str, float | int]:
        d = self.df
        out: dict[str, float | int] = {
            "n_admissions": len(d),
            "n_subjects": int(d.subject_id.nunique()),
            "readmit_rate": float(d.readmit_30d.mean()) if len(d) else float("nan"),
            "n_positive": int(d.readmit_30d.sum()),
            "n_censored_last_admission": int(self.n_censored),
        }
        if len(d):
            # 절단 건을 뺀 재입원률. 실제 값은 이 둘 사이에 있다고 보고함.
            obs = d[d.get("is_last_admission", 0) == 0] if "is_last_admission" in d else d
            out["readmit_rate_excl_censored"] = (
                float(obs.readmit_30d.mean()) if len(obs) else float("nan")
            )
            out["admissions_per_subject"] = round(len(d) / max(1, d.subject_id.nunique()), 2)
        return out


def load_admissions(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """admissions / patients 테이블 로드.

    MIMIC-IV 는 hosp/ 아래에 csv.gz 로 들어있음.
    """
    raw = Path(raw_dir)
    candidates = list(raw.rglob("admissions.csv.gz")) + list(raw.rglob("admissions.csv"))
    if not candidates:
        raise FileNotFoundError(f"admissions 테이블을 찾을 수 없습니다: {raw}")
    adm_path = candidates[0]
    pat_path = next(iter(list(adm_path.parent.glob("patients.csv*"))), None)
    if pat_path is None:
        raise FileNotFoundError(f"patients 테이블을 찾을 수 없습니다: {adm_path.parent}")

    adm = pd.read_csv(adm_path, parse_dates=["admittime", "dischtime", "deathtime"])
    pat = pd.read_csv(pat_path)
    return adm, pat


def build_cohort(
    adm: pd.DataFrame,
    pat: pd.DataFrame,
    cfg: CohortConfig | None = None,
) -> CohortResult:
    """30일 재입원 코호트 구성. 각 단계의 제외 인원을 flow 에 기록함.

    flow 를 남기는 이유: TRIPOD+AI 는 참여자 흐름도를 요구하고,
    "왜 이 코호트를 골랐는가"가 그 자체로 결과의 일부이기 때문임.
    """
    cfg = cfg or CohortConfig()
    flow: list[tuple[str, int, int]] = []

    def record(step: str, d: pd.DataFrame) -> None:
        flow.append((step, len(d), int(d.subject_id.nunique())))

    df = adm.copy()
    record("원본 admissions", df)

    if "anchor_age" in pat.columns:
        df = df.merge(pat[["subject_id", "anchor_age", "gender"]], on="subject_id", how="left")
        df = df[df.anchor_age >= cfg.min_age]
        record(f"성인 {cfg.min_age}세 이상", df)

    df["los_days"] = (df.dischtime - df.admittime).dt.total_seconds() / 86400.0

    # 데이터 품질: dischtime <= admittime 인 레코드가 실제로 존재함
    # (MIMIC-IV v3.1 실측 180건). min_los_days 가 0 이면 그대로 통과하므로
    # 필터와 무관하게 먼저 명시적으로 제거하고 흐름에 남김.
    invalid = df.los_days <= 0
    if invalid.any():
        df = df[~invalid]
        record(f"퇴원≤입원 오류 제거({int(invalid.sum())}건)", df)

    df = df[df.los_days >= cfg.min_los_days]
    record(f"재원 {cfg.min_los_days}일 이상", df)

    # 사망 건을 남기면 "재입원 안 함"으로 잡혀 음성 라벨이 오염됨.
    if cfg.exclude_in_hospital_death:
        dead = df.deathtime.notna()
        if "hospital_expire_flag" in df.columns:
            dead = dead | (df.hospital_expire_flag == 1)
        df = df[~dead]
        record("원내 사망 제외", df)

    df = df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    # 다음 입원은 원본 admissions 전체에서 찾음. 위 필터(나이, 재원 1일, 원내 사망)는
    # index 입원의 조건이지 재입원의 조건이 아님. 필터를 건 뒤의 코호트 안에서 다음 입원을 찾으면
    # 재원 1일 미만인 재입원과 재입원 중 사망한 입원이 재입원으로 세지지 않음(트러블슈팅 24).
    nxt = _next_admission(adm)
    df = df.merge(nxt, on="hadm_id", how="left")

    gap = (df.next_admittime - df.dischtime).dt.total_seconds() / 86400.0
    df["days_to_next_admission"] = gap

    is_readmit = gap.notna() & (gap >= 0) & (gap < cfg.readmission_days)

    # 예정 재입원은 outcome 에서 제외 (CMS HRRP: unplanned only)
    if cfg.elective_admission_types:
        planned = df.next_admission_type.isin(cfg.elective_admission_types)
        df["excluded_planned_readmit"] = is_readmit & planned
        is_readmit = is_readmit & ~planned
    else:
        df["excluded_planned_readmit"] = False

    df["readmit_30d"] = is_readmit.astype(int)

    # 마지막 입원은 관찰 종료로 인해 결과를 볼 수 없는 것이지,
    # 재입원이 없었다고 확인된 것이 아님. 둘을 구분해 기록함.
    df["is_last_admission"] = df.next_admittime.isna().astype(int)
    n_censored = int(df.is_last_admission.sum())

    if cfg.exclude_last_admission:
        df = df[df.is_last_admission == 0].copy()
        record(f"마지막 입원 제외(절단 {n_censored}건)", df)

    # 과거 입원 수도 원본 admissions 전체에서 셈(재원 1일 미만 입원도 입원임).
    df["prior_admissions_12m"] = _prior_admissions(df, months=12, pool=adm)

    record("최종 코호트", df)
    result = CohortResult(df, flow)
    result.n_censored = n_censored
    return result


def _next_admission(adm: pd.DataFrame) -> pd.DataFrame:
    """원본 admissions 전체에서 입원마다 바로 다음 입원(admittime 순)의 시각, 유형.

    반환: [hadm_id, next_admittime, next_admission_type]
    """
    cols = ["subject_id", "hadm_id", "admittime"] + (
        ["admission_type"] if "admission_type" in adm.columns else [])
    a = adm[cols].sort_values(["subject_id", "admittime", "hadm_id"])
    g = a.groupby("subject_id", sort=False)
    out = pd.DataFrame({"hadm_id": a["hadm_id"], "next_admittime": g["admittime"].shift(-1)})
    out["next_admission_type"] = (g["admission_type"].shift(-1)
                                  if "admission_type" in a.columns else pd.NA)
    return out


def _prior_admissions(df: pd.DataFrame, months: int = 12,
                      pool: pd.DataFrame | None = None) -> np.ndarray:
    """각 입원 시점 기준 과거 N개월 내 입원 횟수.

    미래 정보를 쓰면 안 됨. 현재 admittime 이전의 입원만 셈.
    `pool` 을 주면 그 표(원본 admissions 전체)의 입원을 셈. 안 주면 df 안에서 셈.

    주의: 환자별로 admittime 오름차순 정렬을 전제함.
    `times[:i]`(앞쪽 위치)를 후보로 삼고 `delta > 0`(현재보다 과거)으로 거르는
    구조라, 정렬이 깨져도 미래 입원이 새어들지는 않음: 누수는 없음. 대신
    현재보다 앞선 입원이 위치상 뒤에 있으면 후보에서 빠져 과소계산됨.
    에러 없이 값만 틀리는 종류라 호출 시점에 전제를 검사함.
    """
    if len(df) and not df.groupby("subject_id", sort=False).admittime.is_monotonic_increasing.all():
        raise ValueError(
            "_prior_admissions 는 환자별 admittime 오름차순 정렬을 전제한다. "
            "정렬이 깨지면 과거 입원을 빠뜨려 값이 과소계산된다: "
            "df.sort_values(['subject_id', 'admittime']) 후에 호출할 것."
        )

    window_days = months * 30.44
    out = np.zeros(len(df), dtype=int)
    if pool is not None:
        # 환자별 전체 입원 시각을 정렬해 두고, 각 index 시각 t 에 대해 (t - 창, t) 안의 개수를 셈.
        by_subj = {s: np.sort(g.admittime.values) for s, g in pool[["subject_id", "admittime"]]
                   .dropna().groupby("subject_id", sort=False)}
        win = np.timedelta64(int(window_days * 86400), "s")
        subj, times = df.subject_id.to_numpy(), df.admittime.values
        for i in range(len(df)):
            all_t = by_subj.get(subj[i])
            if all_t is None:
                continue
            lo = np.searchsorted(all_t, times[i] - win, side="left")   # delta <= 창
            hi = np.searchsorted(all_t, times[i], side="left")          # delta > 0
            out[i] = int(hi - lo)
        return out
    for _, idx in df.groupby("subject_id", sort=False).indices.items():
        times = df.admittime.values[idx]
        for i, t in enumerate(times):
            delta = (t - times[:i]) / np.timedelta64(1, "D")
            out[idx[i]] = int(((delta > 0) & (delta <= window_days)).sum())
    return out
