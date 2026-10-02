"""입원 분포 요약: 필터 전 전체 입원과 주 분석 코호트의 환자당 입원 수, 원내 사망, 짧은 재원.

    python scripts/admission_profile.py

TRIPOD 20절과 MODEL_CARD 한계 3 이 인용하는 값(환자당 입원 평균 2.44, 최대 238 등)의 출처임.
건수와 비율만 `outputs/admission_profile.json` 에 남김(환자별 값은 남기지 않음).
주 분석 코호트는 run_phase1.py 와 같은 설정으로 다시 만들고, 입원 수가 `phase1_metrics.json` 과 다르면 멈춤.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "admission_profile.json"
P1 = ROOT / "outputs" / "phase1_metrics.json"


def profile(df) -> dict[str, float | int]:
    per = df.groupby("subject_id").size().sort_values(ascending=False)
    top = math.ceil(len(per) * 0.01)
    los_days = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400
    return {
        "n_admissions": int(len(df)),
        "n_subjects": int(len(per)),
        "admissions_per_subject_mean": round(len(df) / len(per), 4),
        "admissions_per_subject_median": float(per.median()),
        "admissions_per_subject_max": int(per.max()),
        "single_admission_subject_pct": round(float((per == 1).mean()) * 100, 1),
        "top1pct_subjects_admission_share_pct": round(float(per.iloc[:top].sum()) / len(df) * 100, 1),
        "in_hospital_death_pct": round(float(df["hospital_expire_flag"].mean()) * 100, 2),
        "los_under_1day_pct": round(float((los_days < 1).mean()) * 100, 1),
    }


def main() -> int:
    from datapaths import RAW

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=str(RAW))
    args = ap.parse_args()

    from cohort import CohortConfig, build_cohort, load_admissions
    from features import ELECTIVE_ADMISSION_TYPES

    adm, pat = load_admissions(Path(args.raw_dir))
    coh = build_cohort(adm, pat, CohortConfig(elective_admission_types=ELECTIVE_ADMISSION_TYPES))
    want = json.loads(P1.read_text(encoding="utf-8"))["cohort"]["n_admissions"]
    if len(coh.df) != want:
        raise SystemExit(f"코호트 입원 수 {len(coh.df)} 가 phase1_metrics.json 의 {want} 와 다르다")

    out = {"all_admissions": profile(adm), "main_cohort": profile(coh.df)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for k, v in out.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
