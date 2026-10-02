"""Phase 2 영상 데이터셋 회귀 테스트."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cxr_dataset import build_cxr_index, view_baseline


def _cohort() -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": [1, 1, 2, 3],
        "hadm_id": [10, 11, 20, 30],
        "admittime": pd.to_datetime(["2110-01-01", "2110-03-01",
                                     "2110-01-05", "2110-02-01"]),
        "dischtime": pd.to_datetime(["2110-01-10", "2110-03-10",
                                     "2110-01-08", "2110-02-05"]),
        "readmit_30d": [1, 0, 1, 0],
    })


def _meta() -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": [1, 1, 1, 2, 3],
        "study_id": [100, 101, 102, 200, 300],
        "dicom_id": ["a", "b", "c", "d", "e"],
        # 100: hadm 10 재원 중 / 101: hadm 10 재원 중(퇴원에 더 가까움)
        # 102: 어느 입원에도 안 걸림 / 200: hadm 20 재원 중 / 300: 퇴원 이후
        "StudyDate": [21100102, 21100109, 21100201, 21100106, 21100210],
        "StudyTime": [120000.0, 90000.0, 120000.0, 80000.0, 100000.0],
        "ViewPosition": ["PA", "AP", "PA", "AP", "PA"],
    })


def _touch(root: Path, subject: int, study: int, dicom: str) -> None:
    p = (root / "files" / f"p{str(subject)[:2]}" / f"p{subject}"
         / f"s{study}" / f"{dicom}.jpg")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_only_within_stay_studies_are_linked(tmp_path: Path) -> None:
    for s, st, d in [(1, 100, "a"), (1, 101, "b"), (1, 102, "c"),
                     (2, 200, "d"), (3, 300, "e")]:
        _touch(tmp_path, s, st, d)
    res = build_cxr_index(_cohort(), _meta(), tmp_path)
    # 퇴원 이후 촬영(300)과 어느 입원에도 안 걸리는 것(102)은 빠져야 함
    assert set(res.df.hadm_id) == {10, 20}
    assert 30 not in set(res.df.hadm_id), "퇴원 이후 영상이 새어들어왔다"


def test_one_study_per_admission_picks_closest_to_discharge(tmp_path: Path) -> None:
    for s, st, d in [(1, 100, "a"), (1, 101, "b"), (2, 200, "d")]:
        _touch(tmp_path, s, st, d)
    res = build_cxr_index(_cohort(), _meta(), tmp_path)
    row = res.df[res.df.hadm_id == 10].iloc[0]
    # 101 이 퇴원(01-10)에 더 가까움
    assert row.study_id == 101
    assert res.df.hadm_id.is_unique, "입원당 1건이어야 한다"


def test_frontal_view_is_preferred_within_same_study(tmp_path: Path) -> None:
    """같은 study 에 정면과 측면이 있으면 정면을 골라야 함."""
    coh = pd.DataFrame({
        "subject_id": [7], "hadm_id": [70],
        "admittime": pd.to_datetime(["2110-01-01"]),
        "dischtime": pd.to_datetime(["2110-01-10"]),
        "readmit_30d": [1],
    })
    # 같은 study, 같은 시각. 사전순으로는 lateral 이 먼저임.
    meta = pd.DataFrame({
        "subject_id": [7, 7],
        "study_id": [700, 700],
        "dicom_id": ["aaa_lateral", "zzz_frontal"],
        "StudyDate": [21100105, 21100105],
        "StudyTime": [120000.0, 120000.0],
        "ViewPosition": ["LATERAL", "PA"],
    })
    for d in ("aaa_lateral", "zzz_frontal"):
        _touch(tmp_path, 7, 700, d)
    res = build_cxr_index(coh, meta, tmp_path)
    assert res.df.iloc[0].ViewPosition == "PA", "측면이 뽑혔다"


def test_linked_studies_matches_index_count() -> None:
    """다운로드 목록과 학습 인덱스가 같은 규칙이어야 함."""
    from modality import linked_studies

    one = linked_studies(_cohort(), _meta(), one_per_admission=True)
    allv = linked_studies(_cohort(), _meta(), one_per_admission=False)
    assert one.hadm_id.is_unique, "입원당 1건이어야 한다"
    assert len(one) <= len(allv)
    # hadm 10 은 study 100/101 두 개 중 퇴원에 가까운 101
    assert one[one.hadm_id == 10].iloc[0].study_id == 101


def test_missing_files_are_dropped_and_counted(tmp_path: Path) -> None:
    """파일이 하나도 없는 입원만 빠짐.

    개별 영상 단위로 없는 것을 세고, 남은 영상 중에서 대표를 고름. "대표로 고른 파일이 없다"를
    세면 그 입원은 다른 영상이 있어도 통째로 탈락함.
    """
    _touch(tmp_path, 1, 101, "b")          # hadm 10 만 파일 존재
    res = build_cxr_index(_cohort(), _meta(), tmp_path)
    assert set(res.df.hadm_id) == {10}
    assert res.dropped["파일 없음 (개별 영상)"] >= 1
    assert "정면 없어 측면 사용" in res.flags


def test_path_layout_matches_mimic_cxr(tmp_path: Path) -> None:
    _touch(tmp_path, 1, 101, "b")
    res = build_cxr_index(_cohort(), _meta(), tmp_path)
    p = Path(res.df.iloc[0].path)
    assert p.parts[-4:] == ("p1", "p1", "s101", "b.jpg") or p.name == "b.jpg"
    assert p.is_file()


def test_view_baseline_returns_empty_when_single_class() -> None:
    df = pd.DataFrame({"readmit_30d": [0, 0, 0, 0],
                       "ViewPosition": ["AP", "PA", "AP", "PA"]})
    assert view_baseline(df, np.array([1, 2, 3, 4])) == {}


def test_view_baseline_detects_perfect_view_signal() -> None:
    n = 200
    view = np.array(["AP"] * (n // 2) + ["PA"] * (n // 2))
    df = pd.DataFrame({"readmit_30d": (view == "AP").astype(int),
                       "ViewPosition": view})
    out = view_baseline(df, np.arange(n))
    assert out["auroc"] > 0.95, "자세가 라벨을 결정하는데 잡아내지 못했다"


@pytest.mark.parametrize("train", [True, False])
def test_transforms_output_shape(train: bool) -> None:
    from PIL import Image

    from cxr_dataset import build_transforms

    tf = build_transforms(train, size=224)
    x = tf(Image.new("RGB", (400, 300)))
    assert tuple(x.shape) == (3, 224, 224)
