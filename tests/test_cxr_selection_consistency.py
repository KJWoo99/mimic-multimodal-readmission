"""다운로드가 고르는 파일과 학습 인덱스가 고르는 파일이 같아야 함.

`download_manifest(frontal_first=True)` 는 `studies` 에 `dicom_view` 컬럼이 있을 때만 정면을
우선 고름. 그 컬럼이 없으면 알파벳순 첫 파일을 받아, 정면을 우선 고르는 학습 인덱스
(`build_cxr_index`)와 서로 다른 파일을 가리키고 그 입원이 학습에서 탈락함(트러블슈팅 7).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cxr_dataset import FRONTAL_VIEWS, build_cxr_index
from modality import download_manifest, linked_studies


def _cohort() -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": [3, 4],
        "hadm_id": [31, 41],
        "admittime": pd.to_datetime(["2150-01-01", "2150-02-01"]),
        "dischtime": pd.to_datetime(["2150-01-10", "2150-02-10"]),
        "readmit_30d": [1, 0],
    })


def _meta() -> pd.DataFrame:
    """study 하나에 측면(알파벳상 먼저)과 정면이 함께 든 상황.

    'aaa' 가 LATERAL, 'zzz' 가 PA: 알파벳순으로 고르면 측면이 뽑힘.
    """
    return pd.DataFrame({
        "subject_id": [3, 3, 4],
        "study_id": [301, 301, 302],
        "dicom_id": ["aaa-lateral", "zzz-frontal", "bbb-frontal"],
        "StudyDate": [21500105, 21500105, 21500205],
        "StudyTime": [120000.0, 120000.0, 120000.0],
        "ViewPosition": ["LATERAL", "PA", "AP"],
    })


def test_linked_studies_exposes_preferred_dicom():
    st = linked_studies(_cohort(), _meta())
    assert "dicom_view" in st.columns, (
        "이 컬럼이 없으면 download_manifest 의 정면 우선 선택이 조용히 꺼진다"
    )
    picked = dict(zip(st.study_id, st.dicom_view, strict=True))
    assert picked[301] == "zzz-frontal", "알파벳순이 아니라 정면을 골라야 한다"


def test_manifest_downloads_the_frontal_view(tmp_path):
    lines = [
        "files/p10/p3/s301/aaa-lateral.jpg",
        "files/p10/p3/s301/zzz-frontal.jpg",
        "files/p10/p4/s302/bbb-frontal.jpg",
    ]
    fn = tmp_path / "IMAGE_FILENAMES"
    fn.write_text("\n".join(lines) + "\n", encoding="utf-8")

    st = linked_studies(_cohort(), _meta())
    paths, manifest = download_manifest(st, fn)
    assert manifest["studies_found"] == 2
    assert "zzz-frontal.jpg" in "\n".join(paths)
    assert "aaa-lateral.jpg" not in "\n".join(paths)


def test_manifest_warns_when_preference_column_missing(tmp_path):
    """컬럼이 없으면 그냥 넘어가지 말고 경고해야 함."""
    fn = tmp_path / "IMAGE_FILENAMES"
    fn.write_text("files/p10/p3/s301/aaa-lateral.jpg\n", encoding="utf-8")
    st = linked_studies(_cohort(), _meta()).drop(columns="dicom_view")
    with pytest.warns(RuntimeWarning, match="dicom_view"):
        download_manifest(st, fn)


def _touch(root: Path, subject: int, study: int, dicom: str) -> None:
    """MIMIC-CXR 의 files/pXX/pSUBJECT/sSTUDY/DICOM.jpg 배치를 만듦.

    식별자는 전부 합성값임: 실제 MIMIC subject_id 는 여덟 자리라 한 자리 수는
    존재할 수 없음. 저장소의 다른 테스트와 같은 관례를 따름.
    """
    d = root / "files" / f"p{str(subject)[:2]}" / f"p{subject}" / f"s{study}"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{dicom}.jpg").write_bytes(b"x")


def test_index_uses_an_available_file_instead_of_dropping(tmp_path):
    """선호하는 정면이 디스크에 없어도 입원을 버리지 않음.

    대표를 먼저 고르고 나서 존재를 확인하면, 고른 파일이 없을 때 그 입원이 통째로 사라짐.
    있는 파일 중에서 고름.
    """
    _touch(tmp_path, 3, 301, "aaa-lateral")   # 측면만 받아둔 상태
    _touch(tmp_path, 4, 302, "bbb-frontal")

    res = build_cxr_index(_cohort(), _meta(), tmp_path)
    assert len(res.df) == 2, f"입원이 탈락했다: {res.dropped}"
    assert set(res.df.hadm_id) == {31, 41}
    assert res.df.path.map(lambda p: p.is_file()).all()


def test_index_reports_lateral_fallback(tmp_path):
    """측면으로 대체된 건수를 빠뜨리지 않고 보고함."""
    _touch(tmp_path, 3, 301, "aaa-lateral")
    _touch(tmp_path, 4, 302, "bbb-frontal")
    res = build_cxr_index(_cohort(), _meta(), tmp_path)
    assert res.flags["정면 없어 측면 사용"] == 1


def test_index_prefers_frontal_when_both_present(tmp_path):
    _touch(tmp_path, 3, 301, "aaa-lateral")
    _touch(tmp_path, 3, 301, "zzz-frontal")
    _touch(tmp_path, 4, 302, "bbb-frontal")
    res = build_cxr_index(_cohort(), _meta(), tmp_path)
    row = res.df[res.df.hadm_id == 31].iloc[0]
    assert row.dicom_id == "zzz-frontal"
    assert row.ViewPosition in FRONTAL_VIEWS
    assert res.flags["정면 없어 측면 사용"] == 0


def test_phase1_5_reads_dicom_id_from_metadata():
    """`run_phase1_5.py` 가 메타데이터에서 `dicom_id` 를 읽어야 함.

    이걸 빠뜨리면 `linked_studies` 가 `dicom_view` 를 만들지 못하고,
    `download_manifest` 의 정면 우선 선택이 경고 없이 꺼짐.

    함수만 따로 검증해서는 이런 결함을 잡지 못함. 호출부가 `usecols` 에서 `dicom_id` 를 빼면
    경고도 `warnings.filterwarnings("ignore")` 에 먹혀 보이지 않으므로, 호출부가 그 함수를
    쓸 수 있는 형태로 부르는지까지 봄.
    """
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_phase1_5.py").read_text(
        encoding="utf-8"
    )
    i = src.index("usecols=")
    block = src[i : i + 200]
    assert '"dicom_id"' in block, (
        "run_phase1_5.py 의 usecols 에 dicom_id 가 없다: "
        "정면 우선 선택이 조용히 비활성화된다"
    )


def test_manifest_records_whether_frontal_first_applied(tmp_path):
    """정면 우선이 실제로 적용됐는지를 산출물에 남김.

    경고만으로는 부족하므로 사후에 확인할 수 있는 값으로도 남김.
    """
    fn = tmp_path / "IMAGE_FILENAMES"
    fn.write_text(
        "files/p3/p3/s301/aaa-lateral.jpg\nfiles/p3/p3/s301/zzz-frontal.jpg\n",
        encoding="utf-8",
    )
    st = linked_studies(_cohort(), _meta())
    _, applied = download_manifest(st, fn)
    assert applied["frontal_first_applied"] is True

    with pytest.warns(RuntimeWarning):
        _, skipped = download_manifest(st.drop(columns="dicom_view"), fn)
    assert skipped["frontal_first_applied"] is False
