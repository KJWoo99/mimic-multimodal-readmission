"""비식별화 검증 회귀 테스트.

새겨진 텍스트 탐지가 실제로 렌더링한 글자를 잡는지, 그리고 흉부 X선
모서리에 흔한 흰 테두리, 통짜 사각형을 오탐하지 않는지를 고정함.

양성 대조군이 없는 탐지기는 "아무것도 못 찾음"과 "고장남"을 구별할 수 없음.
그래서 여기서는 PIL 로 실제 글자를 그려 넣어 검증함. `img[10:30, 10:120] = 255` 같은
통짜 흰 블록은 글자가 아니라 테두리 모양이라 탐지기를 제대로 검증하지 못함.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deid import (
    DeidReport,
    check_jpg_exif,
    check_metadata_phi,
    detect_burned_in_text,
    detect_redaction_boxes,
    verify_date_shift_consistency,
)


def test_metadata_phi_clean():
    df = pd.DataFrame(columns=["subject_id", "study_id", "ViewPosition", "Rows", "Columns"])
    ok, hits = check_metadata_phi(df)
    assert ok and hits == []


@pytest.mark.parametrize("col", [
    "PatientName", "PatientID", "AccessionNumber", "InstitutionName",
    "ReferringPhysician", "PatientBirthDate", "MRN", "SSN",
    "PatientAddress", "ZipCode",
])
def test_metadata_phi_detected(col):
    df = pd.DataFrame(columns=["subject_id", col])
    ok, hits = check_metadata_phi(df)
    assert not ok
    assert col in hits


@pytest.mark.parametrize("col", [
    "hospital_expire_flag",   # 원내 사망 플래그: 기관명이 아님
    "admission_location",     # 입원 경로 범주값: 주소가 아님
    "discharge_location",
])
def test_known_safe_columns_not_flagged(col):
    """위양성 회귀 테스트.

    `hospital` 같은 넓은 패턴은 hospital_expire_flag 를 PHI 로 오탐함.
    위양성이 섞이면 실제 PHI 경고와 구분하기 어려워짐.
    """
    df = pd.DataFrame(columns=["subject_id", col])
    ok, hits = check_metadata_phi(df)
    assert ok, f"{col} 오탐: {hits}"


def test_real_mimic_columns_are_clean():
    """실제 MIMIC-IV admissions 컬럼 구성에서 오탐이 없어야 함."""
    cols = [
        "subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
        "admission_type", "admit_provider_id", "admission_location",
        "discharge_location", "insurance", "language", "marital_status",
        "race", "edregtime", "edouttime", "hospital_expire_flag",
    ]
    ok, hits = check_metadata_phi(pd.DataFrame(columns=cols))
    assert ok, f"오탐: {hits}"


def _clean_xray(h=512, w=512, seed=0):
    """포화되지 않은 흉부 X선 모사 (조직 밝기 범위)."""
    rng = np.random.default_rng(seed)
    img = rng.normal(110, 30, (h, w)).clip(0, 235)
    return img.astype(np.uint8)


def _with_text(text: str, size: int = 40, xy=(14, 14), shape=(1024, 1024)) -> np.ndarray:
    """모서리에 실제 글자를 렌더링한 영상. 탐지기의 양성 대조군."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(_clean_xray(*shape))
    ImageDraw.Draw(im).text(xy, text, fill=255, font=ImageFont.load_default(size))
    return np.asarray(im)


def test_clean_image_not_flagged():
    ok, info = detect_burned_in_text(_clean_xray(1024, 1024))
    assert not ok, f"깨끗한 영상을 오탐: {info}"


@pytest.mark.parametrize("text", ["MRN 12345678", "JOHN DOE", "DOE^JANE 1974-03-11", "PORTABLE"])
def test_rendered_text_is_detected(text):
    """양성 대조군: 실제로 그린 글자는 잡아야 함."""
    ok, info = detect_burned_in_text(_with_text(text))
    assert ok, f"{text!r} 미탐지: {info}"
    assert "top_left" in info["text_corners"]


def test_text_detected_in_each_corner():
    """네 모서리 모두에서 잡히는지.

    모서리 영역은 변의 18% 다(1024px 영상이면 184px). 글자를 그 안에 들어가게
    넣어야 함: 밖에 그려두고 "미탐지"라고 하면 테스트가 틀린 것임.
    문자열/글자크기도 아무거나 쓰면 안 됨. 기본 폰트에서 짧고 큰 글자는
    획 잇기(binary_closing)로 서로 붙어 한 덩어리가 되어 버림.
    """
    from PIL import ImageFont

    size = 28
    text = "ABC123"
    inner = int(1024 * 0.18)
    tw = ImageFont.load_default(size).getbbox(text)[2]
    assert tw < inner, f"글자 폭 {tw} 가 모서리 폭 {inner} 를 넘으면 테스트가 성립하지 않는다"
    near = 1024 - inner + 6
    pos = {
        "top_left": (8, 8),
        "top_right": (near, 8),
        "bottom_left": (8, near),
        "bottom_right": (near, near),
    }
    for corner, xy in pos.items():
        ok, info = detect_burned_in_text(_with_text(text, size=size, xy=xy))
        assert ok, f"{corner} 미탐지: {info['corners'][corner]}"
        assert corner in info["text_corners"], (corner, info["text_corners"])


def test_solid_white_block_is_not_text():
    """통짜 흰 사각형은 콜리메이션 경계, 여백이지 글자가 아님.

    이걸 텍스트로 세면 흉부 X선의 93% 가 의심으로 걸림(실측). 채움률이
    1.0 에 가까운 성분을 제외하는 것이 테두리와 글자를 가르는 핵심임.
    """
    img = _clean_xray(1024, 1024)
    img[10:60, 10:300] = 255
    ok, _ = detect_burned_in_text(img)
    assert not ok


def test_single_marker_is_not_flagged():
    """자세 마커 'R' 한 글자는 PHI 가 아니므로 걸리면 안 됨."""
    ok, _ = detect_burned_in_text(_with_text("R", size=60))
    assert not ok


def test_center_bright_region_not_flagged():
    """중앙의 밝은 해부학적 구조는 주석이 아님: 모서리만 봄."""
    img = _clean_xray(1024, 1024)
    img[400:600, 400:600] = 255
    ok, _ = detect_burned_in_text(img)
    assert not ok


def test_detect_handles_rgb():
    img = np.stack([_clean_xray(1024, 1024)] * 3, axis=-1)
    ok, _ = detect_burned_in_text(img)
    assert not ok


def test_redaction_box_detected():
    """내부의 순수 검정 사각형 = 배포자가 픽셀을 가린 흔적."""
    img = _clean_xray(1024, 1024)
    img[300:400, 200:800] = 0
    ok, info = detect_redaction_boxes(img)
    assert ok
    assert info["n_inner_boxes"] == 1


def test_edge_black_margin_is_not_redaction():
    """가장자리에 닿는 검정 여백은 촬영 여백이지 마스킹이 아님."""
    img = _clean_xray(1024, 1024)
    img[:80, :] = 0
    ok, info = detect_redaction_boxes(img)
    assert not ok
    assert info["n_boxes"] >= 1 and info["n_inner_boxes"] == 0


def test_no_black_region_no_redaction():
    ok, info = detect_redaction_boxes(_clean_xray(1024, 1024))
    assert not ok
    assert info["n_inner_boxes"] == 0



def test_exif_clean_on_generated_jpg(tmp_path):
    from PIL import Image

    p = tmp_path / "x.jpg"
    Image.fromarray(_clean_xray(64, 64)).save(p)
    ok, info = check_jpg_exif(p)
    assert ok
    assert info["n_exif_tags"] == 0


def test_exif_missing_file_returns_error(tmp_path):
    ok, info = check_jpg_exif(tmp_path / "nope.jpg")
    assert not ok
    assert "error" in info


def test_date_shift_consistent_when_aligned():
    """같은 오프셋이면 CXR 이 입원 기간 안에 들어감."""
    cxr = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "StudyDate": [21800103, 21800205, 21800307],
        "StudyTime": [120000.0, 120000.0, 120000.0],
    })
    adm = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "admittime": pd.to_datetime(["2180-01-01", "2180-02-01", "2180-03-01"]),
        "dischtime": pd.to_datetime(["2180-01-10", "2180-02-10", "2180-03-10"]),
    })
    ok, stats = verify_date_shift_consistency(cxr, adm, sample_subjects=10)
    assert ok
    assert stats["overlap_rate"] == 1.0


def test_date_shift_inconsistent_detected():
    """오프셋이 어긋나면 겹침이 사라짐."""
    cxr = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "StudyDate": [21900103, 21900205, 21900307],   # 10년 어긋남
        "StudyTime": [120000.0] * 3,
    })
    adm = pd.DataFrame({
        "subject_id": [1, 2, 3],
        "admittime": pd.to_datetime(["2180-01-01", "2180-02-01", "2180-03-01"]),
        "dischtime": pd.to_datetime(["2180-01-10", "2180-02-10", "2180-03-10"]),
    })
    ok, stats = verify_date_shift_consistency(cxr, adm, sample_subjects=10)
    assert not ok
    assert stats["overlap_rate"] == 0.0


def test_report_tracks_pass_fail_and_na():
    r = DeidReport()
    r.add("A", True)
    r.add("B", False, "이유")
    r.not_applicable.append("DICOM PHI 태그: JPG 배포본이라 대상 없음")
    assert not r.all_passed
    assert len(r.to_frame()) == 2
    s = r.summary()
    assert "1/2 통과" in s
    assert "검증 불가" in s


def test_observations_do_not_affect_pass_fail():
    """관찰은 통과 여부를 바꾸지 않음.

    새겨진 텍스트의 존재처럼 "사실이지만 합격/불합격으로 가를 수 없는 것"을
    FAIL 로 처리하면, 정상 데이터가 전량 실패로 나와 검사기가 무시됨.
    """
    r = DeidReport()
    r.add("EXIF 없음", True)
    r.observe("새겨진 텍스트 259/500장: 자세 마커, 촬영기법 주석")
    assert r.all_passed
    s = r.summary()
    assert "1/1 통과" in s
    assert "관찰" in s and "259/500" in s
