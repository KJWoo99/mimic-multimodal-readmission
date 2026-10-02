"""비식별화 검증.

MIMIC-CXR-JPG 는 이미 비식별화돼 배포되지만 그것을 확인하는 절차를 둠.

DICOM PS3.15 는 DICOM 태그를 전제함. JPG 배포본에는 PHI 태그, UID, private tag 이
아예 없어 검증 대상이 존재하지 않음. 대신 JPG 에서 실제로 확인 가능한 것만 검사함.
픽셀에 새겨진 이름, ID(burned-in annotation), EXIF, 날짜 시프트 일관성,
함께 제공되는 CSV 의 식별자.

무엇을 검증할 수 없는지 명시하는 것도 결과의 일부임.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "KNOWN_SAFE_COLUMNS",
    "PHI_COLUMN_PATTERNS",
    "DeidReport",
    "check_jpg_exif",
    "check_metadata_phi",
    "detect_burned_in_text",
    "detect_redaction_boxes",
    "verify_date_shift_consistency",
]

# 배포 메타데이터에 있으면 안 되는 컬럼명 패턴 (PS3.15 PHI 항목 기준).
#
# 패턴을 넓게 잡으면 위양성이 남(`hospital` 패턴이 MIMIC-IV 의 `hospital_expire_flag` 를 PHI 로 오탐).
#    그래서 DICOM PS3.15 가 실제로 지정하는 항목명에 가깝게 좁힘.
PHI_COLUMN_PATTERNS = (
    "patientname", "patientaddress", "patienttelephone",
    "patientid", "mrn", "medicalrecordnumber",
    "birthdate", "dateofbirth", "dob", "ssn", "socialsecurity",
    "accessionnumber", "institutionname", "institutionaddress", "hospitalname",
    "physicianname", "referringphysician", "operatorname", "performingphysician",
    "streetaddress", "phonenumber", "postalcode", "zipcode",
)

# 위양성이 확인된 안전 컬럼: 패턴에 걸려도 PHI 가 아님.
# 새 위양성이 나오면 여기에 추가하고 왜 안전한지 사유를 적음.
KNOWN_SAFE_COLUMNS = {
    "hospital_expire_flag": "원내 사망 여부 플래그 (0/1). 기관명이 아니다",
    "admission_location": "입원 경로 범주값 (EMERGENCY ROOM 등). 주소가 아니다",
    "discharge_location": "퇴원 경로 범주값 (HOME 등). 주소가 아니다",
}


@dataclass
class DeidReport:
    """검증 결과. 통과 여부와 함께 검증하지 못한 것도 남김."""

    checks: list[dict] = field(default_factory=list)
    not_applicable: list[str] = field(default_factory=list)
    # 통과/실패로 가를 수 없지만 반드시 남겨야 하는 관찰.
    # 예) 새겨진 텍스트의 존재: 있다는 사실만으로는 비식별화 실패가 아님.
    observations: list[str] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append({"check": name, "passed": passed, "detail": detail})

    def observe(self, text: str) -> None:
        self.observations.append(text)

    @property
    def all_passed(self) -> bool:
        return all(c["passed"] for c in self.checks)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks)

    def summary(self) -> str:
        n = len(self.checks)
        ok = sum(c["passed"] for c in self.checks)
        lines = [f"검증 {ok}/{n} 통과"]
        for c in self.checks:
            mark = "PASS" if c["passed"] else "FAIL"
            lines.append(f"  [{mark}] {c['check']}" + (f": {c['detail']}" if c["detail"] else ""))
        if self.observations:
            lines.append("  관찰 (통과/실패로 가를 수 없는 사실):")
            lines += [f"    - {x}" for x in self.observations]
        if self.not_applicable:
            lines.append("  검증 불가 (JPG 배포본이라 대상이 존재하지 않음):")
            lines += [f"    - {x}" for x in self.not_applicable]
        return "\n".join(lines)


def check_metadata_phi(df: pd.DataFrame, name: str = "metadata") -> tuple[bool, list[str]]:
    """배포 메타데이터 컬럼에 PHI 로 보이는 것이 있는지 검사.

    한계: 반드시 함께 보고할 것
      - 컬럼명 기반이라 값 수준의 식별자(자유텍스트에 섞인 이름 등)는 잡지 못함.
      - 알려진 안전 컬럼은 `KNOWN_SAFE_COLUMNS` 로 제외함. 위양성이 쌓이면
        검사기 자체를 무시하게 되므로, 예외는 사유와 함께 명시적으로 관리함.
    """
    hits = []
    for c in df.columns:
        if c in KNOWN_SAFE_COLUMNS:
            continue
        norm = c.lower().replace(" ", "").replace("_", "")
        if any(p in norm for p in PHI_COLUMN_PATTERNS):
            hits.append(c)
    return (len(hits) == 0), hits


def verify_date_shift_consistency(
    cxr_meta: pd.DataFrame,
    admissions: pd.DataFrame,
    sample_subjects: int = 2000,
    seed: int = 42,
) -> tuple[bool, dict]:
    """CXR 과 EHR 의 날짜 시프트가 환자별로 일관되는지 확인.

    두 데이터셋이 같은 오프셋을 쓴다면 같은 환자의 CXR 촬영 시각이
    그 환자의 입원 기간 안에 들어가는 경우가 상당수 존재해야 함.
    반대로 오프셋이 다르면 거의 겹치지 않음.

    이것은 "비식별화가 간격을 보존했는가"를 간접 확인하는 절차이기도 함.
    """
    from modality import parse_study_datetime

    cxr = cxr_meta.copy()
    if "study_datetime" not in cxr.columns:
        cxr["study_datetime"] = parse_study_datetime(cxr["StudyDate"], cxr["StudyTime"])

    common = np.intersect1d(cxr.subject_id.unique(), admissions.subject_id.unique())
    if len(common) == 0:
        return False, {"reason": "공통 환자 없음"}

    rng = np.random.default_rng(seed)
    pick = rng.choice(common, size=min(sample_subjects, len(common)), replace=False)

    c = cxr[cxr.subject_id.isin(pick)][["subject_id", "study_datetime"]]
    a = admissions[admissions.subject_id.isin(pick)][["subject_id", "admittime", "dischtime"]]
    m = c.merge(a, on="subject_id", how="inner")
    inside = (
        (m.study_datetime >= m.admittime) & (m.study_datetime <= m.dischtime)
    )

    subj_with_overlap = m.loc[inside, "subject_id"].nunique()
    rate = subj_with_overlap / len(pick)
    stats = {
        "sampled_subjects": len(pick),
        "subjects_with_in_stay_cxr": int(subj_with_overlap),
        "overlap_rate": float(rate),
        "median_year_cxr": int(pd.to_datetime(c.study_datetime).dt.year.median()),
        "median_year_adm": int(pd.to_datetime(a.admittime).dt.year.median()),
    }
    # 오프셋이 어긋나면 겹침이 사실상 0 이 됨. 보수적으로 1% 를 임계로 둠.
    return bool(rate > 0.01), stats


# 글자 크기 기대치를 고정하기 위해 짧은 변을 이 크기로 맞춰서 봄.
# MIMIC-CXR-JPG 는 3050x2539 같은 원본 해상도라, 절대 픽셀 기준을 쓰면
# 1~3 픽셀짜리 JPEG 잡티가 전부 글자 후보가 되어 버림.
_NORM_SHORT_SIDE = 1024


def _normalise(img: np.ndarray) -> np.ndarray:
    from PIL import Image

    h, w = img.shape[:2]
    scale = _NORM_SHORT_SIDE / min(h, w)
    if scale >= 1.0:
        return img
    resized = Image.fromarray(img.astype(np.uint8)).resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR
    )
    return np.asarray(resized)


def _glyph_like(patch: np.ndarray, saturation_threshold: int) -> list[dict]:
    """포화 영역에서 '읽을 수 있는 글자' 크기의 연결성분만 골라냄.

    임계값 이진화만 하면 글자 획이 안티에일리어싱 때문에 잘게 부서짐.
    binary_closing 으로 획을 이어붙인 뒤 크기, 모양으로 거름.
    """
    from scipy import ndimage

    h, _w = patch.shape[:2]
    binary = patch >= saturation_threshold
    if not binary.any():
        return []
    binary = ndimage.binary_closing(binary, structure=np.ones((3, 3)), iterations=2)
    lab, _ = ndimage.label(binary)
    out = []
    for sy, sx in ndimage.find_objects(lab):
        bh, bw = sy.stop - sy.start, sx.stop - sx.start
        # 정규화 영상 기준 모서리 높이의 5~35%: 사람이 읽을 수 있는 글자 크기
        if not (0.05 * h <= bh <= 0.35 * h):
            continue
        if not (0.2 <= bw / bh <= 3.0):
            continue
        area = int(binary[sy, sx].sum())
        if area / (bh * bw) > 0.92:      # 통짜 사각형은 테두리, 마스킹이지 글자가 아님
            continue
        out.append({"h": bh, "w": bw, "cy": (sy.start + sy.stop) / 2.0,
                    "cx": (sx.start + sx.stop) / 2.0})
    return out


def detect_burned_in_text(
    image: np.ndarray,
    corner_frac: float = 0.18,
    saturation_threshold: int = 250,
    min_glyphs: int = 2,
) -> tuple[bool, dict]:
    """모서리에 새겨진 텍스트가 있는지 봄. PHI 판정이 아님.

    무엇을 세는지 분명히 할 것: 이 함수는 "글자처럼 보이는 것이 있다"까지만
    말함. 그것이 PHI 인지는 읽어봐야 알 수 있고, OCR 없이는 알 수 없음.

    실측으로 확인한 사실 (MIMIC-CXR-JPG)
    ------------------------------------
    모서리를 직접 눈으로 확인해 보니 새겨진 텍스트가 실제로 있음. 다만 내용은
    'R'/'L' 자세 마커(원 안 글자), 'PORTABLE', 'semi upright' 같은 촬영기법 주석,
    방향 화살표였음. 어느 것도 PHI 가 아님. 그리고 PHI 로 보이는 자리는
    배포자가 이미 검은 사각형으로 가려 놓음(`detect_redaction_boxes` 참고).

    따라서 "텍스트가 있다 = 비식별화 실패" 로 판정하면 안 됨. 정상적으로
    비식별화된 데이터가 전량 FAIL 로 나오고, 그러면 아무도 이 검사를 안 봄.
    호출부는 이 결과를 통과/실패가 아니라 관찰로 기록해야 함.

    포화 픽셀 비율을 쓰지 않는 이유
    -----------------------------
    흉부 X선 모서리는 콜리메이션 경계, 흰 여백 때문에 원래 포화돼 있어(중앙값 3.7%) 비율로는
    "흰 테두리"와 "글자"를 가를 수 없음. 그래서 글자의 기하 구조로 봄:
    읽을 수 있는 크기의, 통짜가 아닌, 같은 줄에 정렬된 성분이 몇 개인가.

    한계: 반드시 함께 보고할 것
      - OCR 이 아니므로 내용을 읽지 못함. PHI 여부는 판정할 수 없음.
      - 어두운 배경에 어두운 글자로 새겨진 주석은 포화되지 않아 놓침.
      - 사람이 확인할 후보를 좁히는 용도임.
    """
    img = np.asarray(image)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = _normalise(img)
    h, w = img.shape[:2]
    ch, cw = max(1, int(h * corner_frac)), max(1, int(w * corner_frac))

    corners = {
        "top_left": img[:ch, :cw],
        "top_right": img[:ch, -cw:],
        "bottom_left": img[-ch:, :cw],
        "bottom_right": img[-ch:, -cw:],
    }
    details: dict[str, dict] = {}
    found: list[str] = []
    for name, patch in corners.items():
        glyphs = _glyph_like(patch, saturation_threshold)
        best = 0
        for anchor in glyphs:
            band = max(2.0, anchor["h"] * 0.6)
            best = max(best, sum(1 for g in glyphs if abs(g["cy"] - anchor["cy"]) <= band))
        details[name] = {
            "saturated_ratio": float((patch >= saturation_threshold).mean()),
            "n_glyph_like": len(glyphs),
            "max_aligned_in_row": best,
        }
        if best >= min_glyphs:
            found.append(name)

    return (len(found) > 0), {
        "corners": details,
        "text_corners": sorted(found),
        "min_glyphs": min_glyphs,
        "note": "텍스트 존재 여부만 말한다. 내용이 PHI 인지는 OCR 없이 알 수 없다.",
    }


def detect_redaction_boxes(
    image: np.ndarray,
    max_value: int = 2,
    min_side: int = 12,
    min_area_frac: float = 1e-4,
    min_fill: float = 0.98,
) -> tuple[bool, dict]:
    """배포자가 픽셀을 가린 흔적(순수 검정 직사각형)을 찾음.

    이건 결함이 아니라 비식별화가 실제로 적용됐다는 증거임. 검증 절차가
    "문제를 못 찾았다"만 말하면 검사기가 동작하긴 한 건지 알 수 없음.
    적용의 흔적을 함께 보고해야 결과가 해석 가능해짐.

    가장자리에 닿는 검정 영역은 촬영 여백, 콜리메이션이므로 마스킹과 구분함.
    실측(200장): 순수 검정 직사각형 13.5%, 그중 가장자리에 안 닿는 것 7.5%.
    크기는 대략 90~120px 높이 x 300~800px 너비로, 지워진 텍스트 한 줄 모양임.
    """
    from scipy import ndimage

    img = np.asarray(image)
    if img.ndim == 3:
        img = img.mean(axis=2)
    h, w = img.shape[:2]
    dark = img <= max_value
    if not dark.any():
        return False, {"n_boxes": 0, "boxes": []}

    lab, _ = ndimage.label(dark)
    boxes = []
    for sy, sx in ndimage.find_objects(lab):
        bh, bw = sy.stop - sy.start, sx.stop - sx.start
        if bh < min_side or bw < min_side:
            continue
        area = int(dark[sy, sx].sum())
        if area / (h * w) < min_area_frac or area / (bh * bw) < min_fill:
            continue
        touches = sy.start == 0 or sx.start == 0 or sy.stop == h or sx.stop == w
        boxes.append({"height": int(bh), "width": int(bw), "touches_edge": bool(touches)})

    inner = [b for b in boxes if not b["touches_edge"]]
    return (len(inner) > 0), {
        "n_boxes": len(boxes),
        "n_inner_boxes": len(inner),
        "boxes": inner[:10],
    }


def check_jpg_exif(path: str | Path) -> tuple[bool, dict]:
    """JPG EXIF 에 장비, 시각 등 부가정보가 남아있는지 확인.

    비식별화된 배포본이라면 EXIF 가 비어 있어야 함.
    """
    try:
        from PIL import Image
    except ImportError:
        return True, {"skipped": "Pillow 미설치"}

    try:
        with Image.open(path) as im:
            exif = im.getexif()
            tags = {int(k): str(v)[:80] for k, v in dict(exif).items()} if exif else {}
    except Exception as e:  # 손상 파일 등
        return False, {"error": f"{type(e).__name__}: {e}"}

    return (len(tags) == 0), {"n_exif_tags": len(tags), "tags": tags}
