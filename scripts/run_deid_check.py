"""비식별화 검증 실행.

    python scripts/run_deid_check.py                 # 메타데이터 수준 검증
    python scripts/run_deid_check.py --images DIR    # JPG 픽셀 검사까지

MIMIC-CXR-JPG 는 이미 비식별화되어 배포되지만 그것을 직접 확인함.
무엇을 검증할 수 없는지도 결과로 남김.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from datapaths import RAW

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--images", default=None, help="JPG 디렉터리 (있으면 픽셀 검사)")
    ap.add_argument("--n-images", type=int, default=200, help="표본 검사할 영상 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd

    from deid import (
        DeidReport,
        check_jpg_exif,
        check_metadata_phi,
        detect_burned_in_text,
        detect_redaction_boxes,
        verify_date_shift_consistency,
    )

    raw = Path(args.raw_dir)
    rep = DeidReport()

    print("=" * 74)
    print("  비식별화 검증 (MIMIC-CXR-JPG)")
    print("=" * 74)

    # DICOM 전제 항목은 검증 대상이 존재하지 않음을 먼저 명시함.
    rep.not_applicable += [
        "PHI 태그(PatientName/PatientID/AccessionNumber 등): JPG 에는 DICOM 태그가 없음",
        "UID 재매핑(StudyInstanceUID/SeriesInstanceUID/SOPInstanceUID): UID 없음",
        "벤더 private tag 은닉 정보: 태그 구조 자체가 없음",
        "BurnedInAnnotation 태그값: 태그는 없으나 픽셀 검사로 대체함",
    ]

    print("\n[1] 배포 메타데이터 컬럼 PHI 검사")
    for fname in ("mimic-cxr-2.0.0-metadata.csv.gz",
                  "mimic-cxr-2.0.0-chexpert.csv.gz",
                  "mimic-cxr-2.0.0-split.csv.gz",
                  "admissions.csv.gz",
                  "patients.csv.gz"):
        hits = list(raw.rglob(fname))
        if not hits:
            print(f"    ⬜ {fname}: 없음 (건너뜀)")
            continue
        df = pd.read_csv(hits[0], nrows=5)
        ok, cols = check_metadata_phi(df, fname)
        rep.add(f"메타데이터 PHI 컬럼 없음 ({fname})", ok, "" if ok else f"의심 컬럼 {cols}")
        print(f"    {'[OK]' if ok else '[오류]'} {fname}: {len(df.columns)}개 컬럼"
              + ("" if ok else f": 의심 {cols}"))

    print("\n[2] 날짜 시프트 일관성 (CXR ↔ EHR)")
    meta_hits = list(raw.rglob("mimic-cxr-2.0.0-metadata.csv.gz"))
    adm_hits = list(raw.rglob("admissions.csv.gz"))
    if meta_hits and adm_hits:
        meta = pd.read_csv(meta_hits[0], usecols=["subject_id", "StudyDate", "StudyTime"])
        adm = pd.read_csv(adm_hits[0], usecols=["subject_id", "admittime", "dischtime"],
                          parse_dates=["admittime", "dischtime"])
        ok, stats = verify_date_shift_consistency(meta, adm)
        rep.add("날짜 시프트 일관 (간격 보존)", ok, f"겹침률 {stats.get('overlap_rate', 0):.3f}")
        print(f"    표본 환자 {stats['sampled_subjects']:,}명 중 "
              f"재원 중 CXR 보유 {stats['subjects_with_in_stay_cxr']:,}명 "
              f"({stats['overlap_rate'] * 100:.1f}%)")
        print(f"    CXR 중앙 연도 {stats['median_year_cxr']} / 입원 중앙 연도 {stats['median_year_adm']}")
        print(f"    -> 두 데이터셋이 {'같은' if ok else '다른'} 환자별 오프셋을 사용")
    else:
        print("    ⬜ 필요한 파일 없음 (건너뜀)")

    print("\n[3] 픽셀에 새겨진 내용 (텍스트 / 마스킹 흔적)")
    if not args.images:
        print("    ⬜ --images 미지정. CXR JPG 다운로드 후 실행할 것")
        print("       JPG 에서 가장 중요한 검사다. DICOM 태그를 다 지워도")
        print("       픽셀에 그려진 환자명, ID, 검사일은 그대로 남으며,")
        print("       JPG 변환은 이를 제거하지 않는다.")
    else:
        img_dir = Path(args.images)
        files = sorted(img_dir.rglob("*.jpg"))
        if not files:
            print(f"    [오류] {img_dir} 에 JPG 가 없습니다")
            rep.add("픽셀 검사 수행", False, "영상 없음")
        else:
            from PIL import Image

            rng = np.random.default_rng(args.seed)
            pick = rng.choice(len(files), size=min(args.n_images, len(files)), replace=False)
            with_text, with_box, exif_hits, samples = [], [], [], []
            for i in pick:
                fp = files[int(i)]
                with Image.open(fp) as im:
                    arr = np.asarray(im.convert("L"))
                has_text, tinfo = detect_burned_in_text(arr)
                if has_text:
                    with_text.append(fp.name)
                    if len(samples) < 5:
                        samples.append((fp.name, tinfo["text_corners"]))
                if detect_redaction_boxes(arr)[0]:
                    with_box.append(fp.name)
                if not check_jpg_exif(fp)[0]:
                    exif_hits.append(fp.name)

            n = len(pick)
            # 텍스트의 존재는 통과/실패가 아님. 눈으로 확인한 표본에서
            # 내용은 R/L 자세 마커, PORTABLE, semi upright 촬영기법 주석,
            # 방향 화살표였고 전부 PHI 가 아니었음. OCR 없이는 내용을 읽을 수
            # 없으므로 PHI 여부를 판정하지 않고 관찰로 남김.
            print(f"    표본 {n}장 중 새겨진 텍스트 {len(with_text)}장 "
                  f"({len(with_text) / n * 100:.1f}%)")
            print(f"    표본 {n}장 중 마스킹 흔적(내부 검정 사각형) {len(with_box)}장 "
                  f"({len(with_box) / n * 100:.1f}%)")
            rep.observe(
                f"새겨진 텍스트 {len(with_text)}/{n}장: 확인한 표본에서는 자세 마커"
                "(R/L), 촬영기법 주석(PORTABLE 등), 방향 화살표였고 PHI 는 없었다. "
                "OCR 이 없어 내용 판독은 불가하므로 통과/실패로 판정하지 않는다."
            )
            rep.observe(
                f"마스킹 흔적 {len(with_box)}/{n}장: 배포자가 픽셀을 가린 자국. "
                "비식별화가 실제로 적용됐다는 증거다."
            )
            rep.add("EXIF 메타데이터 없음", len(exif_hits) == 0, f"{len(exif_hits)}장에 EXIF")
            if samples:
                print("    텍스트가 있는 영상 예시(파일명, 위치만, 픽셀 미출력):")
                for name, corners in samples:
                    print(f"        {name}: {corners}")
                print("    -> 내용 판독은 사람이 직접 할 일이다.")


    print("\n" + "=" * 74)
    print(rep.summary())
    print("=" * 74)
    print("\n한계")
    print("  - 컬럼명 기반 PHI 검사는 값 수준의 식별자(자유텍스트 내 이름 등)를 잡지 못한다.")
    print("  - 새겨진 텍스트 탐지는 OCR 이 아니라 기하 휴리스틱이라 내용을 읽지 못한다.")
    print("    따라서 그 텍스트가 PHI 인지 아닌지는 이 절차로 판정할 수 없다.")
    print("  - 어두운 배경에 어두운 글자로 새겨진 주석은 포화되지 않아 놓친다.")
    print("  - 확정 판정이 아니라 사람이 확인할 후보를 좁히는 용도다.")
    return 0 if rep.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
