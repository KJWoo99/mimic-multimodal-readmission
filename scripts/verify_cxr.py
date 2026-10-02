"""내려받은 CXR 영상이 온전한지 검사함.

다운로드를 중간에 끊으면 그 순간 받던 파일이 잘린 채 남음. 잘린 JPEG 은
열리기는 해도 아래쪽이 회색으로 비어 학습에 경고 없이 섞여 들어감.

    python scripts/verify_cxr.py              # 검사만
    python scripts/verify_cxr.py --deep       # 전체 디코딩까지
    python scripts/verify_cxr.py --delete     # 손상 파일 삭제

삭제한 뒤 download_mimic.sh images 를 다시 돌리면 없는 파일만 채움.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _img_root():
    """CXR JPG 루트. datapaths 는 src 에 있어 sys.path 설정 뒤에 임포트함."""
    from datapaths import CXR_JPG

    return CXR_JPG

LIST = ROOT / "outputs" / "phase2_download_list.txt"

# JPEG 은 FFD8 로 시작해 FFD9 로 끝남. 끝 마커가 없으면 잘린 것임.
SOI = b"\xff\xd8"
EOI = b"\xff\xd9"

OK, MISSING, EMPTY, TRUNCATED, BROKEN = "정상", "없음", "0바이트", "잘림", "손상"


def check(rel: str, deep: bool) -> tuple[str, str]:
    p = _img_root() / rel
    if not p.is_file():
        return MISSING, rel
    size = p.stat().st_size
    if size == 0:
        return EMPTY, rel
    with p.open("rb") as f:
        if f.read(2) != SOI:
            return BROKEN, rel
        f.seek(-2, 2)
        if f.read(2) != EOI:
            return TRUNCATED, rel
    if deep:
        try:
            from PIL import Image

            with Image.open(p) as im:
                im.load()
        except Exception:
            return BROKEN, rel
    return OK, rel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true", help="전체 디코딩까지 확인 (느림)")
    ap.add_argument("--delete", action="store_true", help="손상 파일을 지운다")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if not LIST.is_file():
        print(f"목록이 없습니다: {LIST}")
        return 1

    rels = [ln.strip() for ln in LIST.read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"목록 {len(rels):,}장 검사 시작" + (" (전체 디코딩)" if args.deep else ""))

    counts: dict[str, list[str]] = {k: [] for k in (OK, MISSING, EMPTY, TRUNCATED, BROKEN)}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (status, rel) in enumerate(ex.map(lambda r: check(r, args.deep), rels), 1):
            counts[status].append(rel)
            if i % 5000 == 0:
                print(f"  {i:,}/{len(rels):,}", flush=True)

    print("\n=== 결과 ===")
    for k in (OK, MISSING, EMPTY, TRUNCATED, BROKEN):
        print(f"  {k:<8} {len(counts[k]):>7,}장")

    bad = counts[EMPTY] + counts[TRUNCATED] + counts[BROKEN]
    if not bad:
        print("\n손상 파일 없음.")
        if counts[MISSING]:
            print(f"다만 {len(counts[MISSING]):,}장이 아직 안 받아졌습니다.")
        return 0

    print(f"\n손상 {len(bad):,}장. 예시:")
    for rel in bad[:5]:
        print(f"  {rel}")

    if not args.delete:
        print("\n--delete 를 붙이면 지웁니다. 지운 뒤 download_mimic.sh images 로 다시 받으세요.")
        return 1

    gone = 0
    for rel in bad:
        try:
            (_img_root() / rel).unlink()
            gone += 1
        except OSError as e:
            print(f"  삭제 실패 {rel}: {e}")
    print(f"\n{gone:,}장 삭제. download_mimic.sh images 15 로 다시 받으세요.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
