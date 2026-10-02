#!/bin/sh
# =====================================================================
# MIMIC 다운로드: PhysioNet 크리덴셜 필요
#
#   sh scripts/download_mimic.sh meta      # 1단계: CXR 메타데이터만 (~58MB)
#   sh scripts/download_mimic.sh mimiciv   # 2단계: MIMIC-IV 전체 (~9.9GB)
#
# 주의: 비밀번호는 이 파일에 절대 적지 않는다.
#    --ask-password 로 실행 시점에 직접 입력받으며 디스크에 남지 않는다.
#    (pre-commit 훅이 하드코딩된 자격증명을 차단한다)
#
# 주의: 저장 위치는 클라우드 동기화 폴더 밖이어야 한다.
#    이 저장소의 data/ 는 .gitignore + pre-commit 훅으로 이중 차단되어 있다.
#    내려받을 위치가 클라우드 동기화 폴더 안인지 먼저 확인할 것.
#
# wget 옵션
#   -r  재귀        -N  타임스탬프 비교(변경분만)   -c  중단 재개
#   -np 상위디렉터리로 올라가지 않음                 -nH 호스트명 디렉터리 생략
# =====================================================================
set -e

# PhysioNet 계정명. 기본값을 코드에 박지 않는다: 저장소를 공개하면 계정명이
# 그대로 노출되고, 남이 받아 쓸 때도 자기 계정으로 바꿔야 한다는 것을 모른다.
USER_ID="${PHYSIONET_USER:-}"
if [ -z "$USER_ID" ]; then
  printf "PhysioNet 사용자명: "
  read -r USER_ID
  [ -n "$USER_ID" ] || { echo "[ERROR] 사용자명이 비었습니다."; exit 1; }
  echo "  (다음부터는 PHYSIONET_USER 환경변수로 넘길 수 있습니다)"
fi

ROOT=$(cd "$(dirname "$0")/.." && pwd)

# 데이터 위치. 저장소 기준 상대경로이며 MIMIC_DATA_ROOT 로 덮어쓸 수 있다.
# src/datapaths.py 와 같은 규칙을 쓴다.
if [ -n "${MIMIC_DATA_ROOT:-}" ]; then
  DEST="$MIMIC_DATA_ROOT/raw"
else
  DEST="$ROOT/../../데이터/mimic/data/raw"
fi
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)

MIMICIV_URL="https://physionet.org/files/mimiciv/3.1/"
CXR_BASE="https://physionet.org/files/mimic-cxr-jpg/2.1.0"
NOTE_URL="https://physionet.org/files/mimic-iv-note/2.2/"

# CXR 메타데이터: 전체 570GB 중 이것만 먼저 받아 코호트를 정한다.
# "무엇을 받을지" 를 정하는 근거이므로 영상보다 반드시 먼저 받는다.
META_FILES="mimic-cxr-2.0.0-metadata.csv.gz
mimic-cxr-2.0.0-split.csv.gz
mimic-cxr-2.0.0-chexpert.csv.gz
IMAGE_FILENAMES"

command -v wget >/dev/null 2>&1 || {
  echo "[ERROR] wget 이 필요합니다.  winget install JernejSimoncic.Wget"
  exit 1
}

usage() {
  echo "사용법: sh scripts/download_mimic.sh {meta|mimiciv|images|note} [연결수]"
  echo "  meta        CXR 메타데이터 4종 (~58MB): 코호트 설계용, 가장 먼저"
  echo "  mimiciv     MIMIC-IV v3.1 전체 (~9.9GB)"
  echo "  images [N]  Phase 1.5 가 고른 CXR 영상만 (~49GB)"
  echo "              전체 570GB 중 코호트 재원 기간에 촬영된 것만 받는다."
  echo "              run_phase1_5.py 를 먼저 돌려 목록을 만들 것."
  echo "  note        MIMIC-IV-Note v2.2 퇴원요약문+영상판독문 전체 (~4GB)"
  echo "              별도 DUA 서명 필요 (physionet.org/content/mimic-iv-note/2.2/)"
  echo ""
  echo "  N 은 동시 연결 수(기본 8, images 에만 적용). PhysioNet 계정당 최대 15."
  echo "  중단되면 같은 명령을 다시 실행하면 된다. -N -c 가 받은 것을 건너뛴다."
  exit 1
}

[ $# -ge 1 ] && [ $# -le 2 ] || usage

echo "PhysioNet 사용자: $USER_ID"
echo "저장 위치      : $DEST"
echo "비밀번호는 아래에서 직접 입력합니다(파일에 저장되지 않음)."
echo ""

case "$1" in
  meta)
    mkdir -p "$DEST/mimic-cxr-jpg"
    for f in $META_FILES; do
      echo "--- $f ---"
      wget -N -c --user "$USER_ID" --ask-password \
           -P "$DEST/mimic-cxr-jpg" "$CXR_BASE/$f"
    done
    ;;
  mimiciv)
    mkdir -p "$DEST/mimiciv"
    wget -r -N -c -np -nH --cut-dirs=3 \
         --user "$USER_ID" --ask-password \
         -P "$DEST/mimiciv" "$MIMICIV_URL"
    ;;
  images)
    LIST="$ROOT/outputs/phase2_download_list.txt"
    [ -f "$LIST" ] || {
      echo "[ERROR] 목록이 없습니다: $LIST"
      echo "        python scripts/run_phase1_5.py 를 먼저 실행하세요."
      exit 1
    }
    URLS="$ROOT/outputs/phase2_urls.txt"
    sed "s|^|$CXR_BASE/|" "$LIST" > "$URLS"
    IMG_DEST="$DEST/mimic-cxr-jpg"
    mkdir -p "$IMG_DEST"

    TOTAL=$(wc -l < "$URLS")
    DONE=$(find "$IMG_DEST/files" -name '*.jpg' 2>/dev/null | wc -l)
    echo "목록 ${TOTAL}장 / 이미 받은 것 ${DONE}장"

    N="${2:-8}"
    # -x -nH --cut-dirs=3 로 files/pXX/... 구조를 그대로 재현한다.
    WGET_OPTS="-N -c -x -nH --cut-dirs=3"

    if [ "$N" -le 1 ]; then
      wget $WGET_OPTS --user "$USER_ID" --ask-password \
           -P "$IMG_DEST" -i "$URLS"
    else
      # 병렬은 --ask-password 를 쓸 수 없다(프롬프트가 N개 겹친다).
      # 한 번만 입력받아 각 wget 에 넘긴다. 디스크에는 남지 않는다.
      printf "PhysioNet 비밀번호: "
      stty -echo 2>/dev/null
      read -r PW
      stty echo 2>/dev/null
      echo ""
      [ -n "$PW" ] || { echo "[ERROR] 비밀번호가 비었습니다."; exit 1; }

      CHUNKS="$DEST/_chunks"
      rm -rf "$CHUNKS"; mkdir -p "$CHUNKS"
      split -n l/"$N" "$URLS" "$CHUNKS/part_"

      LOG="$DEST/download.log"
      : > "$LOG"
      echo "동시 연결 $N 개로 받습니다. 진행 상황: $LOG"
      echo "중단해도 같은 명령으로 이어받습니다."

      for c in "$CHUNKS"/part_*; do
        wget $WGET_OPTS --user "$USER_ID" --password="$PW" \
             -P "$IMG_DEST" -i "$c" -a "$LOG" &
      done
      wait
      rm -rf "$CHUNKS"
      unset PW

      GOT=$(find "$IMG_DEST/files" -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
      echo "받은 파일 ${GOT}장 / 목록 ${TOTAL}장"
      if [ "$GOT" -lt "$TOTAL" ]; then
        echo "덜 받았습니다. 같은 명령을 다시 실행하면 이어받습니다."
      fi
    fi
    ;;
  note)
    NOTE_DEST="$DEST/mimic-iv-note"
    mkdir -p "$NOTE_DEST"
    # 파일이 discharge/discharge_detail/radiology/radiology_detail 4개뿐이라
    # CXR 처럼 URL 쪼개기가 아니라 파일 단위로 동시에 받는다.
    printf "PhysioNet 비밀번호: "
    stty -echo 2>/dev/null
    read -r PW
    stty echo 2>/dev/null
    echo ""
    [ -n "$PW" ] || { echo "[ERROR] 비밀번호가 비었습니다."; exit 1; }

    mkdir -p "$NOTE_DEST/note"
    for f in note/discharge.csv.gz note/discharge_detail.csv.gz \
             note/radiology.csv.gz note/radiology_detail.csv.gz; do
      wget -N -c --user "$USER_ID" --password="$PW" \
           -P "$NOTE_DEST" -x -nH --cut-dirs=3 "${NOTE_URL}${f}" &
    done
    wait
    unset PW
    echo "받은 파일:"
    ls -la "$NOTE_DEST"/*.gz 2>/dev/null
    ;;
  *)
    usage
    ;;
esac

echo ""
echo "완료. 받은 내용:"
du -sh "$DEST"/* 2>/dev/null || true
echo ""
echo "확인: 이 데이터는 .gitignore + pre-commit 훅으로 커밋이 차단되어 있습니다."
echo "      sh scripts/verify_hooks.sh 로 차단 상태를 재검증할 수 있습니다."
