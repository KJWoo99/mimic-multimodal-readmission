#!/bin/sh
# =====================================================================
# git 훅 설치: clone 직후 반드시 1회 실행할 것
#
# git은 .git/hooks/ 를 추적하지 않으므로, clone한 저장소에는 유출 차단
# 훅이 없는 상태다. 데이터를 받기 전에 이 스크립트를 실행해야 한다.
#
#   sh scripts/install_hooks.sh
# =====================================================================
set -e

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "[ERROR] git 저장소가 아닙니다."; exit 1;
}
cd "$ROOT"

SRC="scripts/hooks"
DST="$(git rev-parse --git-path hooks)"

[ -d "$SRC" ] || { echo "[ERROR] $SRC 가 없습니다."; exit 1; }
mkdir -p "$DST"

for h in "$SRC"/*; do
  [ -f "$h" ] || continue
  name=$(basename "$h")
  cp "$h" "$DST/$name"
  chmod +x "$DST/$name"
  echo "  installed: $name"
done

# 데이터 디렉터리 재생성 (gitignore 때문에 clone 시 따라오지 않음)
mkdir -p data/raw/mimiciv data/raw/mimic-cxr-jpg data/interim data/processed outputs/models
echo "  created:   data/ 하위 디렉터리"

echo ""
echo "완료. 설치된 훅을 검증하려면:"
echo "  sh scripts/verify_hooks.sh"
