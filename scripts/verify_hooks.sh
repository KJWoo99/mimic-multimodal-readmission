#!/bin/sh
# =====================================================================
# 유출 차단 장치 자가검증: 실제로 뚫어보는 테스트
#
# .gitignore와 pre-commit 훅이 살아있는지 확인한다.
# 데이터를 받기 전, 그리고 .gitignore를 수정한 뒤에 실행할 것.
#
#   sh scripts/verify_hooks.sh
# =====================================================================

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
  echo "[ERROR] git 저장소가 아닙니다."; exit 1;
}
cd "$ROOT"

PASS=0; FAIL=0
ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }

TMP="data/raw/mimiciv/__verify_tmp.csv"
mkdir -p data/raw/mimiciv
printf 'subject_id,hadm_id,dischtime\n10001,20001,2180-01-01\n' > "$TMP"

echo "1. .gitignore 가 data/ 를 막는가"
git check-ignore -q "$TMP" && ok "data/ 이하 무시됨" || bad "data/ 가 추적 대상이다 ***위험***"

echo "2. pre-commit 훅이 존재하고 실행 가능한가"
HOOK="$(git rev-parse --git-path hooks)/pre-commit"
[ -f "$HOOK" ] && ok "훅 존재: $HOOK" || bad "훅 없음 - sh scripts/install_hooks.sh 실행 필요"

echo "3. 강제 add(-f) 우회를 훅이 차단하는가"
if [ -f "$HOOK" ]; then
  git add -f "$TMP" 2>/dev/null
  if git commit -m "__verify__ this must be blocked" >/dev/null 2>&1; then
    bad "커밋이 통과했다 ***위험***: 즉시 되돌립니다"
    git reset --soft HEAD~1 2>/dev/null
  else
    ok "훅이 커밋을 차단함"
  fi
  git reset -q 2>/dev/null
else
  bad "훅이 없어 테스트 불가"
fi

echo "4. 대용량 파일 차단이 동작하는가"
BIG="outputs/__verify_big.bin"
head -c 25000000 /dev/urandom > "$BIG" 2>/dev/null
git add -f "$BIG" 2>/dev/null
if git commit -m "__verify__ big file" >/dev/null 2>&1; then
  bad "25MB 파일이 통과했다"
  git reset --soft HEAD~1 2>/dev/null
else
  ok "20MB 초과 파일 차단됨"
fi
git reset -q 2>/dev/null

# 5, 6 은 합성 값만 쓴다. 값을 printf 로 조립하는 것은 이 파일 자체가 검사 도구에 식별자, 원문 모양으로
# 걸리지 않게 하려는 것이다.
echo "5. 파일 안의 MIMIC 식별자를 훅이 차단하는가"
IDF="outputs/__verify_id.txt"
printf 'row 1\n%s-%s-%s\n' 12345678 DS 9 > "$IDF"
git add -f "$IDF" 2>/dev/null
if git commit -m "__verify__ identifier" >/dev/null 2>&1; then
  bad "식별자 값이 든 파일이 통과했다 ***위험***"
  git reset --soft HEAD~1 2>/dev/null
else
  ok "식별자 값 차단됨"
fi
git reset -q 2>/dev/null

echo "6. 허용 목록에 없는 노트 원문 모양 줄을 훅이 차단하는가"
NOTEF="outputs/__verify_note.txt"
printf 'Fakeol 1 mg %s DAILY\n' PO > "$NOTEF"
git add -f "$NOTEF" 2>/dev/null
if git commit -m "__verify__ note-like line" >/dev/null 2>&1; then
  bad "노트 원문 모양 줄이 통과했다"
  git reset --soft HEAD~1 2>/dev/null
else
  ok "허용 목록 밖 원문 모양 줄 차단됨"
fi
git reset -q 2>/dev/null

rm -f "$TMP" "$BIG" "$IDF" "$NOTEF"

echo ""
echo "결과: PASS $PASS / FAIL $FAIL"
[ "$FAIL" -eq 0 ] || {
  echo "*** 실패 항목이 있습니다. 데이터를 받기 전에 반드시 해결하세요. ***"
  echo "*** 참고: PhysioNet DUA: 원문, 파생 데이터는 외부로 내보내지 않는다 ***"
  exit 1
}
echo "모든 차단 장치 정상."
