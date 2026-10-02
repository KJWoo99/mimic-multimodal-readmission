"""원본 데이터 위치: 저장소 기준 상대경로.

데이터는 저장소 바깥에 있음. PhysioNet 배포본이 수백 GB 라 저장소에 넣을 수
없고, DUA 상 넣어서도 안 됨. 그래서 저장소 루트에서 위로 올라가는 상대경로로
가리킴.

    <어딘가>/<폴더>/<이 저장소>/     <- 저장소 루트
    <어딘가>/데이터/mimic/data/raw/   <- 원본

왜 CWD 가 아니라 `__file__` 기준인가
------------------------------------
현재 작업 디렉터리를 기준으로 하면 `python scripts/xxx.py` 를 저장소 밖에서
실행하는 순간 깨짐. `__file__` 에서 저장소 루트를 유도하면 어디서 실행하든
같은 곳을 가리키며, 상대 구조는 그대로 유지됨.

데이터를 다른 위치에 두었다면 환경변수로 덮어쓸 수 있음.

    export MIMIC_DATA_ROOT=/mnt/data/mimic/data
"""
from __future__ import annotations

import os
from pathlib import Path

# 이 파일은 <저장소>/src/ 에 있으므로 저장소 루트는 한 단계 위임.
REPO_ROOT = Path(__file__).resolve().parents[1]

# 저장소 루트에서 두 단계 올라간 폴더 아래에 데이터 폴더가 있음(<상위>/<폴더>/<저장소>, <상위>/데이터).
_DEFAULT = REPO_ROOT / ".." / ".." / "데이터" / "mimic" / "data"
DATA_ROOT = Path(os.environ.get("MIMIC_DATA_ROOT", _DEFAULT)).resolve()

RAW = DATA_ROOT / "raw"
MIMICIV = RAW / "mimiciv"
NOTE = RAW / "mimic-iv-note" / "note"
CXR_JPG = RAW / "mimic-cxr-jpg"
PUBLIC = DATA_ROOT / "public"


def describe() -> str:
    """경로와 그 출처, 존재 여부를 한 줄로. 실패 메시지에 씀."""
    src = "MIMIC_DATA_ROOT" if "MIMIC_DATA_ROOT" in os.environ else "저장소 기준 상대경로"
    return f"{DATA_ROOT}  ({src}, 존재={DATA_ROOT.is_dir()})"
