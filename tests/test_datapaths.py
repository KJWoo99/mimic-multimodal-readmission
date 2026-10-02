"""데이터 경로가 저장소 기준 상대경로로 잡히는지 고정.

절대경로를 코드에 박으면 남이 clone 해서 실행할 수 없음. 경로는 `src/datapaths.py` 하나에 모으고,
그 값을 저장소 루트에서 유도함.

여기서 지키려는 것 세 가지.

1. 드라이브 문자 절대경로가 코드에 없음: 새로 박히면 즉시 실패함.
2. CWD 와 무관함: `__file__` 기준이라 어디서 실행해도 같은 곳을 가리킴.
3. 환경변수로 덮어쓸 수 있음: 데이터를 다른 곳에 둔 사람을 위한 탈출구.
"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BS = chr(92)
DRIVE = re.compile("(?<![A-Za-z0-9])[A-Za-z]:[" + BS + BS + "/]")


def _sources() -> list[Path]:
    """검사 대상 소스. `datapaths.py` 만 예외임.

    그 파일은 "환경변수로 이렇게 덮어쓴다"는 예시 경로를 docstring 에
    싣고 있음. 실제로 쓰는 경로가 아니라 사용법 설명이므로 제외함:
    경로를 한 곳에 모은 목적이 그 파일임.
    """
    out = []
    for d in ("src", "scripts", "tests"):
        base = REPO / d
        if base.is_dir():
            out += [
                f for f in base.rglob("*.py")
                if "__pycache__" not in str(f) and f.name != "datapaths.py"
            ]
    return out


def _code_lines(text: str):
    """주석과 docstring 을 뺀 줄만 내놓음.

    검사하려는 것은 "코드가 그 경로를 실제로 쓰는가"다. 설명문에 적힌 경로는
    대상이 아님. 설명문까지 보면 절대경로 예시를 적어 둔 docstring 이 스스로 걸림.

    삼중따옴표 개수로 안팎을 가림. 한 줄 안에서 열고 닫는 경우까지 세므로
    여는 줄만 세는 방식보다 정확함.
    """
    in_doc = False
    for i, line in enumerate(text.splitlines(), 1):
        marks = line.count('"""') + line.count("'''")
        if in_doc:
            if marks % 2 == 1:
                in_doc = False
            continue
        if marks % 2 == 1:
            in_doc = True
            continue          # 여는 줄 자체도 설명문임
        if line.lstrip().startswith("#"):
            continue          # 주석의 예시 경로는 허용
        yield i, line


def test_no_absolute_paths_in_code():
    """코드에 드라이브 문자 절대경로가 남아 있으면 안 됨."""
    bad = []
    for f in _sources():
        for i, line in _code_lines(f.read_text(encoding="utf-8")):
            if DRIVE.search(line):
                bad.append(f"{f.relative_to(REPO)}:{i}  {line.strip()[:90]}")
    assert not bad, "절대경로가 박혀 있다:\n" + "\n".join(bad)


def test_paths_are_derived_from_repo_root():
    sys.path.insert(0, str(REPO / "src"))
    import datapaths

    importlib.reload(datapaths)
    assert datapaths.REPO_ROOT == REPO
    # 데이터는 저장소 바깥이어야 함: 안에 있으면 DUA 위반임
    assert REPO not in datapaths.DATA_ROOT.parents


def test_env_override_wins(tmp_path, monkeypatch):
    fake = tmp_path / "mydata"
    (fake / "raw" / "mimiciv").mkdir(parents=True)
    monkeypatch.setenv("MIMIC_DATA_ROOT", str(fake))
    sys.path.insert(0, str(REPO / "src"))
    import datapaths

    importlib.reload(datapaths)
    try:
        assert fake.resolve() == datapaths.DATA_ROOT
        assert fake.resolve() / "raw" / "mimiciv" == datapaths.MIMICIV
        assert "MIMIC_DATA_ROOT" in datapaths.describe()
    finally:
        monkeypatch.delenv("MIMIC_DATA_ROOT")
        importlib.reload(datapaths)


def test_resolves_the_same_from_any_cwd(tmp_path):
    """다른 디렉터리에서 실행해도 같은 경로가 나와야 함."""
    code = (
        "import sys; sys.path.insert(0, r'" + str(REPO / "src") + "');"
        "import datapaths; print(datapaths.RAW)"
    )
    # 경로에 한글이 있으면 로캘에 따라 디코딩이 깨질 수 있음.
    # 자식 프로세스와 부모 양쪽에 UTF-8 을 명시함.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    run = lambda cwd: subprocess.run(  # noqa: E731
        [sys.executable, "-c", code], cwd=cwd, env=env,
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    here, away = run(REPO), run(tmp_path)
    assert here.returncode == 0 and away.returncode == 0, (here.stderr, away.stderr)
    assert here.stdout.strip() == away.stdout.strip(), (
        "CWD 가 달라지면 경로도 달라진다: __file__ 기준이 아니라는 뜻이다"
    )


@pytest.mark.skipif(
    not (Path(os.environ.get("MIMIC_DATA_ROOT", "")) / "raw").is_dir()
    and not (REPO / ".." / ".." / "데이터" / "mimic" / "data" / "raw").is_dir(),
    reason="원본 데이터 없음",
)
def test_default_location_actually_exists():
    """이 PC 에서는 기본 상대경로가 실제 데이터를 가리켜야 함."""
    sys.path.insert(0, str(REPO / "src"))
    import datapaths

    importlib.reload(datapaths)
    assert datapaths.RAW.is_dir(), datapaths.describe()
