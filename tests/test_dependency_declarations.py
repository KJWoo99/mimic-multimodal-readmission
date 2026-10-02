"""의존성 선언이 코드가 실제로 쓰는 것과 맞는지 고정.

두 가지를 잡음.

1. 선언 누락: 코드가 임포트하는데 어디에도 선언이 없는 것.
   선언이 빠지면 새 환경에서 그 모듈을 쓰는 명령이 통째로 실행되지 않음.

2. 버전 불일치: 선언한 버전이 실제로 결과를 만든 환경과 다른 것.

CUDA 휠(torch)이나 플랫폼 의존 패키지(mamba_ssm, tensorrt)는 버전을 고정하면
오히려 방해가 되므로, 문서에 설치 안내가 있으면 통과로 봄.
"""
from __future__ import annotations

import importlib.metadata as md
import os
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

def _in_declared_env() -> bool:
    """지금 돌고 있는 인터프리터가 이 저장소가 선언한 conda 환경인가.

    이 검사가 주장하는 것은 "선언과 그 선언이 가리키는 환경이 같다" 임.
    다른 환경에서 돌리면 당연히 어긋나고, 그 실패는 알려주는 것이 없음.
    이 검사가 매번 실패하면 실제 불일치가 생겨도 알아볼 수 없으므로 범위를 좁힘.
    """
    p = REPO / "environment.yml"
    if not p.exists():
        return True
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*name:\s*(\S+)", line)
        if m:
            return os.environ.get("CONDA_DEFAULT_ENV") == m.group(1)
    return True

STD = set(sys.stdlib_module_names)
ALIAS = {"PIL": "pillow", "sklearn": "scikit-learn", "cv2": "opencv-python", "yaml": "pyyaml",
         "imblearn": "imbalanced-learn", "skimage": "scikit-image"}
# 별도 설치를 안내하는 것으로 충분한 패키지
SPECIAL = {"torch", "torchvision", "mamba_ssm", "causal_conv1d", "tensorrt",
           "onnxruntime-gpu", "hf_transfer"}


def _declared() -> set[str]:
    out: set[str] = set()
    p = REPO / "requirements.txt"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "--")):
                out.add(re.split(r"[<>=\[]", line)[0].strip().lower())
    p = REPO / "pyproject.toml"
    if p.exists():
        d = tomllib.loads(p.read_text(encoding="utf-8"))
        proj = d.get("project", {})
        for dep in proj.get("dependencies", []):
            out.add(re.split(r"[<>=\[ ]", dep)[0].lower())
        for grp in (proj.get("optional-dependencies") or {}).values():
            for dep in grp:
                out.add(re.split(r"[<>=\[ ]", dep)[0].lower())
    p = REPO / "environment.yml"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("- --"):
                continue
            m = re.match(r"\s*-\s+([A-Za-z0-9_.\-]+)", line)
            if m:
                out.add(m.group(1).lower())
    return out


def _imported() -> set[str]:
    local = {f.stem for d in ("src", "scripts", "tests") for f in (REPO / d).rglob("*.py")}
    src = REPO / "src"
    if src.exists():
        local |= {d.name for d in src.iterdir() if d.is_dir()}
    out: set[str] = set()
    for d in ("src", "scripts", "tests"):
        for f in (REPO / d).rglob("*.py"):
            if "__pycache__" in str(f):
                continue
            for m in re.findall(
                r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)",
                f.read_text(encoding="utf-8"), re.M,
            ):
                if m in STD or m in local or m == "__future__":
                    continue
                out.add(ALIAS.get(m, m).lower())
    return out


def _docs() -> str:
    txt = ""
    for f in ("requirements.txt", "pyproject.toml", "environment.yml", "README.md"):
        p = REPO / f
        if p.exists():
            txt += p.read_text(encoding="utf-8") + "\n"
    return txt.lower()


def test_every_import_is_declared_or_documented():
    docs = _docs()
    declared = _declared()
    problems = []
    for mod in sorted(_imported() - declared):
        if mod in SPECIAL:
            if mod not in docs:
                problems.append(f"{mod} (별도 설치인데 안내도 없음)")
        else:
            problems.append(f"{mod} (선언 누락)")
    assert not problems, "의존성 문제:\n  " + "\n  ".join(problems)


@pytest.mark.skipif(not _in_declared_env(),
                    reason="이 저장소가 선언한 conda 환경이 아니다")
def test_declared_versions_match_this_environment():
    """선언 버전이 지금 환경과 같아야 함.

    다르면 "이 코드를 돌린 환경"과 "설치하라고 적어둔 환경"이 어긋난 것임.
    이 환경에 없는 패키지는 확인할 수 없으므로 건너뜀.
    """
    mismatched = []
    for fname in ("environment.yml", "requirements.txt", "pyproject.toml"):
        p = REPO / fname
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            # 주석은 선언이 아님. 설치 안내문의 `torch==2.13.0` 같은 예시를 선언으로 읽으면 오탐이 남.
            if line.lstrip().startswith("#"):
                continue
            m = re.search(r"([A-Za-z0-9_.\-]+)==([0-9][^\s\"',]*)", line)
            if not m:
                continue
            name, ver = m.group(1), m.group(2).split("+")[0]
            try:
                real = md.version(name).split("+")[0]
            except Exception:
                continue
            if real != ver:
                mismatched.append(f"{fname}: {name} 선언 {ver} != 실제 {real}")
    assert not mismatched, "선언 버전이 실제 환경과 다르다:\n  " + "\n  ".join(mismatched)


def _declared_pins() -> dict[str, dict[str, list[str]]]:
    """선언 파일들에서 `이름==버전` 을 모음.

    `{패키지: {버전: [그 버전을 적은 파일들]}}` 로 돌려줌. 하나의 dict 에 덮어쓰면
    나중에 읽은 파일이 앞 파일의 값을 지워 파일 간 불일치가 안 보임.
    """
    pins: dict[str, dict[str, list[str]]] = {}
    for fname in ("environment.yml", "requirements.txt", "pyproject.toml"):
        p = REPO / fname
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = re.search(r"([A-Za-z0-9_.\-]+)==([0-9][^\s\"',]*)", line)
            if m:
                name = m.group(1).lower().replace("_", "-")
                ver = m.group(2).split("+")[0]
                pins.setdefault(name, {}).setdefault(ver, []).append(fname)
    return pins


def test_declaration_files_agree_with_each_other():
    """같은 패키지를 여러 파일에 적었으면 버전이 같아야 함.

    `environment.yml` 과 `requirements.txt` 가 다른 버전을 말하면 어느 쪽을
    믿어야 하는지 알 수 없음. 사람이 한쪽만 고치는 일이 흔함.
    """
    bad = [f"{name}: " + ", ".join(f"{v}({', '.join(fs)})" for v, fs in sorted(vers.items()))
           for name, vers in sorted(_declared_pins().items()) if len(vers) > 1]
    assert not bad, "선언 파일마다 버전이 다르다:\n  " + "\n  ".join(bad)


@pytest.mark.skipif(not _in_declared_env(),
                    reason="이 저장소가 선언한 conda 환경이 아니다")
def test_declared_versions_are_mutually_installable():
    """선언한 버전들이 서로 모순되지 않아야 함.

    선언↔설치 일치만 보면 선언끼리 양립하는지는 드러나지 않음. 예: `mlflow 3.15.1` 은
    `pandas<3` 을 요구하는데 `pandas==3.0.5` 를 고정하면, 기존 환경에 하나씩 설치할 때는
    pip 이 경고만 내고 넘어가지만 이 파일로 환경을 처음부터 만들면 실패함.

    네트워크를 쓰지 않음. 설치된 배포판의 `Requires-Dist` 를 읽어 선언된 다른
    패키지의 고정 버전이 그 조건을 어기는지만 봄.
    """
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion, Version

    # 파일마다 다르게 적혔으면 위 테스트가 잡음. 여기서는 아무 값이나 하나 씀.
    pins = {n: sorted(v)[0] for n, v in _declared_pins().items()}
    violations = []
    for name, ver in sorted(pins.items()):
        try:
            reqs = md.requires(name) or []
        except md.PackageNotFoundError:
            continue  # 이 환경에 없으면 확인할 수 없음
        for raw in reqs:
            try:
                req = Requirement(raw)
            except Exception:
                continue
            if req.marker is not None and not req.marker.evaluate():
                continue  # extra 나 플랫폼 조건이라 지금은 해당 없음
            dep = req.name.lower().replace("_", "-")
            if dep not in pins or not req.specifier:
                continue
            try:
                target = Version(pins[dep])
            except InvalidVersion:
                continue
            if not req.specifier.contains(target, prereleases=True):
                violations.append(
                    f"{name}=={ver} 는 {dep}{req.specifier} 를 요구하는데 "
                    f"선언은 {dep}=={pins[dep]}")

    assert not violations, (
        "선언한 버전들이 서로 모순된다. 깨끗한 환경에서 설치가 실패한다:\n  "
        + "\n  ".join(violations))


def test_declared_python_matches_this_interpreter():
    """`environment.yml` 이 선언한 python 버전이 실제와 같아야 함.

    버전 핀만 맞추고 python 줄을 빼먹기 쉬움.
    그냥 어긋나기만 하는 게 아니라 파일을 못 쓰게 됨. 예를 들어 `python=3.11` 인데
    `numpy==2.5.2` 가 python>=3.12 를 요구하면 이 파일로는 환경을 만들 수 없음. 로컬이 3.13
    환경이면 드러나지 않음.
    """
    p = REPO / "environment.yml"
    if not p.exists():
        pytest.skip("environment.yml 없음")
    m = re.search(r"^\s*-\s*python\s*=\s*([0-9]+\.[0-9]+)", p.read_text(encoding="utf-8"), re.M)
    if not m:
        pytest.skip("python 버전 선언 없음")
    declared = m.group(1)
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    assert declared == actual, (
        f"environment.yml 은 python={declared} 인데 지금 인터프리터는 {actual} 다. "
        "이 파일로 환경을 만들면 다른 파이썬이 깔린다")
