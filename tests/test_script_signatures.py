"""스크립트가 라이브러리 함수를 올바른 시그니처로 부르는지 정적 검증.

실행에 GPU, 실데이터가 필요한 스크립트는 테스트에서 돌릴 수 없음. 그래서
"인자를 몇 개 넘기는가", "반환값을 어떻게 다루는가" 를 AST 로 확인함.

예: `build_cxr_index(cohort, img_root, one_per_admission=True)` 처럼 cxr_meta 인자를 빠뜨리거나
반환값(IndexResult)을 DataFrame 처럼 쓰면 실행 즉시 TypeError 지만, GPU 가 필요한 스크립트라
GPU 에서 돌리기 전에는 드러나지 않음.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cxr_dataset  # noqa: E402

SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))

# 스크립트가 부르는 라이브러리 함수 중 시그니처를 검사할 대상
WATCHED = {
    "build_cxr_index": cxr_dataset.build_cxr_index,
    "CXRDataset": cxr_dataset.CXRDataset.__init__,
}


def _calls_in(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _min_required_args(func) -> int:
    """기본값이 없는 위치 인자 수 (self 제외)."""
    sig = inspect.signature(func)
    n = 0
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.default is inspect.Parameter.empty and p.kind in (
            p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD
        ):
            n += 1
    return n


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_watched_calls_supply_enough_arguments(script):
    """감시 대상 함수를 부를 때 필수 인자를 빠뜨리지 않았는가."""
    for call in _calls_in(script):
        name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
        if name not in WATCHED:
            continue
        supplied = len(call.args) + len(call.keywords)
        required = _min_required_args(WATCHED[name])
        assert supplied >= required, (
            f"{script.name}:{call.lineno}: {name}() 에 인자를 {supplied}개만 넘겼다. "
            f"필수 {required}개. 실행 시 TypeError 가 난다."
        )


def test_build_cxr_index_returns_index_result_not_dataframe():
    """반환 타입을 고정함: DataFrame 으로 오해하면 len()/컬럼 접근이 깨짐."""
    sig = inspect.signature(cxr_dataset.build_cxr_index)
    assert sig.return_annotation is cxr_dataset.IndexResult or \
        sig.return_annotation == "IndexResult", (
            "build_cxr_index 의 반환 타입이 바뀌었다: 호출부(.df 접근)를 함께 고칠 것"
        )
    assert {"df", "dropped"} <= set(cxr_dataset.IndexResult.__dataclass_fields__)


def test_embed_script_unwraps_index_result():
    """run_phase2_embed.py 가 IndexResult 를 .df 로 풀어 쓰는지 확인.

    풀지 않고 바로 DataFrame 처럼 쓰면 len()/컬럼 접근에서 깨짐.
    """
    path = ROOT / "scripts" / "run_phase2_embed.py"
    src = path.read_text(encoding="utf-8")
    assert "build_cxr_index(" in src
    assert ".df" in src, "IndexResult 를 .df 로 풀어 쓰지 않았다"
    # 결과를 바로 df 에 대입하는 잘못된 형태가 남아있지 않은지
    assert "df = build_cxr_index(" not in src, (
        "build_cxr_index 의 반환을 곧바로 df 에 대입했다: IndexResult 다"
    )


# ── 문서가 모든 소스 파일을 언급하는가 ────────────────────────────────
#
# deid.py, modality.py 처럼 결론을 좌우하는 동작(비식별 검증, 영상 결측이 MNAR 이라는 진단)을
# 담은 파일은 독자가 어디를 볼지 알아야 함. 파일을 더하고 문서를 안 고치면 빠지므로 고정함.

_DOC_FILES = ("README.md", "docs/report.md", "docs/lowdose.md", "docs/TROUBLESHOOTING.md", "docs/MODEL_CARD.md", "docs/TRIPOD_REPORT.md")


def _docs_text() -> str:
    root = Path(__file__).resolve().parents[1]
    parts = []
    for rel in _DOC_FILES:
        f = root / rel
        if f.is_file():
            parts.append(f.read_text(encoding="utf-8"))
    return "\n".join(parts)


@pytest.mark.parametrize("sub", ["src", "scripts"])
def test_every_source_file_is_named_in_docs(sub):
    root = Path(__file__).resolve().parents[1]
    text = _docs_text()
    assert text, "문서를 하나도 읽지 못했다"
    files = sorted(p for p in (root / sub).iterdir()
                   if p.suffix in (".py", ".sh") and not p.name.startswith("__"))
    assert files, f"{sub}/ 에 파일이 없다"
    missing = [p.name for p in files if p.name not in text]
    assert not missing, (
        f"{sub}/ 의 다음 파일이 README, docs 어디에도 없다: {', '.join(missing)}. "
        "README 의 코드 구조 블록에 역할 한 줄과 함께 넣을 것")
