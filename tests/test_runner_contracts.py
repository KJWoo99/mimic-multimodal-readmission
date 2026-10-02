"""러너 스크립트가 지켜야 할 규칙을 소스에서 확인함.

두 러너는 GPU 학습을 해야 끝까지 돌아 테스트로 실행할 수 없음. 대신 규칙이 걸린 줄이
코드에 남아 있는지 봄. 되돌리면 이 테스트가 실패함.
"""
from __future__ import annotations

import ast
import contextlib
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _calls(path: Path, func: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == func]


def test_ablation_split_uses_split_seed_not_init_seed():
    """절제 시드 반복(43, 44)이 분할까지 바꾸면 판정선에 분할 변동이 섞이고, 다른 분할의 val 에
    주 분할의 test 환자가 들어감. 분할은 --split-seed(기본 42)로만 정함."""
    calls = _calls(SCRIPTS / "run_ablation.py", "patient_split")
    assert len(calls) == 1
    kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert kw.get("seed") == "args.split_seed"


def test_phase2_ehr_comparison_uses_full_cohort_features():
    """같은 부분집합의 EHR 단독은 Phase 3 와 같은 피처(코호트 전체 행에서 만든 것)여야 함.
    build_cxr_index 의 df(5개 컬럼)로 만들면 입원 유형, 과거 입원 수가 빠진 축소판이 됨."""
    calls = _calls(SCRIPTS / "run_phase2.py", "build_features")
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[0]) == "coh.df"


def test_raddino_ehr_uses_full_cohort_features_and_head_is_chosen_on_val():
    """RAD-DINO 도 Phase 2 와 같은 EHR 피처를 쓰고, 추가분을 낼 헤드는 val AUROC 로 고름.
    test AUROC 최고로 고르면 "영상을 봤다" 쪽으로 낙관적임."""
    path = SCRIPTS / "run_phase2_raddino.py"
    calls = _calls(path, "build_features")
    assert len(calls) == 1 and ast.unparse(calls[0].args[0]) == "coh.df"
    src = path.read_text(encoding="utf-8")
    assert "best_name = max(val_auroc, key=val_auroc.get)" in src


def _module_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            with contextlib.suppress(ValueError):
                out[n.targets[0].id] = ast.literal_eval(n.value)
    return out


def test_raddino_uses_the_same_view_thresholds_as_phase2():
    """RAD-DINO 는 "Phase 2 와 같은 판정선" 이라고 적고 값을 따로 들고 있음. 같아야 함."""
    p2 = _module_constants(SCRIPTS / "run_phase2.py")
    rd = _module_constants(SCRIPTS / "run_phase2_raddino.py")
    assert (rd["LIFT_SAW_IMAGE"], rd["LIFT_SAW_VIEW"]) == (p2["LIFT_PASS"], p2["LIFT_FAIL"])
