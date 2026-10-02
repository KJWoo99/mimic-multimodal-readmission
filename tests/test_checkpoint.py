"""체크포인트 저장, 복원 검증.

지문(sha256)이 잘못된 가중치를 실제로 잡아내는지 봄. 지문 대조가
느슨하면 Phase 3~5 가 리포트의 수치를 낸 것과 다른 가중치를 경고 없이 집어들 수
있고, 그러면 "이 숫자는 이 모델에서 나왔다"는 주장 자체가 깨짐.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from checkpoint import (
    load_checkpoint,
    save_checkpoint,
    sha256_of,
)


def _toy_state(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return {"w": torch.randn(4, 3), "b": torch.randn(4)}


def test_save_then_load_roundtrip(tmp_path):
    state = _toy_state()
    info = save_checkpoint(state, "toy", meta={"note": "테스트"}, out_dir=tmp_path)

    loaded, loaded_info = load_checkpoint("toy", out_dir=tmp_path)
    for k in state:
        torch.testing.assert_close(loaded[k], state[k])
    assert loaded_info.sha256 == info.sha256
    assert loaded_info.meta["note"] == "테스트"


def test_parameter_count_is_correct(tmp_path):
    state = _toy_state()
    info = save_checkpoint(state, "toy", out_dir=tmp_path)
    # 4*3 + 4 = 16
    assert info.n_parameters == 16


def test_sidecar_json_written_and_parseable(tmp_path):
    save_checkpoint(_toy_state(), "toy", meta={"phase": "2"}, out_dir=tmp_path)
    sidecar = tmp_path / "toy.json"
    assert sidecar.exists()
    d = json.loads(sidecar.read_text(encoding="utf-8"))
    assert d["meta"]["phase"] == "2"
    assert len(d["sha256"]) == 64  # sha256 는 16진수 64자


def test_fingerprint_mismatch_is_rejected(tmp_path):
    """다른 가중치를 같은 이름으로 덮어쓰면 지문 대조가 반드시 막아야 함."""
    info = save_checkpoint(_toy_state(seed=0), "toy", out_dir=tmp_path)
    original_sha = info.sha256

    # 다른 실행의 가중치로 덮어씀 (실수로 재학습한 상황)
    save_checkpoint(_toy_state(seed=999), "toy", out_dir=tmp_path)

    with pytest.raises(RuntimeError, match="지문이 다릅니다"):
        load_checkpoint("toy", expect_sha256=original_sha, out_dir=tmp_path)


def test_matching_fingerprint_passes(tmp_path):
    info = save_checkpoint(_toy_state(), "toy", out_dir=tmp_path)
    loaded, _ = load_checkpoint("toy", expect_sha256=info.sha256, out_dir=tmp_path)
    assert set(loaded) == {"w", "b"}


def test_missing_checkpoint_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"run_phase2\.py"):
        load_checkpoint("does_not_exist", out_dir=tmp_path)


def test_same_weights_give_same_fingerprint(tmp_path):
    """같은 가중치는 항상 같은 지문이어야 재현성 주장이 성립함."""
    a = save_checkpoint(_toy_state(seed=7), "a", out_dir=tmp_path)
    b = save_checkpoint(_toy_state(seed=7), "b", out_dir=tmp_path)
    assert a.sha256 == b.sha256


def test_different_weights_give_different_fingerprint(tmp_path):
    a = save_checkpoint(_toy_state(seed=1), "a", out_dir=tmp_path)
    b = save_checkpoint(_toy_state(seed=2), "b", out_dir=tmp_path)
    assert a.sha256 != b.sha256


def test_saved_tensors_are_cpu_even_if_source_is_cuda(tmp_path):
    """GPU 로 학습해도 CPU 텐서로 저장돼야 GPU 없는 환경에서 불러올 수 있음."""
    state = _toy_state()
    if torch.cuda.is_available():
        state = {k: v.cuda() for k, v in state.items()}
    save_checkpoint(state, "toy", out_dir=tmp_path)
    loaded, _ = load_checkpoint("toy", out_dir=tmp_path)
    assert all(v.device.type == "cpu" for v in loaded.values())


def test_info_is_json_serializable(tmp_path):
    """CheckpointInfo 를 그대로 지표 JSON 에 넣을 수 있어야 함."""
    info = save_checkpoint(_toy_state(), "toy",
                           meta={"cfg": {"lr": 1e-3}, "path": tmp_path},
                           out_dir=tmp_path)
    s = json.dumps(info.as_dict(), ensure_ascii=False)
    assert "sha256" in s


def test_sha256_of_matches_hashlib(tmp_path):
    import hashlib

    p = tmp_path / "x.bin"
    p.write_bytes(b"hello mimic")
    assert sha256_of(p) == hashlib.sha256(b"hello mimic").hexdigest()
