"""저선량 모사와 입력 격자 검증(사전등록 `docs/prereg_lowdose.md` 3절의 모사 검증).

모사가 틀리면 뒤의 복원과 분류 결과가 전부 엉뚱한 잡음 위에서 나옴. 물리 방향(밝은 곳이 더 시끄럽음),
선량비에 따른 크기, 결정성을 합성 영상으로 고정함.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from lowdose import (
    K_MAIN,
    finding_labels,
    grid518,
    insert_lowdose_noise,
    n0_from_sigma,
    noise_seed,
    to_model_input,
)


def _added_noise_sd(level: int, dose: float, n0: float, k: float = K_MAIN, seed: int = 0) -> float:
    """평탄 영상에 넣은 잡음의 표시 단위 표준편차. 8비트로 되돌린 뒤에 측정함."""
    v = np.full((400, 400), level, dtype=np.uint8)
    out = insert_lowdose_noise(v, dose, n0, k, np.random.default_rng(seed))
    return float(out.astype(float).std())


def test_noise_grows_as_dose_falls_like_inverse_sqrt():
    """평탄 영상에는 원래 잡음이 없으므로 더한 잡음만 남음. 로그 영역 분산은 (1 - r) / (r N) 임."""
    n0 = 4000.0
    sds = {r: _added_noise_sd(128, r, n0) for r in (0.5, 0.25, 0.125)}
    expect = lambda r: math.sqrt((1 - r) / r)  # noqa: E731
    for a, b in ((0.5, 0.25), (0.25, 0.125)):
        assert sds[b] / sds[a] == pytest.approx(expect(b) / expect(a), rel=0.05)


def test_noise_matches_theory_in_display_units():
    n0, r, level = 4000.0, 0.25, 128
    p = level * K_MAIN / 255
    sd_p = math.sqrt((1 - r) / (r * n0 * math.exp(-p)))
    assert _added_noise_sd(level, r, n0) == pytest.approx(sd_p * 255 / K_MAIN, rel=0.05)


def test_bright_regions_get_more_noise_than_dark():
    """밝은 곳(많이 흡수, 광자 적음)이 어두운 곳보다 시끄러워야 함. 픽셀값을 광자 수로 쓰면 반대가 됨."""
    assert _added_noise_sd(200, 0.25, 4000.0) > 2 * _added_noise_sd(50, 0.25, 4000.0)


def test_full_dose_is_identity_and_bad_dose_is_rejected():
    v = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert np.array_equal(insert_lowdose_noise(v, 1.0, 1000.0), v)
    with pytest.raises(ValueError):
        insert_lowdose_noise(v, 0.0, 1000.0)
    with pytest.raises(ValueError):
        insert_lowdose_noise(v, 1.5, 1000.0)


def test_same_seed_same_noise_and_seed_depends_on_every_key():
    v = np.full((64, 64), 100, dtype=np.uint8)
    s = noise_seed("abc", 0.25)
    a = insert_lowdose_noise(v, 0.25, 3000.0, rng=np.random.default_rng(s))
    b = insert_lowdose_noise(v, 0.25, 3000.0, rng=np.random.default_rng(noise_seed("abc", 0.25)))
    assert np.array_equal(a, b)
    assert len({noise_seed("abc", 0.25), noise_seed("abd", 0.25), noise_seed("abc", 0.5),
                noise_seed("abc", 0.25, "r4_k3")}) == 4


def test_n0_from_sigma_reproduces_the_measured_noise():
    """n0_from_sigma 가 낸 N0 로 원 선량 잡음을 다시 계산하면 잰 값이 나와야 함."""
    sigma01, mean_v, k = 0.007, 120.0, K_MAIN
    n0 = n0_from_sigma(sigma01, mean_v, k)
    p = mean_v * k / 255
    back = (255 / k) * math.exp(p / 2) / math.sqrt(n0) / 255
    assert back == pytest.approx(sigma01, rel=1e-9)
    with pytest.raises(ValueError):
        n0_from_sigma(0.0, 100.0)


def test_finding_labels_blank_is_negative_uncertain_is_dropped_or_zero():
    raw = np.array([1, 0, -1, np.nan])
    ign = finding_labels(raw, "ignore")
    assert ign[0] == 1 and ign[1] == 0 and np.isnan(ign[2]) and ign[3] == 0
    assert list(finding_labels(raw, "zeros")) == [1, 0, 0, 0]
    with pytest.raises(ValueError):
        finding_labels(raw, "ones")


@pytest.mark.parametrize("shape", [(3000, 2500), (2500, 3000), (1024, 1024)])
def test_grid518_shape_and_center(shape):
    v = np.zeros(shape, dtype=np.uint8)
    v[shape[0] // 2 - 5: shape[0] // 2 + 5, shape[1] // 2 - 5: shape[1] // 2 + 5] = 255
    g = grid518(v)
    assert g.shape == (518, 518) and g.dtype == np.uint8
    assert g[259, 259] > 200  # 가운데를 자름


def test_model_input_normalization():
    x = to_model_input(np.full((2, 518, 518), 255, dtype=np.uint8))
    assert x.shape == (2, 3, 518, 518) and x.dtype == np.float32
    assert float(x[0, 0, 0, 0]) == pytest.approx((1 - 0.5307) / 0.2583, rel=1e-6)


def test_grid518_matches_the_distributed_processor():
    """격자가 RAD-DINO 배포 프로세서의 출력과 8비트 한 단계 안에서 같아야 함(모델 파일이 있는 곳에서만)."""
    transformers = pytest.importorskip("transformers")
    from PIL import Image
    try:
        proc = transformers.AutoImageProcessor.from_pretrained(
            "microsoft/rad-dino", revision="110cbc18d5133582e320b43d53bf5c44e410c936", local_files_only=True)
    except Exception:
        pytest.skip("RAD-DINO 프로세서 설정이 이 장비에 없다")
    rng = np.random.default_rng(0)
    for shape in ((2544, 3056), (3056, 2544), (2000, 2000)):
        v = (rng.random(shape) * 255).astype(np.uint8)
        ref = proc(images=Image.fromarray(v).convert("RGB"), return_tensors="np")["pixel_values"][0]
        mine = to_model_input(grid518(v))[0]
        assert np.abs(ref - mine).max() <= 1 / 255 / 0.2583 + 1e-5


def test_denoisers_keep_shape_and_range():
    pytest.importorskip("skimage")
    from denoise import classical, psnr

    rng = np.random.default_rng(1)
    clean = np.tile(np.linspace(0.2, 0.8, 64, dtype=np.float32), (64, 1))
    noisy = np.clip(clean + 0.05 * rng.standard_normal(clean.shape).astype(np.float32), 0, 1)
    methods = ["gaussian", "nlm"] + (["bm3d"] if _has("bm3d") else [])
    for m in methods:
        y = classical(m, noisy, 1.0)
        assert y.shape == noisy.shape and y.min() >= 0 and y.max() <= 1
        assert psnr(y, clean) > psnr(noisy, clean)  # 매끈한 영상이면 어느 방법이든 나아져야 함
    assert psnr(clean, clean) == float("inf")
    with pytest.raises(ValueError):
        classical("median", noisy, 1.0)


def test_bm3d_is_single_threaded_and_repeatable():
    # 병렬 작업자 안에서 BM3D 가 CPU 수만큼 스레드를 띄우지 않고, 같은 입력에 같은 결과를 내는지 봄.
    bm3d = pytest.importorskip("bm3d")
    from denoise import classical

    seen = []
    real = bm3d.bm3d

    def spy(*args, **kwargs):
        seen.append(kwargs.get("profile"))
        return real(*args, **kwargs)

    rng = np.random.default_rng(2)
    noisy = np.clip(0.5 + 0.05 * rng.standard_normal((64, 64)), 0, 1).astype(np.float32)
    bm3d.bm3d = spy
    try:
        a = classical("bm3d", noisy, 1.0)
        b = classical("bm3d", noisy, 1.0)
    finally:
        bm3d.bm3d = real
    # 스레드는 호출이 끝나면 닫혀 개수로는 못 잡으므로 넘긴 설정값을 봄.
    assert len(seen) == 2 and all(p is not None and p.num_threads == 1 for p in seen)
    assert np.array_equal(a, b)


def test_wide_grid_keeps_prereg_values_and_separates_outputs():
    # 넓힌 격자(사전등록 이탈, 보조)는 원안 값을 모두 포함하고 양쪽 끝이 더 넓어야 하며,
    # 산출물 이름이 원안과 겹치면 주 분석을 덮어씀.
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_lowdose.py"
    spec = importlib.util.spec_from_file_location("run_lowdose", path)
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    for m, grid in rl.TUNE_GRID.items():
        wide = rl.TUNE_GRID_WIDE[m]
        assert set(grid) <= set(wide) and min(wide) < min(grid) and max(wide) > max(grid)
        assert list(wide) == sorted(wide)
    assert rl.GRIDS["prereg"] == ("", rl.TUNE_GRID)
    assert rl._tagged("tune", "") == "tune.json" and rl._tagged("tune", "wide") == "tune_wide.json"
    # DnCNN: 학습률을 줄일 기회가 조기종료보다 먼저 와야 하고, 줄이다 멈추지 않을 만큼 에폭이 있어야 함.
    cfg = rl.DNCNN
    assert cfg["lr"] <= 1e-4 and cfg["lr_min"] < cfg["lr"] and 0 < cfg["lr_factor"] < 1
    assert cfg["lr_patience"] < cfg["patience"] <= cfg["max_epochs"] // 3


def test_heartbeat_prints_while_work_is_slow(capsys):
    # 한 장이 오래 걸려도 진행 줄이 일정 시간마다 나와야 멈춘 것과 구분됨.
    import importlib.util
    import time
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_lowdose.py"
    spec = importlib.util.spec_from_file_location("run_lowdose", path)
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    with rl._Heartbeat(10, every=0.05) as hb:
        hb.done = 3
        time.sleep(0.3)
    out = capsys.readouterr().out
    assert out.count("(진행 중) 3/10") >= 2


def test_save_json_accepts_numpy_scalars(tmp_path):
    # evaluate 결과에 float32 가 섞여도 저장이 멈추지 않아야 함.
    import importlib.util
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_lowdose.py"
    spec = importlib.util.spec_from_file_location("run_lowdose", path)
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    out = tmp_path / "r.json"
    rl._save_json(out, {"a": np.float32(0.5), "b": [np.int64(3), np.float64(0.25)], "c": np.bool_(True)})
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 0.5, "b": [3, 0.25], "c": True}
    with pytest.raises(TypeError):
        rl._save_json(out, {"x": np.zeros(2)})


def test_dncnn_is_residual_and_keeps_shape():
    torch = pytest.importorskip("torch")
    from denoise import DnCNN

    m = DnCNN(depth=5, width=8)
    x = torch.rand(2, 1, 32, 32)
    assert m(x).shape == x.shape
    torch.nn.init.zeros_(m.body[-1].weight)
    torch.nn.init.zeros_(m.body[-1].bias)
    assert torch.allclose(m(x), x)  # 잡음 예측이 0 이면 입력을 그대로 돌려줌


def test_rad_dino_loads_are_pinned_to_one_revision():
    """RAD-DINO 를 부르는 곳은 모두 같은 판을 넘겨야 함. 판이 없으면 그날의 최신판을 받아 특징값이 달라질 수 있음."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    revs = set()
    for name in ("run_lowdose.py", "run_phase2_raddino.py"):
        src = (root / "scripts" / name).read_text(encoding="utf-8")
        calls = re.findall(r"from_pretrained\(([^)]*)\)", src)
        assert calls, f"{name} 에 from_pretrained 가 없다"
        assert all("revision=MODEL_REVISION" in c for c in calls), f"{name} 에 판 없이 부르는 곳이 있다"
        revs.update(re.findall(r'^MODEL_REVISION = "([0-9a-f]{40})"', src, re.M))
    assert len(revs) == 1, f"두 스크립트의 RAD-DINO 판이 다르다: {revs}"


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False
