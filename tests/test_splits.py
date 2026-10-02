"""개체 단위 분할과 누수 탐지 회귀 테스트.

이 프로젝트에서 가장 중요한 테스트임. 같은 환자가 train 과 test 에 함께 들어가는 분할이
다시 들어오면 여기서 잡혀야 함.
"""
from __future__ import annotations

import numpy as np
import pytest

from splits import (
    LeakageError,
    assert_no_group_overlap,
    check_split,
    patient_kfold,
    patient_split,
    split_with_fixed_folds,
    subject_folds,
)


@pytest.fixture
def multi_admission_groups() -> np.ndarray:
    """환자당 여러 건을 가진 상황(MIMIC 의 실제 형태)."""
    rng = np.random.default_rng(0)
    subs: list[int] = []
    for pid in range(200):
        subs += [pid] * int(rng.integers(1, 6))
    return np.array(subs)


def test_patient_split_no_overlap(multi_admission_groups):
    g = multi_admission_groups
    sp = patient_split(g, test_size=0.2, val_size=0.1, seed=42)
    assert not (set(g[sp.train_idx]) & set(g[sp.test_idx]))
    assert not (set(g[sp.train_idx]) & set(g[sp.val_idx]))
    assert not (set(g[sp.val_idx]) & set(g[sp.test_idx]))


def test_patient_split_covers_all_rows(multi_admission_groups):
    g = multi_admission_groups
    sp = patient_split(g, test_size=0.2, val_size=0.1, seed=42)
    total = len(sp.train_idx) + len(sp.val_idx) + len(sp.test_idx)
    assert total == len(g)
    assert len(set(sp.train_idx) | set(sp.val_idx) | set(sp.test_idx)) == len(g)


def test_patient_split_is_deterministic(multi_admission_groups):
    a = patient_split(multi_admission_groups, seed=7)
    b = patient_split(multi_admission_groups, seed=7)
    assert np.array_equal(a.test_idx, b.test_idx)


def test_row_level_random_split_is_detected(multi_admission_groups):
    """핵심 회귀 테스트: 행 단위로 섞어 나누는 실패 모드.

    행 단위 무작위 분할은 반드시 누수로 탐지되어야 함.
    """
    g = multi_admission_groups
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(g))
    cut = int(len(g) * 0.8)
    with pytest.raises(LeakageError):
        assert_no_group_overlap(g, perm[:cut], perm[cut:], names=("train", "test"))


def test_deliberate_contamination_is_detected(multi_admission_groups):
    g = multi_admission_groups
    sp = patient_split(g, test_size=0.2, val_size=0.1, seed=42)
    contaminated = np.concatenate([sp.train_idx, sp.test_idx[:1]])
    with pytest.raises(LeakageError):
        assert_no_group_overlap(g, contaminated, sp.test_idx, names=("train", "test"))


def test_patient_kfold_no_overlap(multi_admission_groups):
    g = multi_admission_groups
    seen_val: set[int] = set()
    n_folds = 0
    for train_idx, val_idx in patient_kfold(g, n_splits=5):
        assert not (set(g[train_idx]) & set(g[val_idx]))
        seen_val |= set(val_idx.tolist())
        n_folds += 1
    assert n_folds == 5
    # 모든 행이 정확히 한 번씩 검증셋에 등장해야 함
    assert len(seen_val) == len(g)


def test_check_split_reports_subject_counts(multi_admission_groups):
    g = multi_admission_groups
    y = (np.arange(len(g)) % 5 == 0).astype(int)
    sp = patient_split(g, test_size=0.2, val_size=0.1, seed=42)
    info = check_split(y, g, sp)
    assert info["n_subjects_train"] + info["n_subjects_val"] + info["n_subjects_test"] == len(
        np.unique(g)
    )
    for key in ("pos_rate_train", "pos_rate_val", "pos_rate_test"):
        assert 0.0 <= info[key] <= 1.0


def test_val_size_zero(multi_admission_groups):
    sp = patient_split(multi_admission_groups, test_size=0.2, val_size=0.0, seed=3)
    assert len(sp.val_idx) == 0


# --------------------------------------------------------------------------
# 앞선 분할 물려받기: 인코더 누수 회귀
#
# 영상 인코더는 영상 보유 입원만 따로 나눠 그중 train 으로 학습하고, 그 임베딩을 쓰는 실험
# (Phase 3 M1a, Phase 5 M2)은 전체 코호트에서 나눔. 두 분할을 따로 만들면 같은 seed 라도 환자 집합이
# 달라 인코더가 학습한 환자가 그 실험의 test 로 넘어감(트러블슈팅 19).
# --------------------------------------------------------------------------


def _subset_groups():
    """900행 300명. 짝수 번호 환자만 영상을 갖고 있다고 둠."""
    g = np.repeat(np.arange(300), 3)
    sub = np.where(np.isin(g, np.arange(0, 300, 2)))[0]
    return g, sub


def test_fixed_folds_keeps_encoder_train_out_of_test():
    g, sub = _subset_groups()
    sp_sub = patient_split(g[sub], seed=42)
    sp = split_with_fixed_folds(g, subject_folds(g[sub], sp_sub), seed=42)

    enc_train = set(np.unique(g[sub][sp_sub.train_idx]).tolist())
    for name, idx in (("val", sp.val_idx), ("test", sp.test_idx)):
        overlap = enc_train & set(np.unique(g[idx]).tolist())
        assert not overlap, f"인코더 train 환자 {len(overlap)}명이 {name} 에 있다"


def test_fixed_folds_covers_every_row_exactly_once():
    g, sub = _subset_groups()
    sp_sub = patient_split(g[sub], seed=42)
    sp = split_with_fixed_folds(g, subject_folds(g[sub], sp_sub), seed=42)
    allidx = np.concatenate([sp.train_idx, sp.val_idx, sp.test_idx])
    assert len(allidx) == len(g)
    assert len(np.unique(allidx)) == len(g)


def test_fixed_folds_preserves_known_assignment():
    g, sub = _subset_groups()
    sp_sub = patient_split(g[sub], seed=42)
    folds = subject_folds(g[sub], sp_sub)
    sp = split_with_fixed_folds(g, folds, seed=42)
    where = {}
    for name, idx in (("train", sp.train_idx), ("val", sp.val_idx), ("test", sp.test_idx)):
        for i in idx:
            where[int(i)] = name
    for i, gg in enumerate(g):
        if int(gg) in folds:
            assert where[i] == folds[int(gg)], "영상 보유 환자의 소속이 바뀌었다"


def test_naive_resplit_would_have_leaked():
    """전체를 따로 나누는 방식이 실제로 겹치는 것을 남겨 둠.

    이 테스트가 깨진다면 겹침이 사라진 것이므로, 위 함수를 쓸 이유도 다시 봐야 함.
    """
    g, sub = _subset_groups()
    sp_sub = patient_split(g[sub], seed=42)
    sp_naive = patient_split(g, seed=42)          # 전체를 따로 나누는 방식
    enc_train = set(np.unique(g[sub][sp_sub.train_idx]).tolist())
    leaked = enc_train & set(np.unique(g[sp_naive.test_idx]).tolist())
    assert leaked, "예전 방식이 겹치지 않았다면 이 시나리오가 재현되지 않은 것이다"
