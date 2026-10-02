"""데이터 분할: 개체(환자) 단위 분리와 누수 검사.

왜 이 모듈이 따로 있는가
-----------------------
영상이나 슬라이스를 섞어 나누면 같은 환자의 영상이 train 과 test 에 동시에 들어감. 예를 들어
환자당 14장인 CT 슬라이스를 섞어 8:2 로 나누면, 어떤 환자의 14장이 한 장도 train 에 안 들어갈
확률은 0.2^14 ~ 1.6e-10 이라 test 환자는 사실상 모두 train 에도 있음. 그때 나온 성능은 분할
설계의 산물임.

MIMIC-CXR 은 같은 문제가 더 큼. 한 환자가 여러 study, 여러 영상을 갖기 때문임(V1).
그래서 분할은 "실수하지 않도록 조심"이 아니라 틀리면 예외가 나도록 강제함.

원칙
----
1. split 은 반드시 개체(subject_id) 단위. 행/이미지 단위 분할은 V1 비교 실험에서만.
2. 전처리는 train 에만 fit, val/test 는 transform 만.
3. SMOTE 등 리샘플링은 train fold 안에서만 (imblearn Pipeline 사용).
4. 영상 인코더처럼 라벨을 보고 학습한 중간 산출물을 특징으로 쓸 때는, 그
   산출물을 만든 분할과 지금 분할이 서로 어긋나면 안 됨. 어긋나면 인코더가
   외운 라벨이 test 로 흘러 들어감. `subject_folds` + `split_with_fixed_folds`
   로 앞선 분할을 그대로 물려받음.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

__all__ = [
    "LeakageError",
    "SplitResult",
    "assert_no_group_overlap",
    "check_split",
    "patient_kfold",
    "patient_split",
    "split_with_fixed_folds",
    "subject_folds",
]


class LeakageError(AssertionError):
    """개체 단위 분리가 깨졌을 때 발생. 경고가 아니라 예외로 다룸."""


@dataclass
class SplitResult:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {
            "n_train": len(self.train_idx),
            "n_val": len(self.val_idx),
            "n_test": len(self.test_idx),
        }


def assert_no_group_overlap(
    groups: np.ndarray, *index_sets: np.ndarray, names: tuple[str, ...] | None = None
) -> None:
    """분할된 인덱스 집합들 사이에 같은 개체가 섞였는지 검사하고, 있으면 예외.

    이 검사는 모든 분할 직후 반드시 호출함. 그냥 지나가면 누수가 성능 수치에 섞임.
    """
    groups = np.asarray(groups)
    names = names or tuple(f"set{i}" for i in range(len(index_sets)))
    sets = [set(np.unique(groups[np.asarray(idx)]).tolist()) for idx in index_sets]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                sample = sorted(overlap)[:5]
                raise LeakageError(
                    f"개체 단위 분리 위반: {names[i]} 와 {names[j]} 에 같은 개체 "
                    f"{len(overlap)}명이 존재합니다. 예: {sample}"
                )


def patient_split(
    groups: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> SplitResult:
    """개체 단위 train/val/test 분할.

    val_size 는 전체 대비 비율임(train 대비가 아님).
    test 를 먼저 떼고, 남은 것에서 val 을 떼는 순서라 test 는 어떤 튜닝에도 노출되지 않음.
    """
    groups = np.asarray(groups)
    n = len(groups)
    idx_all = np.arange(n)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev_idx, test_idx = next(gss.split(idx_all, groups=groups))

    if val_size > 0:
        # 남은 개발셋 대비 비율로 환산
        rel = val_size / (1.0 - test_size)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel, random_state=seed)
        tr_rel, va_rel = next(gss2.split(dev_idx, groups=groups[dev_idx]))
        train_idx, val_idx = dev_idx[tr_rel], dev_idx[va_rel]
    else:
        train_idx, val_idx = dev_idx, np.array([], dtype=int)

    assert_no_group_overlap(
        groups, train_idx, val_idx, test_idx, names=("train", "val", "test")
    )
    return SplitResult(train_idx, val_idx, test_idx)


def patient_kfold(groups: np.ndarray, n_splits: int = 5):
    """개체 단위 k-fold. (train_idx, val_idx) 를 순서대로 내놓음."""
    groups = np.asarray(groups)
    idx_all = np.arange(len(groups))
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, val_idx in gkf.split(idx_all, groups=groups):
        assert_no_group_overlap(groups, train_idx, val_idx, names=("train", "val"))
        yield train_idx, val_idx


def check_split(
    y: np.ndarray,
    groups: np.ndarray,
    split: SplitResult,
) -> dict[str, float | int]:
    """분할 결과 요약: 개체 수, 양성률, 누수 여부.

    실험 로그에 그대로 남길 수 있는 형태로 반환함.
    분할이 끝날 때마다 이 값을 기록해두면 나중에 "그때 어떻게 나눴더라"를 추적할 수 있음.
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)

    assert_no_group_overlap(
        groups, split.train_idx, split.val_idx, split.test_idx,
        names=("train", "val", "test"),
    )

    out: dict[str, float | int] = dict(split.sizes())
    for name, idx in (
        ("train", split.train_idx), ("val", split.val_idx), ("test", split.test_idx)
    ):
        if len(idx) == 0:
            continue
        out[f"n_subjects_{name}"] = len(np.unique(groups[idx]))
        out[f"pos_rate_{name}"] = float(y[idx].mean())
    return out


def subject_folds(groups: np.ndarray, split: SplitResult) -> dict:
    """분할 결과를 {개체: "train"|"val"|"test"} 표로 바꿈.

    뒤이은 분할이 이 소속을 물려받게 하려고 씀.
    """
    groups = np.asarray(groups)
    out: dict = {}
    for name, idx in (("train", split.train_idx), ("val", split.val_idx),
                      ("test", split.test_idx)):
        for gkey in np.unique(groups[np.asarray(idx, dtype=int)]):
            out[gkey.item() if hasattr(gkey, "item") else gkey] = name
    return out


def split_with_fixed_folds(
    groups: np.ndarray,
    folds: dict,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> SplitResult:
    """이미 소속이 정해진 개체는 그대로 두고, 나머지만 새로 나눔.

    왜 필요한가
    -----------
    영상 인코더는 영상이 있는 입원만으로 학습했고, 그 임베딩을 특징으로 쓰는
    실험은 전체 코호트에서 다시 나눔. 두 분할을 따로 만들면 같은 seed 라도
    개체가 다르게 떨어져, 인코더가 학습한 환자가 다음 실험의 test 로 넘어감.
    임베딩은 그 환자의 라벨을 이미 본 상태이므로 test 성능이 부풀려짐.

    실측(MIMIC-IV v3.1): 전체 분할의 test 에서 영상을 가진
    7,213행 중 5,019행(69.6%)이 인코더의 train 이었음. 그 행들만 보면 픽셀 기여가
    +0.2102 였고, 인코더가 처음 보는 행에서는 -0.0970 이었음. 라벨을 본 적 없는
    공개 인코더(RAD-DINO)로 바꾸면 두 구간의 차이가 사라짐(-0.0176 대 -0.0128).
    부풀림은 인코더가 학습 때 본 환자에서 나옴.
    """
    groups = np.asarray(groups)
    n = len(groups)
    known = np.array([folds.get(x.item() if hasattr(x, "item") else x) for x in groups],
                     dtype=object)
    rest = np.where(known == None)[0]  # noqa: E711  (object 배열이라 is 비교가 안 됨)

    buckets: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    for name in buckets:
        buckets[name].append(np.where(known == name)[0])

    if len(rest):
        sp = patient_split(groups[rest], test_size=test_size, val_size=val_size, seed=seed)
        buckets["train"].append(rest[sp.train_idx])
        buckets["val"].append(rest[sp.val_idx])
        buckets["test"].append(rest[sp.test_idx])

    tr, va, te = (np.sort(np.concatenate(buckets[k])).astype(int)
                  for k in ("train", "val", "test"))
    assert len(tr) + len(va) + len(te) == n, "분할이 전체를 덮지 않는다"
    assert_no_group_overlap(groups, tr, va, te, names=("train", "val", "test"))
    return SplitResult(tr, va, te)
