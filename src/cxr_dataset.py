"""CXR 영상 데이터셋 (Phase 2).

입원 하나에 study 가 여러 개 붙을 수 있음(중앙값 1, 최대 97).
예측 시점이 퇴원이므로 재원 중 촬영만 쓰고, 그중 퇴원에 가장 가까운
study 하나를 대표로 삼음. 여러 장을 평균하면 촬영 횟수가 많은
중환자에게 가중이 실려 중증도를 다시 학습하게 됨.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = [
    "AUG_LEVELS",
    "CXRDataset",
    "build_cxr_index",
    "build_transforms",
    "view_baseline",
]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 정면 촬영. AP 는 누워서, PA 는 서서 찍음. 나머지(LATERAL 등)는 보조.
FRONTAL_VIEWS = ("PA", "AP", "AP AXIAL", "PA LLD", "PA RLD")


@dataclass
class IndexResult:
    df: pd.DataFrame
    dropped: dict[str, int]                       # 코호트에서 빠진 건수
    # 빠지지는 않았지만 알고 있어야 하는 사실. "제외"와 섞으면 출력이
    # 오해를 부름: 측면 대체는 탈락이 아니라 대체임.
    flags: dict[str, int] = field(default_factory=dict)


def build_cxr_index(
    cohort: pd.DataFrame,
    cxr_meta: pd.DataFrame,
    image_root: Path,
    one_per_admission: bool = True,
) -> IndexResult:
    """입원 ↔ 영상 파일 경로를 잇음.

    Parameters
    ----------
    one_per_admission : 입원당 study 1개(퇴원에 가장 가까운 것)만 남김.
    """
    from modality import parse_study_datetime

    cxr = cxr_meta.copy()
    if "study_datetime" not in cxr.columns:
        cxr["study_datetime"] = parse_study_datetime(cxr["StudyDate"], cxr["StudyTime"])
    cxr = cxr.dropna(subset=["study_datetime"])
    cxr["is_ap"] = (cxr.get("ViewPosition") == "AP").astype(int)

    need = ["subject_id", "hadm_id", "admittime", "dischtime", "readmit_30d"]
    merged = cohort[need].merge(
        cxr[["subject_id", "study_id", "dicom_id", "study_datetime", "is_ap",
             "ViewPosition"]],
        on="subject_id", how="inner",
    )
    dropped = {"환자 미매칭": len(cohort) - merged.hadm_id.nunique()}

    keep = (merged.study_datetime >= merged.admittime) & (
        merged.study_datetime <= merged.dischtime
    )
    merged = merged[keep].copy()
    merged["hours_before_discharge"] = (
        merged.dischtime - merged.study_datetime
    ).dt.total_seconds() / 3600.0

    # 경로를 먼저 만들고 디스크에 있는 것만 남긴 뒤 대표를 고름.
    #
    # 순서를 뒤집으면(대표 선택 -> 존재 확인) 고른 파일이 없을 때 그 입원이
    # 통째로 탈락함(트러블슈팅 7). 이미 받아 둔 파일 안에서 최선을 고르게 함.
    sid = merged.subject_id.astype("int64").astype(str)
    merged["path"] = [
        image_root / "files" / f"p{a[:2]}" / f"p{a}" / f"s{b}" / f"{c}.jpg"
        for a, b, c in zip(sid, merged.study_id.astype("int64").astype(str),
                           merged.dicom_id.astype(str), strict=True)
    ]
    exists = merged.path.map(lambda p: p.is_file())
    dropped["파일 없음 (개별 영상)"] = int((~exists).sum())
    merged = merged[exists].copy()

    if one_per_admission:
        # 퇴원에 가장 가까운 study, 그 안에서 정면(PA/AP) 우선.
        # 측면은 보조 촬영이라 표준 판독 화면이 아님.
        merged["_not_frontal"] = (~merged.ViewPosition.isin(FRONTAL_VIEWS)).astype(int)
        merged = merged.sort_values(
            ["hadm_id", "hours_before_discharge", "_not_frontal", "dicom_id"]
        ).drop_duplicates("hadm_id", keep="first").drop(columns="_not_frontal")

    merged = merged.reset_index(drop=True)
    # 정면이 없어 측면으로 대체된 입원 수. 모르는 사이에 섞이면 안 되므로 남김.
    flags = {"정면 없어 측면 사용": int((~merged.ViewPosition.isin(FRONTAL_VIEWS)).sum())}
    return IndexResult(merged, dropped, flags)


AUG_LEVELS = ("none", "light", "medium", "strong", "extreme")


def build_transforms(train: bool, size: int = 224, aug: str = "medium"):
    """학습용 증강. `aug` 로 강도를 고름.

    강도는 절제실험에서 바꿔 볼 수 있게 인자로 둠.

    좌우반전은 어느 강도에서도 쓰지 않음. 흉부X선을 뒤집으면 심장이
    오른쪽에 오는 우심증(dextrocardia)처럼 보이는데, 이는 실제로는 드문 소견이고
    추론 시에는 뒤집힌 영상이 절대 들어오지 않음. 학습에만 존재하는
    해부학적으로 불가능한 배치를 넣으면 모델이 실제와 다른 패턴을 배움.
    (자연 영상에서 hflip 이 표준이라 그대로 가져다 쓰기 쉬운 실수임)
    """
    from torchvision import transforms

    if aug not in AUG_LEVELS:
        raise ValueError(f"알 수 없는 증강 강도 {aug!r}. 지원: {AUG_LEVELS}")

    norm = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    eval_tf = [transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size)]

    if not train or aug == "none":
        # aug="none" 은 평가와 완전히 같은 전처리: 증강 효과를 재는 기준선.
        return transforms.Compose([*eval_tf, *norm])

    if aug == "light":
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomRotation(3),
            *norm,
        ])
    if aug == "medium":   # 기본 설정
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.11)),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            *norm,
        ])
    if aug == "strong":
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.5))], p=0.3),
            *norm,
        ])
    # extreme: light -> medium -> strong 이 단조 증가라 그 위를 재보려고 넣음.
    # RandomErasing 은 정규화 후에 와야 함(0 으로 지우는 것이 정규화된 공간에서
    # 평균값을 뜻하도록). 회전을 25도까지 주는 것은 흉부X선에서는 과할 수 있는데,
    # 그것이 성능을 깎는지 보는 것이 이 실험의 목적임.
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.45, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.45, contrast=0.45),
        transforms.RandomApply([transforms.GaussianBlur(7, (0.1, 2.5))], p=0.5),
        *norm,
        transforms.RandomErasing(p=0.35, scale=(0.02, 0.15)),
    ])


class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, size: int = 224,
                 aug: str = "medium"):
        self.paths = df["path"].tolist()
        self.y = df["readmit_30d"].to_numpy(dtype=np.float32)
        self.tf = build_transforms(train, size, aug)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
            x = self.tf(img)
        return x, torch.tensor(self.y[i])


def view_baseline(df: pd.DataFrame, groups: np.ndarray, seed: int = 42) -> dict:
    """픽셀을 보지 않고 촬영 자세만으로 얼마나 맞히는가.

    AP 는 누워서 찍는 자세라 못 일어나는 환자를 뜻함. 영상 모델이 이 값을
    크게 넘지 못하면 폐가 아니라 자세를 본 것임.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = df["readmit_30d"].to_numpy().astype(int)
    if len(np.unique(y)) < 2:
        return {}

    view = df["ViewPosition"].fillna("UNKNOWN").astype(str)
    cats = sorted(view.unique())
    x = np.zeros((len(df), len(cats)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(cats)}
    for i, v in enumerate(view):
        x[i, idx[v]] = 1.0

    uniq = np.unique(groups)
    fold_of = {g: i % 5 for i, g in enumerate(sorted(uniq, key=str))}
    folds = np.array([fold_of[g] for g in groups])
    pred = np.zeros(len(y), dtype=float)
    for f in range(5):
        tr, te = folds != f, folds == f
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            return {}
        m = LogisticRegression(max_iter=1000, class_weight="balanced")
        m.fit(x[tr], y[tr])
        pred[te] = m.predict_proba(x[te])[:, 1]

    return {
        "auroc": float(roc_auc_score(y, pred)),
        "pr_auc": float(average_precision_score(y, pred)),
        "prevalence": float(y.mean()),
        "categories": {c: int((view == c).sum()) for c in cats},
    }
