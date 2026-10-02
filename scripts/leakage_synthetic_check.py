"""누수 자가검증: 신호가 0인 합성 데이터에서 전처리 순서만 바꿔 봄.

V6(`run_phase5.py`)는 실제 데이터에 누수 컬럼을 넣어 AUROC 가 얼마나 뛰는지 측정함.
그런데 실제 데이터로는 "지금 점수 중 얼마가 누수에서 왔는가"를 잴 수 없음. 누수를 뺀 성능을
모르기 때문임. AUROC 0.70 이 실력인지 누수인지 가를 기준이 없음.

그래서 기준을 만들 수 있는 자리를 하나 둠. feature 와 라벨이 완전히 무관한 난수
데이터에서는 누수가 없는 모델의 AUC 가 0.5 여야 함. 그보다 높은 만큼은
전처리 순서에서 생긴 것임. 이 저장소가 지키는 규칙(환자 단위 분리, 학습셋에서만
fit, 리샘플링은 학습셋에만)이 실제로 그 부풀림을 없애는지 여기서 확인함.

    python scripts/leakage_synthetic_check.py

원본 데이터가 필요 없고 몇 초면 끝남. 결과는 `outputs/leakage_synthetic.json`.
시드 하나로는 우연에 흔들리므로 30회 반복해 평균이 아니라 편향의 '방향'을 봄.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1] / "outputs" / "leakage_synthetic.json"
FEATURES = ["glucose", "spo2", "hba1c", "sbp", "age"]
N_TRIALS = 30
# 판정선: 누수 쪽은 거의 항상 0.5 를 넘어야 하고(체계적 부풀림), 규칙을 지킨 쪽은 0.5 근처여야 함.
LEAKY_ABOVE_HALF_MIN = 0.8
FIXED_MEAN_TOLERANCE = 0.05


def make_synthetic(seed: int, n_patients: int = 300, visits: int = 3) -> pd.DataFrame:
    """feature 와 라벨 사이에 아무 관계가 없는 합성 EHR. 환자당 여러 내원 기록."""
    rng = np.random.default_rng(seed)
    n = n_patients * visits
    df = pd.DataFrame({
        "patient_id": np.repeat(np.arange(n_patients), visits),
        "glucose": rng.normal(110, 30, n),
        "spo2": rng.normal(96, 3, n),
        "hba1c": rng.normal(6.0, 1.2, n),
        "sbp": rng.normal(130, 18, n),
        "age": np.repeat(rng.integers(40, 85, n_patients), visits),
    })
    df.loc[rng.random(n) < 0.1, "hba1c"] = np.nan          # 결측 10%
    label = rng.random(n_patients) < 0.12                   # 양성률 12%, feature 와 무관
    df["readmit"] = np.repeat(label, visits).astype(int)    # 환자별로 고정 -> 같은 환자 기록끼리 라벨이 닮음
    return df


def auc_leaky(df: pd.DataFrame, seed: int) -> float:
    """전처리를 전부 분리 앞에 두고, 행 단위로 나눔. 흔히 보는 잘못된 순서."""
    X, y = df[FEATURES].to_numpy(), df["readmit"].to_numpy()
    X = SimpleImputer(strategy="mean").fit_transform(X)      # 전체로 대치
    X = StandardScaler().fit_transform(X)                    # 전체로 스케일
    X, y = SMOTE(random_state=seed).fit_resample(X, y)       # 전체로 오버샘플링
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))


def auc_fixed(df: pd.DataFrame, seed: int) -> float:
    """이 저장소가 쓰는 순서. 환자 단위 분리 -> 학습셋에서만 fit -> 리샘플링도 학습셋에만."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(df, groups=df["patient_id"]))
    X_tr, X_te = df[FEATURES].iloc[tr], df[FEATURES].iloc[te]
    y_tr, y_te = df["readmit"].iloc[tr], df["readmit"].iloc[te]

    imputer = SimpleImputer(strategy="median").fit(X_tr)
    X_tr, X_te = imputer.transform(X_tr), imputer.transform(X_te)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    X_tr, y_tr = SMOTE(random_state=seed).fit_resample(X_tr, y_tr)   # 평가셋은 건드리지 않음

    model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))


def main() -> int:
    warnings.filterwarnings("ignore")
    leaky = np.array([auc_leaky(make_synthetic(s), s) for s in range(N_TRIALS)])
    fixed = np.array([auc_fixed(make_synthetic(s), s) for s in range(N_TRIALS)])

    result = {
        "what": "feature 와 라벨이 무관한 합성 데이터. 누수가 없는 모델의 AUC 는 0.5 여야 한다",
        "n_trials": N_TRIALS,
        "seeds": [0, N_TRIALS - 1],
        "n_patients": 300,
        "visits_per_patient": 3,
        "positive_rate": 0.12,
        "leaky": {
            "auc_mean": float(leaky.mean()), "auc_std": float(leaky.std()),
            "above_half": int((leaky > 0.5).sum()),
        },
        "fixed": {
            "auc_mean": float(fixed.mean()), "auc_std": float(fixed.std()),
            "above_half": int((fixed > 0.5).sum()),
        },
        "inflation_auc": float(leaky.mean() - fixed.mean()),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    result["verdict"] = ("분리 앞 전처리는 신호가 없는 데이터에서도 점수를 올린다"
                         if (leaky > 0.5).sum() >= N_TRIALS * LEAKY_ABOVE_HALF_MIN
                         and abs(fixed.mean() - 0.5) < FIXED_MEAN_TOLERANCE
                         else "재현되지 않음: 원인을 확인할 것")

    print(f"신호가 0인 데이터: 누수가 없는 모델의 AUC 는 0.5 여야 한다 (시드 {N_TRIALS}회)\n")
    print(f"  분리 전 전처리(누수)  평균 AUC {leaky.mean():.4f} (+/- {leaky.std():.4f})  "
          f"0.5 초과 {(leaky > 0.5).sum()}/{N_TRIALS}회")
    print(f"  분리 후 전처리(규칙)  평균 AUC {fixed.mean():.4f} (+/- {fixed.std():.4f})  "
          f"0.5 초과 {(fixed > 0.5).sum()}/{N_TRIALS}회")
    print(f"  판정: {result['verdict']}")

    # 봐야 할 것은 격차의 크기보다 편향의 방향임. 누수 쪽은 거의 항상 0.5 위로 치우치고,
    # 규칙을 지킨 쪽은 0.5 를 중심으로 위아래로 흩어짐.
    assert (leaky > 0.5).sum() >= N_TRIALS * LEAKY_ABOVE_HALF_MIN, "누수 쪽의 체계적 부풀림이 재현되지 않았다"
    assert abs(fixed.mean() - 0.5) < FIXED_MEAN_TOLERANCE, "규칙을 지킨 쪽이 0.5 를 중심으로 있지 않다"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(Path(__file__).resolve().parents[1])}")
    return 0


if __name__ == "__main__":
    import argparse

    # 인자는 없음. 읽어 두어야 --help 가 설명만 찍고 끝남.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    raise SystemExit(main())
