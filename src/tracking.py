"""실험 추적: MLflow 로컬 sqlite 백엔드.

왜 MLflow 로컬인가:
  PhysioNet DUA는 "제3자와의 데이터 공유(API 전송, 온라인 플랫폼 사용 포함)"를 금지함(README 의 데이터 공개 방침).
  W&B 등 클라우드 기반 추적 도구는 기본 동작이 실시간 원격 전송이라 이 조항에 걸림.
  오프라인 모드가 있긴 하나 매 실행마다 잊지 않고 켜야 하고, 한 번 전송되면 되돌릴 수 없음.
  MLflow 로컬 백엔드는 원격 전송 경로 자체가 없음.
  아래 `_assert_local_uri()`가 비로컬 URI를 런타임에 차단함(화이트리스트 방식).

왜 파일이 아니라 sqlite인가:
  MLflow 3.x부터 파일 백엔드(./mlruns)는 유지보수 모드가 되어 예외를 던짐.
  sqlite도 로컬 단일 파일이므로 DUA 관점에서는 동일하게 안전함.

왜 커밋 해시를 필수로 남기는가:
  결과 JSON 만으로는 "이 결과가 어느 시점 코드에서 나왔는가"를 추적할 수 없음.
  실험 1건을 코드 1상태로 묶어 둠.

사용:
    from tracking import ExperimentLogger

    with ExperimentLogger("phase1-ehr-baseline", params={...}) as run:
        run.log_split(n_train=..., n_val=..., n_test=..., method="patient")
        run.log_metrics({"auroc": 0.68, "pr_auc": 0.31})
        run.log_subgroup("sex", {"M": {...}, "F": {...}})
"""
from __future__ import annotations

import contextlib
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import mlflow

# MLflow 3.x부터 파일 백엔드(./mlruns)는 유지보수 모드라 예외를 던짐.
# sqlite 백엔드로 감: 여전히 로컬 단일 파일이라 DUA 관점에서 동일하게 안전함.
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = _REPO_ROOT / "experiments" / "mlflow.db"
DEFAULT_ARTIFACTS = _REPO_ROOT / "experiments" / "artifacts"

# 이 프로젝트에서 절대 로깅하면 안 되는 것 (DUA)
_FORBIDDEN_ARTIFACT_SUFFIXES = {
    ".csv", ".parquet", ".dcm", ".npy", ".npz", ".h5", ".hdf5",
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".nii", ".gz",
}

# 허용하는 백엔드는 화이트리스트로 관리함.
# 블랙리스트는 새 원격 스킴이 생기면 뚫리므로, 아는 것만 허용함.
_ALLOWED_URI_PREFIXES = ("file://", "sqlite:///")


def _assert_local_uri(uri: str) -> None:
    """로컬 백엔드가 아니면 차단함 (DUA 방어선).

    화이트리스트 방식: file:// 과 sqlite:/// 만 통과시킴.
    postgresql, mysql은 sqlalchemy URI라 MLflow가 받아들이지만 원격 서버일 수 있어 막음.
    """
    lowered = uri.lower()
    if lowered.startswith(_ALLOWED_URI_PREFIXES):
        return
    raise RuntimeError(
        f"로컬 백엔드가 아닌 tracking URI가 설정되었습니다: {uri}\n"
        "PhysioNet DUA는 제3자 온라인 플랫폼으로의 전송을 금지합니다.\n"
        f"허용: {', '.join(_ALLOWED_URI_PREFIXES)}"
    )


def _git_state() -> dict[str, str]:
    """현재 코드 상태를 기록함. 실험 결과와 코드를 묶음."""
    def _run(*args: str) -> str:
        try:
            out = subprocess.run(
                args, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else ""
        except Exception:
            return ""

    commit = _run("git", "rev-parse", "HEAD")
    if not commit:
        # git 저장소 밖이거나 git 이 없음. 이때 dirty 를 False("깨끗함")로 적으면 기록이 오히려
        # 재현 가능해 보이므로, 알 수 없으면 알 수 없다고 적음.
        warnings.warn("git 커밋을 읽지 못해 코드 상태를 기록하지 못했습니다 (git_commit=unknown).",
                      stacklevel=3)
        return {"git_commit": "unknown", "git_commit_short": "unknown",
                "git_branch": "unknown", "git_dirty": "unknown"}
    dirty = bool(_run("git", "status", "--porcelain"))
    if dirty:
        warnings.warn(
            "커밋되지 않은 변경이 있는 상태로 실험을 기록합니다. "
            "이 실험은 정확히 재현할 수 없습니다 (git_dirty=True).",
            stacklevel=3,
        )
    return {
        "git_commit": commit or "unknown",
        "git_commit_short": commit[:8] if commit else "unknown",
        "git_branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": str(dirty),
    }


def _env_state() -> dict[str, str]:
    """재현에 필요한 환경 정보."""
    info: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}"
    except Exception:
        # torch 가 없거나 GPU 조회가 실패해도 실험을 막지 않음: 여기서 모으는 건
        # 재현용 부가 정보이지 실험의 전제 조건이 아님. 아래 선택적 라이브러리와 같은 취지.
        pass
    # 선택적 라이브러리는 없어도 기록만 건너뜀(실험 자체를 막지 않음).
    for mod in ("numpy", "pandas", "sklearn", "xgboost"):
        with contextlib.suppress(Exception):
            info[mod] = __import__(mod).__version__
    return info


class ExperimentLogger:
    """MLflow run 컨텍스트 매니저.

    Parameters
    ----------
    experiment : str
        실험 그룹명. Phase 단위를 권장 (예: "phase1-ehr-baseline").
    params : dict
        하이퍼파라미터. 모델, lr, batch, seed 등.
    run_name : str, optional
        미지정 시 MLflow가 자동 생성.
    db : Path, optional
        sqlite DB 위치. 기본 `experiments/mlflow.db`.
    """

    def __init__(
        self,
        experiment: str,
        params: dict[str, Any] | None = None,
        run_name: str | None = None,
        db: Path | str | None = None,
    ):
        self.experiment = experiment
        self.params = dict(params or {})
        self.run_name = run_name
        self.db = Path(db) if db else DEFAULT_DB
        self._run = None

    def __enter__(self) -> ExperimentLogger:
        # 환경변수로 원격이 주입됐을 수 있으므로 먼저 검사
        env_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if env_uri:
            _assert_local_uri(env_uri)

        self.db.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{self.db.resolve().as_posix()}"
        _assert_local_uri(uri)
        mlflow.set_tracking_uri(uri)
        # 아티팩트 위치는 experiment 생성 시에만 지정 가능함.
        # 기본값(mlflow-artifacts:)을 쓰면 서버 경유를 시도하므로 로컬 경로를 명시함.
        if mlflow.get_experiment_by_name(self.experiment) is None:
            # 상대경로로 넘기지만 MLflow sqlite 저장소는 실험을 만들 때 이것을
            # 현재 디렉터리 기준 절대경로로 풀어 DB 에 저장함(sqlalchemy_store 의
            # create_experiment 가 resolve_uri_if_local 을 부름). 그래서 DB 는
            # 만든 컴퓨터에서만 씀. 저장소를 옮기면 DB 를 새로 만듦.
            # 이 저장소의 실행 스크립트는 전부 저장소 루트에서 돌리므로 풀리는
            # 위치는 항상 <저장소>/experiments/artifacts 다.
            rel = os.path.relpath(DEFAULT_ARTIFACTS, Path.cwd())
            mlflow.create_experiment(
                self.experiment,
                artifact_location=Path(rel).as_posix(),
            )
        mlflow.set_experiment(self.experiment)

        self._run = mlflow.start_run(run_name=self.run_name)

        # 코드 상태 + 환경을 태그로 (params와 분리해 검색 가능하게)
        mlflow.set_tags(_git_state())
        mlflow.set_tags({f"env.{k}": v for k, v in _env_state().items()})
        if self.params:
            mlflow.log_params(self.params)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # 정리 중에 터진 오류가 원래 오류를 덮으면 안 됨. 몇 시간짜리 학습이
        # 실패했을 때 알아야 하는 것은 MLflow 가 태그를 못 썼다는 사실이 아니라
        # 학습이 왜 죽었는가임. 기록 실패는 경고로 낮추고 원래 예외를 올려보냄.
        try:
            if exc_type is not None:
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error", f"{exc_type.__name__}: {exc}"[:500])
            else:
                mlflow.set_tag("status", "finished")
        except Exception as tag_err:
            warnings.warn(f"MLflow 상태 태그 기록 실패: {tag_err}", stacklevel=2)
        try:
            mlflow.end_run()
        except Exception as end_err:
            warnings.warn(f"MLflow run 종료 실패: {end_err}", stacklevel=2)
        return False  # 예외를 삼키지 않음

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_split(
        self,
        n_train: int,
        n_val: int,
        n_test: int,
        method: str,
        pos_rate: dict[str, float] | None = None,
        n_subjects: dict[str, int] | None = None,
    ) -> None:
        """분할 정보. `method`는 반드시 명시함: V1의 핵심 기록.

        method : "patient" | "image" | "temporal"
            "patient"가 아닌 값을 쓰는 경우는 V1 비교 실험뿐이어야 함.
        """
        if method not in {"patient", "image", "temporal"}:
            raise ValueError(f"method는 patient/image/temporal 중 하나여야 함: {method!r}")
        mlflow.log_params({
            "split_method": method,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        })
        if n_subjects:
            mlflow.log_params({f"n_subjects_{k}": v for k, v in n_subjects.items()})
        if pos_rate:
            mlflow.log_metrics({f"pos_rate_{k}": float(v) for k, v in pos_rate.items()})

    def log_subgroup(self, name: str, results: dict[str, dict[str, float]]) -> None:
        """서브그룹별 지표 (V4). `results = {"M": {"auroc": .., "ece": ..}, ...}`"""
        for group, metrics in results.items():
            safe = str(group).replace("/", "_").replace(" ", "_")
            mlflow.log_metrics({f"{name}.{safe}.{k}": float(v) for k, v in metrics.items()})

    def log_artifact(self, path: Path | str, allow_data: bool = False) -> None:
        """집계 결과물만 로깅함. 환자 데이터, 영상은 차단 (DUA)."""
        p = Path(path)
        if not allow_data:
            suffixes = "".join(p.suffixes).lower()
            if any(suffixes.endswith(s) for s in _FORBIDDEN_ARTIFACT_SUFFIXES):
                raise RuntimeError(
                    f"데이터/영상 파일은 아티팩트로 남기지 않습니다: {p.name}\n"
                    "MIMIC 파생물은 DUA 대상입니다. 집계 지표만 기록하세요.\n"
                    "공개 데이터(PneumoniaMNIST 등)라면 allow_data=True 로 명시하세요."
                )
        mlflow.log_artifact(str(p))

    def note(self, text: str) -> None:
        """실험 메모. 왜 이 설정을 시도했는지 남김."""
        mlflow.set_tag("mlflow.note.content", text)

    @property
    def run_id(self) -> str:
        return self._run.info.run_id if self._run else ""
