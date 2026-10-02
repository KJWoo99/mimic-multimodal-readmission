"""학습된 모델 가중치 저장, 복원: 재사용과 증거 보존.

## 왜 필요한가

Phase 2 는 CXR 모델을 학습해 지표만 남기고 가중치는 버림. 두 가지 문제가 있음.

1. 재사용 불가: Phase 3(Fusion), 4(Grad-CAM), 5(윈도우 실험)가 전부 같은 인코더를
   필요로 하는데, 매번 50분씩 재학습해야 함.
2. 증거 부재: "AUROC 0.5422" 가 *어떤* 가중치에서 나왔는지 재현할 방법이 없음.
   TRIPOD+AI 는 모델을 특정할 수 있어야 한다고 요구함(22번, 모델 명세).

## 지문(fingerprint)을 따로 두는 이유

가중치 파일 자체는 MIMIC 파생물이라 `.gitignore` 로 저장소에서 막힘(README 의 데이터 공개 방침).
그래서 SHA-256 해시만 지표 JSON 에 함께 적음. 해시는 데이터가 아니라 16진수
문자열이므로 공개 저장소에 올려도 안전하고, "리포트의 이 숫자는 지문이 `a3f2...` 인
가중치에서 나왔다" 를 증명함. 나중에 같은 체크포인트를 불러올 때 해시를 재검증하므로,
파일이 바뀌었거나 다른 실행의 가중치를 잘못 집어든 경우를 즉시 잡음.

지문은 파일 바이트가 아니라 텐서 값 에서 뽑음(`sha256_of_state`). `torch.save`
는 zip 형식이라 아카이브 안에 파일 이름이 들어가고, 그래서 같은 가중치도 이름이
다르면 파일 해시가 달라짐. 증명하려는 것은 "이 파일" 이 아니라 "이 가중치 값" 이므로
이름, 저장 포맷에 흔들리지 않는 내용 해시를 씀.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "CheckpointInfo", "load_checkpoint", "save_checkpoint",
    "sha256_of", "sha256_of_state",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = _REPO_ROOT / "outputs" / "models"


def _display_path(p: Path) -> str:
    """저장소 안이면 상대경로, 밖이면 절대경로.

    기본 위치(outputs/models)에 저장하면 상대경로가 나와 지표 JSON 에 그대로
    실어도 머신에 무관함. 반면 저장소 밖(임시 폴더, 외부 디스크)을 지정하는
    경우도 있어야 하므로, 그때는 절대경로로 떨어뜨림.
    """
    p = p.resolve()
    try:
        return str(p.relative_to(_REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


@dataclass
class CheckpointInfo:
    """저장된 체크포인트를 가리키는 정보. 그대로 지표 JSON 에 넣어도 안전함.

    가중치 자체는 들어있지 않고 경로, 해시, 메타만 담음.
    """

    path: str            # 저장소 기준 상대경로
    sha256: str          # 가중치 값 의 지문 (파일이 아니라 텐서 내용)
    n_parameters: int
    saved_at: str        # UTC ISO8601
    meta: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_of(path: Path | str, chunk: int = 1 << 20) -> str:
    """파일의 SHA-256. 큰 파일이라 조각내어 읽음."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_of_state(state_dict: dict) -> str:
    """가중치 값 의 SHA-256: 파일이 아니라 모델의 지문.

    `torch.save` 는 zip 형식이라 아카이브 안에 파일 이름이 들어감. 그래서 같은
    가중치라도 다른 이름으로 저장하면 파일 해시가 달라짐. 우리가 증명하려는 것은
    "이 파일" 이 아니라 "이 가중치 값 이 이 수치를 냈다" 이므로, 이름, 저장
    포맷, torch 버전에 흔들리지 않도록 텐서 내용만으로 해시함.

    키를 정렬하고 dtype, shape 까지 넣어, 값이 같아도 구조가 다르면 다른 지문이 되게 함.
    """
    h = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _jsonable(obj: Any) -> Any:
    """dataclass, Path 등을 JSON 에 넣을 수 있는 형태로."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def save_checkpoint(
    state_dict: dict,
    name: str,
    *,
    meta: dict[str, Any] | None = None,
    out_dir: Path | str | None = None,
) -> CheckpointInfo:
    """가중치를 저장하고 지문, 메타를 담은 정보를 돌려줌.

    Parameters
    ----------
    state_dict : 모델 `state_dict()` (CPU 텐서 권장: 로드 환경을 안 가리게)
    name : 파일 이름(확장자 제외). 예: "phase2_cxr_densenet121"
    meta : 함께 남길 설정, 지표. dataclass 도 그대로 넘길 수 있음.
    """
    import torch

    out = Path(out_dir) if out_dir else DEFAULT_DIR
    out.mkdir(parents=True, exist_ok=True)
    weight_path = out / f"{name}.pth"

    # CPU 로 옮겨 저장함: GPU 없는 환경에서도 불러올 수 있어야 함.
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    torch.save(cpu_state, weight_path)

    digest = sha256_of_state(cpu_state)
    n_params = int(sum(v.numel() for v in cpu_state.values()))
    info = CheckpointInfo(
        path=_display_path(weight_path),
        sha256=digest,
        n_parameters=n_params,
        saved_at=datetime.now(UTC).isoformat(timespec="seconds"),
        meta=_jsonable(meta or {}),
    )

    # 사이드카 JSON: 가중치 파일 옆에 메타를 같이 둠. 나중에 이 폴더만 보고도
    # 어떤 실행의 산출물인지 알 수 있어야 하기 때문임.
    (out / f"{name}.json").write_text(
        json.dumps(info.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return info


def load_checkpoint(
    name: str,
    *,
    expect_sha256: str | None = None,
    out_dir: Path | str | None = None,
    map_location: str = "cpu",
) -> tuple[dict, CheckpointInfo]:
    """가중치를 불러옴. 지문이 주어지면 대조해 다른 실행의 것을 집어들지 않게 막음.

    Returns
    -------
    (state_dict, info)
    """
    import torch

    out = Path(out_dir) if out_dir else DEFAULT_DIR
    weight_path = out / f"{name}.pth"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"체크포인트가 없습니다: {weight_path}\n"
            "Phase 2 를 먼저 실행하세요: python scripts/run_phase2.py"
        )

    state = torch.load(weight_path, map_location=map_location)
    digest = sha256_of_state(state)
    if expect_sha256 and digest != expect_sha256:
        raise RuntimeError(
            "체크포인트 지문이 다릅니다: 리포트의 수치를 낸 가중치가 아닙니다.\n"
            f"  기대: {expect_sha256}\n  실제: {digest}\n"
            "Phase 2 를 다시 돌렸다면 지표 JSON 의 sha256 도 갱신되어야 합니다."
        )

    sidecar = out / f"{name}.json"
    if sidecar.exists():
        info = CheckpointInfo(**json.loads(sidecar.read_text(encoding="utf-8")))
    else:
        info = CheckpointInfo(
            path=_display_path(weight_path),
            sha256=digest, n_parameters=-1, saved_at="", meta={},
        )
    return state, info
