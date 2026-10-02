"""git 상태 기록이 모르는 것을 모른다고 적는지 봄.

git 저장소 밖에서 돌려 커밋을 못 읽었는데 git_dirty 를 False(깨끗함)로 적으면, 재현 가능한
기록처럼 보이지만 사실은 코드 상태를 모르는 것임.
"""
from __future__ import annotations

import pytest


def test_outside_a_git_repo_every_field_is_unknown(tmp_path, monkeypatch):
    # 건너뛰기를 함수 안에서 함. 모듈에서 하면 mlflow 가 없는 환경에서 수집 개수가 줄어
    # 문서의 테스트 개수 검사(test_output_claims)가 환경마다 달라짐.
    tracking = pytest.importorskip("tracking")
    monkeypatch.setattr(tracking, "_REPO_ROOT", tmp_path)
    with pytest.warns(UserWarning, match="unknown"):
        state = tracking._git_state()
    assert state == {"git_commit": "unknown", "git_commit_short": "unknown",
                     "git_branch": "unknown", "git_dirty": "unknown"}
