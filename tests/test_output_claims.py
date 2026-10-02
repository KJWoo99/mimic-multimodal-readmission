"""산출물 JSON 이 README 주장, 판정 규칙과 어긋나지 않는지 검증.

Phase 1, 1.5, 2 는 GPU, 실데이터가 필요해 단위 테스트로 재실행할 수 없음. 대신
그것들이 남긴 지표 JSON 을 검사해, 수치가 서로 모순되지 않고 판정 규칙이
저장된 값으로부터 다시 유도되는지 확인함.

특히 잡으려는 것: 저장된 요약값이 저장된 원자료와 어긋나는 경우.
그런 어긋남은 에러 없이 리포트 숫자만 틀리게 만듦.

산출물이 없으면 건너뜀.
"""
from __future__ import annotations

import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

OUT = Path(__file__).resolve().parents[1] / "outputs"
P1 = OUT / "phase1_metrics.json"
P1_NORX = OUT / "phase1_metrics_no_rx.json"
P15 = OUT / "phase1_5_metrics.json"
P2 = OUT / "phase2_metrics.json"
P3 = OUT / "phase3_metrics.json"
P4 = OUT / "phase4_metrics.json"
P4B = OUT / "phase4_gradcam.json"
P5 = OUT / "phase5_metrics.json"
P2_RN = OUT / "phase2_metrics_resnet50.json"
P2_EF = OUT / "phase2_metrics_efficientnet_b0.json"
P2C = OUT / "phase2_raddino.json"
P3R = OUT / "phase3_metrics_raddino.json"
P4B_EF = OUT / "phase4_gradcam_efficientnet_b0.json"
P5R = OUT / "phase5_metrics_raddino.json"
LEAK_SYN = OUT / "leakage_synthetic.json"

# run_phase2.py 의 판정선 (자세 기준선 대비 추가분)
LIFT_SAW_IMAGE = 0.05
LIFT_SAW_VIEW = 0.02

# run_phase3.py 의 판정선
LIFT_REAL = 0.02       # Fusion, 픽셀 기여를 인정하는 선
INFLATION_BIG = 0.02   # 분할 방식이 결과를 만들었다고 보는 선

# run_phase4.py / run_phase5.py 의 판정선
ECE_ACCEPTABLE = 0.05
SUBGROUP_GAP_BIG = 0.05
DCA_MIN_RANGE = 0.02
LEAK_INFLATION_BIG = 0.05
TEMPORAL_DROP_BIG = 0.03
ALERT_CAPACITY = 0.10

# 절제 기록 중 이 번호 이상은 "설정을 바꾼 실험"이 아니라 같은 설정을 시드만
# 바꿔 다시 돌린 회차임. 절제 표가 아니라 별도 표에 들어가므로 표 대조에서 뺌
# (scripts/render_experiment_log.py 와 같은 규약).
REPEAT_ID_FROM = 90


# 산출물 JSON 의 키를 그대로 적음. scripts/run_phase5.py 의 라벨과 같아야 함.
SUBTLE_LEAK_KEY = "은근한 누수(퇴원처, 마지막입원)"


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ── Phase 1 ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not P1.exists(), reason="phase1_metrics.json 없음")
def test_phase1_split_subjects_sum_to_cohort():
    """분할별 환자 수 합계가 코호트 전체 환자 수와 같아야 함.

    합계가 더 크면 같은 환자가 여러 분할에 들어간 것임: 누수.
    """
    d = _load(P1)
    total = sum(d["split"][f"n_subjects_{s}"] for s in ("train", "val", "test"))
    assert total == d["cohort"]["n_subjects"], (
        f"분할 환자 합계 {total} != 코호트 {d['cohort']['n_subjects']}: "
        "초과하면 환자가 분할을 가로지른 것이다"
    )


@pytest.mark.skipif(not P1.exists(), reason="phase1_metrics.json 없음")
def test_phase1_split_rows_sum_to_cohort():
    d = _load(P1)
    total = sum(d["split"][f"n_{s}"] for s in ("train", "val", "test"))
    assert total == d["cohort"]["n_admissions"]


@pytest.mark.skipif(not P1.exists(), reason="phase1_metrics.json 없음")
def test_phase1_pr_auc_baseline_equals_positive_rate():
    """PR-AUC 의 무작위 기준선은 정의상 양성률과 같아야 함."""
    t = _load(P1)["test"]
    assert t["pr_auc_baseline"] == pytest.approx(t["pos_rate"], rel=1e-9)
    assert t["n_pos"] / t["n"] == pytest.approx(t["pos_rate"], rel=1e-9)


@pytest.mark.skipif(not P1.exists(), reason="phase1_metrics.json 없음")
def test_phase1_auroc_not_suspiciously_high():
    """AUROC 0.85 이상이면 누수를 먼저 의심하라는 것이 이 프로젝트의 규칙임(docs/report.md 성능 기대치 절)."""
    auroc = _load(P1)["test"]["auroc"]
    assert auroc < 0.85, f"AUROC {auroc:.4f}: 문헌 범위를 크게 넘었다. 누수 점검 필요"
    assert auroc > 0.5, "무작위보다 나빠서는 안 된다"


@pytest.mark.skipif(not P1.exists(), reason="phase1_metrics.json 없음")
def test_phase1_cohort_readmit_rate_matches_counts():
    c = _load(P1)["cohort"]
    assert c["n_positive"] / c["n_admissions"] == pytest.approx(c["readmit_rate"], rel=1e-9)


# ── Phase 1.5 ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not P15.exists(), reason="phase1_5_metrics.json 없음")
def test_phase15_risk_ratio_recomputes_from_rates():
    """위험비가 저장된 두 집단 재입원률의 비와 같아야 함."""
    e = _load(P15)["mnar"]["effect"]
    assert e["risk_ratio"] == pytest.approx(
        e["rate_exposed"] / e["rate_unexposed"], rel=1e-9
    )


@pytest.mark.skipif(not P15.exists(), reason="phase1_5_metrics.json 없음")
def test_phase15_ci_brackets_point_estimate_and_excludes_one():
    """신뢰구간이 점추정을 감싸야 하고, MNAR 판정을 하려면 1 을 제외해야 함."""
    m = _load(P15)["mnar"]
    e = m["effect"]
    assert e["rr_ci_low"] < e["risk_ratio"] < e["rr_ci_high"]
    if "MNAR" in m["verdict"]:
        assert e["rr_ci_low"] > 1.0 or e["rr_ci_high"] < 1.0, (
            "MNAR 이라 주장하려면 위험비 신뢰구간이 1 을 제외해야 한다"
        )


@pytest.mark.skipif(not P15.exists(), reason="phase1_5_metrics.json 없음")
def test_phase15_cxr_rate_matches_counts():
    lk = _load(P15)["linkage"]
    assert lk["n_with_cxr"] / lk["n_admissions"] == pytest.approx(lk["cxr_rate"], rel=1e-9)


# ── Phase 2 ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not P2.exists(), reason="phase2_metrics.json 없음")
def test_phase2_lift_recomputes_and_verdict_follows_rule():
    """추가분이 두 AUROC 차이와 같고, 판정이 그 값에서 규칙대로 나와야 함."""
    d = _load(P2)
    lift = d["image_only"]["auroc"] - d["view_baseline"]["auroc"]
    assert d["lift_over_view"] == pytest.approx(lift, rel=1e-9)

    if lift >= LIFT_SAW_IMAGE:
        expected = "영상을 봤다"
    elif lift < LIFT_SAW_VIEW:
        expected = "자세를 본 것"
    else:
        expected = "판단 보류"
    assert d["verdict"] == expected, (
        f"추가분 {lift:.4f} 인데 판정이 '{d['verdict']}': 규칙상 '{expected}'"
    )


@pytest.mark.skipif(not P2.exists(), reason="phase2_metrics.json 없음")
def test_phase2_subset_readmit_rate_higher_than_cohort():
    """영상 보유군의 재입원률이 전체 코호트보다 높다는 것이 MNAR 발견의 핵심임."""
    s = _load(P2)["subset"]
    assert s["readmit_rate"] > s["cohort_readmit_rate"]


@pytest.mark.skipif(not P2.exists(), reason="phase2_metrics.json 없음")
def test_phase2_split_subjects_sum_to_subset():
    d = _load(P2)
    total = sum(d["split"][f"n_subjects_{s}"] for s in ("train", "val", "test"))
    assert total == d["subset"]["n_subjects"]


@pytest.mark.skipif(not P2.exists(), reason="phase2_metrics.json 없음")
def test_phase2_best_epoch_is_within_history():
    d = _load(P2)
    epochs = [h["epoch"] for h in d["history"]]
    assert d["best_epoch"] in epochs
    best = next(h for h in d["history"] if h["epoch"] == d["best_epoch"])
    assert best["val_metric"] == max(h["val_metric"] for h in d["history"]), (
        "best_epoch 이 val 지표 최댓값 지점이 아니다"
    )


# ── Phase 3 ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not P3.exists(), reason="phase3_metrics.json 없음")
def test_phase3_v1_inflation_recomputes_and_verdict_follows_rule():
    """부풀림이 두 분할의 AUROC 차이와 같고, 판정이 그 값에서 규칙대로 나와야 함."""
    d = _load(P3)
    infl = d["v1_random_split"]["auroc"] - d["v1_patient_split"]["auroc"]
    assert d["v1_inflation_auroc"] == pytest.approx(infl, rel=1e-9)
    expected = ("분할 방식이 결과를 만든다" if infl >= INFLATION_BIG
                else "분할 방식으로 설명되지 않는다")
    assert d["v1_verdict"] == expected


@pytest.mark.skipif(not P3.exists(), reason="phase3_metrics.json 없음")
def test_phase3_v1_signal_ratio_recomputes():
    """부풀림/신호 비는 부풀림을 '무작위 대비 실제 신호'로 나눈 값이어야 함.

    이 비가 1 을 넘으면 분할만 잘못 잡아도 실제 신호보다 큰 양이 만들어진다는 뜻이라,
    사전등록 판정선을 못 넘어도 한계로 적어야 함.
    """
    d = _load(P3)
    signal = d["v1_patient_split"]["auroc"] - 0.5
    assert d["v1_signal_above_chance"] == pytest.approx(signal, rel=1e-9)
    assert d["v1_inflation_to_signal_ratio"] == pytest.approx(
        d["v1_inflation_auroc"] / signal, rel=1e-9)


@pytest.mark.skipif(not P3.exists(), reason="phase3_metrics.json 없음")
def test_phase3_fusion_lift_recomputes_and_verdict_follows_rule():
    d = _load(P3)
    lift = d["fusion_pca"]["auroc"] - d["ehr_only_subset"]["auroc"]
    assert d["fusion_lift_over_ehr"] == pytest.approx(lift, rel=1e-9)
    expected = "영상이 보탠 것" if lift >= LIFT_REAL else "영상이 보태지 않는다"
    assert d["fusion_verdict"] == expected


@pytest.mark.skipif(not P3.exists(), reason="phase3_metrics.json 없음")
def test_phase3_v3_verdict_follows_rule():
    """영상이 메타데이터를 넘었는지는 두 AUROC 와 판정선으로 다시 유도돼야 함."""
    d = _load(P3)
    img = d["image_embedding_only"]["auroc"]
    meta = d["v3_metadata_only"]["auroc"]
    expected = ("메타데이터를 넘지 못함" if img <= meta + LIFT_REAL
                else "픽셀이 메타데이터를 넘음")
    assert d["v3_verdict"] == expected


@pytest.mark.skipif(not P3.exists(), reason="phase3_metrics.json 없음")
def test_phase3_m1a_deltas_recompute():
    """부분집합별 델타가 저장된 AUROC 들로부터 다시 나와야 함.

    특히 `delta_pixels` 는 영상 유무 정보를 양쪽에 똑같이 준 뒤의 차이여야 함.
    EHR 단독과 비교하면 픽셀의 몫에 '영상이 있다는 사실을 안 것'이 섞임.
    """
    d = _load(P3)
    assert d["m1a_by_subset"], "M1a 부분집합 결과가 비어 있다"
    for r in d["m1a_by_subset"]:
        assert r["delta_vs_ehr"] == pytest.approx(
            r["fusion_m1_auroc"] - r["ehr_auroc"], rel=1e-9), r["subset"]
        assert r["delta_pixels"] == pytest.approx(
            r["fusion_m1_auroc"] - r["ehr_plus_hasimage_auroc"], rel=1e-9), r["subset"]


@pytest.mark.skipif(not P3.exists(), reason="phase3_metrics.json 없음")
def test_phase3_m1a_verdicts_use_the_right_subsets():
    """결측 처리 판정은 '영상 없음', 픽셀 기여 판정은 '영상 있음' 에서 나와야 함.

    두 질문이 다르므로 하나의 판정으로 묶으면 안 됨.
    """
    d = _load(P3)
    by = {r["subset"]: r["delta_pixels"] for r in d["m1a_by_subset"]}
    d_missing, d_present = by["영상 없음"], by["영상 있음"]

    assert d["m1a_fallback_verdict"] == (
        "영상 없는 입원에서 무너지지 않는다" if d_missing > -LIFT_REAL
        else "결측 입원에서 성능이 깎인다")

    if d_present >= LIFT_REAL:
        expected = "픽셀이 보탠다"
    elif d_present <= -LIFT_REAL:
        expected = "픽셀이 오히려 깎는다"
    else:
        expected = "픽셀이 보태지도 깎지도 않는다"
    assert d["m1a_pixel_verdict"] == expected


@pytest.mark.skipif(not P3.exists(), reason="phase3_metrics.json 없음")
def test_phase3_pr_auc_baseline_equals_positive_rate():
    """PR-AUC 의 무작위 기준선은 정의상 양성률과 같음: 모든 지표 블록에서."""
    d = _load(P3)
    blocks = ["v3_metadata_only", "image_embedding_only", "v1_patient_split",
              "v1_random_split", "ehr_only_subset", "fusion_pca", "fusion_raw"]
    for b in blocks:
        m = d[b]
        assert m["pr_auc_baseline"] == pytest.approx(m["pos_rate"], rel=1e-9), b


@pytest.mark.skipif(not (P3.exists() and P2.exists()),
                    reason="phase2/3 metrics 없음")
def test_phase3_embeddings_came_from_the_reported_phase2_model():
    """Phase 3 이 쓴 임베딩이 Phase 2 가 보고한 바로 그 체크포인트에서 나와야 함.

    체크포인트를 다시 학습했는데 임베딩을 갱신하지 않으면, Phase 3 의 모든 수치가
    이미 폐기된 인코더의 것이 됨. 에러 없이 어긋나는 종류라 지문으로 막음.
    """
    d3, d2 = _load(P3), _load(P2)
    assert d3["checkpoint_sha256"] == d2["checkpoint"]["sha256"]


P2B = OUT / "phase2_embed_meta.json"

@pytest.mark.skipif(not (P5.exists() and P2.exists()), reason="phase2/5 metrics 없음")
def test_phase5_embeddings_came_from_the_reported_phase2_model():
    """Phase 5 의 M2(촬영 시점 창) 분석도 기록상의 인코더에서 나와야 함.

    M2 수치가 임베딩 위에 얹히므로, 인코더를 바꾼 뒤 임베딩을 다시 안 뽑으면 옛 인코더의
    결과가 그대로 리포트로 감(트러블슈팅 17).

    cnn 인코더로 돌린 결과에만 적용함. raddino 는 별도 파일로 나감.
    """
    d5 = _load(P5)
    if d5.get("encoder") != "cnn":
        pytest.skip("cnn 인코더 결과가 아니다")
    if "checkpoint_sha256" not in d5:
        pytest.fail("phase5_metrics.json 에 checkpoint_sha256 이 없다: "
                    "임베딩 출처를 기록하기 전의 실행이다. Phase 5 를 다시 돌릴 것")
    assert d5["checkpoint_sha256"] == _load(P2)["checkpoint"]["sha256"], (
        "Phase 5 가 쓴 임베딩이 기록상의 Phase 2 체크포인트에서 나오지 않았다")



@pytest.mark.skipif(not P2B.exists(), reason="phase2_embed_meta.json 없음")
def test_embedding_came_from_the_model_of_record():
    """임베딩은 기록상의 모델(`phase2_metrics.json`)에서 뽑아야 함.

    Phase 2 는 `--tag` 로 여러 설정을 나란히 남길 수 있음. 그때 Phase 2-B 가
    꼬리표 없는 옛 파일만 읽으면, 새 설정으로 학습해 놓고 임베딩은 옛 인코더에서
    뽑는 일이 벌어짐. Phase 3, 4, 5 가 전부 그 위에 얹히는데 오류는 안 남.

    지문 대조(위 테스트)로는 못 잡음: 옛 인코더끼리는 지문이 일치하기
    때문임. 그래서 "어느 파일에서 왔는가"를 따로 봄.
    """
    meta = _load(P2B)
    src = meta.get("source_metrics")
    assert src is not None, (
        "phase2_embed_meta.json 에 source_metrics 가 없다: "
        "꼬리표 기능이 붙기 전의 실행이다. run_phase2_embed.py 를 다시 돌릴 것")
    assert src == P2.name, (
        f"임베딩이 {src} 에서 나왔다. 기록상의 모델은 {P2.name} 이다. "
        "Phase 2 의 승자 설정을 꼬리표 없이 한 번 더 돌려 기록 모델로 삼든지, "
        "Phase 2-B 를 그 꼬리표로 다시 돌리든지 해야 한다")
    assert meta["checkpoint_sha256"] == _load(P2)["checkpoint"]["sha256"]


@pytest.mark.skipif(not (P3.exists() and P2.exists()),
                    reason="phase2/3 metrics 없음")
def test_phase3_image_subset_matches_phase2_cohort():
    """영상 보유 입원 수가 Phase 2 의 부분집합 크기와 같아야 함."""
    d3, d2 = _load(P3), _load(P2)
    assert d3["n_image_subset"] == d2["subset"]["n_admissions"]


# ── README 와 산출물의 일치 ────────────────────────────────────────────

README = Path(__file__).resolve().parents[1] / "README.md"
REPORT = README.parent / "docs" / "report.md"
TROUBLE = README.parent / "docs" / "TROUBLESHOOTING.md"
LOWDOSE_DOC = README.parent / "docs" / "lowdose.md"


def _md() -> str:
    """README 와 docs/report.md, docs/TROUBLESHOOTING.md, docs/lowdose.md 를 합친 글.

    Phase 별 결과와 트러블슈팅, 저선량 부가 분석은 README 가 아니라 이 문서들에 있음.
    수치가 어느 문서에 있든 산출물과 같아야 함.
    """
    return "\n".join(p.read_text(encoding="utf-8") for p in (README, REPORT, TROUBLE, LOWDOSE_DOC) if p.exists())


# README 가 인용하는 수치와 그 출처. 재실행으로 값이 바뀌었는데 README 를 고치지
# 않으면 여기서 걸림.
_README_NUMBERS = [
    (P2, ("view_baseline", "auroc"), 4),
    (P2, ("image_only", "auroc"), 4),
    (P2, ("image_only", "ece"), 4),
    (P2, ("ehr_same_subset", "auroc"), 4),
    (P2, ("subgroup_sex", "F", "auroc"), 4),
    (P2, ("subgroup_sex", "M", "auroc"), 4),
    (P3, ("v3_metadata_only", "auroc"), 4),
    (P3, ("image_embedding_only", "auroc"), 4),
    (P3, ("v1_random_split", "auroc"), 4),
    (P3, ("ehr_only_subset", "auroc"), 4),
    (P3, ("fusion_pca", "auroc"), 4),
    (P3, ("fusion_raw", "auroc"), 4),
    (P4, ("main_model", "auroc"), 4),
    (P4, ("main_model", "pr_auc"), 4),
    (P4, ("calibration", "원본", "ece"), 4),
    (P4, ("calibration", "Platt", "ece"), 4),
    (P4, ("calibration", "Isotonic", "ece"), 4),
    (P4, ("v4_subgroups", "연령대", "auroc_gap"), 4),
    (P4, ("v4_subgroups", "보험", "auroc_gap"), 4),
    (P4, ("v4_subgroups_calibrated", "보험", "ece_gap"), 4),
    (P4, ("model_auroc_spread"), 4),
    (P4, ("shap_top15", 0, "mean_abs_shap"), 4),
    (P4B, ("center_mass", "mean"), 4),
    (P4B, ("attention_entropy", "mean"), 4),
    (P4B, ("peak_radius", "mean"), 4),
    (P4B_EF, ("center_mass", "mean"), 4),
    (P4B_EF, ("attention_entropy", "mean"), 4),
    (P4B_EF, ("peak_radius", "mean"), 4),
    # 결론이 뒤집힌 대목이라 특히 낡으면 안 됨: 백본별 영상 성능과
    # 파운데이션 모델 재검증 수치를 전부 등록함.
    (P2_RN, ("image_only", "auroc"), 4),
    (P2_EF, ("image_only", "auroc"), 4),
    # 결론을 뒤집은 바로 그 수치들: 여기가 낡으면 리포트가 틀린 결론을 말함.
    (P2C, ("heads", "PCA 32차원", "auroc"), 4),
    (P2C, ("heads", "선형 프로브", "auroc"), 4),
    (P2C, ("lift_over_view"), 4),
    (P2C, ("view_baseline", "auroc"), 4),
    (P2C, ("ehr_same_subset", "auroc"), 4),
    (P3R, ("image_embedding_only", "auroc"), 4),
    (P3R, ("fusion_pca", "auroc"), 4),
    (P3R, ("fusion_late", "auroc"), 4),
    (P5R, ("m2_window_spread_size_matched"), 4),
    (P5, ("baseline", "auroc"), 4),
    (P5, ("v6_leakage_ab", SUBTLE_LEAK_KEY, "auroc"), 4),
    (P5, ("v6_leakage_ab", "완전 누수(다음입원까지 일수)", "auroc"), 4),
    (P5, ("v7_temporal", "temporal_auroc"), 4),
    (P5, ("v7_temporal", "control_random_auroc_mean"), 4),
    # 누수 자가검증(합성 데이터): 스크립트를 다시 돌려 값이 바뀌면 README 도 같이 고쳐야 함
    (LEAK_SYN, ("leaky", "auc_mean"), 4),
    (LEAK_SYN, ("fixed", "auc_mean"), 4),
    (P5, ("operating_points", 1, "ppv"), 3),
    (P5, ("operating_points", 1, "sensitivity"), 3),
    (P5, ("prevalence"), 3),
]


def _dig(d: dict, path):
    """중첩 dict/list 를 따라 내려감. 정수는 리스트 인덱스로 씀."""
    if isinstance(path, str):
        path = (path,)
    for k in path:
        d = d[k]
    return d


@pytest.mark.skipif(not (README.exists() and P2.exists() and P3.exists()),
                    reason="README 또는 산출물 없음")
@pytest.mark.parametrize("src,path,nd", _README_NUMBERS)
def test_readme_quotes_current_numbers(src, path, nd):
    """README 와 docs 가 인용한 지표가 현재 산출물의 값과 같아야 함."""
    md = _md()
    value = f"{_dig(_load(src), path):.{nd}f}"
    where = path if isinstance(path, str) else ".".join(str(k) for k in path)
    assert value in md, (
        f"{src.name} 의 {where} = {value} 인데 README 에 그 값이 없다: "
        "재실행 후 README 를 갱신하지 않았을 가능성이 높다"
    )


@pytest.mark.skipif(not (README.exists() and P1.exists()), reason="README 또는 phase1 산출물 없음")
def test_readme_sensitivity_table_matches_phase1():
    """코호트 기준 민감도 표의 재입원률 네 줄이 phase1_metrics.json 의 sensitivity 와 같아야 함.

    손으로 옮긴 표는 라벨을 고치면 낡으므로 산출물과 대조함.
    """
    rows = _load(P1).get("sensitivity")
    if not rows:
        pytest.skip("--sensitivity 로 돌린 산출물이 아니다")
    lines = _md().splitlines()
    for r in rows:
        pct = f"{r['readmit_rate'] * 100:.2f}%"
        # 같은 표가 여러 문서에 있을 수 있음. 기준 이름으로 시작하는 줄은 전부 맞아야 함.
        found = [ln for ln in lines if ln.startswith(f"| {r['criteria']}") and "%" in ln]
        assert found, f"문서 민감도 표에 '{r['criteria']}' 줄이 없다"
        wrong = [ln for ln in found if pct not in ln]
        assert not wrong, f"'{r['criteria']}' 은 {pct} 인데 README 에 다른 값이 있다: {wrong}"


@pytest.mark.skipif(not (README.exists() and P1.exists() and P1_NORX.exists()),
                    reason="README 또는 투약 제외 산출물 없음")
def test_readme_quotes_medication_ablation():
    """투약 테이블을 뺀 실행의 AUROC, 빠진 피처 수, 차이가 README 문장과 같아야 함."""
    d, dn = _load(P1), _load(P1_NORX)
    md = _md()
    full, norx = d["test"]["auroc"], dn["test"]["auroc"]
    assert d["split"] == dn["split"], "투약 제외 실행이 다른 분할에서 돌았다"
    for want in (f"{full:.4f} 에서 {norx:.4f}", f"{full - norx:.4f} 만큼",
                 f"투약 피처 {d['n_features'] - dn['n_features']}개"):
        assert want in md, f"README 에 '{want}' 가 없다"


# ── Phase 4 ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not P4.exists(), reason="phase4_metrics.json 없음")
def test_phase4_calibration_best_is_chosen_on_val_not_test():
    """'최선' 보정법은 val 5겹 밖 예측 ECE 로 고름. test ECE 로 고르면 선택이 test 에
    기대 보고 ECE 가 낙관적이 됨.
    판정 문장은 고른 방식의 test ECE 로 씀."""
    d = _load(P4)
    cal = d["calibration"]
    sel = d["calibration_selection"]["val_oof_ece"]
    assert set(sel) == set(cal)
    best = min(sel, key=sel.get)
    assert d["calibration_best"] == best
    expected = ("확률로 읽어도 된다" if cal[best]["ece"] < ECE_ACCEPTABLE
                else "확률로 읽으면 안 된다")
    assert d["calibration_verdict"] == expected


@pytest.mark.skipif(not P4.exists(), reason="phase4_metrics.json 없음")
def test_phase4_calibration_preserves_ranking():
    """보정은 확률만 고치고 순위는 건드리지 않아야 함: AUROC 가 그대로여야 함.

    보정 후 AUROC 가 눈에 띄게 움직였다면 그건 보정이 아니라 다른 모델임.
    """
    d = _load(P4)
    cal = d["calibration"]
    base = cal["원본"]["auroc"]
    for name, v in cal.items():
        assert abs(v["auroc"] - base) < 0.01, (
            f"{name} 보정이 AUROC 를 {v['auroc'] - base:+.4f} 움직였다")


@pytest.mark.skipif(not P4.exists(), reason="phase4_metrics.json 없음")
def test_phase4_subgroup_gap_recomputes_and_verdict_follows_rule():
    """서브그룹 격차가 저장된 그룹별 AUROC 로부터 다시 나와야 함.

    표본 미달 그룹은 격차 계산에서 빠짐: 31건짜리 그룹의 AUROC 로 '차이 있음'
    을 선언하면 잡음을 발견으로 착각하게 됨.
    """
    d = _load(P4)
    for axis, blk in d["v4_subgroups"].items():
        big = [gg["auroc"] for k, gg in blk["groups"].items()
               if k not in blk["small_groups"]]
        if len(big) < 2:
            continue
        assert blk["auroc_gap"] == pytest.approx(max(big) - min(big), rel=1e-9), axis
        expected = ("집단 간 차이 있음" if blk["auroc_gap"] >= SUBGROUP_GAP_BIG
                    else "집단 간 차이 없음")
        assert blk["verdict"] == expected, axis


@pytest.mark.skipif(not P4.exists(), reason="phase4_metrics.json 없음")
def test_phase4_calibrated_subgroup_verdict_follows_rule():
    """전체 보정이 좋아도 집단별로 어긋날 수 있음: 판정이 최대 격차에서 나와야 함."""
    d = _load(P4)
    gaps = [v["ece_gap"] for v in d["v4_subgroups_calibrated"].values()]
    assert d["v4_calibrated_ece_gap_max"] == pytest.approx(max(gaps), rel=1e-9)
    expected = ("모든 집단에서 확률을 믿어도 된다"
                if max(gaps) < ECE_ACCEPTABLE else "일부 집단에서 확률이 어긋난다")
    assert d["v4_calibrated_verdict"] == expected


@pytest.mark.skipif(not P4.exists(), reason="phase4_metrics.json 없음")
def test_phase4_dca_treat_all_matches_formula():
    """'전부 치료' 순편익은 유병률과 임계값만으로 정해짐: 공식대로여야 함."""
    d = _load(P4)
    prev = d["main_model"]["pos_rate"]
    for pt in d["dca_curve"][::10]:
        t = pt["threshold"]
        expected = prev - (1 - prev) * (t / (1 - t))
        assert pt["treat_all"] == pytest.approx(expected, abs=1e-9), t


@pytest.mark.skipif(not P4.exists(), reason="phase4_metrics.json 없음")
def test_phase4_dca_range_and_verdict_follow_rule():
    """모델이 우세한 구간이 곡선에서 다시 나오고, 판정이 그 폭에서 나와야 함."""
    d = _load(P4)
    useful = [pt["threshold"] for pt in d["dca_curve"]
              if pt["model"] > max(pt["treat_all"], pt["treat_none"])]
    if useful:
        assert d["dca_useful_threshold_range"][0] == pytest.approx(min(useful))
        assert d["dca_useful_threshold_range"][1] == pytest.approx(max(useful))
        width = max(useful) - min(useful)
    else:
        width = 0.0
    assert d["dca_range_width"] == pytest.approx(width, abs=1e-9)
    expected = ("쓸모 있는 임계값 구간이 있다" if width >= DCA_MIN_RANGE
                else "아무것도 안 하는 것보다 나은 구간이 없다")
    assert d["dca_verdict"] == expected


@pytest.mark.skipif(not P4.exists(), reason="phase4_metrics.json 없음")
def test_phase4_model_spread_recomputes():
    d = _load(P4)
    a = [v["auroc"] for v in d["model_comparison"].values()]
    assert d["model_auroc_spread"] == pytest.approx(max(a) - min(a), rel=1e-9)


@pytest.mark.skipif(not P4.exists(), reason="phase4_metrics.json 없음")
def test_phase4_shap_is_sorted_descending():
    """상위 기여 목록이 내림차순이어야 함: 정렬이 깨지면 '무엇을 보는가' 가 뒤집힘."""
    d = _load(P4)
    vals = [r["mean_abs_shap"] for r in d["shap_top15"]]
    assert vals == sorted(vals, reverse=True)
    assert all(v >= 0 for v in vals), "SHAP 절대평균은 음수일 수 없다"


# ── Phase 4-B (Grad-CAM) ───────────────────────────────────────────────

@pytest.mark.skipif(not (P4B.exists() and P2.exists()),
                    reason="phase4_gradcam.json 없음")
def test_phase4b_uses_reported_checkpoint():
    assert _load(P4B)["checkpoint_sha256"] == _load(P2)["checkpoint"]["sha256"]


@pytest.mark.skipif(not P4B.exists(), reason="phase4_gradcam.json 없음")
def test_phase4b_center_mass_is_a_proportion():
    """중심부 주의 비율은 확률질량의 부분합이므로 0~1 이어야 함."""
    d = _load(P4B)
    cm = d["center_mass"]
    assert 0.0 <= cm["mean"] <= 1.0
    assert cm["p25"] <= cm["p75"]
    assert 0.0 <= d["attention_entropy"]["mean"] <= 1.0
    for v in d["by_view"].values():
        assert 0.0 <= v["center_mass"] <= 1.0


# ── Phase 5 ────────────────────────────────────────────────────────────

@pytest.mark.skipif(not P5.exists(), reason="phase5_metrics.json 없음")
def test_phase5_leak_inflation_recomputes_and_verdict_follows_rule():
    """누수 부풀림이 기준 모델 대비 차이와 같고, 판정이 최대값에서 나와야 함."""
    d = _load(P5)
    base = d["baseline"]["auroc"]
    for name, v in d["v6_leakage_ab"].items():
        assert v["inflation_auroc"] == pytest.approx(v["auroc"] - base, rel=1e-9), name
    worst = max(v["inflation_auroc"] for v in d["v6_leakage_ab"].values())
    assert d["v6_max_inflation"] == pytest.approx(worst, rel=1e-9)
    expected = ("빼는 것이 필수였다" if worst >= LEAK_INFLATION_BIG
                else "빼지 않아도 큰 차이는 없었다")
    assert d["v6_verdict"] == expected


@pytest.mark.skipif(not P5.exists(), reason="phase5_metrics.json 없음")
def test_phase5_full_leak_is_near_perfect():
    """라벨 정의에 쓰인 변수를 넣으면 거의 완벽해져야 함.

    그렇지 않다면 '완전 누수' 라 이름 붙인 변수가 실제로는 라벨과 무관하다는 뜻이라,
    이 A/B 가 증명하려던 것을 증명하지 못한 것임.
    """
    d = _load(P5)
    full = [v for k, v in d["v6_leakage_ab"].items() if "완전 누수" in k]
    if full:
        assert full[0]["auroc"] > 0.95, full[0]["auroc"]


@pytest.mark.skipif(not P5.exists(), reason="phase5_metrics.json 없음")
def test_phase5_temporal_drop_recomputes_and_verdict_follows_rule():
    d = _load(P5)
    t = d.get("v7_temporal") or {}
    if not t:
        pytest.skip("시간 분할을 수행하지 않음")
    # 대조군은 연도를 무시한 같은 크기 무작위 분할의 평균이어야 함.
    assert t["control_random_auroc_mean"] == pytest.approx(
        sum(t["control_random_aurocs"]) / len(t["control_random_aurocs"]), rel=1e-9)
    assert t["drop"] == pytest.approx(
        t["control_random_auroc_mean"] - t["temporal_auroc"], rel=1e-9)
    expected = "시간에 취약" if t["drop"] >= TEMPORAL_DROP_BIG else "시간 이동에 견딘다"
    assert d["v7_verdict"] == expected


@pytest.mark.skipif(not P5.exists(), reason="phase5_metrics.json 없음")
def test_phase5_operating_points_are_internally_consistent():
    """운영점의 PPV, 민감도, 선별필요수가 서로 맞아야 함.

    알림 비율이 오르면 PPV 는 내리고 민감도는 올라야 함: 위험 순 정렬이
    깨지면 이 단조성이 무너지므로 여기서 잡힘.
    """
    d = _load(P5)
    ops = d["operating_points"]
    for o in ops:
        assert o["nns"] == pytest.approx(1 / o["ppv"], rel=1e-9)
        assert 0 <= o["ppv"] <= 1 and 0 <= o["sensitivity"] <= 1
    for a, b in itertools.pairwise(ops):
        assert a["alert_rate"] < b["alert_rate"]
        assert a["ppv"] >= b["ppv"] - 1e-9, "알림을 늘렸는데 PPV 가 올랐다"
        assert a["sensitivity"] <= b["sensitivity"] + 1e-9, "알림을 늘렸는데 민감도가 내렸다"


@pytest.mark.skipif(not P5.exists(), reason="phase5_metrics.json 없음")
def test_phase5_chosen_operating_point_matches_capacity():
    d = _load(P5)
    assert d["chosen_operating_point"]["alert_rate"] == pytest.approx(ALERT_CAPACITY)
    assert d["chosen_operating_point"]["ppv"] > d["prevalence"], (
        "운영점 PPV 가 유병률보다 낮으면 무작위로 고르는 것만도 못하다")


@pytest.mark.skipif(not P5.exists(), reason="phase5_metrics.json 없음")
def test_phase5_m2_spread_recomputes():
    d = _load(P5)
    if not d.get("m2_windows"):
        pytest.skip("M2 를 수행하지 않음")
    a = [r["auroc"] for r in d["m2_windows"]]
    assert d["m2_window_spread"] == pytest.approx(max(a) - min(a), rel=1e-9)
    # 표본을 맞춘 폭도 저장된 값에서 다시 나와야 하고, 판정은 그쪽으로 해야 함.
    # 맞추지 않은 폭은 창마다 학습 표본이 7배까지 달라 시점이 아니라 데이터 양을 측정함.
    if "auroc_size_matched" in d["m2_windows"][0]:
        am = [r["auroc_size_matched"] for r in d["m2_windows"]]
        assert d["m2_window_spread_size_matched"] == pytest.approx(
            max(am) - min(am), rel=1e-9)
        expected = ("촬영 시점은 결과를 바꾸지 않는다"
                    if d["m2_window_spread_size_matched"] < 0.03
                    else "촬영 시점이 결과를 바꾼다")
        assert d["m2_verdict"] == expected
        for r in d["m2_windows"]:
            assert r["n_train_matched"] <= r["n_train"]


# ── 대시보드 신선도 ────────────────────────────────────────────────────

DASHBOARD = OUT / "dashboard.html"


@pytest.mark.skipif(not (DASHBOARD.exists() and P5.exists() and P3.exists()),
                    reason="dashboard.html 또는 산출물 없음")
def test_dashboard_is_not_stale():
    """대시보드가 현재 산출물의 값을 보여줘야 함.

    대시보드는 JSON 을 다시 읽어 그리므로, 생성 순서가 저장보다 앞서면 직전
    실행의 값을 담은 채로 남음. 에러가 나지 않으므로 여기서 대조함.
    """
    html = DASHBOARD.read_text(encoding="utf-8")
    d5, d3 = _load(P5), _load(P3)
    checks = [("V7 시간분할 차이", f"{d5['v7_temporal']['drop']:+.4f}"),
              ("V6 최대 부풀림", f"{d5['v6_max_inflation']:+.4f}"),
              ("V1 부풀림", f"{d3['v1_inflation_auroc']:+.4f}")]
    for label, value in checks:
        assert value in html, (
            f"대시보드에 {label} 의 현재 값 {value} 이 없다: "
            "산출물 저장보다 먼저 그려져 한 실행 뒤처졌을 가능성이 높다")


# ── 확인 실행이 절제실험과 같은 조건인가 ───────────────────────────────

P2_ABL = OUT / "phase2_metrics_ablated.json"
ABL_DIR = OUT / "ablation"


def _ablation_winner() -> dict | None:
    """절제실험 기록 중 val 이 가장 높은 회차.

    승자를 파일 이름으로 박아 두면 그 기록이 없어질 때 테스트가 건너뛰어져, 확인 실행이
    승자와 다른 설정으로 돌아도 잡지 못함(트러블슈팅 22). 이름 대신 값에서 뽑음.

    되풀이 회차(시드만 바꾼 것, id >= 90)는 후보에서 뺌: 그건 같은 설정임.
    """
    best = None
    for f in sorted(ABL_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("id", 0) >= 90:
            continue
        v = (d.get("result") or {}).get("best_val_pr_auc")
        if v is None:
            continue
        if best is None or v > best["result"]["best_val_pr_auc"]:
            best = d
    return best


@pytest.mark.skipif(not (P2_ABL.exists() and ABL_DIR.is_dir()
                         and any(ABL_DIR.glob("*.json"))),
                    reason="확인 실행 또는 절제실험 기록 없음")
def test_confirmation_config_matches_ablation():
    """확인 실행의 학습 조건이 절제실험 승자와 같아야 함.

    설정 자체는 어긋나면 안 됨(예: `drop_last=True` 가 한쪽에만 있으면 학습 배치 구성이 달라짐).

    수치의 완전 일치는 요구하지 않음. CUDA 커널이 기본 설정에서 비결정적이라 완전 일치는
    `torch.use_deterministic_algorithms(True)` 없이는 보장되지 않음(트러블슈팅 16). 그래서 아래 `test_confirmation_val_within_seed_spread` 로
    "재현되는가"를 실측된 시드간 변동폭에 견줘 봄.
    """
    conf = _load(P2_ABL)
    abl = _ablation_winner()
    assert abl is not None, "절제실험 기록에서 승자를 찾지 못했다"
    c = abl["config"]
    who = f"절제 승자 #{abl['id']} {abl['name']}"
    assert conf["aug"] == c["aug"], f"{who} 의 aug 는 {c['aug']}, 확인 실행은 {conf['aug']}"
    assert conf["split_seed"] == c["seed"], f"{who} 와 분할 시드가 다르다"
    assert conf["weight_decay"] == c["weight_decay"], f"{who} 와 weight_decay 가 다르다"
    assert conf["freeze_epochs"] == c["freeze_epochs"], (
        f"{who} 의 freeze_epochs 는 {c['freeze_epochs']}, 확인 실행은 {conf['freeze_epochs']}")
    assert conf["img_size"] == c["size"], (
        f"{who} 의 size 는 {c['size']}, 확인 실행은 {conf['img_size']}")


def _confirmation_vals() -> list[float]:
    """확인 실행 시드별 최고 val. 파일이 없으면 빈 목록."""
    out = []
    for p in sorted(OUT.glob("phase2_metrics_ablated*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("aug") == "strong" and d.get("history"):
            out.append(max(h["val_metric"] for h in d["history"]))
    return out


@pytest.mark.skipif(not (ABL_DIR.is_dir() and any(ABL_DIR.glob("*.json"))),
                    reason="절제실험 기록 없음")
def test_confirmation_val_within_seed_spread():
    """절제실험 승자의 val 이 확인 실행 시드들이 만드는 폭 안에 들어와야 함.

    같은 분할, 같은 설정을 초기화 시드만 바꿔 돌린 값들이 얼마나 흩어지는지가
    이 학습의 잡음 하한임. 절제실험의 값이 그 폭 밖으로 나가면 흩어짐이 아니라
    조건이 달라진 것임.

    폭은 값에서 직접 계산함. 상수를 박아두면 재학습할 때 문장만 낡음.
    """
    vals = _confirmation_vals()
    if len(vals) < 3:
        pytest.skip("확인 실행이 3시드 미만이다")
    import statistics as st

    mean, sd = st.mean(vals), st.stdev(vals)
    winner = _ablation_winner()
    assert winner is not None, "절제실험 기록에서 승자를 찾지 못했다"
    abl_val = winner["result"]["best_val_pr_auc"]
    # 3시드로 잰 표준편차라 그 자체가 거칠음. 4배까지 허용함.
    assert abs(abl_val - mean) <= 4 * sd, (
        f"절제실험 val {abl_val:.4f} 이 확인 실행 평균 {mean:.4f} +/- {sd:.4f} 의 "
        f"4배 폭({4 * sd:.4f}) 밖이다. 시드 흩어짐이 아니라 조건이 다르다")


@pytest.mark.skipif(not P2_ABL.exists(), reason="확인 실행 없음")
def test_confirmation_used_the_ablation_winner():
    """확인 실행은 절제실험에서 이긴 설정으로 돌려야 함."""
    assert _load(P2_ABL)["aug"] == "strong"


README = OUT.parent / "README.md"


@pytest.mark.skipif(not ABL_DIR.exists(), reason="절제실험 기록 없음")
def test_readme_ablation_table_matches_records():
    """문서의 절제실험 표가 기록과 어긋나면 실패시킴.

    표는 손으로 적은 것이라 실험이 하나 늘거나 다시 돌면 낡음. 낡은 요약을 믿으면 틀린 결론을
    적게 되므로(트러블슈팅 11), 사람이 눈으로 맞추는 대신 대조를 코드로 둠.
    """
    text = _md()
    recs = {}
    for p in ABL_DIR.glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        recs[r["id"]] = r
    assert recs, "절제실험 기록이 비어 있다"

    base = recs[1]["result"]["best_val_pr_auc"]
    missing = []
    for i, r in sorted(recs.items()):
        # 90번대는 설정을 바꾼 실험이 아니라 같은 설정을 시드만 바꿔 다시 돌린
        # 회차임. 절제 표가 아니라 별도 표(시드 재현)에 들어가므로 여기서 빼지
        # 않으면, 표를 올바르게 쓴 상태에서도 실패함.
        if i >= REPEAT_ID_FROM:
            continue
        res = r["result"]
        cells = [f"{res['best_val_pr_auc']:.4f}",
                 f"{res['best_epoch']}/{res['epochs_run']}"]
        if i != 1:
            # 문서(report 절제 표)는 유니코드 빼기표(−)를 씀
            cells.append(f"{res['best_val_pr_auc'] - base:+.4f}".replace("-", "−"))
        for c in cells:
            if c not in text:
                missing.append(f"#{i}: {c}")
    assert not missing, (
        "문서 절제실험 표가 기록과 다르다. 없는 값: " + ", ".join(missing))


def _baseline_seed_vals() -> list[float]:
    """기준선(#1)과 시드만 다른 기록(#1, #90, #91 등)의 최고 val."""
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ABL_DIR.glob("*.json"))]
    base = next((d for d in recs if d.get("id") == 1), None)
    if base is None:
        return []

    def key(d):
        return {k: v for k, v in d["config"].items() if k != "seed"}

    return [d["result"]["best_val_pr_auc"] for d in recs
            if key(d) == key(base) and (d.get("result") or {}).get("best_val_pr_auc") is not None]


@pytest.mark.skipif(not (ABL_DIR.is_dir() and any(ABL_DIR.glob("*.json"))),
                    reason="절제실험 기록 없음")
def test_report_threshold_matches_seed_spread():
    """판정선 문장이 기준선 시드 반복의 표준편차와 맞아야 함.

    시드 하나로 얻은 차이를 한계 없이 적으면 과장임. 기준선을 시드만 바꿔 되풀이했으면 report 의
    판정선(표준편차의 2배)이 그 기록에서 나와야 하고, 한 번뿐이면 단일 시드 한계 문장을 요구함.
    표준편차는 표본 표준편차(n-1)임. 상수를 박아 두면 재학습할 때 문장만 낡으므로 값에서 계산함.
    """
    vals = _baseline_seed_vals()
    if len(vals) < 2:
        text = _md()
        assert "seed 42" in text or "시드 42" in text
        assert "PR-AUC 쪽 변동폭은" in text and "재지 않았다" in text, (
            "AUROC 스케일 변동폭(0.012)을 PR-AUC 델타에 갖다 쓰지 않았다는 단서가 "
            "문서에 없다")
        return
    import statistics as st

    sd = st.stdev(vals)
    # 줄바꿈 위치가 바뀌어도 문구가 끊기지 않게 공백을 펴서 찾음.
    text = " ".join(REPORT.read_text(encoding="utf-8").split())
    want = [f"판정선 {2 * sd:.4f} 은 기준선을 시드만 바꿔", f"표준편차({sd:.4f})의 2배"]
    missing = [w for w in want if w not in text]
    assert not missing, (
        f"기준선 {len(vals)}시드의 표준편차 {sd:.4f}, 판정선 {2 * sd:.4f} 이 report 와 다르다. "
        "없는 문구: " + ", ".join(missing))


MODEL_CARD = OUT.parent / "docs" / "MODEL_CARD.md"
P4 = OUT / "phase4_metrics.json"


@pytest.mark.skipif(not (MODEL_CARD.exists() and P4.exists()),
                    reason="모델카드 또는 Phase 4 산출물 없음")
def test_model_card_numbers_match_outputs():
    """모델카드의 주요 수치가 산출물과 같아야 함.

    모델카드는 손으로 쓰는 문서라 재학습하면 낡음. README 와 서로 다른 말을 하지 않게 대조함.
    """
    text = MODEL_CARD.read_text(encoding="utf-8")
    d4 = _load(P4)
    want = {
        "주 모델 AUROC": f"{d4['main_model']['auroc']:.4f}",
        "원본 ECE": f"{d4['calibration']['원본']['ece']:.4f}",
        "Platt ECE": f"{d4['calibration']['Platt']['ece']:.4f}",
        "집단별 ECE 최대격차": f"{d4['v4_calibrated_ece_gap_max']:.4f}",
    }
    for axis in ("성별", "연령대", "보험"):
        for g, v in d4["v4_subgroups_calibrated"][axis]["groups"].items():
            want[f"{axis}/{g} AUROC"] = f"{v['auroc']:.4f}"
    missing = [f"{k}={v}" for k, v in want.items() if v not in text]
    assert not missing, "모델카드에 없는 산출물 수치: " + ", ".join(missing)


# 학습이 실패했던 DenseNet 실행의 수치. 그 산출물은 재학습이 덮어써 남아 있지 않고,
# 경위는 문서에만 있음(MODEL_CARD.md 의 "이 표는 한 번 뒤집힌 결론이다",
# README.md 의 "학습이 실패한 인코더(best epoch 1)"). 그래서 기준을 상수로 고정함.
FAILED_RUN = {
    "영상 AUROC": "0.5037",
    "V3 영상": "0.5115",
}



@pytest.mark.skipif(not MODEL_CARD.exists(), reason="모델카드 없음")
def test_model_card_does_not_cite_the_failed_run():
    """뒤집힌 결론의 수치를 현행 성능으로 인용하면 안 됨.

    학습이 실패한 DenseNet 실행(best epoch 1, verdict "자세를 본 것")의 수치를
    누가 옮겨 적는 경로를 막음. 경위를 설명하는 문단 안에서의 인용은 허용함.
    """
    text = MODEL_CARD.read_text(encoding="utf-8")
    banned = dict(FAILED_RUN)
    # 인용을 아예 금지하지는 않음. 경위를 설명하는 문단은 있어야 함.
    found = [f"{k}={v}" for k, v in banned.items()
             if v in text and "뒤집" not in text[max(0, text.find(v) - 600):text.find(v) + 200]]
    assert not found, (
        "학습 실패 실행의 수치가 경위 설명 없이 인용돼 있다: " + ", ".join(found))
    assert "학습이 실패한 상태" in text or "학습이 실패" in text, (
        "뒤집힌 결론의 경위가 모델카드에 없다")


TRIPOD = OUT.parent / "docs" / "TRIPOD_REPORT.md"


@pytest.mark.skipif(not (TRIPOD.exists() and P4.exists() and P5.exists()),
                    reason="TRIPOD 또는 산출물 없음")
def test_tripod_numbers_match_outputs():
    """TRIPOD 보고서의 수치가 산출물과 같아야 함.

    README 와 규제 보고서가 서로 다른 단계를 말하지 않게 대조함.
    """
    text = TRIPOD.read_text(encoding="utf-8")
    d4, d5 = _load(P4), _load(P5)
    want = {
        "주 모델 AUROC": f"{d4['main_model']['auroc']:.4f}",
        "Platt ECE": f"{d4['calibration']['Platt']['ece']:.4f}",
        "은근한 누수 AUROC": f"{d5['v6_leakage_ab'][SUBTLE_LEAK_KEY]['auroc']:.4f}",
        "완전 누수 AUROC": f"{d5['v6_leakage_ab']['완전 누수(다음입원까지 일수)']['auroc']:.4f}",
        "운영점 PPV": f"{d5['chosen_operating_point']['ppv']:.4f}",
        "운영점 민감도": f"{d5['chosen_operating_point']['sensitivity']:.4f}",
        "시간분할 하락폭": f"{d5['v7_temporal']['drop']:.4f}",
    }
    missing = [f"{k}={v}" for k, v in want.items() if v not in text]
    assert not missing, "TRIPOD 에 없는 산출물 수치: " + ", ".join(missing)


ADM_PROFILE = OUT / "admission_profile.json"


@pytest.mark.skipif(not (TRIPOD.exists() and MODEL_CARD.exists() and ADM_PROFILE.exists()),
                    reason="TRIPOD, 모델카드 또는 입원 분포 산출물 없음")
def test_admission_profile_numbers_match_outputs():
    """TRIPOD 20절 표와 26절 한계 3, 모델카드 한계 3 의 입원 분포 값이 admission_profile.json 과 같은가.

    필터 전 전체 입원 기준 값(평균 2.44, 최대 238 등)과 코호트 기준(2.29)은 모집단이 다름.
    문서 전체에서 부분 문자열을 찾으면 두 열이 바뀌거나
    같은 숫자가 다른 절에 다른 뜻으로 있어도 통과하므로, 표는 20절 구간의 행과 열로 대조함.
    """
    d = _load(ADM_PROFILE)
    a, c = d["all_admissions"], d["main_cohort"]
    assert a["n_admissions"] == 546028 and c["n_admissions"] == _load(P1)["cohort"]["n_admissions"]
    # 코호트는 원내 사망과 재원 1일 미만을 제외 기준으로 뺌. 표의 "없음(제외 기준)" 은 이 0 에 기댐.
    assert c["in_hospital_death_pct"] == 0 and c["los_under_1day_pct"] == 0

    tripod = TRIPOD.read_text(encoding="utf-8")
    i = tripod.index("### 20. Participants")
    sec = tripod[i:tripod.index("\n### ", i + 1)]
    rows = {}
    for ln in sec.splitlines():
        cells = [x.strip() for x in ln.strip().strip("|").split("|")]
        if ln.startswith("|") and len(cells) == 3 and not set(cells[1]) <= set("-"):
            rows[cells[0]] = cells[1:]

    def per(s):
        return (f"평균 {s['admissions_per_subject_mean']:.2f} / 중앙값 {s['admissions_per_subject_median']:.0f}"
                f" / 최대 {s['admissions_per_subject_max']}")

    want = {
        "입원 / 환자": [f"{s['n_admissions']:,} / {s['n_subjects']:,}" for s in (a, c)],
        "환자당 입원 수": [per(a), per(c)],
        "1회만 입원한 환자": [f"{s['single_admission_subject_pct']}%" for s in (a, c)],
        "상위 1% 환자가 차지하는 입원": [f"{s['top1pct_subjects_admission_share_pct']}%" for s in (a, c)],
        "원내 사망률": [f"{a['in_hospital_death_pct']}%", "없음(제외 기준)"],
        "재원 1일 미만": [f"{a['los_under_1day_pct']}%", "없음(제외 기준)"],
    }
    # 두 열을 위치로 대조하므로 열 제목도 봄(제목만 바뀌면 값이 맞아도 뜻이 뒤집힘).
    assert rows.get("항목") == ["필터 전 전체 입원", "주 분석 코호트"], rows.get("항목")
    for k, v in want.items():
        assert rows.get(k) == v, f"TRIPOD 20절 '{k}' 행이 {rows.get(k)} 인데 산출물은 {v}"

    # 26절 한계 3 은 분석 코호트 기준이어야 함(필터 전 최대 238 을 코호트 값처럼 쓰지 않음).
    j = tripod.index("3. 행 단위 과대대표")
    lim = " ".join(tripod[j:tripod.index("\n4. ", j)].split())
    assert f"최대 {c['admissions_per_subject_max']}행" in lim and "238" not in lim, lim
    assert f"입원의 {c['top1pct_subjects_admission_share_pct']}% 를" in lim, lim

    card = MODEL_CARD.read_text(encoding="utf-8")
    j = card.index("3. 행 단위 과대대표")
    lim = " ".join(card[j:card.index("\n4. ", j)].split())
    assert (f"주 분석 코호트({c['n_admissions']:,}건)에서 환자당 입원 수가 평균 "
            f"{c['admissions_per_subject_mean']:.2f}, 최대 {c['admissions_per_subject_max']}회이고") in lim, lim
    assert f"입원의 {c['top1pct_subjects_admission_share_pct']}% 를" in lim, lim
    assert (f"(필터 전 전체 입원에서는 평균 {a['admissions_per_subject_mean']:.2f}, 최대 {a['admissions_per_subject_max']}회, "
            f"상위 1% 가 {a['top1pct_subjects_admission_share_pct']}%") in lim, lim


@pytest.mark.skipif(not TRIPOD.exists(), reason="TRIPOD 없음")
def test_tripod_has_no_unfilled_placeholders():
    """`미완` 표시가 남아 있으면 안 됨.

    Phase 0~6 이 끝났으므로 자리표시자는 낡은 것임. 안 한 것은 `미완` 이 아니라
    19절 한계에 이유와 함께 적음.
    """
    text = TRIPOD.read_text(encoding="utf-8")
    assert "미완" not in text, (
        "TRIPOD 에 `미완` 자리표시자가 남아 있다. 안 한 항목은 19절 에 명시할 것")


DOC_FILES = [OUT.parent / "README.md",
             OUT.parent / "docs" / "report.md",
             OUT.parent / "docs" / "lowdose.md",
             OUT.parent / "docs" / "TROUBLESHOOTING.md",
             OUT.parent / "docs" / "MODEL_CARD.md",
             OUT.parent / "docs" / "TRIPOD_REPORT.md"]


def _heading_slugs(text: str) -> set[str]:
    """GitHub 이 제목에서 만드는 앵커를 흉내냄."""
    import re
    out = set()
    for m in re.finditer(r"^#{1,6}\s+(.+)$", text, re.M):
        h = re.sub(r"[^\w\s가-힣-]", "", m.group(1).strip().lower())
        out.add(re.sub(r"\s+", "-", h).strip("-"))
    return out


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.name)
def test_internal_anchors_resolve(path: Path):
    """문서 안 `](#앵커)` 링크가 실제 제목을 가리켜야 함.

    깨진 앵커는 에러를 내지 않고 그냥 아무 데도 안 감. 제목을 고칠 때마다
    경고 없이 끊기므로 사람이 눈으로 볼 게 아님.
    """
    import re
    if not path.exists():
        pytest.skip(f"{path.name} 없음")
    text = path.read_text(encoding="utf-8")
    slugs = _heading_slugs(text)
    broken = [a for a in re.findall(r"\]\(#([^)]+)\)", text) if a not in slugs]
    assert not broken, f"{path.name} 의 깨진 내부 링크: {broken}"


def test_outputs_contain_no_absolute_paths():
    """산출물에 장비별 절대경로가 박히면 안 됨.

    산출물은 저장소에 들어가 다른 컴퓨터에서 읽힘. 거기에 `D:/...` 나
    `/home/사용자/...` 가 들어 있으면 그 경로는 남의 기계에서 존재하지 않음.

    예: MLflow 는 실험을 만들 때 아티팩트 위치를 절대 경로 문자열로 DB 에 저장해서, 그 DB 를
    만든 자리와 여는 자리가 다르면 `file:///D:/...` 가 `/D:` 로 읽혀 `PermissionError` 가 남.
    """
    import re
    pat = re.compile(r'"[A-Za-z]:[\/]|"/home/|"/Users/|file:///[A-Za-z]:')
    bad = []
    for p in OUT.rglob("*.json"):
        if "_before" in str(p):
            continue          # 옛 산출물 보관함은 검사 대상이 아님
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        m = pat.search(text)
        if m:
            bad.append(f"{p.relative_to(OUT)}: {text[m.start():m.start()+50]}")
    assert not bad, "산출물에 절대경로가 있다:\n  " + "\n  ".join(bad[:10])


# ── 누수 자가검증 (합성 데이터) ────────────────────────────────────────

@pytest.mark.skipif(not LEAK_SYN.exists(), reason="leakage_synthetic.json 없음")
def test_leakage_synthetic_inflation_recomputes_and_verdict_follows_rule():
    """부풀림이 두 평균의 차이와 같고, 판정이 저장된 값에서 다시 유도되는가.

    신호가 0인 데이터이므로 분리 뒤 전처리는 0.5 근처여야 하고, 분리 앞 전처리는
    거의 항상 0.5 를 넘어야 함. 둘 중 하나라도 깨지면 판정 문장이 성립하지 않음.
    """
    d = _load(LEAK_SYN)
    assert d["inflation_auc"] == pytest.approx(
        d["leaky"]["auc_mean"] - d["fixed"]["auc_mean"], abs=1e-9)
    assert abs(d["fixed"]["auc_mean"] - 0.5) < 0.05, "규칙을 지킨 쪽이 0.5 를 중심으로 있지 않다"
    assert d["leaky"]["above_half"] >= d["n_trials"] * 0.8, "누수 쪽의 체계적 부풀림이 없다"
    assert d["verdict"] == "분리 앞 전처리는 신호가 없는 데이터에서도 점수를 올린다"


# ── 부가 분석: 저선량 모사와 복원(outputs/lowdose/) ─────────────────────

LOWDOSE = OUT / "lowdose"


def _readme_section(title: str) -> str:
    # 저선량 절은 docs/lowdose.md 에 있음. 문서 전체가 그 절이라 제목이 있는지만 보고 전체를 돌려줌.
    text = (Path(__file__).resolve().parents[1] / "docs" / "lowdose.md").read_text(encoding="utf-8")
    assert title in text, f"docs/lowdose.md 에 제목이 없음: {title}"
    return text


@pytest.mark.skipif(not (LOWDOSE / "results.json").exists(), reason="lowdose results.json 없음")
def test_readme_lowdose_numbers_match_outputs():
    """docs/lowdose.md 의 수치가 results.json, quality.json, dncnn_seed*.json 에서 그대로 나오는가.

    report 와 README 에 남긴 요약(1/4 하락과 구간)도 같은 값인지 함께 봄.
    """
    sec = _readme_section("# 부가 분석. 저선량 모사와 복원이 소견 분류에 주는 영향")
    r = _load(LOWDOSE / "results.json")
    q = _load(LOWDOSE / "quality.json")["conds"]
    f4 = lambda x: f"{x:.4f}"  # noqa: E731
    ci = lambda c: f"[{f4(c[0])}, {f4(c[1])}]"  # noqa: E731
    clean = r["auroc"]["clean"]
    assert f"{f4(clean['mean'])} {ci(clean['ci'])}" in sec
    for d, label in (("r2", "1/2"), ("r4", "1/4"), ("r8", "1/8"), ("r32", "1/32")):
        v = r["doses"][d]
        assert not v["drop_confirmed"]
        assert f"| {label} | {f4(r['auroc'][d]['mean'])} | {f4(v['drop'])} {ci(v['drop_ci'])} |" in sec
        dn = sum(q[f"dncnn{s}_{d}"]["psnr"] for s in (1, 2, 3)) / 3
        row = [q[d]["psnr"], q[f"gaussian_{d}"]["psnr"], q[f"nlm_{d}"]["psnr"], q[f"bm3d_{d}"]["psnr"], dn]
        assert "| " + label + " | " + " | ".join(f"{x:.2f}" for x in row) + " |" in sec
        wide = [f"{q[f'{m}wide_{d}']['psnr']:.2f}" for m in ("gaussian", "nlm", "bm3d")]
        assert all(w in sec for w in wide)
    r4 = r["doses"]["r4"]
    summary = f"{f4(r4['drop'])} {ci(r4['drop_ci'])}"
    # 줄바꿈 위치가 바뀌어도 수치로만 판정하도록 공백을 하나로 합쳐 찾음.
    flat = lambda p: " ".join(p.read_text(encoding="utf-8").split())  # noqa: E731
    assert f"1/4 하락 {summary}" in flat(REPORT), "report 의 저선량 요약이 산출물과 다름"
    assert f"1/4 하락 {summary}" in flat(README), "README 결과 표의 저선량 행이 산출물과 다름"
    assert "선량비 1/4 에서는 저선량이 분류를 해치지 않았다" in r["primary"]["verdict"]
    assert "회복률은 없음" in sec
    assert f"{r['psnr_up_auroc_down']['count']}건" in sec
    gains = [_load(LOWDOSE / f"dncnn_seed{s}.json")["val_gain_db"] for s in (1, 2, 3)]
    assert ", ".join(f"{g:.2f}" for g in gains) + " dB" in sec


# ── 문서가 말하는 테스트 개수가 실제와 같은가 ──────────────────────────
#
# 테스트를 더하거나 지울 때마다 문장만 낡는 종류라 여기서 고정함.
# 이 테스트가 깨지면 문서의 숫자를 고치라는 뜻임.

_COUNT_RE = re.compile(r"회귀 테스트 (\d+)개")
# 본문에서 "테스트 4개로 고정했다" 처럼 이번에 더한 수를 말하는 문장이 있음.
# 그건 스위트 전체 개수 주장이 아님. 스위트는 수백 개라 작은 수는 걸러냄.
_SUITE_MIN = 50
_COLLECTED_RE = re.compile(r"^tests[/\\]\S+\.py: (\d+)$", re.M)


def _documented_counts() -> list[tuple[str, int]]:
    root = Path(__file__).resolve().parents[1]
    out = []
    for rel in ("README.md", "docs/TRIPOD_REPORT.md"):
        f = root / rel
        if f.is_file():
            for m in _COUNT_RE.finditer(f.read_text(encoding="utf-8")):
                n = int(m.group(1))
                if n >= _SUITE_MIN:
                    out.append((rel, n))
    return out


def test_documented_test_count_matches_collection():
    root = Path(__file__).resolve().parents[1]
    claims = _documented_counts()
    assert claims, "문서에서 '회귀 테스트 N개' 를 하나도 찾지 못했다"

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=root, capture_output=True, text=True, timeout=300,
    )
    # "tests/test_x.py: 12" 꼴의 줄을 더함. 마지막 요약 줄은 형식이 버전마다
    # 달라 쓰지 않음.
    actual = sum(int(m.group(1)) for m in _COLLECTED_RE.finditer(r.stdout))
    assert actual > 0, "수집 개수를 읽지 못했다:\n" + r.stdout[-800:]

    for rel, n in claims:
        assert n == actual, (
            f"{rel} 이 회귀 테스트 {n}개라고 적었는데 실제 수집은 {actual}개다. "
            "테스트를 더했거나 지웠다면 문서의 숫자를 고칠 것")


# ── 문장 속 판정어와 부호 ─────────────────────────────────────
# 위 등록부는 양수 수치가 README 어딘가에 있는지만 봄. 그래서 V3 를 "넘음"으로 쓰거나 RAD-DINO 융합 최고를
# -0.0014 로 쓰거나, "지금 값" 이라며 옛 값을 적어도 통과함. 판정어와 부호, 현재 값 문장을 따로 대조함.

def _signed(x: float, nd: int = 4) -> tuple[str, ...]:
    s = f"{abs(x):.{nd}f}"
    return (f"+{s}",) if x >= 0 else (f"-{s}", f"\u2212{s}")


def _has(md: str, *cands: str) -> bool:
    return any(c in md for c in cands)


@pytest.mark.skipif(not (README.exists() and P3.exists() and P3R.exists()), reason="README 또는 산출물 없음")
def test_readme_v3_verdict_words_match_outputs():
    md = _md()
    row = next(ln for ln in md.splitlines() if ln.startswith("| V3 영상 vs 메타데이터"))
    cnn, rad = [c.strip() for c in row.strip("|").split("|")[1:3]]
    cnn_pass = _load(P3)["v3_verdict"] != "메타데이터를 넘지 못함"
    rad_pass = _load(P3R)["v3_verdict"] != "메타데이터를 넘지 못함"
    assert cnn.endswith("넘음") and (("못 넘음" not in cnn) == cnn_pass), cnn
    assert rad.endswith("넘음") and (("못 넘음" not in rad) == rad_pass), rad


@pytest.mark.skipif(not (README.exists() and P3.exists() and P3R.exists()), reason="README 또는 산출물 없음")
def test_readme_fusion_signs_match_outputs():
    md = _md()
    for src in (P3, P3R):
        lift = _load(src)["fusion_best_lift"]
        assert _has(md, *_signed(lift)), f"{src.name} 융합 최고 {lift:+.4f} 가 README 에 없다"
        wrong = _signed(-lift)
        assert not _has(md, *wrong), f"{src.name} 융합 최고의 부호가 반대로 적힌 곳이 있다: {wrong}"


@pytest.mark.skipif(not (README.exists() and P3.exists() and P4.exists()), reason="README 또는 산출물 없음")
def test_readme_current_value_sentences_match_outputs():
    md = _md()
    m1a = _load(P3)["m1a_pixel_contribution"]["영상 있음"]
    assert any(f"지금 값은 M1a 픽셀 기여 {s}" in md for s in _signed(m1a))
    p4 = _load(P4)
    lgb = p4["model_comparison"]["lightgbm"]["auroc"]
    assert f"{lgb:.4f}, {p4['model_auroc_spread']:.4f} 임" in md   # 트러블슈팅의 "지금 값" 문장


@pytest.mark.skipif(not (README.exists() and P15.exists()), reason="README 또는 phase1.5 산출물 없음")
def test_readme_quotes_adjusted_rr_and_patient_bootstrap():
    md = _md()
    mnar = _load(P15)["mnar"]
    assert f"{mnar['los_adjusted']['mh_risk_ratio']:.3f}" in md
    b = mnar["rr_patient_bootstrap"]
    assert f"[{b['ci_low']:.3f}, {b['ci_high']:.3f}]" in md
    assert b["ci_low"] > 1.0
    # 주 수치(보정 RR)에도 같은 부트스트랩의 구간이 붙어야 함. 구간이 점추정을 품는지도 봄.
    adj = mnar["los_adjusted"]
    ab = adj["patient_bootstrap"]
    assert f"[{ab['ci_low']:.3f}, {ab['ci_high']:.3f}]" in md
    assert ab["ci_low"] < adj["mh_risk_ratio"] < ab["ci_high"] and ab["ci_low"] > 1.0
    assert f"{adj['mh_risk_ratio']:.2f}" in mnar["verdict_adjusted"]


@pytest.mark.skipif(not (README.exists() and P4.exists()), reason="README 또는 phase4 산출물 없음")
def test_readme_explains_the_small_group_behind_the_insurance_gap():
    # 교정 후 보험 축 격차가 한 작은 집단에서 나왔다는 설명과, 그 집단을 뺀 격차를 산출물로 다시 계산해 대조함.
    md = _md()
    groups = _load(P4)["v4_subgroups_calibrated"]["보험"]["groups"]
    small = min(groups, key=lambda g: groups[g]["n"])
    rest = [v["ece"] for g, v in groups.items() if g != small]
    assert f"'{small}' 집단 하나({groups[small]['n']}건" in md
    assert f"{max(rest) - min(rest):.4f}" in md


LABEL_SENS = OUT / "label_sensitivity_cms.json"


@pytest.mark.skipif(not (README.exists() and LABEL_SENS.exists()), reason="README 또는 라벨 민감도 산출물 없음")
def test_readme_quotes_cms_label_sensitivity_from_repo_output():
    # CMS식 라벨 민감도의 문서 수치가 저장소 산출물에서 나오는지 봄.
    md = _md()
    d = _load(LABEL_SENS)
    diff = d["definition_diff"]
    assert f"관찰 입원 {diff['positive_next_is_observation']:,}건" in md
    assert f"전원 {diff['index_transfer_acute_hospital']:,}건" in md
    assert f"자의 퇴원 {diff['index_against_advice']:,}건" in md
    for name in ("main", "obs_both", "cms"):
        r = d["phase1"][name]
        assert f"| {r['n_index']:,} | {r['readmit_rate'] * 100:.2f}% | {r['test_auroc']:.4f} |" in md
    # 퇴원처 DIED 인데 사망 기록이 없는 입원과 그중 30일 안 재입원(원내 사망 판정 규칙의 근거)
    assert diff["discharge_died_with_deathtime_or_flag"] == 0
    assert f"사망 시각도 없는 {diff['discharge_died_without_death_record']}건" in md
    assert f"그중 {diff['discharge_died_next_admission_within_30d']}건은" in md
    ex = d["exclusions"]
    for k in ("obs_excluded_index", "transfer_ama_excluded_index", "same_day_not_counted"):
        assert f"{ex[k]:,}건" in md
    for enc in ("cnn", "raddino"):
        f = d["fusion"][f"{enc}_cms"]
        assert f["verdict"] == "영상이 보태지 않는다" and f["best_lift"] < d["lift_threshold"]
    assert _has(md, *(f"{a} / {b}" for a in _signed(d["fusion"]["cnn_cms"]["best_lift"])
                      for b in _signed(d["fusion"]["raddino_cms"]["best_lift"])))
