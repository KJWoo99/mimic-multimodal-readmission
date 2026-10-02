"""Phase 4-B: V5 Grad-CAM 정량화: 모델이 어디를 보는가.

Phase 2, 3 에서 영상 단독 신호는 약했고(자세 기준선 대비 +0.03~0.07) EHR 에는 보태지 않음.
그럼 이 모델은 무엇을 보고 점수를 내는가. Grad-CAM 을 그림으로 몇 장 보고 "폐를 보는 것 같다"고
말하는 것은 근거가 아님. 숫자로 측정함.

  중심부 주의 비율  주의 질량이 가운데(가로세로 각 60% 의 정사각형. 폐 분할이 아님)에 얼마나 모이는가.
                    테두리에 몰리면 마커, 글자, 기기 같은 촬영 부수물을 본 것임.
  주의 엔트로피      주의가 한 곳에 모였는가, 전면에 퍼졌는가. 퍼져 있으면
                    특정 부위를 보지 않는다는 뜻임.
  자세별 비교        AP 와 PA 에서 보는 곳이 다른가. 다르면 자세 신호를
                    보고 있을 수 있음.

    python scripts/run_phase4_gradcam.py            # 기본 300장
    python scripts/run_phase4_gradcam.py --check    # 준비 상태만 확인

## DUA

개별 환자의 Grad-CAM 이미지는 저장하지 않음. 원본 영상 위에 겹친 오버레이는
그 자체가 MIMIC 파생물임. 남기는 것은 수백 장을 평균한 집계 통계와, 환자 식별이
불가능한 평균 히트맵 한 장뿐이며 그마저 `.gitignore` 로 막힌 경로에 둠.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapaths import RAW  # noqa: E402

OUT_DIR = ROOT / "outputs"
# 산출물은 모델별로 나눔. 백본마다 학습이 된 정도가 달라 주의맵의 의미도
# 다르므로(학습이 안 된 모델의 주의맵은 해석 대상이 아님) 덮어쓰면 안 됨.
#
# `run_tag` 는 Phase 2 의 `--tag` 와 같은 값임. 같은 백본이라도 해상도, 증강이
# 다르면 다른 모델이므로 산출물도 분리함.
def _paths(model_name: str, run_tag: str = ""):
    tag = "" if model_name == "densenet121" else f"_{model_name}"
    if run_tag:
        tag = f"{tag}_{run_tag}"
    metrics = ("phase2_metrics.json" if model_name == "densenet121" and not run_tag
               else f"phase2_metrics_{run_tag or model_name}.json")
    return (OUT_DIR / f"phase4_gradcam{tag}.json",
            OUT_DIR / "figures" / f"gradcam_mean{tag}.png",
            OUT_DIR / metrics)

CENTER_FRAC = 0.60          # 중심 영역 한 변 비율
CENTER_MASS_EXPECTED = 0.36  # 균등 주의일 때 중심부가 갖는 질량 = 0.6^2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--model", default="densenet121")
    ap.add_argument("--tag", default="",
                    help="Phase 2 실행 꼬리표 (예: efficientnet_b0). 비우면 기본 실행")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n", type=int, default=300, help="표본 장수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import torch

    from checkpoint import load_checkpoint
    from cohort import CohortConfig, build_cohort, load_admissions
    from cxr_dataset import CXRDataset, build_cxr_index
    from features import ELECTIVE_ADMISSION_TYPES
    from models import build_image_model
    from splits import patient_split

    raw = Path(args.raw_dir)
    t0 = time.time()
    print("=" * 74)
    print("  Phase 4-B: V5 Grad-CAM 정량화")
    print("=" * 74)

    print("\n[0] 입력 점검")
    METRICS_PATH, FIG_PATH, p2 = _paths(args.model, args.tag)
    if not p2.exists():
        print(f"  {p2.name} 이 없다. run_phase2.py --model {args.model} 을 먼저 실행할 것.")
        return 1
    p2_json = json.loads(p2.read_text(encoding="utf-8"))
    ck = p2_json["checkpoint"]
    trained_size = p2_json.get("img_size")
    if trained_size and trained_size != args.size:
        print(f"  이 체크포인트는 {trained_size}px 로 학습됐다. --size {trained_size} 로 실행할 것.")
        return 1
    print(f"  [OK] 체크포인트 지문 {ck['sha256'][:16]}…  (출처 {p2.name})")
    meta_hits = list(raw.rglob("mimic-cxr-2.0.0-metadata.csv.gz"))
    if not meta_hits:
        print("  CXR 메타데이터가 없다.")
        return 1
    print("  [OK] cxr metadata")
    if args.check:
        print("\n준비 완료.")
        return 0

    print("\n[1] 코호트 + 영상 인덱스 (Phase 2 와 동일 규칙)")
    adm, pat = load_admissions(raw)
    coh = build_cohort(adm, pat,
                       CohortConfig(elective_admission_types=ELECTIVE_ADMISSION_TYPES))
    meta = pd.read_csv(meta_hits[0],
                       usecols=["subject_id", "study_id", "dicom_id",
                                "StudyDate", "StudyTime", "ViewPosition"])
    idx = build_cxr_index(coh.df, meta, raw / "mimic-cxr-jpg")
    df = idx.df.reset_index(drop=True)
    sp = patient_split(df["subject_id"].to_numpy(), seed=args.seed)
    print(f"    영상 연결 {len(df):,} 입원 / test {len(sp.test_idx):,}")

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(sp.test_idx, size=min(args.n, len(sp.test_idx)),
                      replace=False)
    sample = df.iloc[pick].reset_index(drop=True)
    print(f"    표본 {len(sample):,}장")

    print("\n[2] 체크포인트 로드")
    state, info = load_checkpoint(
        f"phase2_cxr_{args.model}" + (f"_{args.tag}" if args.tag else ""),
        expect_sha256=ck["sha256"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_image_model(args.model, in_channels=3, num_classes=1)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"    장치 {device}  파라미터 {info.n_parameters:,}")

    # Grad-CAM 대상: 마지막 합성곱 블록의 출력.
    #
    # `model.features` 를 그대로 잡으면 안 됨. torchvision 의 DenseNet 은
    # forward 에서 그 출력에 `F.relu(..., inplace=True)` 를 걸어 제자리 수정을
    # 하는데, 그러면 backward 훅이 "view 를 제자리 수정했다"며 죽음. 제자리
    # 수정이 닿지 않는 한 단계 안쪽(denseblock4)을 잡음.
    if hasattr(model, "features"):
        target = getattr(model.features, "denseblock4", None) or model.features[-1]
    else:
        target = model.layer4
    acts: dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _i, o):
        acts["v"] = o

    h1 = target.register_forward_hook(fwd_hook)

    print("\n[3] Grad-CAM 계산")
    ds = CXRDataset(sample, False, args.size)
    S = args.size
    lo, hi = int(S * (1 - CENTER_FRAC) / 2), int(S * (1 + CENTER_FRAC) / 2)

    center_mass, entropy, peak_r, logits = [], [], [], []
    mean_cam = np.zeros((S, S), dtype=np.float64)
    t1 = time.time()
    for i in range(len(ds)):
        x, _y = ds[i]
        x = x.unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        out = model(x)
        # 훅으로 기울기를 받지 않고 직접 구함. 모듈 backward 훅은 제자리 수정과
        # 얽히면 오류 없이 틀린 값을 줄 수 있는데, autograd.grad 는 그 위험이 없음.
        a = acts["v"]
        gr = torch.autograd.grad(out.sum(), a, retain_graph=False)[0]
        w = gr.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * a).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=(S, S), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy().astype(np.float64)
        total = cam.sum()
        if not np.isfinite(total) or total <= 0:
            continue
        p = cam / total
        mean_cam += p
        center_mass.append(float(p[lo:hi, lo:hi].sum()))
        nz = p[p > 0]
        entropy.append(float(-(nz * np.log(nz)).sum() / np.log(p.size)))
        yy, xx = np.unravel_index(int(np.argmax(p)), p.shape)
        peak_r.append(float(np.hypot(yy - S / 2, xx - S / 2) / (S / 2)))
        logits.append(float(out.item()))
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(ds)}  ({time.time() - t1:.0f}s)")
    h1.remove()

    n = len(center_mass)
    if n == 0:
        print("  유효한 CAM 이 하나도 없다.")
        return 1
    cm = np.array(center_mass)
    en = np.array(entropy)
    pr = np.array(peak_r)
    print(f"\n[4] 집계 (n={n})")
    print(f"    중심부 주의 비율  평균 {cm.mean():.4f}  "
          f"(균등 주의라면 {CENTER_MASS_EXPECTED:.2f})")
    print(f"    주의 엔트로피     평균 {en.mean():.4f}  (1 에 가까울수록 전면에 퍼짐)")
    print(f"    최대점 중심거리   평균 {pr.mean():.4f}  (0=중앙, 1=가장자리)")

    verdict_center = ("중심(폐야)에 모인다" if cm.mean() > CENTER_MASS_EXPECTED + 0.10
                      else ("테두리로 쏠린다" if cm.mean() < CENTER_MASS_EXPECTED - 0.10
                            else "균등 주의와 구분되지 않는다"))
    verdict_focus = ("주의가 퍼져 있다" if en.mean() >= 0.90 else "주의가 모여 있다")
    print(f"    판정(위치): {verdict_center}")
    print(f"    판정(집중): {verdict_focus}")

    out_json = {
        "phase": "4-B", "n": int(n), "img_size": S,
        "checkpoint_sha256": ck["sha256"],
        "center_fraction": CENTER_FRAC,
        "center_mass_uniform_expected": CENTER_MASS_EXPECTED,
        "center_mass": {"mean": float(cm.mean()), "std": float(cm.std()),
                        "p25": float(np.percentile(cm, 25)),
                        "p75": float(np.percentile(cm, 75))},
        "attention_entropy": {"mean": float(en.mean()), "std": float(en.std())},
        "peak_radius": {"mean": float(pr.mean()), "std": float(pr.std())},
        "verdict_location": verdict_center,
        "verdict_focus": verdict_focus,
    }

    # 자세별 비교: AP 와 PA 에서 주의가 다른지
    views = sample["ViewPosition"].fillna("UNKNOWN").astype(str).to_numpy()[:n]
    by_view = {}
    for v in sorted(set(views)):
        m = views == v
        if m.sum() < 20:
            continue
        by_view[v] = {"n": int(m.sum()), "center_mass": float(cm[m].mean()),
                      "entropy": float(en[m].mean())}
        print(f"    [{v}] n={m.sum():>4}  중심부 {cm[m].mean():.4f}  "
              f"엔트로피 {en[m].mean():.4f}")
    out_json["by_view"] = by_view
    if len(by_view) > 1:
        vals = [v["center_mass"] for v in by_view.values()]
        out_json["view_center_mass_gap"] = float(max(vals) - min(vals))

    print("\n[5] 저장")
    METRICS_PATH.write_text(json.dumps(out_json, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"    {METRICS_PATH}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(mean_cam / n, cmap="inferno")
        ax.add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, fill=False,
                                   edgecolor="white", lw=1.2, ls="--"))
        ax.set_title(f"평균 Grad-CAM (n={n})\n흰 점선 = 중심 {CENTER_FRAC:.0%} 영역",
                     fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(FIG_PATH, dpi=130)
        plt.close(fig)
        print(f"    {FIG_PATH}  (수백 장 평균: 개별 환자 영상 아님)")
    except Exception as e:
        print(f"    (그림 저장 건너뜀: {type(e).__name__})")

    print(f"\n  소요 {time.time() - t0:.0f}s")
    print("=" * 74)
    print("  Phase 4-B 완료")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
