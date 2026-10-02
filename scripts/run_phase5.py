"""Phase 5: 완결: V6 누수 A/B, V7 시간분할, M2 촬영시점 윈도우, 운영점, 대시보드.

Phase 4 까지가 "이 모델을 믿어도 되는가"였다면 Phase 5 는 "그래서 어떻게 쓰는가"를 보고,
피하려 한 누수가 실제로 성능을 부풀리는지 측정함.

  V6  누수 컬럼을 일부러 넣어 성능이 얼마나 부풀려지는지 측정함. 빼는 것이
      옳았다는 주장은, 넣었을 때 얼마나 오르는지를 보여야 근거가 됨.
  V7  과거로 학습해 미래를 맞히는가. MIMIC-IV 는 환자별 날짜 시프트라 `admittime`
      연도를 못 씀: `patients.anchor_year_group` 으로 나눔.
  M2  영상을 "언제 찍힌 것"으로 제한하느냐에 따라 결과가 달라지는가.
  운영 임계값을 어디에 둘 것인가. 순편익만이 아니라 감당 가능한 알림량으로도 봄.
  대시 위 결과를 한 장의 로컬 HTML 로 모음.

    python scripts/run_phase5.py            # 전체
    python scripts/run_phase5.py --check    # 입력 준비 상태만 확인

## DUA

대시보드는 로컬 파일로만 만듦. 집계 지표뿐이라도 MIMIC 파생물이므로 외부에
게시하지 않음. 지표 JSON, HTML 에 환자 단위 정보는 들어가지 않음.
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
# M2(촬영 시점 윈도우)만 임베딩을 씀. 인코더에 따라 결론이 달라질 수 있으므로
# Phase 3 와 같은 선택지를 둠.
EMBEDDINGS = {
    "cnn": OUT_DIR / "models" / "phase2_cxr_embeddings.npz",
    "raddino": OUT_DIR / "models" / "phase2_raddino_embeddings.npz",
}
DASHBOARD_PATH = OUT_DIR / "dashboard.html"

# 판정선: 결과를 보기 전에 고정함.
LEAK_INFLATION_BIG = 0.05   # 누수로 이만큼 오르면 "빼는 것이 필수였다"
TEMPORAL_DROP_BIG = 0.03    # 시간 분할에서 이만큼 떨어지면 "시간에 취약"
ALERT_CAPACITY = 0.10       # 현실적으로 감당 가능한 알림 비율 가정


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", default="cnn", choices=sorted(EMBEDDINGS),
                    help="M2 에 쓸 영상 임베딩")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    embed_path = EMBEDDINGS[args.encoder]
    metrics_path = (OUT_DIR / "phase5_metrics.json" if args.encoder == "cnn"
                    else OUT_DIR / f"phase5_metrics_{args.encoder}.json")

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd

    from cohort import CohortConfig, build_cohort, load_admissions
    from features import ELECTIVE_ADMISSION_TYPES, build_features, feature_columns
    from metrics import classification_metrics
    from models import build_tabular_model
    from splits import assert_no_group_overlap, patient_split, subject_folds
    from tracking import ExperimentLogger

    raw = Path(args.raw_dir)
    t0 = time.time()
    print("=" * 74)
    print("  Phase 5: V6 누수 A/B, V7 시간분할, M2 윈도우, 운영점, 대시보드")
    print("=" * 74)

    print("\n[0] 입력 점검")
    dx_hits = list(raw.rglob("diagnoses_icd.csv.gz"))
    adm_hits = list(raw.rglob("admissions.csv.gz"))
    for name, hits in (("admissions", adm_hits), ("diagnoses_icd", dx_hits)):
        if not hits:
            print(f"  {name} 파일이 없다. download_mimic.sh 를 먼저 실행할 것.")
            return 1
        print(f"  [OK] {name}")
    print(f"  {'[OK]' if embed_path.exists() else '[없음]'} 임베딩 캐시"
          f"{'' if embed_path.exists() else ' (없으면 M2 는 건너뜀)'}")
    if args.check:
        print("\n준비 완료.")
        return 0

    print("\n[1] 코호트 + 피처 + 분할")
    adm, pat = load_admissions(raw)
    coh = build_cohort(adm, pat,
                       CohortConfig(elective_admission_types=ELECTIVE_ADMISSION_TYPES))
    dx = pd.read_csv(dx_hits[0])
    feat, _ = build_features(coh.df, pat, dx, None, None)
    cols = [c for c in feature_columns(feat)
            if c not in ("has_cxr", "n_cxr", "cxr_view_ap", "is_ap")]
    X = feat[cols].to_numpy(dtype=np.float32)
    y = feat["readmit_30d"].to_numpy().astype(int)
    g = feat["subject_id"].to_numpy()
    sp = patient_split(g, seed=args.seed)
    te = sp.test_idx
    n_pos = int(y[sp.train_idx].sum())
    pw = float((len(sp.train_idx) - n_pos) / max(n_pos, 1))
    print(f"    {len(feat):,} 입원 / 피처 {len(cols)}개 / test {len(te):,}")

    results: dict = {"phase": "5", "seed": args.seed, "encoder": args.encoder}

    def fit_eval(Xm, tr_idx, te_idx, weight=None):
        mdl = build_tabular_model("xgboost", scale_pos_weight=weight or pw)
        mdl.fit(Xm[tr_idx], y[tr_idx])
        pp = mdl.predict_proba(Xm[te_idx])[:, 1]
        return classification_metrics(y[te_idx], pp), pp

    # 기준(누수 없음)
    m_base, p_base = fit_eval(X, sp.train_idx, te)
    print(f"    기준 모델 AUROC {m_base.auroc:.4f}  PR-AUC {m_base.pr_auc:.4f}")
    results["baseline"] = m_base.as_dict()

    # ---------------------------------------------------------- V6
    print("\n[2] V6: 누수 A/B (일부러 넣어 부풀림을 잰다)")
    leak_specs = []
    if "is_last_admission" in feat.columns:
        leak_specs.append(("is_last_admission",
                           feat["is_last_admission"].to_numpy(dtype=np.float32)))
    if "discharge_location" in feat.columns:
        # pandas 3.0 부터 `astype(str)` 은 결측을 문자열 'nan' 으로 바꾸지 않고
        # NaN 그대로 둠. 그대로 sorted() 에 넣으면 float 과 str 을 비교하다 죽음.
        # 원소마다 명시적으로 문자열로 만듦.
        dl_raw = feat["discharge_location"]
        if isinstance(dl_raw, pd.DataFrame):
            dl_raw = dl_raw.iloc[:, 0]
        dl = pd.Series(["결측" if pd.isna(v) else str(v) for v in np.asarray(dl_raw)],
                       index=feat.index)
        for v in sorted(dl.unique())[:8]:
            leak_specs.append((f"discharge_location={v}",
                               (dl == v).to_numpy(dtype=np.float32)))
    if "days_to_next_admission" in feat.columns:
        d = feat["days_to_next_admission"].to_numpy(dtype=np.float64)
        leak_specs.append(("days_to_next_admission",
                           np.nan_to_num(d, nan=9999.0).astype(np.float32)))

    v6 = {}
    if leak_specs:
        # (a) 라벨 정의에 직접 쓰인 변수: 완전 누수
        direct = [v for k, v in leak_specs if k == "days_to_next_admission"]
        # (b) 결과와 가까운 운영 변수: 은근한 누수
        soft = [v for k, v in leak_specs if k != "days_to_next_admission"]
        for label, extra in (("은근한 누수(퇴원처, 마지막입원)", soft),
                             ("완전 누수(다음입원까지 일수)", direct)):
            if not extra:
                continue
            Xl = np.column_stack([X, *extra]).astype(np.float32)
            mm, _ = fit_eval(Xl, sp.train_idx, te)
            infl = mm.auroc - m_base.auroc
            v6[label] = {**mm.as_dict(), "inflation_auroc": float(infl),
                         "n_added": len(extra)}
            print(f"    {label:<28} AUROC {mm.auroc:.4f}  부풀림 {infl:+.4f}")
    results["v6_leakage_ab"] = v6
    worst = max((v["inflation_auroc"] for v in v6.values()), default=0.0)
    results["v6_max_inflation"] = float(worst)
    results["v6_verdict"] = ("빼는 것이 필수였다" if worst >= LEAK_INFLATION_BIG
                             else "빼지 않아도 큰 차이는 없었다")
    print(f"    판정: {results['v6_verdict']} (최대 부풀림 {worst:+.4f})")

    # ---------------------------------------------------------- V7
    print("\n[3] V7: 시간 분할 (anchor_year_group, 과거로 학습해 미래를 맞히는가)")
    yg_raw = feat["subject_id"].map(
        dict(zip(pat["subject_id"], pat["anchor_year_group"], strict=False)))
    # 매핑되지 않은 환자는 NaN 으로 남음. pandas 3.0 의 astype(str) 은 그 NaN 을
    # 'nan' 문자열로 바꾸지 않으므로, 여기서 명시적으로 문자열을 만들어야
    # 아래 sorted() 가 float 과 str 을 비교하다 죽지 않음.
    yg = np.array(["결측" if pd.isna(v) else str(v) for v in np.asarray(yg_raw)],
                  dtype=object)
    groups_sorted = sorted({v for v in yg if v != "결측"})
    print(f"    연도 구간: {groups_sorted}")
    if len(groups_sorted) >= 2:
        last = groups_sorted[-1]
        tr_t = np.where((yg != last) & (yg != "결측"))[0]
        te_t = np.where(yg == last)[0]
        # 시간 분할이어도 환자가 양쪽에 걸치면 안 됨. anchor_year_group 은
        # 환자 속성이라 원래 겹칠 수 없지만, 깨져도 오류가 나지 않으므로 검사함.
        assert_no_group_overlap(g, tr_t, te_t, names=("과거", "미래"))
        n_pos_t = int(y[tr_t].sum())
        pw_t = float((len(tr_t) - n_pos_t) / max(n_pos_t, 1))
        m_t, _ = fit_eval(X, tr_t, te_t, weight=pw_t)

        # 대조군: 연도를 무시하고 같은 크기로 무작위 분할한 것.
        #
        # 전체 무작위 분할로 학습한 모델을 te_t 에 그대로 평가하면 te_t 환자의 상당수가 그 모델의 학습에
        # 들어가 있어 대조군이 부풀려지고 시간 이동의 효과가 과장됨. 같은 표본 풀에서 같은 비율로,
        # 환자 단위로 나눠 측정함. 분할 운에 흔들리지 않도록 seed 를 바꿔 3회 평균함.
        pool = np.concatenate([tr_t, te_t])
        frac = len(te_t) / len(pool)
        ctrl_aurocs = []
        for k in range(3):
            spc = patient_split(g[pool], test_size=frac, val_size=0.0,
                                seed=args.seed + 100 + k)
            tr_c, te_c = pool[spc.train_idx], pool[spc.test_idx]
            if len(np.unique(y[te_c])) < 2:
                continue
            npc = int(y[tr_c].sum())
            pwc = float((len(tr_c) - npc) / max(npc, 1))
            mc, _ = fit_eval(X, tr_c, te_c, weight=pwc)
            ctrl_aurocs.append(mc.auroc)
        ctrl = float(np.mean(ctrl_aurocs)) if ctrl_aurocs else float("nan")
        drop = ctrl - m_t.auroc
        results["v7_temporal"] = {
            "train_groups": [x for x in groups_sorted[:-1]], "test_group": last,
            "n_train": len(tr_t), "n_test": len(te_t),
            "temporal_auroc": m_t.auroc,
            "control_random_auroc_mean": ctrl,
            "control_random_aurocs": [float(a) for a in ctrl_aurocs],
            "control_test_fraction": float(frac),
            "drop": float(drop),
        }
        results["v7_verdict"] = ("시간에 취약" if drop >= TEMPORAL_DROP_BIG
                                 else "시간 이동에 견딘다")
        print(f"    과거->미래 AUROC {m_t.auroc:.4f}  vs 같은크기 무작위분할 "
              f"{ctrl:.4f}(3회 평균)   차이 {drop:+.4f}")
        print(f"    판정: {results['v7_verdict']}")
    else:
        print("    연도 구간이 하나뿐이라 건너뜀")
        results["v7_temporal"] = {}

    # ---------------------------------------------------------- M2
    print("\n[4] M2: 촬영 시점 윈도우 (영상을 언제 찍힌 것으로 제한하는가)")
    m2 = []
    if embed_path.exists():
        from sklearn.decomposition import PCA

        from cxr_dataset import build_cxr_index
        z = np.load(embed_path, allow_pickle=False)
        emb_all = z["embedding"].astype(np.float32)
        # 어느 인코더에서 나온 임베딩인지 남김. M2(촬영 시점 창) 결과가 이
        # 임베딩 위에 얹히므로, 인코더가 바뀐 뒤 임베딩을 다시 안 뽑으면 옛
        # 인코더의 결과가 그대로 리포트로 감.
        if "checkpoint_sha256" in z:
            results["checkpoint_sha256"] = str(z["checkpoint_sha256"])
            print(f"    임베딩 출처 지문 {str(z['checkpoint_sha256'])[:16]}…")
        else:
            print("    [경고] 임베딩에 체크포인트 지문이 없다: 옛 형식이다. "
                  "run_phase2_embed.py 를 다시 돌릴 것.")
        pos = pd.Series(np.arange(len(z["hadm_id"])), index=z["hadm_id"])
        row = feat["hadm_id"].map(pos)
        has_img = row.notna().to_numpy()
        emb = np.zeros((len(feat), emb_all.shape[1]), dtype=np.float32)
        emb[has_img] = emb_all[row[has_img].to_numpy().astype(int)]

        # Phase 2 인코더가 쓴 분할을 그대로 재현해 소속표를 만듦
        # (같은 환자 집합, 같은 seed 라 인코더의 분할과 일치함).
        sub_all = np.where(has_img)[0]
        enc_fold = subject_folds(g[sub_all], patient_split(g[sub_all], seed=args.seed))

        meta_hits = list(raw.rglob("mimic-cxr-2.0.0-metadata.csv.gz"))
        idx_res = build_cxr_index(coh.df, pd.read_csv(
            meta_hits[0], usecols=["subject_id", "study_id", "dicom_id",
                                   "StudyDate", "StudyTime", "ViewPosition"]),
            raw / "mimic-cxr-jpg")
        hrs = feat["hadm_id"].map(
            idx_res.df.set_index("hadm_id")["hours_before_discharge"]
        ).to_numpy(dtype=np.float64)

        # 창마다 표본 수가 7배까지 차이남(24h 4,939 vs 전체 35,599). 그대로
        # 비교하면 "촬영 시점의 효과"와 "학습 표본이 많아서 좋아진 효과"가 섞임.
        # 그래서 두 팔을 측정함: 있는 대로 쓴 것과, 가장 작은 창의 학습 크기에
        # 맞춰 무작위로 줄인 것. 후자만이 시점의 효과를 말함.
        windows = []
        for win in (24, 48, 72, 168, None):
            sel = has_img & (np.isfinite(hrs))
            if win is not None:
                sel = sel & (hrs <= win)
            ii = np.where(sel)[0]
            if len(ii) < 500:
                continue
            # 창마다 새로 나누면 안 됨. 이 임베딩을 만든 인코더는 영상 전체를
            # 한 번 나눠 그중 train 으로 학습했는데, 창 안에서 다시 나누면 그
            # train 환자가 여기 test 로 들어옴. 임베딩이 그 환자의 라벨을 이미
            # 본 상태라 AUROC 가 부풀려짐(실측은 splits.py 주석 참조).
            tr_i = ii[np.array([enc_fold.get(x) == "train" for x in g[ii]])]
            te_i = ii[np.array([enc_fold.get(x) == "test" for x in g[ii]])]
            if len(tr_i) < 100 or len(te_i) < 50:
                continue
            if len(np.unique(y[te_i])) < 2:
                continue
            windows.append((win, ii, tr_i, te_i))
        min_train = min(len(t) for _, _, t, _ in windows) if windows else 0
        print(f"    표본 맞춤 기준 학습 크기: {min_train:,}건 (가장 좁은 창)")

        rng_m2 = np.random.default_rng(args.seed)
        for win, ii, tr_i, te_i in windows:
            pca = PCA(n_components=32, random_state=args.seed)
            pca.fit(emb[tr_i])
            E = pca.transform(emb).astype(np.float32)

            # E, te_i 를 기본값으로 묶음. 닫힘이 이름만 들고 있으면 다음 창으로
            # 넘어간 뒤에 불릴 때 엉뚱한 창의 값을 읽음. 지금은 같은 반복 안에서만
            # 부르므로 결과가 달라지지 않지만, 호출 위치가 한 줄만 움직여도 경고 없이
            # 틀리는 종류라 묶어둠.
            def _fit(tr, E=E, te_i=te_i):
                np_i = int(y[tr].sum())
                w = float((len(tr) - np_i) / max(np_i, 1))
                mm_, _ = fit_eval(E, tr, te_i, weight=w)
                return mm_

            mm = _fit(tr_i)
            tr_m = (tr_i if len(tr_i) <= min_train
                    else rng_m2.choice(tr_i, size=min_train, replace=False))
            mm_m = _fit(tr_m)
            m2.append({"window_hours": win, "n": len(ii),
                       "n_train": len(tr_i), "n_test": len(te_i),
                       "auroc": mm.auroc, "pr_auc": mm.pr_auc,
                       "auroc_size_matched": mm_m.auroc,
                       "n_train_matched": len(tr_m)})
            print(f"    퇴원 전 {str(win) + 'h' if win else '전체':>6}"
                  f"  n={len(ii):>6,}  AUROC {mm.auroc:.4f}"
                  f"   표본맞춤 {mm_m.auroc:.4f}")
        if m2:
            a = [r["auroc"] for r in m2]
            am = [r["auroc_size_matched"] for r in m2]
            results["m2_window_spread"] = float(max(a) - min(a))
            results["m2_window_spread_size_matched"] = float(max(am) - min(am))
            # 판정은 표본을 맞춘 쪽으로 함. 맞추지 않은 폭은 시점이 아니라
            # 데이터 양의 차이를 재고 있을 수 있음.
            results["m2_verdict"] = (
                "촬영 시점은 결과를 바꾸지 않는다"
                if results["m2_window_spread_size_matched"] < 0.03
                else "촬영 시점이 결과를 바꾼다")
            print(f"    판정: {results['m2_verdict']}  "
                  f"(폭: 그대로 {results['m2_window_spread']:.4f} / "
                  f"표본맞춤 {results['m2_window_spread_size_matched']:.4f})")
    else:
        print("    임베딩 캐시가 없어 건너뜀")
    results["m2_windows"] = m2

    # ------------------------------------------------------- 운영점
    print("\n[5] 운영점: 임계값을 어디에 둘 것인가")
    yt = y[te]
    order = np.argsort(p_base)[::-1]
    ops = []
    for rate in (0.05, ALERT_CAPACITY, 0.20, 0.30):
        k = max(1, int(len(yt) * rate))
        flag = np.zeros(len(yt), dtype=bool)
        flag[order[:k]] = True
        tp = int((flag & (yt == 1)).sum())
        ppv = tp / k
        sens = tp / max(int(yt.sum()), 1)
        ops.append({"alert_rate": rate, "n_alerts": int(k), "ppv": float(ppv),
                    "sensitivity": float(sens),
                    "threshold": float(p_base[order[k - 1]]),
                    "nns": float(1 / ppv) if ppv else float("inf")})
        print(f"    알림 {rate * 100:>4.0f}%  n={k:>6,}  PPV {ppv:.3f}  "
              f"민감도 {sens:.3f}  선별필요수 {1 / ppv if ppv else float('inf'):.1f}")
    results["operating_points"] = ops
    prev = float(yt.mean())
    chosen = next(o for o in ops if o["alert_rate"] == ALERT_CAPACITY)
    results["chosen_operating_point"] = chosen
    results["prevalence"] = prev
    results["operating_verdict"] = (
        f"알림 {ALERT_CAPACITY:.0%} 에서 PPV {chosen['ppv']:.3f}: "
        f"무작위({prev:.3f}) 대비 {chosen['ppv'] / prev:.2f}배")
    print(f"    선택: {results['operating_verdict']}")

    print("\n[6] 저장")
    results["elapsed_sec"] = round(time.time() - t0, 1)
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")
    print(f"    {metrics_path}")

    # ----------------------------------------------------- 대시보드
    # 반드시 위 저장 뒤에 만듦. 대시보드는 산출물 JSON 을 다시 읽어
    # 그리므로, 저장 전에 만들면 이번 실행이 아니라 직전 실행의 값을 보여줌.
    print("\n[7] 대시보드 (로컬 HTML)")
    _write_dashboard(DASHBOARD_PATH, OUT_DIR)
    print(f"    {DASHBOARD_PATH}")

    try:
        with ExperimentLogger("phase5-final",
                              params={"phase": "5", "seed": args.seed},
                              run_name=f"final-seed{args.seed}") as log:
            log.log_metrics({"baseline_auroc": m_base.auroc,
                             "v6_max_inflation": worst,
                             "ppv_at_capacity": chosen["ppv"]})
    except Exception as e:
        print(f"    (실험 기록 건너뜀: {type(e).__name__})")

    print("\n" + "=" * 74)
    print("  Phase 5 완료")
    print("=" * 74)
    return 0


def _write_dashboard(path: Path, out_dir: Path) -> None:
    """산출물 JSON 들을 한 장짜리 로컬 HTML 로 모은다 (외부 게시 금지)."""
    import html as _html

    def load(name):
        p = out_dir / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            # 말없이 건너뛰면 대시보드에서 그 단계가 통째로 사라지고, 보는 사람은
            # 그 단계를 안 돌린 것으로 읽음. 파일이 있는데 못 읽은 것은 말함.
            print(f"  [경고] {name} 을 읽지 못해 대시보드에서 뺀다: {e}")
            return None

    d1, d15 = load("phase1_metrics.json"), load("phase1_5_metrics.json")
    d2, d3 = load("phase2_metrics.json"), load("phase3_metrics.json")
    d4, d5 = load("phase4_metrics.json"), load("phase5_metrics.json")

    rows = []

    def add(phase, item, value):
        rows.append((phase, item, value))

    if d1:
        add("1", "EHR 단독 AUROC", f"{d1.get('test', {}).get('auroc', float('nan')):.4f}"
            if isinstance(d1.get("test"), dict) else "없음")
    if d15:
        eff = (d15.get("mnar") or {}).get("effect") or {}
        rr = eff.get("risk_ratio")
        add("1.5", "영상 유무 위험비",
            f"{rr:.3f} [{eff.get('rr_ci_low', float('nan')):.3f}, "
            f"{eff.get('rr_ci_high', float('nan')):.3f}]" if rr else "없음")
        add("1.5", "MNAR 판정", (d15.get("mnar") or {}).get("verdict", "없음"))
    if d2:
        add("2", "영상 단독 AUROC", f"{d2['image_only']['auroc']:.4f}")
        add("2", "자세 기준선 AUROC", f"{d2['view_baseline']['auroc']:.4f}")
        add("2", "판정", d2.get("verdict", ": "))
    if d3:
        add("3", "메타데이터 AUROC", f"{d3['v3_metadata_only']['auroc']:.4f}")
        add("3", "Fusion 최고 대비 EHR (세 방식 중 최고, 판정에 쓰는 값)", f"{d3['fusion_best_lift']:+.4f}")
        add("3", "V1 부풀림", f"{d3['v1_inflation_auroc']:+.4f}")
        add("3", "M1a 픽셀 기여(영상 있음)",
            f"{d3['m1a_pixel_contribution'].get('영상 있음', float('nan')):+.4f}")
    if d4:
        add("4", "주 모델 AUROC", f"{d4['main_model']['auroc']:.4f}")
        add("4", "보정 최선", f"{d4['calibration_best']} "
                              f"(ECE {d4['calibration'][d4['calibration_best']]['ece']:.4f})")
        add("4", "DCA 판정", d4.get("dca_verdict", ": "))
        add("4", "모델 간 AUROC 폭", f"{d4['model_auroc_spread']:.4f}")
    if d5:
        add("5", "V6 최대 부풀림", f"{d5['v6_max_inflation']:+.4f}")
        if d5.get("v7_temporal"):
            add("5", "V7 시간분할 차이", f"{d5['v7_temporal']['drop']:+.4f}")
        cop = d5.get("chosen_operating_point", {})
        if cop:
            add("5", f"운영점 PPV(알림 {cop['alert_rate']:.0%})", f"{cop['ppv']:.3f}")

    body = "\n".join(
        f"<tr><td>{_html.escape(p)}</td><td>{_html.escape(i)}</td>"
        f"<td class='v'>{_html.escape(str(v))}</td></tr>" for p, i, v in rows)

    path.write_text(f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<title>MIMIC 재입원: 결과 대시보드 (로컬)</title>
<style>
 body{{font:14px/1.6 system-ui,sans-serif;margin:2rem;max-width:860px;color:#222}}
 h1{{font-size:1.3rem}} .warn{{background:#fff4e5;border-left:4px solid #e8a33d;
 padding:.8rem 1rem;margin:1rem 0}}
 table{{border-collapse:collapse;width:100%;margin-top:1rem}}
 th,td{{border-bottom:1px solid #ddd;padding:.45rem .6rem;text-align:left}}
 th{{background:#f6f6f6}} td.v{{font-variant-numeric:tabular-nums;font-weight:600}}
 footer{{margin-top:2rem;color:#666;font-size:.85rem}}
</style></head><body>
<h1>MIMIC 멀티모달 30일 재입원: 결과 대시보드</h1>
<div class="warn"><strong>로컬 전용.</strong> 집계 지표뿐이라도 MIMIC 파생물이므로
PhysioNet DUA 에 따라 외부에 게시하지 않는다.</div>
<table><thead><tr><th>Phase</th><th>항목</th><th>값</th></tr></thead>
<tbody>
{body}
</tbody></table>
<footer>산출물 JSON 에서 자동 생성: <code>scripts/run_phase5.py</code></footer>
</body></html>
""", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
