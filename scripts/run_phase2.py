"""Phase 2: CXR 영상 단독 모델 + 모달리티 기여도 분해.

영상이 있는 입원은 전체의 8.6% 뿐이고 그 집단은 더 아픈 환자로 치우쳐 있음
(Phase 1.5: RR 1.340, 재원일수 보정 후 1.202). 그래서 비교를 같은 부분집합
안에서 함. 전체 코호트 EHR 성능과 8.6% 영상 성능을 나란히 놓으면 안 됨.

학습 전에 자세(ViewPosition)만으로 얻는 AUROC 를 먼저 측정함. AP 는 누워서
찍는 자세라 못 일어나는 환자를 뜻함. 영상 모델이 이 값을 넘지 못하면
폐가 아니라 자세를 본 것임.

    python scripts/run_phase2.py --check
    python scripts/run_phase2.py
    python scripts/run_phase2.py --epochs 15 --model resnet50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_dataset import AUG_LEVELS  # noqa: E402

LIFT_PASS = 0.05        # 자세 기준선 대비 이만큼 넘으면 영상을 봤다고 봄
LIFT_FAIL = 0.02        # 이 미만이면 자세를 본 것임


def verdict(lift: float) -> str:
    if lift != lift:
        return "기준선 없음"
    if lift >= LIFT_PASS:
        return "영상을 봤다"
    if lift < LIFT_FAIL:
        return "자세를 본 것"
    return "판단 보류"


def main() -> int:
    from datapaths import RAW

    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--model", default="densenet121")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42,
                    help="모델 초기화, 학습 시드")
    # 분할 시드를 따로 둠. 하나로 묶으면 시드를 바꿀 때 test 집합까지 바뀌어,
    # 앞선 실행의 val 환자가 뒤 실행의 test 로 들어감. 설정을 val 로 골랐으므로
    # 그건 약한 오염임. 홀드아웃은 한 번 정하면 고정해야 함.
    ap.add_argument("--split-seed", type=int, default=42,
                    help="환자 분할 시드. 시드를 여러 개 돌려도 이 값은 고정할 것")
    # 절제실험과 조건을 맞추려면 이 둘도 여기서 지정할 수 있어야 함.
    # 기본값은 TrainConfig 와 같으므로 옵션을 안 주면 기존 실행과 동일함.
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--freeze-epochs", type=int, default=3)
    ap.add_argument("--check", action="store_true", help="데이터 상태만 확인")
    # 아래는 "영상에 공정한 기회를 줬는가"를 확인하기 위한 강화 학습용 옵션임.
    # 기본값은 지금까지 보고한 실행과 동일하게 두어, 옵션을 주지 않으면 결과가
    # 재현되도록 함.
    ap.add_argument("--patience", type=int, default=5, help="조기종료 인내 에폭")
    ap.add_argument("--scheduler", default="none", choices=("none", "cosine"),
                    help="LR 스케줄. 긴 학습에서는 cosine 권장")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--tag", default="", help="산출물 파일명 꼬리표 (비교 실행 보존용)")
    # 증강 강도. 절제실험에서 고른 값을 여기로 넘겨 확인 실행을 함.
    # 기본값 medium 은 지금까지 보고한 실행과 같은 조건임.
    ap.add_argument("--aug", default="medium", choices=AUG_LEVELS)
    args = ap.parse_args()

    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from checkpoint import save_checkpoint
    from cohort import CohortConfig, build_cohort, load_admissions
    from cxr_dataset import CXRDataset, build_cxr_index, view_baseline
    from features import ELECTIVE_ADMISSION_TYPES, build_features, feature_columns
    from metrics import classification_metrics, subgroup_metrics
    from models import build_image_model, build_tabular_model
    from splits import check_split, patient_split
    from tracking import ExperimentLogger
    from train import TrainConfig, predict_proba, set_seed, train_image_model

    raw = Path(args.raw_dir)
    img_root = raw / "mimic-cxr-jpg"
    print("=" * 74)
    print("  Phase 2: CXR 영상 단독 모델")
    print("=" * 74)

    meta_hits = list(raw.rglob("mimic-cxr-2.0.0-metadata.csv.gz"))
    adm_hits = list(raw.rglob("admissions.csv.gz"))
    if not meta_hits or not adm_hits:
        print("  필요한 파일이 없습니다. download_mimic.sh meta / mimiciv 를 먼저 실행할 것.")
        return 1

    print("\n[1] 코호트 + 영상 인덱스")
    adm, pat = load_admissions(raw)
    coh = build_cohort(adm, pat,
                       CohortConfig(elective_admission_types=ELECTIVE_ADMISSION_TYPES))
    meta = pd.read_csv(meta_hits[0],
                       usecols=["subject_id", "study_id", "dicom_id",
                                "StudyDate", "StudyTime", "ViewPosition"])
    idx = build_cxr_index(coh.df, meta, img_root)
    df = idx.df
    print(f"    전체 코호트 {len(coh.df):,} 입원")
    print(f"    영상 연결    {len(df):,} 입원 ({len(df) / len(coh.df) * 100:.1f}%)")
    for k, v in idx.dropped.items():
        print(f"      제외 {k}: {v:,}")
    for k, v in idx.flags.items():
        print(f"      참고 {k}: {v:,}")
    if df.empty:
        print("\n  영상 파일이 없습니다. sh scripts/download_mimic.sh images 를 실행할 것.")
        return 1
    print(f"    재입원률 {df.readmit_30d.mean() * 100:.2f}%  "
          f"(전체 코호트 {coh.df.readmit_30d.mean() * 100:.2f}%)")
    print(f"    AP 비율 {df.is_ap.mean() * 100:.1f}%")
    print(f"    마지막 촬영~퇴원 중앙값 {df.hours_before_discharge.median():.1f}시간")

    print("\n[2] 환자 단위 분할")
    groups = df["subject_id"].to_numpy()
    y = df["readmit_30d"].to_numpy().astype(int)
    sp = patient_split(groups, test_size=0.2, val_size=0.1, seed=args.split_seed)
    info = check_split(y, groups, sp)
    for k, v in info.items():
        print(f"    {k:22s} {v}")

    print("\n[3] 자세 기준선: 픽셀 없이 ViewPosition 만으로")
    vb = view_baseline(df, groups, args.seed)
    if vb:
        print(f"    AUROC {vb['auroc']:.4f}   PR-AUC {vb['pr_auc']:.4f}   "
              f"양성률 {vb['prevalence']:.4f}")
        print(f"    자세 분포 {vb['categories']}")
        print(f"    판정선: 추가분 >= {LIFT_PASS} 영상을 봤다 / "
              f"< {LIFT_FAIL} 자세를 본 것")
    else:
        print(f"    잴 수 없다: 표본 {len(df):,}건, 양성 {int(y.sum()):,}건.")
        print("    판정 없이 영상 성능만 보고하게 된다.")

    if args.check:
        print("\n  --check 모드. 학습은 건너뜁니다.")
        return 0

    print(f"\n[4] 영상 단독 모델 ({args.model})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_df, va_df, te_df = (df.iloc[sp.train_idx], df.iloc[sp.val_idx],
                           df.iloc[sp.test_idx])
    kw = {"batch_size": args.batch, "num_workers": args.workers,
          "pin_memory": device.type == "cuda",
          "persistent_workers": args.workers > 0}
    # drop_last 를 쓰지 않음. 절제실험 러너(run_ablation.py)가 쓰지 않으므로,
    # 여기서만 마지막 자투리 배치를 버리면 학습 조건이 달라져 val 이 재현되지 않음.
    # 24,920 / 32 = 778 배치 + 24 건인데, 그 24 건을 매 에폭 버리는 차이임.
    tr = DataLoader(CXRDataset(tr_df, True, args.size, args.aug), shuffle=True, **kw)
    va = DataLoader(CXRDataset(va_df, False, args.size, args.aug), shuffle=False, **kw)
    te = DataLoader(CXRDataset(te_df, False, args.size, args.aug), shuffle=False, **kw)

    n_pos = int(y[sp.train_idx].sum())
    pw = (len(sp.train_idx) - n_pos) / max(1, n_pos)
    # 모델을 만들기 직전에 시드를 설정함. 이 스크립트는 앞에서 자세 베이스라인을
    # 적합하느라 난수를 더 쓰므로, 여기서 고정하지 않으면 같은 시드를 줘도
    # run_ablation.py 와 다른 헤드 초기값으로 시작함.
    set_seed(args.seed)
    model = build_image_model(args.model, in_channels=3, num_classes=1)
    cfg = TrainConfig(epochs=args.epochs, batch=args.batch, seed=args.seed,
                      early_stop_patience=args.patience, scheduler=args.scheduler,
                      lr_head=args.lr_head, lr_backbone=args.lr_backbone,
                      weight_decay=args.weight_decay,
                      freeze_epochs=args.freeze_epochs)
    print(f"    장치 {device}  pos_weight {pw:.2f}  epochs {args.epochs}"
          f"  patience {args.patience}  sched {args.scheduler}"
          f"  lr {args.lr_head:g}/{args.lr_backbone:g}  size {args.size}"
          f"  aug {args.aug}  seed {args.seed}/split {args.split_seed}")
    res = train_image_model(model, args.model, tr, va, cfg, device=device,
                            pos_weight=pw)
    model.load_state_dict(res.best_state)
    prob_img, y_img = predict_proba(model, te, device)
    m_img = classification_metrics(y_img, prob_img)
    print(f"\n    영상 단독  AUROC {m_img.auroc:.4f}  PR-AUC {m_img.pr_auc:.4f}  "
          f"ECE {m_img.ece:.4f}")

    # 가중치를 남김: Phase 3(Fusion), 4(Grad-CAM), 5(윈도우 실험)가 같은 인코더를
    # 쓰므로 매번 재학습하지 않기 위함이고, 동시에 "이 수치를 낸 모델"을 특정할 수
    # 있어야 하기 때문임(TRIPOD+AI). 파일 자체는 .gitignore 로 막히고,
    # 지문(sha256)만 지표 JSON 에 남아 저장소에 올라감.
    # 태그가 있으면 체크포인트도 따로 남김. 같은 이름에 덮어쓰면 Phase 3, 4-B 가
    # 지문으로 대조하는 "그 모델"이 경고 없이 바뀜(테스트가 잡긴 하지만, 애초에
    # 덮어쓰지 않도록 이름을 나눔).
    ckpt_name = (f"phase2_cxr_{args.model}_{args.tag}" if args.tag
                 else f"phase2_cxr_{args.model}")
    ckpt = save_checkpoint(
        res.best_state,
        ckpt_name,
        meta={
            "phase": "2", "model": args.model, "seed": args.seed,
            "img_size": args.size, "epochs_requested": args.epochs,
            "best_epoch": res.best_epoch, "best_val_pr_auc": res.best_metric,
            "pos_weight": pw, "train_config": cfg,
            "test_metrics": m_img.as_dict(),
            "n_train": len(sp.train_idx), "n_val": len(sp.val_idx),
            "n_test": len(sp.test_idx),
        },
    )
    print(f"    체크포인트 {ckpt.path}")
    print(f"      지문 sha256 {ckpt.sha256[:16]}…  파라미터 {ckpt.n_parameters:,}")

    print("\n[5] 같은 부분집합에서 EHR 단독")
    dx_hits = list(raw.rglob("diagnoses_icd.csv.gz"))
    m_ehr = None
    if dx_hits:
        dx = pd.read_csv(dx_hits[0])
        # 피처는 코호트 전체 행(coh.df)에서 만들고 영상 부분집합의 행 순서로 가져옴.
        # Phase 3 의 "EHR 단독" 과 같은 피처가 됨. build_cxr_index 가 돌려준 df(5개 컬럼)에서 만들면
        # 입원 유형, 과거 입원 수 등이 빠진 축소판 EHR 이 됨.
        feat_all, _ = build_features(coh.df, pat, dx, None, None)
        feat = df[["hadm_id"]].merge(feat_all, on="hadm_id", how="left", validate="1:1")
        cols = [c for c in feature_columns(feat)
                if c not in ("has_cxr", "n_cxr", "cxr_view_ap", "is_ap")]
        X = feat[cols].to_numpy(dtype=np.float32)
        gb = build_tabular_model("xgboost", scale_pos_weight=pw)
        gb.fit(X[sp.train_idx], y[sp.train_idx])
        prob_ehr = gb.predict_proba(X[sp.test_idx])[:, 1]
        m_ehr = classification_metrics(y[sp.test_idx], prob_ehr)
        print(f"    EHR 단독   AUROC {m_ehr.auroc:.4f}  PR-AUC {m_ehr.pr_auc:.4f}  "
              f"ECE {m_ehr.ece:.4f}   피처 {len(cols)}개")
    else:
        print("    diagnoses_icd.csv.gz 가 없어 건너뜁니다.")

    print("\n[6] 판정")
    lift = m_img.auroc - vb["auroc"] if vb else float("nan")
    v = verdict(lift)
    if vb:
        print(f"    자세 기준선 {vb['auroc']:.4f}  ->  영상 모델 {m_img.auroc:.4f}"
              f"  추가분 {lift:+.4f}   [{v}]")
    else:
        # 기준선을 못 재도 여기서 죽으면 안 됨. 학습이 이미 끝난 시점이라
        # 크래시하면 몇 시간짜리 결과를 저장도 못 하고 통째로 잃음.
        print(f"    자세 기준선 없음  ->  영상 모델 {m_img.auroc:.4f}   [{v}]")
    if m_ehr:
        print(f"    같은 집단 EHR {m_ehr.auroc:.4f}  ->  영상이 EHR 대비 "
              f"{m_img.auroc - m_ehr.auroc:+.4f}")

    print("\n[7] 서브그룹")
    sub = {}
    # 성별은 df 에 없음: build_cxr_index 는 코호트에서 subject_id, hadm_id,
    # admittime, dischtime, readmit_30d 만 가져옴. patients 에서
    # 직접 붙임(진단 파일 유무와 무관하게 항상 가능함).
    sex_of = dict(zip(pat["subject_id"], pat["gender"], strict=False))
    sex = df["subject_id"].map(sex_of).to_numpy()
    te_idx = np.asarray(sp.test_idx)
    known = np.array([str(x) in ("F", "M") for x in sex[te_idx]])
    if known.any():
        g = sex[te_idx][known].astype(str)
        for name, gm in subgroup_metrics(y[te_idx][known], prob_img[known], g).items():
            print(f"    {name}: n={gm.n:,} auroc={gm.auroc:.4f} ece={gm.ece:.4f}")
            sub[name] = {"n": gm.n, "auroc": gm.auroc, "ece": gm.ece}
        if (~known).sum():
            print(f"    성별 미상 {int((~known).sum()):,}건은 제외")
    else:
        print("    성별 정보를 붙이지 못했다 (patients.csv.gz 확인 필요)")

    art = {
        "phase": "2", "model": args.model, "seed": args.seed,
        "epochs": args.epochs, "img_size": args.size, "aug": args.aug,
        "split_seed": args.split_seed,
        "weight_decay": args.weight_decay, "freeze_epochs": args.freeze_epochs,
        "subset": {
            "n_admissions": len(df),
            "n_subjects": int(df.subject_id.nunique()),
            "pct_of_cohort": float(len(df) / len(coh.df)),
            "readmit_rate": float(df.readmit_30d.mean()),
            "cohort_readmit_rate": float(coh.df.readmit_30d.mean()),
            "ap_rate": float(df.is_ap.mean()),
        },
        "split": dict(info),
        "view_baseline": vb,
        "image_only": m_img.as_dict(),
        "ehr_same_subset": m_ehr.as_dict() if m_ehr else None,
        "lift_over_view": float(lift),
        "verdict": v,
        # 이 수치를 낸 가중치를 특정하는 지문. 가중치 파일 자체는 DUA 로 저장소에
        # 올리지 않지만, 해시는 데이터가 아니므로 함께 공개해 재현성을 담보함.
        "checkpoint": ckpt.as_dict(),
        "subgroup_sex": sub,
        "history": [vars(h) for h in res.history],
        "best_epoch": res.best_epoch,
    }
    # 기본 모델(densenet121)의 결과가 이 프로젝트의 주 산출물임. 다른 백본을
    # 같은 조건으로 돌려볼 때 그 파일을 덮어쓰면 주 결과를 잃으므로, 모델이
    # 기본이 아니면 파일 이름에 모델명을 붙임.
    # --tag 를 주면 그 이름으로 따로 남김. 강화 학습처럼 조건을 바꾼 실행이
    # 주 산출물을 덮어쓰면, 지금까지의 모든 판정이 소리 없이 다른 조건의 것이 됨.
    if args.tag:
        fname = f"phase2_metrics_{args.tag}.json"
    else:
        fname = ("phase2_metrics.json" if args.model == "densenet121"
                 else f"phase2_metrics_{args.model}.json")
    p = ROOT / "outputs" / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(art, ensure_ascii=False, indent=2, default=float),
                 encoding="utf-8")
    print(f"\n  저장: {p}")

    # 실험 추적. 코드 상태(git 커밋), 환경, 하이퍼파라미터가 지표와 함께 묶여야 나중에
    # "이 숫자가 어느 코드에서 나왔는지" 를 답할 수 있음.
    print("\n[8] 실험 기록 (MLflow 로컬 sqlite)")
    with ExperimentLogger(
        "phase2-cxr",
        params={
            "phase": "2", "model": args.model, "seed": args.seed,
            "img_size": args.size, "epochs_requested": args.epochs,
            "batch": args.batch, "pos_weight": round(pw, 4),
            "checkpoint_sha256": ckpt.sha256,
        },
        run_name=f"{args.model}-seed{args.seed}",
    ) as run:
        run.note(
            "Phase 2: CXR 영상 단독. 자세(ViewPosition) 기준선 대비 추가분으로 "
            "'영상을 봤는가 자세를 봤는가'를 판정한다."
        )
        run.log_split(
            n_train=len(sp.train_idx), n_val=len(sp.val_idx), n_test=len(sp.test_idx),
            method="patient",
            pos_rate={k.replace("pos_rate_", ""): v
                      for k, v in info.items() if k.startswith("pos_rate_")},
            n_subjects={k.replace("n_subjects_", ""): v
                        for k, v in info.items() if k.startswith("n_subjects_")},
        )
        run.log_metrics({f"view_baseline_{k}": v
                         for k, v in (vb or {}).items() if isinstance(v, (int, float))})
        run.log_metrics({f"image_{k}": v for k, v in m_img.as_dict().items()})
        if m_ehr:
            run.log_metrics({f"ehr_{k}": v for k, v in m_ehr.as_dict().items()})
        run.log_metrics({"lift_over_view": float(lift), "best_epoch": res.best_epoch})
        for h in res.history:
            run.log_metrics({"train_loss": h.train_loss, "val_metric": h.val_metric},
                            step=h.epoch)
        if sub:
            run.log_subgroup("sex", {k: {"auroc": g["auroc"], "ece": g["ece"]}
                                     for k, g in sub.items()})
        run.log_artifact(p)  # 집계 지표 JSON 만 (환자 데이터 없음)
        print(f"    run_id {run.run_id}")

    print("\n" + "=" * 74)
    print("  Phase 2 완료")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
