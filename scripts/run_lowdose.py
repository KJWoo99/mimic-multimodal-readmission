"""부가 분석: 흉부 X선 저선량 모사와 복원이 소견 분류(RAD-DINO 선형 프로브)에 주는 영향.

사전등록 `docs/prereg_lowdose.md`. 단계마다 결과를 파일로 남기므로 순서대로 한 번씩 돌리면 되고,
중간에 멈추면 그 단계부터 다시 돌림.

    python scripts/run_lowdose.py index              # 정면 영상 목록, 분할, 라벨
    python scripts/run_lowdose.py n0                 # 원본 잡음으로 N0 추정
    python scripts/run_lowdose.py grid --split train # 518 격자(깨끗한, 저선량). val, test 도 같게
    python scripts/run_lowdose.py check-grid         # 격자가 RAD-DINO 프로세서와 같은지
    python scripts/run_lowdose.py tune               # 고전 복원의 세기를 val PSNR 로 고름
    python scripts/run_lowdose.py classical          # test 에 고전 복원 셋
    python scripts/run_lowdose.py tune --grid wide   # 보조(사전등록 이탈): 넓힌 격자, tune_wide.json
    python scripts/run_lowdose.py classical --grid wide  # 보조: 넓힌 격자로 고른 세기, 조건 이름 bm3dwide_r4 등
    python scripts/run_lowdose.py dncnn --seed 1     # DnCNN 학습과 test 적용(시드 1, 2, 3)
    python scripts/run_lowdose.py embed --split test # RAD-DINO 특징(train 은 깨끗한 영상과 저선량만)
    python scripts/run_lowdose.py quality            # test 의 PSNR, SSIM
    python scripts/run_lowdose.py evaluate           # 프로브, 부트스트랩, 판정 -> outputs/lowdose/results.json

## DUA

잡음 영상, 복원 영상, 특징, 가중치, 영상 목록은 MIMIC 파생물임. `outputs/models/lowdose/` 에 두고 `.gitignore` 로
막음. `outputs/lowdose/` 에는 개수와 요약 수치만 씀.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 병렬 작업자(28개)마다 수치 라이브러리가 스레드를 또 띄우면 32코어에서 스레드가 수백 개가 되어 오히려 느려짐.
# numpy 를 불러오기 전에 작업자당 한 스레드로 묶음.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datapaths import CXR_JPG  # noqa: E402

OUT = ROOT / "outputs" / "lowdose"
WORK = ROOT / "outputs" / "models" / "lowdose"
MODEL_ID = "microsoft/rad-dino"
# 결과를 만든 판. 판을 주지 않으면 그날의 최신판을 받아 특징값이 달라질 수 있음.
MODEL_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"
SPLIT_SEED = 42
N0_SAMPLE = 500
TUNE_N = 1000
BOOT_N = 1000
# 조건 이름: 선량비 1/2, 1/4, 1/8 을 r2, r4, r8 로 씀. 민감도는 주 선량비(1/4)에서만.
DOSE_KEYS = {"r2": 0.5, "r4": 0.25, "r8": 0.125}
# 보조 부하 조건(사전등록 3절 5번): 1/32. val, test 에만 두고 DnCNN 학습과 다시 학습한 비교군에는 넣지 않음.
STRESS = {"r32": 1 / 32}
ALL_DOSES = {**DOSE_KEYS, **STRESS}
SENS = {"r4_n0half": ("n0", 0.5), "r4_n0x2": ("n0", 2.0), "r4_k3": ("k", 3.0), "r4_k6": ("k", 6.0)}
# 사전등록 4절: 조정 격자. 결과를 본 뒤 넓히지 않음.
TUNE_GRID = {"gaussian": (0.5, 0.75, 1.0, 1.5, 2.0),
             "nlm": (0.6, 0.8, 1.0, 1.2, 1.5),
             "bm3d": (0.75, 1.0, 1.25, 1.5)}
# 사전등록 이탈(prereg 수정 이력): 위 격자에서 12건 중 11건이 끝에 걸려, 양쪽으로 넓힌 격자를 보조 분석으로
# 따로 돌림. 원래 값을 모두 포함함. 주 분석(위 격자)은 바꾸지 않고, 결과는 "wide" 이름으로 따로 남김.
TUNE_GRID_WIDE = {"gaussian": (0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
                  "nlm": (0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0),
                  "bm3d": (0.1, 0.25, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)}
GRIDS = {"prereg": ("", TUNE_GRID), "wide": ("wide", TUNE_GRID_WIDE)}
# DnCNN 학습 설정. lr 1e-3 고정은 train 손실이 크게 튀고 한 시드가 배우지 못함(트러블슈팅 26).
# 공개 구현(KAIR train_dncnn.json: Adam 1e-4, 반씩 줄임)과 원 논문(학습률을 줄여 가며 50에폭)을 따라
# 1e-4 에서 시작해 val 이 멈추면 반으로 줄이고, 더 오래 기다림.
DNCNN = dict(patch=96, batch=64, steps=2000, max_epochs=60, patience=10, lr=1e-4,
             lr_factor=0.5, lr_patience=3, lr_min=1e-6, val_n=500)


def _json_default(o):
    # numpy 숫자(float32 등)는 json 이 못 씀. 결과를 다 계산한 뒤 저장에서 멈추지 않게 바꿔 줌.
    if hasattr(o, "item") and getattr(o, "ndim", 1) == 0:
        return o.item()
    raise TypeError(f"json 으로 못 쓴다: {type(o).__name__}")


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default) + "\n").encode("utf-8"))


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _index():
    import pandas as pd
    return pd.read_parquet(WORK / "index.parquet")


def _conds_for(split: str) -> list[str]:
    if split == "train":
        return ["clean", *DOSE_KEYS]
    return ["clean", *ALL_DOSES] + (list(SENS) if split == "test" else [])


def _grid_path(split: str, cond: str) -> Path:
    return WORK / f"grid_{split}_{cond}.npy"


class _Heartbeat:
    """CPU 만 쓰는 긴 단계에서 진행 줄을 일정 시간마다 찍음.

    로그가 오래 그대로면 교착과 구분되지 않음. 영상 몇 장마다 찍는 방식은 한 장이 오래 걸리면 간격도 길어지므로,
    계산과 따로 도는 스레드가 5분마다 지금까지 끝난 수를 찍음.
    """

    def __init__(self, total: int, every: float = 300.0):
        import threading
        self.done, self.total, self.every, self.t0 = 0, total, every, time.time()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.every):
            print(f"  (진행 중) {self.done:,}/{self.total:,}  {time.time() - self.t0:.0f}s", flush=True)

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join()
        return False


def _tagged(stem: str, tag: str) -> str:
    """주 분석은 tune.json, 넓힌 격자 보조 분석은 tune_wide.json 처럼 이름을 나눔."""
    return f"{stem}_{tag}.json" if tag else f"{stem}.json"


# ------------------------------------------------------------------ index
def cmd_index(args) -> int:
    import numpy as np
    import pandas as pd

    from lowdose import FINDINGS
    from splits import patient_split

    files = sorted(CXR_JPG.glob("files/p1*/p*/s*/*.jpg"))
    if not files:
        print(f"영상이 없다: {CXR_JPG}")
        return 1
    img = pd.DataFrame({"dicom_id": [f.stem for f in files],
                        "subject_id": [int(f.parent.parent.name[1:]) for f in files],
                        "study_id": [int(f.parent.name[1:]) for f in files],
                        "path": [str(f.relative_to(CXR_JPG)) for f in files]})
    meta = pd.read_csv(CXR_JPG / "mimic-cxr-2.0.0-metadata.csv.gz", usecols=["dicom_id", "ViewPosition"])
    lab = pd.read_csv(CXR_JPG / "mimic-cxr-2.0.0-chexpert.csv.gz", usecols=["study_id", *FINDINGS])
    df = img.merge(meta, on="dicom_id", how="left").merge(lab, on="study_id", how="left")
    n_all = len(df)
    df = df[df["ViewPosition"].isin(["PA", "AP"])].reset_index(drop=True)
    sp = patient_split(df["subject_id"].to_numpy(), test_size=0.2, val_size=0.1, seed=SPLIT_SEED)
    split = np.empty(len(df), dtype=object)
    for name, idx in (("train", sp.train_idx), ("val", sp.val_idx), ("test", sp.test_idx)):
        split[idx] = name
    df["split"] = split
    WORK.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WORK / "index.parquet", index=False)

    summ = {"n_images_downloaded": n_all, "n_frontal": len(df), "n_subjects": int(df["subject_id"].nunique()),
            "views": {k: int(v) for k, v in df["ViewPosition"].value_counts().items()},
            "splits": {}, "findings": {}}
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        summ["splits"][s] = {"images": len(d), "subjects": int(d["subject_id"].nunique())}
    for f, name in FINDINGS.items():
        summ["findings"][name] = {s: {"pos": int((df.loc[df["split"] == s, f] == 1).sum()),
                                      "uncertain": int((df.loc[df["split"] == s, f] == -1).sum())}
                                  for s in ("train", "val", "test")}
    _save_json(OUT / "index_summary.json", summ)
    print(json.dumps(summ, ensure_ascii=False)[:600])
    return 0


# ------------------------------------------------------------------ n0
def _read_gray(rel: str):
    import numpy as np
    from PIL import Image
    with Image.open(CXR_JPG / rel) as im:
        return np.asarray(im.convert("L"))


def _n0_one(rel: str):
    from denoise import estimate_noise
    v = _read_gray(rel)
    return estimate_noise(v / 255.0), float(v.mean())


def cmd_n0(args) -> int:
    from multiprocessing import Pool

    import numpy as np

    from lowdose import K_MAIN, n0_from_sigma

    df = _index()
    tr = df[df["split"] == "train"].sample(n=N0_SAMPLE, random_state=SPLIT_SEED)
    with Pool(args.workers) as pool:
        got = pool.map(_n0_one, tr["path"].tolist(), chunksize=8)
    sig = np.array([g[0] for g in got])
    mv = np.array([g[1] for g in got])
    out = {"sample": N0_SAMPLE, "sigma01_median": float(np.median(sig)),
           "sigma01_iqr": [float(np.percentile(sig, 25)), float(np.percentile(sig, 75))],
           "mean_v_median": float(np.median(mv)), "n0": {}}
    for name, k in (("main", K_MAIN), ("k3", 3.0), ("k6", 6.0)):
        n0s = np.array([n0_from_sigma(s, m, k) for s, m in zip(sig, mv, strict=True)])
        out["n0"][name] = {"k": k, "median": float(np.median(n0s)),
                           "iqr": [float(np.percentile(n0s, 25)), float(np.percentile(n0s, 75))]}
    _save_json(OUT / "n0.json", out)
    print(json.dumps(out, ensure_ascii=False))
    return 0


# ------------------------------------------------------------------ grid
_N0 = None


def _grid_one(job):
    import numpy as np

    from lowdose import K_MAIN, grid518, insert_lowdose_noise, noise_seed

    rel, dicom_id, conds = job
    v = _read_gray(rel)
    out = {}
    for c in conds:
        if c == "clean":
            out[c] = grid518(v)
            continue
        if c in ALL_DOSES:
            dose, n0, k, tag = ALL_DOSES[c], _N0["main"]["median"], K_MAIN, "main"
        else:
            kind, val = SENS[c]
            dose, tag = 0.25, c
            if kind == "n0":
                n0, k = _N0["main"]["median"] * val, K_MAIN
            else:
                key = "k3" if val == 3.0 else "k6"
                n0, k = _N0[key]["median"], val
        rng = np.random.default_rng(noise_seed(dicom_id, dose, tag))
        out[c] = grid518(insert_lowdose_noise(v, dose, n0, k, rng))
    return out


def _init_grid(n0):
    global _N0
    _N0 = n0


def cmd_grid(args) -> int:
    from multiprocessing import Pool

    import numpy as np

    df = _index()
    d = df[df["split"] == args.split].reset_index(drop=True)
    n0 = _load_json(OUT / "n0.json")["n0"]
    conds = _conds_for(args.split)
    mm = {c: np.lib.format.open_memmap(_grid_path(args.split, c), mode="w+", dtype=np.uint8,
                                       shape=(len(d), 518, 518)) for c in conds}
    jobs = [(r, i, conds) for r, i in zip(d["path"], d["dicom_id"], strict=True)]
    t0 = time.time()
    with Pool(args.workers, initializer=_init_grid, initargs=(n0,)) as pool:
        for i, res in enumerate(pool.imap(_grid_one, jobs, chunksize=4)):
            for c in conds:
                mm[c][i] = res[c]
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1:,}/{len(d):,}  {time.time() - t0:.0f}s", flush=True)
    for m in mm.values():
        m.flush()
    _save_json(OUT / f"grid_{args.split}.json", {"split": args.split, "n": len(d), "conds": conds,
                                                   "seconds": round(time.time() - t0)})
    print(f"끝 {args.split} {len(d):,}장 {conds}")
    return 0


# ------------------------------------------------------------------ check-grid
def cmd_check_grid(args) -> int:
    import numpy as np
    from PIL import Image
    from transformers import AutoImageProcessor

    from lowdose import grid518, to_model_input

    proc = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    df = _index()
    rows = df[df["split"] == "test"].head(args.n)
    diffs = []
    for rel in rows["path"]:
        v = _read_gray(rel)
        ref = proc(images=Image.fromarray(v).convert("RGB"), return_tensors="np")["pixel_values"][0]
        mine = to_model_input(grid518(v))[0]
        diffs.append(float(np.abs(ref - mine).max()))
    # 8비트 한 단계는 정규화 뒤 1/255/0.2583 = 0.0152 다. 그 안이면 같은 입력으로 봄.
    out = {"processor": type(proc).__name__, "n": len(diffs), "max_abs_diff": max(diffs),
           "one_level": 1 / 255 / 0.2583, "same": max(diffs) <= 1 / 255 / 0.2583 + 1e-6}
    _save_json(OUT / "check_grid.json", out)
    print(out)
    return 0 if out["same"] else 1


# ------------------------------------------------------------------ tune
_TUNE = {}


def _tune_one(i):
    from denoise import classical, estimate_noise, psnr

    ref = _TUNE["clean"][i] / 255.0
    res = {}
    for c in ALL_DOSES:
        x = _TUNE[c][i] / 255.0
        s = estimate_noise(x)
        res[c] = {"noisy": psnr(x, ref)}
        for m, grid in _TUNE["grid"].items():
            res[c][m] = [psnr(classical(m, x, f, s), ref) for f in grid]
    return res


def _init_tune(grid):
    import numpy as np
    _TUNE["grid"] = grid
    for c in ["clean", *ALL_DOSES]:
        _TUNE[c] = np.load(_grid_path("val", c), mmap_mode="r")


def cmd_tune(args) -> int:
    from multiprocessing import Pool

    import numpy as np

    tag, grid = GRIDS[args.grid]
    n = np.load(_grid_path("val", "clean"), mmap_mode="r").shape[0]
    idx = np.random.default_rng(SPLIT_SEED).choice(n, size=min(TUNE_N, n), replace=False)
    t0 = time.time()
    got = []
    # CPU 만 쓰는 단계가 오래 조용하면 멈춘 것과 구분되지 않아 5분마다 진행 줄을 찍음.
    with Pool(args.workers, initializer=_init_tune, initargs=(grid,)) as pool, _Heartbeat(len(idx)) as hb:
        for k, g in enumerate(pool.imap(_tune_one, idx.tolist(), chunksize=2), 1):
            got.append(g)
            hb.done = k
            if k % 10 == 0:
                print(f"  {k}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)
    out = {"n": len(idx), "grid_name": args.grid, "grid": grid, "seconds": round(time.time() - t0), "doses": {}}
    for c in ALL_DOSES:
        d = {"noisy_psnr": float(np.mean([g[c]["noisy"] for g in got])), "methods": {}}
        for m, factors in grid.items():
            means = np.mean([g[c][m] for g in got], axis=0)
            best = int(np.argmax(means))
            d["methods"][m] = {"psnr_by_factor": [float(x) for x in means], "factor": factors[best],
                               "psnr": float(means[best]), "at_grid_edge": best in (0, len(factors) - 1)}
        out["doses"][c] = d
    _save_json(OUT / _tagged("tune", tag), out)
    print(json.dumps(out, ensure_ascii=False)[:1500])
    return 0


# ------------------------------------------------------------------ classical
_CL = {}


def _cl_one(i):
    import numpy as np

    from denoise import classical, estimate_noise

    res = {}
    for c in ALL_DOSES:
        x = _CL[c][i] / 255.0
        s = estimate_noise(x)
        for m, f in _CL["factors"][c].items():
            y = classical(m, x, f, s)
            res[f"{m}{_CL['tag']}_{c}"] = np.clip(np.rint(y * 255), 0, 255).astype(np.uint8)
    return res


def _init_cl(factors, tag=""):
    import numpy as np
    _CL["factors"] = factors
    _CL["tag"] = tag
    for c in ALL_DOSES:
        _CL[c] = np.load(_grid_path("test", c), mmap_mode="r")


def cmd_classical(args) -> int:
    from multiprocessing import Pool

    import numpy as np

    tag, _ = GRIDS[args.grid]
    tune = _load_json(OUT / _tagged("tune", tag))
    factors = {c: {m: v["factor"] for m, v in tune["doses"][c]["methods"].items()} for c in ALL_DOSES}
    n = np.load(_grid_path("test", "clean"), mmap_mode="r").shape[0]
    names = [f"{m}{tag}_{c}" for c in ALL_DOSES for m in TUNE_GRID]
    mm = {k: np.lib.format.open_memmap(_grid_path("test", k), mode="w+", dtype=np.uint8, shape=(n, 518, 518))
          for k in names}
    t0 = time.time()
    # 한 장에 BM3D 가 선량마다 한 번씩 돌아 느림(단일 스레드 약 27초). 50장마다 찍고, 오래 조용하지 않게
    # 5분마다 진행 줄도 찍음.
    with Pool(args.workers, initializer=_init_cl, initargs=(factors, tag)) as pool, _Heartbeat(n) as hb:
        for i, res in enumerate(pool.imap(_cl_one, range(n), chunksize=4)):
            for k in names:
                mm[k][i] = res[k]
            hb.done = i + 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1:,}/{n:,}  {time.time() - t0:.0f}s", flush=True)
    for m in mm.values():
        m.flush()
    _save_json(OUT / _tagged("classical", tag),
               {"grid_name": args.grid, "factors": factors, "n": n, "seconds": round(time.time() - t0)})
    return 0


# ------------------------------------------------------------------ dncnn
def cmd_dncnn(args) -> int:
    import numpy as np
    import torch

    from denoise import DnCNN

    cfg = DNCNN
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr = {c: np.load(_grid_path("train", c), mmap_mode="r") for c in ["clean", *DOSE_KEYS]}
    va = {c: np.load(_grid_path("val", c), mmap_mode="r") for c in ["clean", *DOSE_KEYS]}
    n_tr = tr["clean"].shape[0]
    val_idx = np.random.default_rng(SPLIT_SEED + 1).choice(va["clean"].shape[0], size=cfg["val_n"], replace=False)
    model = DnCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg["lr_factor"],
                                                       patience=cfg["lr_patience"], min_lr=cfg["lr_min"])
    keys = list(DOSE_KEYS)
    P = cfg["patch"]

    def batch():
        ii = rng.integers(0, n_tr, cfg["batch"])
        cc = rng.integers(0, len(keys), cfg["batch"])
        yy = rng.integers(0, 518 - P, cfg["batch"])
        xx = rng.integers(0, 518 - P, cfg["batch"])
        noisy = np.stack([tr[keys[c]][i, y:y + P, x:x + P] for i, c, y, x in zip(ii, cc, yy, xx, strict=True)])
        clean = np.stack([tr["clean"][i, y:y + P, x:x + P] for i, y, x in zip(ii, yy, xx, strict=True)])
        return (torch.from_numpy(noisy).float().div(255).unsqueeze(1).to(dev),
                torch.from_numpy(clean).float().div(255).unsqueeze(1).to(dev))

    def val_loss(identity=False):
        # identity=True 는 복원 없이 잡음 영상 그대로의 오차임. 학습이 이것보다 낮아야 뭔가 배운 것임.
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for c in keys:
                for s in range(0, len(val_idx), 16):
                    ii = np.sort(val_idx[s:s + 16])
                    x = torch.from_numpy(va[c][ii]).float().div(255).unsqueeze(1).to(dev)
                    y = torch.from_numpy(va["clean"][ii]).float().div(255).unsqueeze(1).to(dev)
                    tot += torch.nn.functional.mse_loss(x if identity else model(x), y, reduction="sum").item()
                    n += y.numel()
        model.train()
        return tot / n

    val_identity = val_loss(identity=True)
    print(f"  잡음 그대로의 val 오차 {val_identity:.4e}", flush=True)

    WORK.mkdir(parents=True, exist_ok=True)
    wpath = WORK / f"dncnn_seed{args.seed}.pt"
    hist, best, bad, t0 = [], float("inf"), 0, time.time()
    for ep in range(1, cfg["max_epochs"] + 1):
        run = 0.0
        for _ in range(cfg["steps"]):
            x, y = batch()
            loss = torch.nn.functional.mse_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
        vl = val_loss()
        lr_now = opt.param_groups[0]["lr"]
        hist.append({"epoch": ep, "train_mse": run / cfg["steps"], "val_mse": vl, "lr": lr_now})
        print(f"  epoch {ep} train {run / cfg['steps']:.4e} val {vl:.4e} lr {lr_now:.1e}  {time.time() - t0:.0f}s",
              flush=True)
        sched.step(vl)
        if vl < best:
            best, bad = vl, 0
            torch.save(model.state_dict(), wpath)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(torch.load(wpath, map_location=dev))
    model.eval()
    te_conds = [*ALL_DOSES, *SENS]
    for c in te_conds:
        src = np.load(_grid_path("test", c), mmap_mode="r")
        dst = np.lib.format.open_memmap(_grid_path("test", f"dncnn{args.seed}_{c}"), mode="w+",
                                        dtype=np.uint8, shape=src.shape)
        with torch.no_grad():
            for s in range(0, src.shape[0], 16):
                x = torch.from_numpy(np.asarray(src[s:s + 16])).float().div(255).unsqueeze(1).to(dev)
                y = model(x).clamp(0, 1).mul(255).round().byte().squeeze(1).cpu().numpy()
                dst[s:s + 16] = y
        dst.flush()
    best_ep = min(hist, key=lambda h: h["val_mse"])["epoch"]
    _save_json(OUT / f"dncnn_seed{args.seed}.json",
               {"seed": args.seed, "config": cfg, "best_epoch": best_ep, "epochs_run": len(hist),
                "stopped_early": len(hist) < cfg["max_epochs"], "best_val_mse": best,
                "val_identity_mse": val_identity, "val_gain_db": 10 * float(np.log10(val_identity / best)),
                "seconds": round(time.time() - t0), "history": hist})
    return 0


# ------------------------------------------------------------------ embed
def cmd_embed(args) -> int:
    import numpy as np
    import torch
    from transformers import AutoModel

    from lowdose import to_model_input

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).to(dev).eval()
    if args.split == "train":
        conds = ["clean", *DOSE_KEYS]
    else:
        conds = sorted(p.stem[len(f"grid_{args.split}_"):] for p in WORK.glob(f"grid_{args.split}_*.npy"))
    done = []
    for c in conds:
        out = WORK / f"emb_{args.split}_{c}.npy"
        if out.exists() and not args.force:
            continue
        g = np.load(_grid_path(args.split, c), mmap_mode="r")
        feats = np.empty((g.shape[0], 768), dtype=np.float16)
        t0 = time.time()
        with torch.no_grad():
            for s in range(0, g.shape[0], args.batch):
                x = torch.from_numpy(to_model_input(np.asarray(g[s:s + args.batch]))).to(dev)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    h = model(pixel_values=x).last_hidden_state[:, 0]
                feats[s:s + args.batch] = h.float().cpu().numpy().astype(np.float16)
        np.save(out, feats)
        done.append(c)
        print(f"  {args.split} {c} {g.shape[0]:,}장 {time.time() - t0:.0f}s", flush=True)
    print(f"끝 {args.split}: 새로 {len(done)}개")
    return 0


# ------------------------------------------------------------------ quality
_Q = {}


def _q_one(i):
    from denoise import psnr, ssim
    ref = _Q["clean"][i] / 255.0
    return {c: (psnr(_Q[c][i] / 255.0, ref), ssim(_Q[c][i] / 255.0, ref)) for c in _Q["conds"]}


def _init_q(conds):
    import numpy as np
    _Q["conds"] = conds
    for c in ["clean", *conds]:
        _Q[c] = np.load(_grid_path("test", c), mmap_mode="r")


def cmd_quality(args) -> int:
    from multiprocessing import Pool

    import numpy as np

    conds = sorted(p.stem[len("grid_test_"):] for p in WORK.glob("grid_test_*.npy") if p.stem != "grid_test_clean")
    n = np.load(_grid_path("test", "clean"), mmap_mode="r").shape[0]
    got, t0 = [], time.time()
    with Pool(args.workers, initializer=_init_q, initargs=(conds,)) as pool:
        for k, g in enumerate(pool.imap(_q_one, range(n), chunksize=8), 1):
            got.append(g)
            if k % 500 == 0:
                print(f"  {k:,}/{n:,}  {time.time() - t0:.0f}s", flush=True)
    out = {"n": n, "conds": {c: {"psnr": float(np.mean([g[c][0] for g in got])),
                                 "ssim": float(np.mean([g[c][1] for g in got]))} for c in conds}}
    _save_json(OUT / "quality.json", out)
    return 0


# ------------------------------------------------------------------ evaluate
def _auc(y, s):
    """순위로 계산한 AUROC(동점은 평균 순위). sklearn roc_auc_score 와 같은 값이고 부트스트랩에서 빠름."""
    import numpy as np
    from scipy.stats import rankdata
    pos = y == 1
    n1 = int(pos.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cmd_evaluate(args) -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from lowdose import FINDINGS, finding_labels

    df = _index()
    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    emb = lambda split, c: np.load(WORK / f"emb_{split}_{c}.npy").astype(np.float32)  # noqa: E731
    conds = sorted(p.stem[len("emb_test_"):] for p in WORK.glob("emb_test_*.npy"))
    X_te = {c: emb("test", c) for c in conds}
    X_tr = {c: emb("train", c) for c in ["clean", *DOSE_KEYS]}

    def probe(X, y):
        m = ~np.isnan(y)
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X[m], y[m])

    scores, base_scores, labels = {}, {}, {}
    for mode in ("ignore", "zeros"):
        for f in FINDINGS:
            ytr = finding_labels(tr[f].to_numpy(), mode)
            yte = finding_labels(te[f].to_numpy(), mode)
            labels[(mode, f)] = yte
            clf = probe(X_tr["clean"], ytr)
            print(f"  프로브 {mode} {f}", flush=True)
            for c in conds:
                scores[(mode, f, c)] = clf.predict_proba(X_te[c])[:, 1]
            if mode == "ignore":
                for d in DOSE_KEYS:
                    base_scores[(f, d)] = probe(X_tr[d], ytr).predict_proba(X_te[d])[:, 1]

    def mean_auc(idx, cond, mode="ignore", base=False):
        vals = []
        for f in FINDINGS:
            y = labels[(mode, f)][idx]
            s = (base_scores[(f, cond)] if base else scores[(mode, f, cond)])[idx]
            m = ~np.isnan(y)
            vals.append(_auc(y[m], s[m]))
        return float(np.mean(vals)), vals

    all_idx = np.arange(len(te))
    point = {c: mean_auc(all_idx, c) for c in conds}
    base_point = {d: mean_auc(all_idx, d, base=True) for d in DOSE_KEYS}
    zeros_point = {c: mean_auc(all_idx, c, mode="zeros")[0] for c in ["clean", *ALL_DOSES]}
    strata = {v: np.where(te["ViewPosition"].to_numpy() == v)[0] for v in ("AP", "PA")}
    view_point = {v: {c: mean_auc(ix, c)[0] for c in ["clean", *ALL_DOSES]} for v, ix in strata.items()}

    # 환자 단위 짝지은 부트스트랩: 한 표본에서 모든 조건을 같이 측정함.
    subj = te["subject_id"].to_numpy()
    uniq, inv = np.unique(subj, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(SPLIT_SEED)
    boot = {c: [] for c in conds}
    boot_base = {d: [] for d in DOSE_KEYS}
    t0 = time.time()
    for b in range(BOOT_N):
        pick = rng.integers(0, len(uniq), len(uniq))
        ix = np.concatenate([members[k] for k in pick])
        for c in conds:
            boot[c].append(mean_auc(ix, c)[0])
        for d in DOSE_KEYS:
            boot_base[d].append(mean_auc(ix, d, base=True)[0])
        if (b + 1) % 100 == 0:
            print(f"  부트스트랩 {b + 1}/{BOOT_N}  {time.time() - t0:.0f}s", flush=True)
    boot = {c: np.array(v) for c, v in boot.items()}
    boot_base = {d: np.array(v) for d, v in boot_base.items()}
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731

    def dn_mean(d):
        ks = [f"dncnn{s}_{d}" for s in (1, 2, 3)]
        ks = [k for k in ks if k in boot]
        return (float(np.mean([point[k][0] for k in ks])), np.mean([boot[k] for k in ks], axis=0)) if ks else (None, None)

    methods = {m: (lambda d, m=m: (point[f"{m}_{d}"][0], boot[f"{m}_{d}"])) for m in TUNE_GRID}
    methods["dncnn"] = dn_mean
    # 넓힌 격자(사전등록 이탈, 보조 분석)는 산출물이 모두 있을 때만 따로 적음.
    for m in TUNE_GRID:
        if all(f"{m}wide_{d}" in boot for d in ALL_DOSES):
            methods[f"{m}wide"] = lambda d, m=m: (point[f"{m}wide_{d}"][0], boot[f"{m}wide_{d}"])
    res = {"n_test": len(te), "n_subjects": len(uniq), "boot": BOOT_N, "findings": list(FINDINGS.values()),
           "auroc": {c: {"mean": point[c][0], "ci": ci(boot[c]),
                         "by_finding": dict(zip(FINDINGS.values(), point[c][1], strict=True))} for c in conds},
           "baseline_retrained": {d: {"mean": base_point[d][0], "ci": ci(boot_base[d]),
                                      "by_finding": dict(zip(FINDINGS.values(), base_point[d][1], strict=True))} for d in DOSE_KEYS},
           "u_zeros_mean_auroc": zeros_point, "by_view_mean_auroc": view_point, "doses": {}}
    for d in ALL_DOSES:
        drop = boot["clean"] - boot[d]
        drop_pt = point["clean"][0] - point[d][0]
        gate = bool(np.percentile(drop, 2.5) > 0)
        dd = {"drop": drop_pt, "drop_ci": ci(drop), "drop_confirmed": gate, "methods": {}}
        for m, fn in methods.items():
            pt, bs = fn(d)
            if pt is None:
                continue
            rec_pt = (pt - point[d][0]) / drop_pt if drop_pt != 0 else None
            rec_bs = (bs - boot[d]) / np.where(drop == 0, np.nan, drop)
            has_base = d in boot_base
            dd["methods"][m] = {"auroc": pt, "recovery": rec_pt if gate else None,
                                "recovery_ci": ci(rec_bs[~np.isnan(rec_bs)]) if gate else None,
                                "vs_retrained": pt - base_point[d][0] if has_base else None,
                                "vs_retrained_ci": ci(bs - boot_base[d]) if has_base else None}
        res["doses"][d] = dd
    prim = res["doses"]["r4"]
    pm = prim["methods"].get("dncnn")
    if not prim["drop_confirmed"]:
        verdict = "선량비 1/4 에서는 저선량이 분류를 해치지 않았다(하락의 95% 신뢰구간이 0 을 포함)"
    elif pm is None:
        verdict = "DnCNN 결과가 없다"
    else:
        lo, hi = pm["vs_retrained_ci"]
        cmp_ = "복원이 낫다" if lo > 0 else ("다시 학습하는 편이 낫다" if hi < 0 else "복원과 다시 학습의 차이를 확인하지 못했다")
        verdict = f"회복률 {pm['recovery']:.3f} [{pm['recovery_ci'][0]:.3f}, {pm['recovery_ci'][1]:.3f}], {cmp_}"
    res["primary"] = {"dose": "r4", "method": "dncnn(시드 3 평균)", "verdict": verdict}

    qpath = OUT / "quality.json"
    if qpath.exists():
        q = _load_json(qpath)["conds"]
        flips = []
        for d in ALL_DOSES:
            for m in [*TUNE_GRID, *(f"{x}wide" for x in TUNE_GRID)]:
                k = f"{m}_{d}"
                if k in q and q[k]["psnr"] > q[d]["psnr"]:
                    for fi, fname in enumerate(FINDINGS.values()):
                        if point[k][1][fi] < point[d][1][fi]:
                            flips.append({"method": m, "dose": d, "finding": fname})
            ks = [f"dncnn{s}_{d}" for s in (1, 2, 3) if f"dncnn{s}_{d}" in q]
            if ks and np.mean([q[k]["psnr"] for k in ks]) > q[d]["psnr"]:
                for fi, fname in enumerate(FINDINGS.values()):
                    if np.mean([point[k][1][fi] for k in ks]) < point[d][1][fi]:
                        flips.append({"method": "dncnn", "dose": d, "finding": fname})
        res["psnr_up_auroc_down"] = {"count": len(flips), "cases": flips}
    _save_json(OUT / "results.json", res)
    print(res["primary"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    for name in ("n0", "tune", "classical", "quality"):
        sp = sub.add_parser(name)
        sp.add_argument("--workers", type=int, default=28)
        if name in ("tune", "classical"):
            sp.add_argument("--grid", choices=list(GRIDS), default="prereg")
    g = sub.add_parser("grid")
    g.add_argument("--split", choices=["train", "val", "test"], required=True)
    g.add_argument("--workers", type=int, default=28)
    sub.add_parser("check-grid").add_argument("--n", type=int, default=20)
    d = sub.add_parser("dncnn")
    d.add_argument("--seed", type=int, choices=[1, 2, 3], required=True)
    e = sub.add_parser("embed")
    e.add_argument("--split", choices=["train", "test"], required=True)
    e.add_argument("--batch", type=int, default=32)
    e.add_argument("--force", action="store_true")
    sub.add_parser("evaluate")
    args = ap.parse_args()
    return {"index": cmd_index, "n0": cmd_n0, "grid": cmd_grid, "check-grid": cmd_check_grid,
            "tune": cmd_tune, "classical": cmd_classical, "dncnn": cmd_dncnn, "embed": cmd_embed,
            "quality": cmd_quality, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
