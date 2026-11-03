#!/usr/bin/env python3
"""
Per-patient slice counts for the f-AnoGAN cohorts, computed DIRECTLY from the
masks (independent of any benchmark run). For every patient it reports:
  total_slices       all <pid>_*.png images present
  slices_with_mask   slices whose placenta mask has >= min_blob_size nonzero px
                     (>=200 px is the threshold the boundary extractor uses, i.e.
                     the slices that actually contribute patches to the score)

Output:
  <out>/dataset_slice_summary.csv     one row per (cohort, patient)
  + printed per-cohort distribution and the full per-patient list.

joblib (--workers, default 20) + tqdm over patients.

Usage:
  conda run --no-capture-output -n monai_placenta python \\
      f-AnoGAN-pytorch/dataset_slice_summary.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    from joblib import Parallel, delayed
    from tqdm import tqdm
    _JOBLIB = True
except ImportError:
    _JOBLIB = False
    def tqdm(x, **k):
        return x

PROJECT = Path(__file__).resolve().parent.parent
DATASETS = PROJECT / "dataset" / "mri_png"
COHORTS = {
    "rebro":         "DATASET_EXTERNAL_REBRO",
    "mendeley_btfe": "DATASET_BTFE",
    "mendeley_tse":  "DATASET_SSH_TSE",
}
MIN_BLOB = 200


def count_patient(cohort, pid, img_dir, msk_dir):
    imgs = sorted(img_dir.glob(f"{pid}_*.png"))
    with_mask = 0
    for ip in imgs:
        mp = msk_dir / ip.name
        if mp.is_file():
            m = np.asarray(Image.open(mp).convert("L"))
            if int((m > 0).sum()) >= MIN_BLOB:
                with_mask += 1
    return {"cohort": cohort, "patient_id": pid,
            "total_slices": len(imgs), "slices_with_mask": with_mask}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=PROJECT / "shareable" / "figures", type=Path)
    p.add_argument("--workers", type=int, default=20)
    args = p.parse_args()

    tasks = []
    for cohort, ds in COHORTS.items():
        img_root = DATASETS / ds / "images"
        msk_root = DATASETS / ds / "masks"
        if not img_root.is_dir():
            print(f"  SKIP {cohort}: {img_root} not found")
            continue
        for pdir in sorted(x for x in img_root.iterdir() if x.is_dir()):
            tasks.append((cohort, pdir.name, pdir, msk_root / pdir.name))

    if _JOBLIB and args.workers > 1 and len(tasks) > 1:
        rows = Parallel(n_jobs=args.workers)(
            delayed(count_patient)(c, pid, idir, mdir)
            for (c, pid, idir, mdir) in tqdm(tasks, desc="patients"))
    else:
        rows = [count_patient(c, pid, idir, mdir)
                for (c, pid, idir, mdir) in tqdm(tasks, desc="patients")]

    df = pd.DataFrame(rows).sort_values(["cohort", "patient_id"]).reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "dataset_slice_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== slices_with_mask per patient — distribution by cohort ===")
    for coh, sub in df.groupby("cohort"):
        s = sub.slices_with_mask
        print(f"{coh:14s}  n_patients={len(sub):3d}  total_slices={int(sub.total_slices.sum())}"
              f"  slices_with_mask: sum={int(s.sum())}  min={s.min()}  "
              f"median={int(s.median())}  max={s.max()}")

    print("\n=== full per-patient list (slices_with_mask | total_slices) ===")
    for coh, sub in df.groupby("cohort"):
        print(f"\n--- {coh} (n={len(sub)}) ---")
        cells = [f"{r.patient_id}={r.slices_with_mask}/{r.total_slices}"
                 for r in sub.itertuples()]
        # wrap 5 per line for readability
        for i in range(0, len(cells), 5):
            print("  " + "   ".join(cells[i:i + 5]))
    print(f"\nDONE. CSV -> {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
