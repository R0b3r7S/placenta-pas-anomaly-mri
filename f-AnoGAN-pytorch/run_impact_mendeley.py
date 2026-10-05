#!/usr/bin/env python3
"""
Fig-6 Mendeley row: manual (GT) vs algorithmic (predicted) placenta mask ->
effect on the f-AnoGAN anomaly score, on the Mendeley SSH_TSE cohort.

To avoid segmentation-training leakage we use ONLY the 19 Mendeley TEST patients
(truly unseen by the seg model; the 91 train + 20 val patients are excluded --
val was used for early stopping so it is not clean). Predicted masks come from
the base TSE_dynunet_afa_mixup model (runs/..._mendeleyTSE_pred). Anomaly scoring
uses the SAME honest encoder as everything else (phase3_150ep_lf00).

Mirrors run_impact_predmask.py (Rebro). AUC difference is NOT computed for
Mendeley (positive-only, no normals). Only mean |Δ anomaly score|.

Output: comparison_results/impact_mendeley_comparison.csv
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATASETS = PROJECT / "dataset" / "mri_png"
EXTRACT = HERE / "extract_boundary_patches.py"
SCORE = HERE / "score.py"
LABELS = PROJECT / "dataset" / "dicom_converted" / "patient_labels.csv"
PY = sys.executable

TSE_IMG = DATASETS / "DATASET_SSH_TSE" / "images"
PRED = PROJECT / "runs" / "TSE_dynunet_afa_mixup_mendeleyTSE_pred" / "inference_raw_masks"
REF = PROJECT / "runs_fanogan_loo" / "phase3_150ep_lf00"
ENC = REF / "encoder" / "encoder_final.pth"
WGAN = REF / "wgan" / "wgan_final.pth"
GT_SCORES = REF / "scores_kappa1.0" / "scores.csv"
PRED_DS = DATASETS / "DATASET_MENDELEY_TSE_PREDMASK"
PATCH_ROOT = PROJECT / "boundary_patches_loo" / "impact_mendeley_test_lf00"
SCORE_OUT = PROJECT / "runs_fanogan_loo" / "impact_mendeley_test_lf00" / "scores_kappa1.0"


def run(cmd, desc):
    print(f"\n>>> {desc}", flush=True)
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"FAILED: {desc}")


def test_patients():
    return sorted(json.load(open(DATASETS / "DATASET_SSH_TSE" / "splits.json"))["test"])


def build_pred_dataset(pids):
    if PRED_DS.exists():
        shutil.rmtree(PRED_DS)
    (PRED_DS / "images").mkdir(parents=True); (PRED_DS / "masks").mkdir(parents=True)
    n = 0
    for pid in pids:
        (PRED_DS / "images" / pid).mkdir(); (PRED_DS / "masks" / pid).mkdir()
        for pm in sorted((PRED / pid).glob("*_pred.png")):
            sid = pm.name.replace("_pred.png", "")
            img = TSE_IMG / pid / f"{sid}.png"
            if not img.exists():
                continue
            (PRED_DS / "images" / pid / f"{sid}.png").symlink_to(img.resolve())
            (PRED_DS / "masks" / pid / f"{sid}.png").symlink_to(pm.resolve())
            n += 1
    print(f"  linked {n} predicted-mask slices for {len(pids)} test patients")


def pid_of(p):
    s = Path(str(p)).stem.split("_")
    return s[2] if s[0] == "mendeley" else s[0]


def gt_means(pids):
    d = pd.read_csv(GT_SCORES)
    parts = d.patch_path.apply(lambda p: Path(str(p)).stem.split("_"))
    d = d[parts.apply(lambda s: s[0] == "mendeley" and s[1] == "tse")].copy()
    d["pid"] = parts[d.index].apply(lambda s: s[2])
    d = d[d.pid.isin(pids)]
    return d.groupby("pid").score.mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--skip-extract", action="store_true"); args = ap.parse_args()
    pids = test_patients()
    print(f"Mendeley TEST patients (unseen by seg model): {len(pids)}")

    if not args.skip_extract:
        build_pred_dataset(pids)
        if PATCH_ROOT.exists():
            shutil.rmtree(PATCH_ROOT)
        PATCH_ROOT.mkdir(parents=True)
        run([PY, str(EXTRACT), "--mode", "extract",
             "--images_dir", str(PRED_DS / "images"), "--masks_dir", str(PRED_DS / "masks"),
             "--labels_csv", str(LABELS), "--out_root", str(PATCH_ROOT),
             "--manifest_out", str(PATCH_ROOT / "manifest.csv"),
             "--patch_size", "64", "--stride_px", "32", "--lower_fraction", "0.0",
             "--workers", str(args.workers), "--patients", ",".join(pids),
             "--force_subdir", "test_anomaly"], "extract patches from predicted Mendeley masks")
        SCORE_OUT.mkdir(parents=True, exist_ok=True)
        run([PY, str(SCORE), "--encoder_ckpt", str(ENC), "--wgan_ckpt", str(WGAN),
             "--anom_root", str(PATCH_ROOT / "test_anomaly"),
             "--out_dir", str(SCORE_OUT), "--kappa", "1.0"], "score predicted-mask patches")

    ps = pd.read_csv(SCORE_OUT / "scores.csv")
    ps["pid"] = ps.patch_path.apply(pid_of)
    pred = ps.groupby("pid").score.mean()
    gt = gt_means(pids)
    common = [p for p in pids if p in gt.index and p in pred.index]
    m = pd.DataFrame({"pid": common,
                      "gt_mean_score": [round(gt[p], 4) for p in common],
                      "pred_mean_score": [round(pred[p], 4) for p in common]})
    m["abs_diff"] = (m.gt_mean_score - m.pred_mean_score).abs()
    m.round(4).to_csv(PROJECT / "comparison_results" / "impact_mendeley_comparison.csv", index=False)

    print(f"\n=== Mendeley (19 test) manual vs predicted mask ===")
    print(f"  patients compared: {len(common)}")
    print(f"  mean |Δ anomaly score|: {m.abs_diff.mean():.4f}")
    print(f"  GT mean score {m.gt_mean_score.mean():.3f}  vs predicted {m.pred_mean_score.mean():.3f}")
    print(f"  (AUC difference: n/a for Mendeley -- positive-only, no normals)")
    print(f"\nsaved -> comparison_results/impact_mendeley_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
