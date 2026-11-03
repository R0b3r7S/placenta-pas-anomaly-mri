#!/usr/bin/env python3
"""
Impact analysis: re-run the Rebro anomaly validation using
the Seg-3 fine-tuned segmentation model's PREDICTED placenta masks instead of the
radiologist's GROUND-TRUTH masks -- with NO anomaly retraining. This tests whether
the PAS boundary-anomaly pipeline still works when the placenta mask is produced
fully automatically (the clinically realistic, no-manual-segmentation scenario).

Pipeline (all params identical to the phase3_150ep_lf00 leave-healthy-out run;
the ONLY thing that changes is the mask source):
  1. Build a predicted-mask dataset that mirrors DATASET_EXTERNAL_REBRO: images
     symlinked, masks = Seg-3 predicted masks (renamed <pid>_<sid>.png).
  2. extract_boundary_patches.py on those masks -> test_normal + test_anomaly.
  3. score.py with the SAME encoder_final/wgan_final (lf=0.0, kappa=1.0).
  4. Aggregate patch scores -> per-patient mean; compare GT vs predicted
     side-by-side + Rebro PAS-vs-healthy AUC.

Outputs:
  boundary_patches_loo/impact_predmask_lf00/      (patches + manifest)
  runs_fanogan_loo/impact_predmask_lf00/scores_kappa1.0/scores.csv
  comparison_results/impact_predmask_comparison.csv   + printed side-by-side.

Reusable, parallel (joblib workers passed through to extraction). No on-the-fly.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATASETS = PROJECT / "dataset" / "mri_png"
EXTRACT = HERE / "extract_boundary_patches.py"
SCORE = HERE / "score.py"
LABELS_CSV = PROJECT / "dataset" / "dicom_converted" / "patient_labels.csv"
PY = sys.executable

# --- fixed groups (identical to run_leave_healthy_out.py) -------------------
TEST_NORMAL = ["reb004", "reb005", "reb010", "reb011"]   # held-out healthy
TEST_ANOM = ["reb001", "reb007", "reb008"]               # PAS
DIAG = {"reb001": "PAS percreta", "reb007": "PAS acreta", "reb008": "PAS percreta+previa",
        "reb004": "previa", "reb005": "myoma", "reb010": "appendicitis", "reb011": "healthy"}
CONFOUNDER = "reb010"  # appendicitis — hot but not PAS; report AUC with & without

REF_RUN = PROJECT / "runs_fanogan_loo" / "phase3_150ep_lf00"
GT_SCORES = REF_RUN / "scores_kappa1.0" / "scores.csv"
ENCODER = REF_RUN / "encoder" / "encoder_final.pth"
WGAN = REF_RUN / "wgan" / "wgan_final.pth"
SEG_RUN = PROJECT / "runs" / "TSE_dynunet_afa_mixup_rebroFT" / "test_on_heldout" / "inference_raw_masks"
SRC_IMAGES = DATASETS / "DATASET_EXTERNAL_REBRO" / "images"

PRED_DS = DATASETS / "DATASET_REBRO_PREDMASK"
PATCH_ROOT = PROJECT / "boundary_patches_loo" / "impact_predmask_lf00"
SCORE_OUT = PROJECT / "runs_fanogan_loo" / "impact_predmask_lf00" / "scores_kappa1.0"


def run(cmd, desc):
    print(f"\n>>> {desc}\n    {' '.join(str(c) for c in cmd)}", flush=True)
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"FAILED: {desc}")


def build_pred_dataset() -> None:
    """Mirror DATASET_EXTERNAL_REBRO but with Seg-3 predicted masks. Only slices
    that have BOTH a source image and a predicted mask are linked (guarantees the
    predicted-mask run uses exactly the slices the seg model produced)."""
    if PRED_DS.exists():
        shutil.rmtree(PRED_DS)
    (PRED_DS / "images").mkdir(parents=True)
    (PRED_DS / "masks").mkdir(parents=True)
    counts = {}
    for pid in TEST_NORMAL + TEST_ANOM:
        preds = sorted((SEG_RUN / pid).glob("*_pred.png"))
        (PRED_DS / "images" / pid).mkdir()
        (PRED_DS / "masks" / pid).mkdir()
        n = 0
        for pm in preds:
            sid = pm.name.replace("_pred.png", "")          # e.g. reb001_019
            img = SRC_IMAGES / pid / f"{sid}.png"
            if not img.exists():
                continue
            (PRED_DS / "images" / pid / f"{sid}.png").symlink_to(img.resolve())
            (PRED_DS / "masks" / pid / f"{sid}.png").symlink_to(pm.resolve())
            n += 1
        counts[pid] = n
    print("  predicted-mask slices linked per patient:",
          " ".join(f"{k}={v}" for k, v in counts.items()))


def extract_group(patients, subdir, workers) -> None:
    manifest_out = PATCH_ROOT / f"manifest__{subdir}.csv"
    cmd = [PY, str(EXTRACT), "--mode", "extract",
           "--images_dir", str(PRED_DS / "images"),
           "--masks_dir", str(PRED_DS / "masks"),
           "--labels_csv", str(LABELS_CSV),
           "--out_root", str(PATCH_ROOT),
           "--manifest_out", str(manifest_out),
           "--patch_size", "64", "--stride_px", "32", "--lower_fraction", "0.0",
           "--workers", str(workers),
           "--patients", ",".join(patients),
           "--force_subdir", subdir]
    run(cmd, f"extract {subdir} (predicted masks, {len(patients)} patients)")


def patient_of(patch_path: str) -> str:
    parts = Path(str(patch_path)).stem.split("_")
    return parts[2] if parts[0] == "mendeley" else parts[0]


def per_patient_mean(scores_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    df["pid"] = df.patch_path.apply(patient_of)
    df = df[df.pid.isin(TEST_NORMAL + TEST_ANOM)].copy()
    g = df.groupby("pid").agg(mean_score=("score", "mean"),
                              n_patches=("score", "size")).reset_index()
    return g


def rebro_auc(tbl: pd.DataFrame, col: str, exclude=None) -> float:
    t = tbl.copy()
    if exclude:
        t = t[~t.pid.isin(exclude)]
    y = t.pid.isin(TEST_ANOM).astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, t[col]))


def make_figure(m: pd.DataFrame, path: Path) -> None:
    """Per-patient anomaly score, GT vs predicted mask; PAS bars outlined."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(m)); w = 0.38
    ax.bar(x - w / 2, m.gt_mean_score, w, label="GT mask", color="#c6dbef",
           edgecolor=["#d62728" if p else "none" for p in m.is_PAS], linewidth=1.6)
    ax.bar(x + w / 2, m.pred_mean_score, w, label="predicted mask (Seg-3)", color="#2171b5",
           edgecolor=["#d62728" if p else "none" for p in m.is_PAS], linewidth=1.6)
    lab = [f"{p}\n{d}" for p, d in zip(m.pid, m.diagnosis)]
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean anomaly score"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.18)  # headroom so the legend clears the tallest bars
    ax.legend(fontsize=8, loc="upper right", title="red outline = PAS", title_fontsize=8)
    a_gt = rebro_auc(m, "gt_mean_score", [CONFOUNDER]); a_pr = rebro_auc(m, "pred_mean_score", [CONFOUNDER])
    ax.set_title("Impact: anomaly score with GT vs predicted placenta mask\n"
                 f"PAS-vs-healthy AUC (excl. appendicitis) — GT {a_gt:.3f} / predicted {a_pr:.3f}",
                 fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--skip-extract", action="store_true",
                    help="reuse existing patches/scores, only rebuild the comparison")
    args = ap.parse_args()

    for p in (GT_SCORES, ENCODER, WGAN, SEG_RUN):
        if not p.exists():
            sys.exit(f"missing required input: {p}")

    if not args.skip_extract:
        print("=== [1/3] build predicted-mask dataset ===")
        build_pred_dataset()
        print("=== [2/3] extract boundary patches from predicted masks ===")
        if PATCH_ROOT.exists():
            shutil.rmtree(PATCH_ROOT)
        PATCH_ROOT.mkdir(parents=True)
        extract_group(TEST_NORMAL, "test_normal", args.workers)
        extract_group(TEST_ANOM, "test_anomaly", args.workers)
        print("=== [3/3] score predicted-mask patches (same encoder, kappa=1.0) ===")
        SCORE_OUT.mkdir(parents=True, exist_ok=True)
        run([PY, str(SCORE), "--encoder_ckpt", str(ENCODER), "--wgan_ckpt", str(WGAN),
             "--normal_root", str(PATCH_ROOT / "test_normal"),
             "--anom_root", str(PATCH_ROOT / "test_anomaly"),
             "--out_dir", str(SCORE_OUT), "--kappa", "1.0"],
            "score predicted-mask patches")

    # ---- comparison ----
    gt = per_patient_mean(GT_SCORES).rename(
        columns={"mean_score": "gt_mean_score", "n_patches": "gt_n"})
    pr = per_patient_mean(SCORE_OUT / "scores.csv").rename(
        columns={"mean_score": "pred_mean_score", "n_patches": "pred_n"})
    m = gt.merge(pr, on="pid", how="outer")
    m["diagnosis"] = m.pid.map(DIAG)
    m["is_PAS"] = m.pid.isin(TEST_ANOM)
    m["score_delta"] = m.pred_mean_score - m.gt_mean_score
    order = TEST_ANOM + TEST_NORMAL
    m["ord"] = m.pid.apply(lambda p: order.index(p) if p in order else 99)
    m = m.sort_values("ord").drop(columns="ord").reset_index(drop=True)

    out = PROJECT / "comparison_results"; out.mkdir(exist_ok=True)
    m.round(4).to_csv(out / "impact_predmask_comparison.csv", index=False)
    make_figure(m, out / "impact_predmask_comparison.png")

    print("\n=== IMPACT: anomaly scores with GT vs PREDICTED (Seg-3) placenta masks "
          "(per-patient mean, kappa=1.0, lf=0.0) ===")
    show = m[["pid", "diagnosis", "is_PAS", "gt_mean_score", "pred_mean_score",
              "score_delta", "gt_n", "pred_n"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nRebro PAS-vs-healthy patient AUC (mean-aggregation):")
    for label, exc in [("all 7 (incl. reb010 appendicitis)", None),
                       ("6, excl. reb010 appendicitis confounder", [CONFOUNDER])]:
        print(f"  {label:45s}  GT={rebro_auc(m,'gt_mean_score',exc):.3f}"
              f"   PRED={rebro_auc(m,'pred_mean_score',exc):.3f}")
    print(f"\nsaved -> {out / 'impact_predmask_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
