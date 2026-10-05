#!/usr/bin/env python3
"""
Leave-healthy-out experiment + lower_fraction ablation + kappa sweep for f-AnoGAN.

Key question: if we train only on a subset of healthy Rebro
patients, do the HELD-OUT healthy patients (especially previa-without-accreta,
reb004) score LOW while PAS scores HIGH? If yes, the anomaly signal is
pathology, not hospital/scanner.

Patient groups (fixed):
  Train-normal (5):  reb003 reb006 reb009 reb013 reb014   (clean normal placentas)
  Test-normal  (4):  reb004 reb005 reb010 reb011          (held-out healthy, label 0)
                     reb004 = placenta previa WITHOUT accreta  <-- KEY negative control
  Test-anomaly:      reb001 reb007 reb008                  (Rebro PAS, label 1)
                     + all Mendeley BTFE + SSH_TSE PAS      (label 1, extra positives)

Pipeline per lower_fraction value:
  1. Extract boundary patches into three group folders (uses --patients +
     --force_subdir so train-normal and test-normal healthy land separately):
       boundary_patches_loo/lf<F>/{train_normal,test_normal,test_anomaly}/
  2. Train WGAN-GP on train_normal/  (--epochs configurable)
  3. Train izi_f encoder (50000 iters) on train_normal/
  4. Score test_normal/ (label 0) vs test_anomaly/ (label 1)
  5. Kappa sweep {0.5, 1.0, 2.0} by re-scoring (no retrain)
  6. Per-patient summary + the KEY check: does reb004 score low?

Parallelism: the EXTRACTION step uses joblib (--workers, default 20) + tqdm.
Training/scoring are GPU-bound and run sequentially.

Usage:
  # Phase 1 — controlled quick test (150 epochs, lower_fraction 0.5 only):
  conda run -n monai_placenta python f-AnoGAN-pytorch/run_leave_healthy_out.py \\
      --lower-fractions 0.5 --epochs 150 --tag phase1_150ep

  # Phase 2 — full ablation benchmark (300 epochs, all four fractions):
  conda run -n monai_placenta python f-AnoGAN-pytorch/run_leave_healthy_out.py \\
      --lower-fractions 0.5 0.0 0.3 0.7 --epochs 300 --tag phase2_300ep
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    from tqdm import tqdm
    _JOBLIB = True
except ImportError:
    _JOBLIB = False
    def tqdm(x, **k):
        return x

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATASETS = PROJECT / "dataset" / "mri_png"
EXTRACT = HERE / "extract_boundary_patches.py"
TRAIN_WGAN = HERE / "train_wgan.py"
TRAIN_ENC = HERE / "train_encoder.py"
SCORE = HERE / "score.py"
LABELS_CSV = PROJECT / "dataset" / "dicom_converted" / "patient_labels.csv"

PY = sys.executable

# --- fixed patient groups ---------------------------------------------------
TRAIN_NORMAL = ["reb003", "reb006", "reb009", "reb013", "reb014"]
TEST_NORMAL  = ["reb004", "reb005", "reb010", "reb011"]   # reb004 = previa, no accreta
TEST_ANOM_REBRO = ["reb001", "reb007", "reb008"]
KEY_NEGATIVE = "reb004"   # previa-without-accreta — must score LOW

REBRO_DS = "DATASET_EXTERNAL_REBRO"
MENDELEY = [("DATASET_BTFE", "mendeley_btfe"), ("DATASET_SSH_TSE", "mendeley_tse")]


def run(cmd: list[str], desc: str) -> None:
    print(f"\n>>> {desc}\n    {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"FAILED: {desc} (exit {r.returncode})")


def mendeley_pas_ids(dataset: str) -> list[str]:
    """All Mendeley patient IDs (every Mendeley patient is PAS)."""
    img_root = DATASETS / dataset / "images"
    if not img_root.is_dir():
        return []
    return sorted(p.name for p in img_root.iterdir() if p.is_dir())


def extract_group(out_root: Path, dataset: str, patients: list[str],
                  subdir: str, lower_fraction: float, prefix: str,
                  workers: int) -> None:
    if not patients:
        return
    # Write each group's manifest to a DISTINCT file so successive groups do not
    # clobber one another (every group extracts into the same out_root). The
    # per-group manifests are concatenated into manifest.csv after all groups
    # finish (see combine_manifests), so no coordinate information is lost.
    manifest_out = out_root / f"manifest__{subdir}__{prefix or 'main'}.csv"
    cmd = [
        PY, str(EXTRACT), "--mode", "extract",
        "--images_dir", str(DATASETS / dataset / "images"),
        "--masks_dir",  str(DATASETS / dataset / "masks"),
        "--labels_csv", str(LABELS_CSV),
        "--out_root",   str(out_root),
        "--manifest_out", str(manifest_out),
        "--patch_size", "64",
        "--stride_px",  "32",
        "--lower_fraction", str(lower_fraction),
        "--workers",    str(workers),
        "--patients",   ",".join(patients),
        "--force_subdir", subdir,
    ]
    if prefix:
        cmd += ["--patch_prefix", prefix]
    run(cmd, f"extract {subdir} from {dataset} ({len(patients)} patients, lf={lower_fraction})")


def combine_manifests(patch_root: Path, workers: int = 20) -> None:
    """Concatenate all per-group manifest__*.csv into one manifest.csv so the
    full coordinate set (every group, every cohort) is preserved in one file."""
    parts = sorted(patch_root.glob("manifest__*.csv"))
    if not parts:
        return
    if _JOBLIB and workers > 1 and len(parts) > 1:
        frames = Parallel(n_jobs=workers)(
            delayed(pd.read_csv)(p) for p in tqdm(parts, desc="combine manifests"))
    else:
        frames = [pd.read_csv(p) for p in parts]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(patch_root / "manifest.csv", index=False)
    print(f"  combined {len(parts)} group manifests -> {patch_root / 'manifest.csv'} "
          f"({len(combined)} rows)")


def patient_of(patch_path: str) -> str:
    """Recover patient id from a patch filename stem.
    Stems look like '<pid>_<sid>_bNNN' or '<prefix>_<pid>_<sid>_bNNN'."""
    stem = Path(patch_path).stem
    parts = stem.split("_")
    # mendeley prefix patches: mendeley_btfe_sub001_29_b000 -> pid=sub001
    if parts[0] == "mendeley":
        return parts[2]
    return parts[0]


def score_once(encoder_ckpt, wgan_ckpt, test_normal, test_anom, out_dir,
               kappa, save_heatmaps=False):
    cmd = [
        PY, str(SCORE),
        "--encoder_ckpt", str(encoder_ckpt),
        "--wgan_ckpt",    str(wgan_ckpt),
        "--normal_root",  str(test_normal),
        "--anom_root",    str(test_anom),
        "--out_dir",      str(out_dir),
        "--kappa",        str(kappa),
    ]
    if save_heatmaps:
        cmd.append("--save_heatmaps")
    run(cmd, f"score (kappa={kappa}) -> {out_dir}")


def analyze(scores_csv: Path, lower_fraction: float, kappa: float) -> dict:
    """Per-patient + AUC analysis. Returns a summary dict."""
    df = pd.read_csv(scores_csv)
    df["pid"] = df.patch_path.apply(patient_of)
    # patch-level AUC
    summary = {"lower_fraction": lower_fraction, "kappa": kappa,
               "n_patches": len(df)}
    try:
        from sklearn.metrics import roc_auc_score
        summary["patch_auc"] = float(roc_auc_score(df.is_anom, df.score))
    except Exception:
        summary["patch_auc"] = float("nan")

    # per-patient mean
    pp = df.groupby(["pid", "is_anom"]).score.mean().reset_index()
    # patient-level AUC
    try:
        from sklearn.metrics import roc_auc_score
        summary["patient_auc"] = float(roc_auc_score(pp.is_anom, pp.score))
    except Exception:
        summary["patient_auc"] = float("nan")

    # KEY check: reb004 (previa, no accreta) mean score + rank among negatives
    neg = pp[pp.is_anom == 0].sort_values("score")
    pos = pp[pp.is_anom == 1]
    summary["test_normal_mean"] = float(neg.score.mean()) if len(neg) else float("nan")
    summary["test_normal_max"]  = float(neg.score.max()) if len(neg) else float("nan")
    summary["pas_mean"]         = float(pos.score.mean()) if len(pos) else float("nan")
    summary["pas_min"]          = float(pos.score.min()) if len(pos) else float("nan")
    key = pp[pp.pid == KEY_NEGATIVE]
    summary["reb004_score"] = float(key.score.iloc[0]) if len(key) else float("nan")
    # separation: does the lowest PAS exceed the highest held-out healthy?
    summary["clean_separation"] = bool(
        len(neg) and len(pos) and pos.score.min() > neg.score.max())
    summary["_per_patient"] = pp.to_dict("records")
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lower-fractions", nargs="+", type=float, default=[0.5],
                   help="lower_fraction values to run, in order (e.g. 0.5 0.0 0.3 0.7).")
    p.add_argument("--epochs", type=int, default=150,
                   help="WGAN epochs (150 for Phase 1 controlled test, 300 for benchmark).")
    p.add_argument("--encoder-iters", type=int, default=50000)
    p.add_argument("--kappas", nargs="+", type=float, default=[0.5, 1.0, 2.0],
                   help="kappa values for the re-scoring sweep.")
    p.add_argument("--score-kappa", type=float, default=1.0,
                   help="primary kappa used for the per-patient summary + heatmaps.")
    p.add_argument("--workers", type=int, default=20,
                   help="joblib workers for the extraction step.")
    p.add_argument("--tag", type=str, default="loo",
                   help="run tag, used in output dir names.")
    p.add_argument("--out-base", type=str, default="runs_fanogan_loo")
    args = p.parse_args()

    out_base = PROJECT / args.out_base
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  LEAVE-HEALTHY-OUT  tag={args.tag}  epochs={args.epochs}")
    print(f"  lower_fractions: {args.lower_fractions}")
    print(f"  train-normal (5): {TRAIN_NORMAL}")
    print(f"  test-normal  (4): {TEST_NORMAL}   (KEY: {KEY_NEGATIVE} = previa, no accreta)")
    print(f"  test-anomaly: Rebro {TEST_ANOM_REBRO} + all Mendeley PAS")
    print("=" * 80)

    all_summaries = []
    for lf in args.lower_fractions:
        lf_tag = f"lf{lf}".replace(".", "")
        patch_root = PROJECT / "boundary_patches_loo" / f"{args.tag}_{lf_tag}"
        # fresh extraction
        import shutil
        if patch_root.exists():
            shutil.rmtree(patch_root)
        patch_root.mkdir(parents=True, exist_ok=True)

        # 1. EXTRACT (joblib + tqdm inside extractor)
        extract_group(patch_root, REBRO_DS, TRAIN_NORMAL, "train_normal", lf, "", args.workers)
        extract_group(patch_root, REBRO_DS, TEST_NORMAL,  "test_normal",  lf, "", args.workers)
        extract_group(patch_root, REBRO_DS, TEST_ANOM_REBRO, "test_anomaly", lf, "", args.workers)
        for ds, prefix in MENDELEY:
            extract_group(patch_root, ds, mendeley_pas_ids(ds), "test_anomaly", lf, prefix, args.workers)

        # preserve EVERY group's coordinates in one manifest (no clobbering)
        combine_manifests(patch_root, args.workers)

        n_train = len(list((patch_root / "train_normal").glob("*.png")))
        n_tnorm = len(list((patch_root / "test_normal").glob("*.png")))
        n_tanom = len(list((patch_root / "test_anomaly").glob("*.png")))
        print(f"\n[lf={lf}] patches: train_normal={n_train}  "
              f"test_normal={n_tnorm}  test_anomaly={n_tanom}")

        run_dir = out_base / f"{args.tag}_{lf_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # 2. WGAN
        wgan_dir = run_dir / "wgan"
        run([PY, str(TRAIN_WGAN),
             "--data_root", str(patch_root / "train_normal"),
             "--out_dir",   str(wgan_dir),
             "--epochs",    str(args.epochs),
             "--batch_size", "64", "--num_workers", "4",
             "--amp", "--sample_every", "200", "--seed", "42"],
            f"train WGAN lf={lf} ({args.epochs} epochs)")
        wgan_ckpt = wgan_dir / "wgan_final.pth"

        # 3. ENCODER
        enc_dir = run_dir / "encoder"
        run([PY, str(TRAIN_ENC),
             "--data_root", str(patch_root / "train_normal"),
             "--wgan_ckpt", str(wgan_ckpt),
             "--out_dir",   str(enc_dir),
             "--iters",     str(args.encoder_iters),
             "--kappa",     "1.0",
             "--batch_size", "64", "--num_workers", "4",
             "--amp", "--seed", "42"],
            f"train encoder lf={lf}")
        enc_ckpt = enc_dir / "encoder_final.pth"

        # 4. SCORE (primary kappa, with heatmaps)
        score_dir = run_dir / f"scores_kappa{args.score_kappa}"
        score_once(enc_ckpt, wgan_ckpt,
                   patch_root / "test_normal", patch_root / "test_anomaly",
                   score_dir, args.score_kappa, save_heatmaps=True)
        primary = analyze(score_dir / "scores.csv", lf, args.score_kappa)
        all_summaries.append(primary)

        # 5. KAPPA SWEEP (re-score only)
        for kp in args.kappas:
            if abs(kp - args.score_kappa) < 1e-9:
                continue
            kdir = run_dir / f"scores_kappa{kp}"
            score_once(enc_ckpt, wgan_ckpt,
                       patch_root / "test_normal", patch_root / "test_anomaly",
                       kdir, kp, save_heatmaps=False)
            all_summaries.append(analyze(kdir / "scores.csv", lf, kp))

        # per-lf report
        print(f"\n{'='*70}\n  lf={lf}  RESULT (kappa={args.score_kappa})")
        print(f"  patch AUC={primary['patch_auc']:.4f}  patient AUC={primary['patient_auc']:.4f}")
        print(f"  held-out healthy mean={primary['test_normal_mean']:.3f} "
              f"(max={primary['test_normal_max']:.3f})")
        print(f"  PAS mean={primary['pas_mean']:.3f} (min={primary['pas_min']:.3f})")
        print(f"  KEY reb004 (previa,no accreta)={primary['reb004_score']:.3f}")
        print(f"  clean separation (lowest PAS > highest healthy)? "
              f"{'YES' if primary['clean_separation'] else 'NO'}")
        print("=" * 70)

    # global summary CSV
    flat = [{k: v for k, v in s.items() if k != "_per_patient"} for s in all_summaries]
    summ_df = pd.DataFrame(flat)
    out_csv = out_base / f"{args.tag}_summary.csv"
    summ_df.to_csv(out_csv, index=False)
    # also dump per-patient json
    with open(out_base / f"{args.tag}_per_patient.json", "w") as f:
        json.dump([s for s in all_summaries], f, indent=2, default=str)
    print(f"\nDONE. Summary -> {out_csv}")
    print(summ_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
