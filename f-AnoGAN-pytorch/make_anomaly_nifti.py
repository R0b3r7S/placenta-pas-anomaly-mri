#!/usr/bin/env python3
"""
Build an ITK/NIfTI anomaly SEGMENTATION for the Rebro PAS patients from the
f-AnoGAN per-patch scores, so a radiologist can open it over the real MRI in
ITK-SNAP and judge whether the flagged boundary regions make clinical sense.

Deliverable per patient (all NEW files; the original MRI / mask are never
touched), in the ORIGINAL MRI geometry (same grid/affine as mask.nii.gz):
  mri.nii.gz                original T2 volume (copied)
  placenta_mask.nii.gz      original placenta segmentation (copied)
  anomaly_binary.nii.gz     1 = anomalous boundary region, 0 = rest  (the ask)
  anomaly_continuous.nii.gz smoothed f-AnoGAN risk field (float, bonus)
  qc_overlay_<pid>.png       visual QC montage in OUR spine-left orientation
  README.txt                 how to open + what "1" means + provenance

Method (per masked slice):
  per-patch anomaly score A = A_R + kappa*A_D  (f-AnoGAN, kappa=1.0)
  -> splat each score at its patch centre, Gaussian-smooth (sigma) -> risk field
  -> threshold at the 95th percentile of TRUE-normal patch scores (held-out
     healthy reb004/005/011, EXCLUDING the reb010 appendicitis confounder)
  -> clip to the placenta mask dilated by ~5 mm (keeps the band on the
     placenta-myometrium interface where PAS invades) -> light dilation for
     visibility.
Scores are computed in our 512x512 spine-left space; the orientation transform
back to the original MRI grid is AUTO-DETECTED per patient by matching our
placenta mask to mask.nii.gz (IoU must be ~1), so there is no hard-coded flip.

joblib (--workers, default 20) + tqdm over patients.

Usage:
  conda run --no-capture-output -n monai_placenta python \\
      f-AnoGAN-pytorch/make_anomaly_nifti.py --run-tag phase3_150ep --lf 0.5
  # also the whole-contour config:
  ... --run-tag phase3_150ep --lf 0.0
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage.transform import resize as sk_resize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from joblib import Parallel, delayed
    from tqdm import tqdm
    _JOBLIB = True
except ImportError:
    _JOBLIB = False
    def tqdm(x, **k):
        return x

PROJECT = Path(__file__).resolve().parent.parent
DICOM = PROJECT / "dataset" / "dicom_converted" / "reb"
DS = PROJECT / "dataset" / "mri_png" / "DATASET_EXTERNAL_REBRO"

PAS = ["reb001", "reb007", "reb008"]           # all Rebro PAS
NORMAL_REF = ["reb004", "reb005", "reb011"]     # true normals (excl. reb010 appendicitis)
TRAIN_HEALTHY = ["reb003", "reb006", "reb009", "reb013", "reb014"]   # model TRAINED on these
HELDOUT_HEALTHY = ["reb004", "reb005", "reb010", "reb011"]           # held-out controls


def role_of(pid):
    """Role label so training patients are never mistaken for an independent check."""
    if pid in TRAIN_HEALTHY:
        return "TRAINING-SEEN"      # model saw this -> low by construction, NOT validation
    if pid in HELDOUT_HEALTHY:
        return "held-out-test"
    if pid in PAS:
        return "PAS"
    return "other"


def lf_tag(lf: float) -> str:
    return f"lf{lf}".replace(".", "")


# ---------- orientation: auto-detect our(row,col) -> nifti[:,:,z] transform ----
_CANDS = {
    "as_is":   lambda a: a,
    "fliplr":  lambda a: np.fliplr(a),
    "flipud":  lambda a: np.flipud(a),
    "T":       lambda a: a.T,
    "fliplr.T": lambda a: np.fliplr(a).T,     # inverse of nifti->ours = fliplr(x.T)
    "flipud.T": lambda a: np.flipud(a).T,
    "rot90":   lambda a: np.rot90(a),
    "rot90_3": lambda a: np.rot90(a, 3),
}


def _iou(a, b):
    a = a.astype(bool); b = b.astype(bool)
    if a.shape != b.shape:
        return -1.0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def detect_transform(ds_mask_slice, nif_mask_slice):
    """Return the function mapping OUR (row,col) array -> nifti[:,:,z] array,
    chosen by best IoU of the (transformed) placenta mask vs mask.nii.gz."""
    best_name, best_iou, best_fn = None, -1.0, None
    for name, fn in _CANDS.items():
        s = _iou(fn(ds_mask_slice), nif_mask_slice)
        if s > best_iou:
            best_name, best_iou, best_fn = name, s, fn
    return best_fn, best_name, best_iou


def _apply_map(arr512, op, is_mask):
    """Map a 512 DATASET-space array to the native nifti grid using a calibrated
    op = {S, crop=(h,w), fn}: resize 512->SxS (inverse of the resize-to-512),
    center-crop to (h,w) (inverse of pad-to-square), then orient (fn)."""
    S = op["S"]; h, w = op["crop"]
    order = 0 if is_mask else 1
    a = sk_resize(arr512.astype(np.float32), (S, S), order=order,
                  preserve_range=True, anti_aliasing=(order > 0))
    pr, pc = (S - h) // 2, (S - w) // 2
    a = a[pr:pr + h, pc:pc + w]
    if is_mask:
        a = a > 0.5
    return op["fn"](a)


def calibrate_map(ds_msk512, M_native):
    """Find the op (resize + center-crop + orientation) mapping the 512 DATASET
    placenta mask onto the native mask.nii.gz, maximizing IoU. Handles square,
    NON-SQUARE (pad-inverse crop), and native-512 (identity) uniformly."""
    H, W = M_native.shape
    S = max(H, W)
    best = None
    for crop in {(H, W), (W, H)}:
        for name, fn in _CANDS.items():
            op = {"S": S, "crop": crop, "fn": fn, "name": f"{name}|crop{crop}"}
            try:
                cand = _apply_map(ds_msk512, op, is_mask=True)
            except Exception:
                continue
            if cand.shape != M_native.shape:
                continue
            iou = _iou(cand, M_native)
            if best is None or iou > best[0]:
                best = (iou, op)
    return (best[1], best[0]) if best else (None, -1.0)


# ---------- heatmap in OUR 512 space ----------
def slice_heatmap(shape, cxs, cys, scores, sigma):
    h, w = shape
    heat = np.zeros((h, w), np.float32)
    wgt = np.zeros((h, w), np.float32)
    for cx, cy, sc in zip(cxs, cys, scores):
        y, x = int(round(cy)), int(round(cx))
        if 0 <= y < h and 0 <= x < w:
            heat[y, x] += float(sc); wgt[y, x] += 1.0
    heat = gaussian_filter(heat, sigma); wgt = gaussian_filter(wgt, sigma)
    return np.divide(heat, wgt, out=np.zeros_like(heat), where=wgt > 1e-6)


def jet(v):
    v = np.clip(v, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], -1)


def process_patient(pid, patient_rows, threshold, sigma, dilate_mm, band_px,
                    out_dir, vmax):
    """Build the anomaly volumes for one patient and write the package.
    The heatmap is computed in the 512 DATASET space; if the patient's ORIGINAL
    MRI is not 512x512 it is resized to the native grid (square patients only for
    now; non-square would need pad-inverse). Placenta IoU is checked per patient."""
    base = DICOM / pid
    nii_files = sorted((base / "nifti").glob("*.nii.gz"))
    if not nii_files or not (base / "mask.nii.gz").is_file():
        return {"pid": pid, "status": "missing nifti/mask"}
    mri = nib.load(str(nii_files[0]))
    msk = nib.load(str(base / "mask.nii.gz"))
    affine, header = msk.affine, msk.header
    M = (msk.get_fdata() > 0)
    H, W, Z = M.shape
    role = role_of(pid)
    # per-patient in-plane spacing -> dilation in px (native res, not the PAS 0.78)
    sp_inplane = float(np.sqrt((affine[:3, 0] ** 2).sum()))
    dilate_px = max(1, int(round(dilate_mm / sp_inplane)))

    cont = np.zeros((H, W, Z), np.float32)          # continuous risk (nifti grid)
    binv = np.zeros((H, W, Z), np.uint8)            # binary anomaly (nifti grid)
    op = None                                       # calibrated 512->native mapping
    placenta_iou = []
    qc = []   # (z, ds_img, ds_mask, ds_heat, ds_band) for the overlay montage

    ds_img_dir = DS / "images" / pid
    ds_msk_dir = DS / "masks" / pid

    for sid, sub in patient_rows.groupby("slice_id"):
        z = int(sid)
        if z < 0 or z >= Z:
            continue
        # our 512 slice image + placenta mask (spine-left)
        ip = ds_img_dir / f"{pid}_{z:03d}.png"
        mp = ds_msk_dir / f"{pid}_{z:03d}.png"
        if not ip.is_file() or not mp.is_file():
            continue
        ds_img = np.asarray(Image.open(ip).convert("L"))
        ds_msk = (np.asarray(Image.open(mp).convert("L")) > 0)

        heat = slice_heatmap(ds_img.shape, sub.center_x.values, sub.center_y.values,
                             sub.score.values, sigma)
        band = heat > threshold                                   # 512-space band (for QC)

        # calibrate the 512 -> native mapping once (per patient) on the placenta mask
        # (handles square, non-square pad-inverse, and native-512 uniformly)
        if op is None:
            op, _ = calibrate_map(ds_msk, M[:, :, z])
        if op is None:
            continue
        placenta_iou.append(_iou(_apply_map(ds_msk, op, True), M[:, :, z]))

        # -> native nifti grid + orientation, threshold, clip to placenta dilated ~5 mm
        nif_heat = _apply_map(heat, op, False)
        nif_band = nif_heat > threshold
        placenta_dil = binary_dilation(M[:, :, z], iterations=dilate_px)
        nif_band = nif_band & placenta_dil
        if band_px > 0:                                            # visibility dilation, re-clip
            nif_band = binary_dilation(nif_band, iterations=band_px) & placenta_dil
        cont[:, :, z] = nif_heat
        binv[:, :, z] = nif_band.astype(np.uint8)
        qc.append((z, ds_img, ds_msk, heat, band))

    out = out_dir / pid
    out.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(binv, affine, header), str(out / "anomaly_binary.nii.gz"))
    nib.save(nib.Nifti1Image(cont, affine, header), str(out / "anomaly_continuous.nii.gz"))
    shutil.copy(str(nii_files[0]), str(out / "mri.nii.gz"))
    shutil.copy(str(base / "mask.nii.gz"), str(out / "placenta_mask.nii.gz"))

    # ---- QC overlay montage (OUR spine-left orientation, most intuitive) ----
    qc.sort(key=lambda t: t[0])
    n = len(qc)
    if n:
        ncol = min(6, n); nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.6, nrow * 2.6))
        axes = np.atleast_1d(axes).ravel()
        for i, (z, img, pmask, heat, band) in enumerate(qc):
            ax = axes[i]
            rgb = np.stack([img] * 3, -1).astype(np.float32) / 255.0
            hn = np.clip(heat / vmax, 0, 1)
            m = heat > (0.15 * vmax)
            rgb[m] = 0.5 * rgb[m] + 0.5 * jet(hn)[m]               # continuous risk tint
            ax.imshow(rgb)
            # placenta contour (cyan) + anomaly band outline (red)
            ax.contour(pmask, levels=[0.5], colors="cyan", linewidths=0.6)
            if band.any():
                ax.contour(band, levels=[0.5], colors="red", linewidths=1.0)
            ax.set_title(f"z={z}", fontsize=7); ax.axis("off")
        for j in range(n, len(axes)):
            axes[j].axis("off")
        seen = "  [TRAINING - model saw this, low by construction]" if role == "TRAINING-SEEN" else ""
        fig.suptitle(f"{pid}  [{role}]{seen}   native={H}x{W}  "
                     f"anomaly QC (spine-left)  thr={threshold:.3f} sigma={sigma}\n"
                     f"cyan=placenta  red=anomaly band  tint=risk  "
                     f"(map={op['name'] if op else '?'}, placenta IoU={np.mean(placenta_iou):.3f})",
                     fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(str(out / f"qc_overlay_{pid}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    return {"pid": pid, "role": role, "status": "ok", "n_slices": n,
            "native": f"{H}x{W}", "map": op["name"] if op else "?",
            "placenta_iou": round(float(np.mean(placenta_iou)), 4) if placenta_iou else None,
            "anom_voxels": int((binv > 0).sum()),
            "placenta_voxels": int(M.sum()),
            "anom_frac_of_placenta": round((binv > 0).sum() / max(1, int(M.sum())), 4)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", default="phase3_150ep")
    p.add_argument("--lf", type=float, default=0.5, help="lower_fraction of the run")
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=8.0)
    p.add_argument("--thr-pct", type=float, default=95.0,
                   help="percentile of TRUE-normal patch scores used as the 1/0 threshold")
    p.add_argument("--dilate-mm", type=float, default=5.0,
                   help="clip anomaly to placenta dilated by this many mm (myometrium interface)")
    p.add_argument("--band-px", type=int, default=2, help="visibility dilation of the band (px)")
    p.add_argument("--patients", type=str, default=None,
                   help="comma-separated patient ids to render (default: the 3 PAS). "
                        "Pass held-out healthy (reb004,reb005,reb010,reb011) for the "
                        "healthy visual-inspection maps. vmax/threshold stay PAS/true-normal "
                        "based so healthy maps use the SAME colour scale as PAS.")
    p.add_argument("--extra-scores", type=str, default=None,
                   help="extra scores.csv to merge in (e.g. training-patient scores from "
                        "score.py on train_normal) so TRAINING-SEEN patients can be rendered.")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--out", default=None, type=Path)
    args = p.parse_args()

    tag = f"{args.run_tag}_{lf_tag(args.lf)}"
    manifest = pd.read_csv(PROJECT / "boundary_patches_loo" / tag / "manifest.csv")
    scores = pd.read_csv(PROJECT / "runs_fanogan_loo" / tag /
                         f"scores_kappa{args.kappa}" / "scores.csv")
    if args.extra_scores:                       # merge training-patient scores if provided
        extra = pd.read_csv(args.extra_scores)
        scores = pd.concat([scores, extra], ignore_index=True).drop_duplicates("patch_path")
    scores["patch_id"] = scores.patch_path.map(lambda s: Path(s).stem)
    ms = manifest.merge(scores[["patch_id", "score", "is_anom"]], on="patch_id", how="inner")
    ms = ms[~ms.patch_id.str.startswith("mendeley")]            # Rebro only
    if ms.empty:
        print("ERROR: no Rebro rows after join."); return 1

    # threshold from TRUE-normal patch scores (dilation is now per-patient, native)
    ref_scores = ms[ms.patient_id.isin(NORMAL_REF)].score.values
    threshold = float(np.percentile(ref_scores, args.thr_pct))
    vmax = float(np.quantile(ms[ms.patient_id.isin(PAS)].score.values, 0.90))
    print(f"  tag={tag}  dilate={args.dilate_mm:g}mm (per-patient native)  "
          f"threshold(p{args.thr_pct:g} of true-normal, n={len(ref_scores)})={threshold:.3f}")

    out_dir = args.out or (PROJECT / "shareable" / "anomaly_nifti_rebro" /
                           f"{lf_tag(args.lf)}_sigma{int(round(args.sigma))}")
    out_dir.mkdir(parents=True, exist_ok=True)

    render_patients = ([x.strip() for x in args.patients.split(",") if x.strip()]
                       if args.patients else PAS)
    missing = [pid for pid in render_patients if not (ms.patient_id == pid).any()]
    if missing:
        print(f"  NOTE: skipping {missing} (no scored patches in this run — "
              f"training-healthy patients are not scored).")
    print(f"  rendering patients: {[p for p in render_patients if p not in missing]}")
    tasks = [(pid, ms[ms.patient_id == pid].copy()) for pid in render_patients
             if (ms.patient_id == pid).any()]

    def _job(pid, rows):
        return process_patient(pid, rows, threshold, args.sigma, args.dilate_mm,
                               args.band_px, out_dir, vmax)

    if _JOBLIB and args.workers > 1 and len(tasks) > 1:
        results = Parallel(n_jobs=args.workers)(
            delayed(_job)(pid, rows) for pid, rows in tqdm(tasks, desc="patients"))
    else:
        results = [_job(pid, rows) for pid, rows in tqdm(tasks, desc="patients")]

    print("\n  results:")
    for r in sorted(results, key=lambda d: d["pid"]):
        print(f"    {r}")
    # role-labeled summary CSV (so TRAINING-SEEN patients are never confused with held-out)
    ok = [r for r in results if r.get("status") == "ok"]
    if ok:
        pd.DataFrame(ok)[["pid", "role", "native", "placenta_iou",
                          "anom_frac_of_placenta"]].sort_values(["role", "pid"]).to_csv(
            out_dir / "summary_roles.csv", index=False)
    # README for the doctor
    readme = out_dir / "README.txt"
    readme.write_text(
        "f-AnoGAN anomaly segmentation - Rebro PAS patients\n"
        "==================================================\n\n"
        f"Config: lower_fraction={args.lf}, {args.run_tag}, kappa={args.kappa}, "
        f"sigma={args.sigma}, threshold=p{args.thr_pct:g} of normal-boundary scores.\n\n"
        "Per patient folder (open in ITK-SNAP):\n"
        "  mri.nii.gz               - original T2 MRI (main image)\n"
        "  placenta_mask.nii.gz     - placenta segmentation (1=placenta)\n"
        "  anomaly_binary.nii.gz    - 1 = ANOMALOUS placenta-myometrium boundary region, 0 = rest\n"
        "  anomaly_continuous.nii.gz- smoothed anomaly risk (float; higher=more anomalous)\n"
        "  qc_overlay_<pid>.png     - quick visual check (spine-left view)\n\n"
        "How to view: load mri.nii.gz as the main image, then load\n"
        "anomaly_binary.nii.gz (and/or placenta_mask.nii.gz) as segmentations.\n\n"
        "Notes / caveats:\n"
        " - The anomaly is detected along the LOWER placenta boundary (placenta-\n"
        "   myometrium interface) by an unsupervised model trained only on non-PAS\n"
        "   placentas; '1' marks boundary regions that deviate from normal.\n"
        " - It is a COARSE region proposal (64x64 patch / 512 grid), a hint of\n"
        "   'look here', NOT a pixel-precise lesion outline.\n"
        " - The binary uses a fixed threshold; the continuous map can be re-\n"
        "   thresholded interactively in ITK-SNAP.\n")
    print(f"\nDONE. Package under {out_dir}/  (README.txt written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
