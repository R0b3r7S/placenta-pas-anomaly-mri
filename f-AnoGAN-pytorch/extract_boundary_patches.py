#!/usr/bin/env python3
"""
Extract boundary-aligned patches from placenta segmentation masks for f-AnoGAN.

Pipeline (one slice at a time):
  1. Read 512x512 grayscale MRI image + binary placenta mask.
  2. Find the largest contour of the placenta (sub-pixel, scikit-image).
  3. Filter to the LOWER portion of that contour — the placenta-myometrium
     interface side (configurable: --lower_fraction; default 0.5 = bottom half
     of the placenta's bounding box). This is "donja granica placente".
  4. Resample the lower contour at constant arc-length (--stride_px pixels).
  5. At each sample point, compute the local tangent angle from neighbours,
     then extract a --patch_size × --patch_size patch from the MRI rotated so
     the boundary runs horizontally through the patch centre ("izravnamo ih").
  6. Write each patch as a PNG and one manifest row per patch capturing the
     original location + orientation (needed later to project scores back to a
     full-slice heatmap).

Modes:
  --mode extract       (default) Produces patches + manifest.csv.
  --mode project_back  Reads f-AnoGAN's scores.csv + manifest.csv and produces
                       per-slice heatmap PNGs by Gaussian-splatting each
                       patch's anomaly score onto its original (x, y) location.

Optional labels split:
  --labels_csv patient_labels.csv with columns: patient_id,is_pas[,is_previa]
                       Patches from is_pas=0 patients go to normal_patches/,
                       is_pas=1 to anomalous_patches/. If not given, everything
                       lands in a single folder.

Typical training workflow:
  # 1. extract patches (uses ground-truth masks; configurable to use predicted)
  python f-AnoGAN-pytorch/extract_boundary_patches.py \\
      --images_dir dataset/mri_png/DATASET_BTFE/images/ \\
      --masks_dir  dataset/mri_png/DATASET_BTFE/masks/ \\
      --labels_csv patient_labels.csv \\
      --out_root   dataset/boundary_patches/BTFE/ \\
      --patch_size 64 --stride_px 32 --lower_fraction 0.5

  # 2. train f-AnoGAN on the normal patches
  python f-AnoGAN-pytorch/train_wgan.py     --data_root dataset/boundary_patches/BTFE/normal_patches/ ...
  python f-AnoGAN-pytorch/train_encoder.py  --data_root dataset/boundary_patches/BTFE/normal_patches/ ...

  # 3. score test patches and produce per-slice heatmaps
  python f-AnoGAN-pytorch/score.py ... --normal_root .../normal_test/ --anom_root .../pas_test/
  python f-AnoGAN-pytorch/extract_boundary_patches.py --mode project_back \\
      --scores_csv runs_fanogan/scores_v1/scores.csv \\
      --manifest   dataset/boundary_patches/BTFE/manifest.csv \\
      --images_dir dataset/mri_png/DATASET_BTFE/images/ \\
      --out_root   runs_fanogan/heatmaps_v1/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from skimage.measure import find_contours
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

try:
    from scipy.ndimage import map_coordinates, gaussian_filter, label
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from joblib import Parallel, delayed
    _JOBLIB = True
except ImportError:
    _JOBLIB = False


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def find_main_contour(mask_bin: np.ndarray) -> np.ndarray | None:
    """Return the largest contour as an (N, 2) array of (y, x) sub-pixel points.

    `mask_bin` must be a 2D boolean / 0-1 array.
    """
    if not _SKIMAGE:
        raise ImportError("scikit-image required. pip install scikit-image")
    contours = find_contours(mask_bin.astype(float), level=0.5)
    if not contours:
        return None
    return max(contours, key=lambda c: len(c))


def filter_lower_boundary(
    contour: np.ndarray, mask_bin: np.ndarray, lower_fraction: float = 0.5
) -> np.ndarray:
    """Keep only contour points whose y-coordinate is below the threshold.

    "Below" in image coordinates means a LARGER y (PIL/NumPy origin is top-left).
    The threshold is:  y_top + lower_fraction * (y_bot - y_top), where y_top
    and y_bot bound the placenta mask. lower_fraction=0.5 keeps the bottom half
    of the placenta's bounding box; 0.3 keeps the bottom 70%; 0.7 keeps only
    the bottom 30%.
    """
    if contour.size == 0:
        return contour
    ys, xs = np.where(mask_bin)
    if ys.size == 0:
        return contour[:0]
    y_top, y_bot = ys.min(), ys.max()
    y_thresh = y_top + lower_fraction * (y_bot - y_top)
    keep = contour[:, 0] >= y_thresh
    return contour[keep]


def resample_arc_length(contour: np.ndarray, stride_px: float) -> np.ndarray:
    """Pick contour indices evenly spaced by `stride_px` along the path.

    Returns the indices into `contour` (NOT the resampled points themselves —
    we keep indices so we can still compute local tangents from neighbours).
    """
    if len(contour) < 2:
        return np.arange(len(contour))
    # Cumulative arc length along the polyline
    diffs = np.diff(contour, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = float(arc[-1])
    if total < stride_px:
        return np.array([len(contour) // 2])  # one mid-point
    n = max(2, int(np.floor(total / stride_px)))
    targets = np.linspace(0, total, n)
    # For each target, the contour index closest to that arc length.
    idxs = np.searchsorted(arc, targets)
    idxs = np.clip(idxs, 0, len(contour) - 1)
    return np.unique(idxs)


def local_tangent_angle(contour: np.ndarray, idx: int, window: int = 3) -> float:
    """Tangent at contour[idx] estimated by finite differences over [idx-w, idx+w].

    contour is (y, x) so the tangent vector is (dy, dx). The patch is rotated so
    the tangent direction becomes the patch's local x-axis ("horizontal").
    Returned angle is in radians, measured CCW from +x axis (NumPy convention).
    """
    n = len(contour)
    a = (idx - window) % n
    b = (idx + window) % n
    py, px = contour[a]
    qy, qx = contour[b]
    dx = qx - px
    dy = qy - py
    return float(np.arctan2(dy, dx))


def extract_oriented_patch(
    image: np.ndarray, cx: float, cy: float, angle: float, size: int = 64
) -> np.ndarray:
    """Bilinear-sample a size×size patch centred at (cx, cy), with the patch's
    local x-axis pointing along `angle` (the boundary tangent).

    Result: the boundary runs horizontally through the middle of the patch —
    "izravnano". Pixels outside the image are filled with 0.
    """
    if not _SCIPY:
        raise ImportError("scipy required. pip install scipy")
    # Patch-local grid: rx ∈ [-S/2, S/2), ry ∈ [-S/2, S/2)
    half = size / 2.0
    rx, ry = np.meshgrid(
        np.arange(size, dtype=np.float64) - half + 0.5,
        np.arange(size, dtype=np.float64) - half + 0.5,
        indexing="xy",
    )
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    # Rotate the patch-local coords into image-space coords
    src_x = cx + cos_a * rx - sin_a * ry
    src_y = cy + sin_a * rx + cos_a * ry
    # map_coordinates wants [row, col] = [y, x]
    return map_coordinates(image, [src_y, src_x], order=1, mode="constant", cval=0.0)


# ---------------------------------------------------------------------------
# Patient-label parsing
# ---------------------------------------------------------------------------
def load_patient_labels(csv_path: Path) -> dict[str, dict]:
    """Read patient_labels.csv with at minimum columns patient_id, is_pas."""
    if not csv_path.exists():
        return {}
    out = {}
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            pid = row.get("patient_id") or row.get("pid")
            if not pid:
                continue
            # pandas-written CSVs stringify ints as "1.0" / "0.0" / "" when the
            # column has any NaN, so go through float() before int() to tolerate
            # both styles. Empty / unparseable values default to 0 (non-PAS).
            raw = (row.get("is_pas") or "").strip().lower()
            try:
                is_pas = int(float(raw)) if raw not in ("", "nan", "none") else 0
            except (ValueError, TypeError):
                is_pas = 0
            # Spread the raw row first, then OVERRIDE is_pas with the int.
            # If we did {"is_pas": is_pas, **row} the spread would re-stringify it.
            out[pid] = {**row, "is_pas": is_pas}
    return out


# ---------------------------------------------------------------------------
# Per-slice worker
# ---------------------------------------------------------------------------
def _read_gray(path: Path) -> np.ndarray:
    """Read a grayscale PNG as a uint8 ndarray."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _read_mask(path: Path) -> np.ndarray:
    """Read a mask PNG as bool. Tolerates either {0,1} (toolkit `Write masks
    as: 01`) or {0,255} (typical RGB-binarised) storage."""
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return arr > 0


def process_one_slice(
    image_path: Path,
    mask_path: Path,
    out_root: Path,
    patient_id: str,
    slice_id: str,
    is_pas: int,
    patch_size: int,
    stride_px: float,
    lower_fraction: float,
    min_blob_size: int,
    patch_prefix: str = "",
    force_subdir: str = "",
    manifest_only: bool = False,
) -> list[dict]:
    """Generate patch PNGs + return a list of manifest rows.
    If manifest_only=True, the geometry (centres/angles) is computed and the
    manifest rows are returned, but no patch PNGs are written — used to rebuild
    a coordinate manifest cheaply (e.g. for heatmaps) without re-writing images."""
    image = _read_gray(image_path)
    mask = _read_mask(mask_path)
    if mask.sum() < min_blob_size:
        return []

    contour = find_main_contour(mask)
    if contour is None or len(contour) < 4:
        return []

    lower = filter_lower_boundary(contour, mask, lower_fraction)
    if len(lower) < 2:
        return []

    idxs = resample_arc_length(lower, stride_px)
    if len(idxs) == 0:
        return []

    # Decide output folder. If force_subdir is given, ALL patches go there
    # regardless of is_pas (needed for leave-healthy-out where train-normal and
    # test-normal are both healthy but must land in different folders).
    if force_subdir:
        subdir = force_subdir
    else:
        subdir = "anomalous_patches" if is_pas else "normal_patches"
    patch_dir = out_root / subdir
    if not manifest_only:
        patch_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for k, idx in enumerate(idxs):
        cy, cx = lower[idx]  # contour is (y, x)
        angle = local_tangent_angle(lower, idx, window=3)
        # Optional prefix lets the caller disambiguate identical patient IDs
        # across cohorts (e.g. Mendeley BTFE vs SSH_TSE both use 'sub001').
        stem = (f"{patch_prefix}_{patient_id}_{slice_id}_b{k:03d}"
                if patch_prefix
                else f"{patient_id}_{slice_id}_b{k:03d}")
        if not manifest_only:
            patch = extract_oriented_patch(image, cx, cy, angle, size=patch_size)
            # Normalize to uint8 (boundary patches already span 0..255 in source)
            p8 = np.clip(patch, 0, 255).astype(np.uint8)
            Image.fromarray(p8).save(patch_dir / f"{stem}.png")
        rows.append({
            "patch_id": stem,
            "patient_id": patient_id,
            "slice_id": slice_id,
            "is_pas": is_pas,
            "src_image": str(image_path),
            "src_mask": str(mask_path),
            "patch_path": str(patch_dir / f"{stem}.png"),
            "center_x": float(cx),
            "center_y": float(cy),
            "angle_rad": float(angle),
            "patch_size": int(patch_size),
            "boundary_index": int(idx),
        })
    return rows


# ---------------------------------------------------------------------------
# Discovery (mirrors build_items in the train script)
# ---------------------------------------------------------------------------
def discover_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[str, str, Path, Path]]:
    """Yield (patient_id, slice_id, image_path, mask_path) for every paired PNG."""
    out = []
    for img in sorted(images_dir.rglob("*.png")):
        rel = img.relative_to(images_dir)
        mask = masks_dir / rel
        if not mask.exists():
            continue
        # Subfolder layout: <pid>/<pid>_<sid>.png
        if img.parent != images_dir:
            patient_id = img.parent.name
        else:
            # Flat layout: <pid>_<sid>.png
            patient_id = img.stem.rsplit("_", 1)[0]
        slice_id = img.stem.replace(patient_id + "_", "", 1)
        out.append((patient_id, slice_id, img, mask))
    return out


# ---------------------------------------------------------------------------
# Mode 1: extract
# ---------------------------------------------------------------------------
def cmd_extract(args) -> None:
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    labels = load_patient_labels(Path(args.labels_csv)) if args.labels_csv else {}
    if args.labels_csv and not labels:
        print(f"[extract] WARNING: labels_csv {args.labels_csv} not found / empty — "
              f"everything will go to normal_patches/")

    pairs = discover_pairs(images_dir, masks_dir)
    if not pairs:
        sys.exit(f"No image/mask pairs found under {images_dir} (looking for "
                 f"*.png with a sibling under {masks_dir})")

    # Optional patient filter (comma-separated IDs). Used by the leave-healthy-out
    # orchestrator to extract only a specific subset of patients per group.
    if args.patients:
        keep = {p.strip() for p in args.patients.split(",") if p.strip()}
        pairs = [t for t in pairs if t[0] in keep]
        if not pairs:
            sys.exit(f"No pairs left after --patients filter {sorted(keep)}")

    print(f"[extract] {len(pairs)} (image, mask) pairs; "
          f"labels for {len(labels)} patients"
          + (f"; force_subdir={args.force_subdir}" if args.force_subdir else ""))

    def _job(pid, sid, img, mask):
        is_pas = labels.get(pid, {}).get("is_pas", 0)
        return process_one_slice(
            img, mask, out_root, pid, sid,
            is_pas=is_pas,
            patch_size=args.patch_size,
            stride_px=args.stride_px,
            lower_fraction=args.lower_fraction,
            min_blob_size=args.min_blob_size,
            patch_prefix=args.patch_prefix,
            force_subdir=args.force_subdir,
            manifest_only=args.manifest_only,
        )

    if _JOBLIB and args.workers > 1:
        all_rows = Parallel(n_jobs=args.workers)(
            delayed(_job)(pid, sid, img, mask)
            for (pid, sid, img, mask) in tqdm(pairs, desc="extracting")
        )
    else:
        all_rows = [_job(pid, sid, img, mask) for (pid, sid, img, mask)
                    in tqdm(pairs, desc="extracting")]

    rows = [r for sub in all_rows for r in sub]

    manifest = Path(args.manifest_out) if args.manifest_out else out_root / "manifest.csv"
    if args.manifest_out:
        manifest.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        n_norm = sum(1 for r in rows if r["is_pas"] == 0)
        n_anom = sum(1 for r in rows if r["is_pas"] == 1)
        print(f"\n[extract] {len(rows)} patches  "
              f"(normal: {n_norm}, anomalous: {n_anom})")
        print(f"[extract]    manifest: {manifest}")
        print(f"[extract]    normal:   {out_root / 'normal_patches'}")
        if n_anom:
            print(f"[extract]    anom:     {out_root / 'anomalous_patches'}")
    else:
        print("[extract] WARNING: no patches produced (all masks empty or too small?)")


# ---------------------------------------------------------------------------
# Mode 2: project_back  — score CSV + manifest → per-slice heatmap PNGs
# ---------------------------------------------------------------------------
def cmd_project_back(args) -> None:
    """Project per-patch anomaly scores back onto each source slice as a heatmap.

    For every slice that has at least one scored patch, builds a 2D heatmap by
    Gaussian-splatting each score at the patch's original (center_x, center_y),
    then saves the heatmap as a PNG (and a 2-panel composite: MRI | heatmap).
    """
    import json
    images_dir = Path(args.images_dir) if args.images_dir else None
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Manifest: patch_id → (patient_id, slice_id, center_x, center_y, src_image)
    manifest_rows = list(csv.DictReader(open(args.manifest)))
    by_patch_id = {r["patch_id"]: r for r in manifest_rows}

    # Scores: patch_path → score
    # f-AnoGAN's scores.csv has columns: patch_path, is_anom, A_R, A_D, score
    scores: dict[str, float] = {}
    with open(args.scores_csv) as f:
        for row in csv.DictReader(f):
            p = Path(row["patch_path"])
            # The patch_id matches the patch filename's stem
            scores[p.stem] = float(row["score"])

    # Group by (patient_id, slice_id)
    by_slice: dict[tuple, list] = {}
    for pid_sid, r in [(((r["patient_id"], r["slice_id"]), r)) for r in manifest_rows]:
        if r["patch_id"] not in scores:
            continue
        by_slice.setdefault(pid_sid, []).append({
            "x": float(r["center_x"]),
            "y": float(r["center_y"]),
            "score": scores[r["patch_id"]],
            "src_image": r["src_image"],
        })

    print(f"[project] {len(by_slice)} slice(s) with scored patches")
    for (pid, sid), items in tqdm(by_slice.items(), desc="projecting"):
        # Build a heatmap of the original slice size
        src = Path(items[0]["src_image"])
        if not src.exists():
            continue
        img = _read_gray(src)
        H, W = img.shape
        heatmap = np.zeros((H, W), dtype=np.float32)
        # Place one point per patch; Gaussian-smooth the whole map afterwards
        for it in items:
            xi, yi = int(round(it["x"])), int(round(it["y"]))
            if 0 <= xi < W and 0 <= yi < H:
                heatmap[yi, xi] = max(heatmap[yi, xi], it["score"])
        if heatmap.max() > 0:
            heatmap = gaussian_filter(heatmap, sigma=args.heatmap_sigma)
        # Normalize for display
        hm_max = float(heatmap.max())
        hm_n = (heatmap / hm_max if hm_max > 0 else heatmap)
        # Save 3-panel: MRI | heatmap | overlay
        out_dir = out_root / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_heatmap_composite(img, hm_n, hm_max, out_dir / f"{pid}_{sid}_heatmap.png")

    print(f"[project] heatmaps under {out_root}")


def _save_heatmap_composite(img: np.ndarray, hm_norm: np.ndarray,
                            hm_max: float, out_path: Path) -> None:
    """Save MRI | heatmap | overlay collage as one PNG."""
    H, W = img.shape
    gutter = 4
    composite = np.zeros((H, 3 * W + 2 * gutter, 3), dtype=np.uint8)
    # 1. MRI grayscale → RGB
    rgb_img = np.stack([img] * 3, axis=-1)
    composite[:, 0:W] = rgb_img
    # 2. Heatmap (jet)
    def jet(v):
        r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
        return np.stack([r, g, b], axis=-1)
    hm_rgb = (jet(hm_norm) * 255).astype(np.uint8)
    composite[:, W + gutter:2 * W + gutter] = hm_rgb
    # 3. Overlay: alpha-blend
    alpha = (hm_norm[..., None] * 0.6)
    overlay = (rgb_img.astype(np.float32) * (1 - alpha) +
               hm_rgb.astype(np.float32) * alpha).astype(np.uint8)
    composite[:, 2 * (W + gutter):] = overlay
    Image.fromarray(composite).save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["extract", "project_back"], default="extract")
    # Extract args
    p.add_argument("--images_dir", type=str,
                   help="Source MRI slices folder (subfolder layout or flat).")
    p.add_argument("--masks_dir", type=str,
                   help="Mask folder (mirrors images_dir layout).")
    p.add_argument("--out_root", type=str, required=True,
                   help="Output root. Patches land in normal_patches/ and/or "
                        "anomalous_patches/; manifest.csv at the root.")
    p.add_argument("--labels_csv", type=str, default=None,
                   help="Optional CSV with columns: patient_id, is_pas[, is_previa]. "
                        "Without it, every patch is treated as normal.")
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride_px", type=float, default=32.0,
                   help="Arc-length spacing between consecutive sample points. "
                        "Default 32 px = 50%% overlap for 64-px patches.")
    p.add_argument("--lower_fraction", type=float, default=0.5,
                   help="Keep contour points with y >= y_top + lower_fraction*(y_bot-y_top). "
                        "0.5 = bottom half of placenta bbox. Lower values → more of the "
                        "contour kept (e.g. 0.0 = entire contour).")
    p.add_argument("--min_blob_size", type=int, default=200,
                   help="Skip slices whose placenta mask has fewer than this many pixels.")
    p.add_argument("--workers", type=int, default=20,
                   help="joblib parallelism for the extract loop (default 20).")
    p.add_argument("--patch_prefix", type=str, default="",
                   help="Optional prefix prepended to every patch filename. Use to "
                        "disambiguate identical patient IDs across cohorts "
                        "(e.g. Mendeley BTFE vs SSH_TSE both use 'sub001').")
    p.add_argument("--patients", type=str, default="",
                   help="Comma-separated patient IDs to include (others skipped). "
                        "Empty = all patients found.")
    p.add_argument("--manifest_only", action="store_true",
                   help="Compute geometry + write manifest only; do NOT write patch "
                        "PNGs. Used to rebuild a coordinate manifest (e.g. for "
                        "heatmaps) without re-writing images.")
    p.add_argument("--manifest_out", type=str, default=None,
                   help="Write the manifest to this exact path instead of "
                        "out_root/manifest.csv (lets several cohorts write distinct "
                        "manifests for later concatenation).")
    p.add_argument("--force_subdir", type=str, default="",
                   help="Write ALL patches into out_root/<this>/ regardless of is_pas. "
                        "Used by the leave-healthy-out orchestrator to route train- "
                        "vs test-normal healthy patients into separate folders.")
    # Project-back args
    p.add_argument("--scores_csv", type=str,
                   help="(project_back) f-AnoGAN scores.csv file.")
    p.add_argument("--manifest", type=str,
                   help="(project_back) manifest.csv produced by --mode extract.")
    p.add_argument("--heatmap_sigma", type=float, default=8.0,
                   help="(project_back) Gaussian σ in pixels for splatting scores.")

    args = p.parse_args()

    if args.mode == "extract":
        if not args.images_dir or not args.masks_dir:
            sys.exit("--mode extract requires --images_dir and --masks_dir.")
        cmd_extract(args)
    elif args.mode == "project_back":
        if not args.scores_csv or not args.manifest:
            sys.exit("--mode project_back requires --scores_csv and --manifest.")
        cmd_project_back(args)


if __name__ == "__main__":
    main()
