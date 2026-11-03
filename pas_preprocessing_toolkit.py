#!/usr/bin/env python3
"""
pas_preprocessing_toolkit.py
Unified preprocessing toolkit for Placenta Accreta Spectrum (PAS) MRI datasets.

This script consolidates four major functionalities:
1. Dataset Manager & Preprocessor (scan, pad/crop, resize, binarize masks)
2. Interactive Overlays (red mask over grayscale image)
3. Advanced Patient Splits (patient-level or slice-balanced train/val/test)
4. Dataset Merging (combine BTFE and ssh_TSE into a COMBINED dataset)

Designed for preparing placenta segmentation datasets for MONAI and other pipelines.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import shutil
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except Exception:
    _JOBLIB_AVAILABLE = False


# =============================================================================
# GLOBAL CONSTANTS & CONFIGURATION
# =============================================================================
IMG_EXTS_DEFAULT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MSK_EXTS_DEFAULT = {".png", ".bmp", ".tif", ".tiff", ".webp"}
PAIR_REGEX_DEFAULT = r"^(?P<pid>.+)_(?P<sid>\d+)$"  # Regex to extract patient id (pid) and slice id (sid)

# Resampling maps for PIL
RESAMPLE_MAP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "box": Image.BOX,  # Area-like downsampling
}

@dataclass(frozen=True)
class Pair:
    """Dataclass to hold matched image and mask pairs."""
    pid: str
    sid: str
    image_path: Path
    mask_path: Path
    stem: str


# =============================================================================
# SHARED INTERACTIVE HELPERS
# These functions handle user input via the terminal securely and cleanly.
# =============================================================================

def get_input(prompt: str, default: Optional[str] = None, required: bool = True) -> str:
    """Prompt the user for a string input, offering a default value."""
    if default is not None:
        s = input(f"{prompt} [default: {default}]: ").strip()
        return s if s else default
    if not required:
        return input(f"{prompt}: ").strip()
    while True:
        s = input(f"{prompt}: ").strip()
        if s:
            return s
        print("  WARNING: This field is required.")

def get_yes_no(prompt: str, default: str = "y") -> bool:
    """Prompt the user for a yes/no boolean input."""
    default = default.lower()
    default_text = "Y/n" if default == "y" else "y/N"
    while True:
        s = input(f"{prompt} [{default_text}]: ").strip().lower()
        if not s:
            s = default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("  WARNING: Please answer 'y' or 'n'.")

def get_int(prompt: str, default: Optional[int] = None, min_val: Optional[int] = None, max_val: Optional[int] = None, allow_blank: bool = False) -> Optional[int]:
    """Prompt the user for an integer input with bounds checking."""
    d = f" [default: {default}]" if default is not None else ""
    while True:
        s = input(f"{prompt}{d}: ").strip()
        if not s:
            if default is not None:
                return default
            if allow_blank:
                return None
            print("  WARNING: Please enter a number.")
            continue
        try:
            v = int(s)
            if min_val is not None and v < min_val:
                print(f"  WARNING: Must be >= {min_val}.")
                continue
            if max_val is not None and v > max_val:
                print(f"  WARNING: Must be <= {max_val}.")
                continue
            return v
        except ValueError:
            print("  WARNING: Please enter a valid integer.")

def get_float(prompt: str, default: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Prompt the user for a floating point number."""
    while True:
        s = input(f"{prompt} [default: {default}]: ").strip()
        if not s:
            return float(default)
        try:
            v = float(s)
            if v < min_val or v > max_val:
                print(f"  WARNING: Must be between {min_val} and {max_val}.")
                continue
            return v
        except ValueError:
            print("  WARNING: Please enter a valid number.")

def choose(prompt: str, options: List[str], default_idx: int = 0) -> str:
    """Display a numbered list of options and prompt the user to choose one."""
    print(prompt)
    for i, opt in enumerate(options, start=1):
        mark = " (default)" if (i - 1) == default_idx else ""
        print(f"  {i}) {opt}{mark}")
    while True:
        s = input("Choose option number: ").strip()
        if not s:
            return options[default_idx]
        if s.isdigit():
            k = int(s)
            if 1 <= k <= len(options):
                return options[k - 1]
        print("  WARNING: Please enter a valid option number.")

def detect_layout(images_root: Path) -> str:
    """Detect if images_root contains patient subfolders or is flat."""
    subdirs = [p for p in images_root.iterdir() if p.is_dir()]
    return "patient_subfolders" if subdirs else "flat"


# =============================================================================
# MODULE 1: DATASET MANAGER (from placenta_dataset_manager_v2.py)
# Handles pairing, padding, resizing, binarization, and logging.
# =============================================================================

def is_ext(path: Path, exts: Iterable[str]) -> bool:
    """Check if a file's extension is in the valid list."""
    return path.suffix.lower() in {e.lower() for e in exts}

def scan_files(root: Path, exts: Iterable[str]) -> List[Path]:
    """Recursively scan a directory for files matching given extensions."""
    return [p for p in root.rglob("*") if p.is_file() and is_ext(p, exts)]

def extract_patient_number(stem: str) -> Optional[str]:
    """Fallback logic to extract patient ID from a filename if regex fails."""
    patterns = [
        r"(sub\d+)",
        r"(patient\d+)",
        r"(p\d+)",
        r"(case\d+)",
        r"(id\d+)",
    ]
    low = stem.lower()
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            return m.group(1)
    return None

def match_pairs(
    images_dir: Path,
    masks_dir: Path,
    img_exts: Iterable[str],
    mask_exts: Iterable[str],
    pair_re: re.Pattern,
    prefer_mask_ext: str = ".png",
    allow_fallback_patient_id: bool = False,
) -> Tuple[List[Pair], List[str]]:
    """Pairs image files with their corresponding mask files based on filename stem."""
    errors: List[str] = []

    img_files = scan_files(images_dir, img_exts)
    msk_files = scan_files(masks_dir, mask_exts)

    imgs_by_stem: Dict[str, List[Path]] = defaultdict(list)
    msks_by_stem: Dict[str, List[Path]] = defaultdict(list)

    for p in img_files:
        if pair_re.match(p.stem):
            imgs_by_stem[p.stem].append(p)
        elif allow_fallback_patient_id:
            imgs_by_stem[p.stem].append(p)

    for p in msk_files:
        if pair_re.match(p.stem):
            msks_by_stem[p.stem].append(p)
        elif allow_fallback_patient_id:
            msks_by_stem[p.stem].append(p)

    all_stems = sorted(set(imgs_by_stem.keys()) | set(msks_by_stem.keys()))
    pairs: List[Pair] = []

    for stem in all_stems:
        imgs = imgs_by_stem.get(stem, [])
        msks = msks_by_stem.get(stem, [])

        if not imgs:
            errors.append(f"Missing image for '{stem}' (mask(s) exist).")
            continue
        if not msks:
            errors.append(f"Missing mask for '{stem}' (image(s) exist).")
            continue

        img = sorted(imgs)[0]
        msks_sorted = sorted(msks, key=lambda p: (p.suffix.lower() != prefer_mask_ext.lower(), p.name))
        msk = msks_sorted[0]

        m = pair_re.match(stem)
        if m:
            pid, sid = m.group("pid"), m.group("sid")
        else:
            pid = extract_patient_number(stem) or "unknown"
            sid = "0"

        pairs.append(Pair(pid=pid, sid=sid, image_path=img, mask_path=msk, stem=stem))

    return pairs, errors

def open_meta(p: Path) -> Tuple[Tuple[int, int], str]:
    """Retrieve the height, width, and color mode of an image."""
    with Image.open(p) as im:
        w, h = im.size
        return (h, w), im.mode

def summarize_pairs(pairs: List[Pair], open_n: int = 200) -> dict:
    """Summarize a sample of pairs to identify sizes, modes, and any mismatches."""
    n = len(pairs)
    idxs = list(range(n))
    if open_n and open_n < n:
        idxs = idxs[:open_n]

    img_sizes = Counter()
    msk_sizes = Counter()
    img_modes = Counter()
    msk_modes = Counter()
    mismatches: List[dict] = []

    for i in tqdm(idxs, desc="Reading sample metadata", leave=False):
        pr = pairs[i]
        (hi, wi), mi = open_meta(pr.image_path)
        (hm, wm), mm = open_meta(pr.mask_path)
        img_sizes[(hi, wi)] += 1
        msk_sizes[(hm, wm)] += 1
        img_modes[mi] += 1
        msk_modes[mm] += 1
        if (hi, wi) != (hm, wm):
            mismatches.append({
                "pid": pr.pid, "sid": pr.sid, "stem": pr.stem,
                "image_size": [hi, wi], "mask_size": [hm, wm],
                "image_path": str(pr.image_path), "mask_path": str(pr.mask_path),
            })

    max_h = max((h for (h, w) in img_sizes.keys()), default=0)
    max_w = max((w for (h, w) in img_sizes.keys()), default=0)
    max_square = int(max(max_h, max_w)) if (max_h and max_w) else 0

    return {
        "n_pairs": n,
        "sample_open_n": len(idxs),
        "img_sizes": {f"{h}x{w}": c for (h, w), c in img_sizes.most_common()},
        "msk_sizes": {f"{h}x{w}": c for (h, w), c in msk_sizes.most_common()},
        "img_modes": dict(img_modes.most_common()),
        "msk_modes": dict(msk_modes.most_common()),
        "size_mismatch_count_in_sample": len(mismatches),
        "size_mismatch_examples": mismatches[:50],
        "max_observed_square_size_in_sample": max_square,
    }

def to_grayscale(im: Image.Image) -> Image.Image:
    """Convert an image to grayscale if it isn't already."""
    return im if im.mode == "L" else im.convert("L")

def binarize_mask(mask: Image.Image, mode: str, red_thresh: int, g_max: int, b_max: int) -> np.ndarray:
    """Convert masks to a 0/1 binary array based on color thresholds."""
    if mode == "nonzero":
        m = np.array(mask.convert("L"))
        return (m > 0).astype(np.uint8)

    m = np.array(mask.convert("RGB"))
    r, g, b = m[..., 0], m[..., 1], m[..., 2]
    fg = (r >= red_thresh) & (g <= g_max) & (b <= b_max)
    return fg.astype(np.uint8)

def center_pad_or_crop(arr: np.ndarray, out_hw: Tuple[int, int], pad_value: int = 0, pad_only_smaller: bool = False) -> np.ndarray:
    """Pad or crop an image array to a fixed target size."""
    out_h, out_w = out_hw
    if arr.ndim == 2:
        h, w = arr.shape
        c = None
    else:
        h, w, c = arr.shape

    if pad_only_smaller and (h > out_h or w > out_w):
        return arr

    y0 = max(0, (h - out_h) // 2)
    x0 = max(0, (w - out_w) // 2)
    y1 = min(h, y0 + out_h)
    x1 = min(w, x0 + out_w)
    cropped = arr[y0:y1, x0:x1] if arr.ndim == 2 else arr[y0:y1, x0:x1, :]

    ch, cw = cropped.shape[0], cropped.shape[1]
    pad_top = max(0, (out_h - ch) // 2)
    pad_bottom = max(0, out_h - ch - pad_top)
    pad_left = max(0, (out_w - cw) // 2)
    pad_right = max(0, out_w - cw - pad_left)

    if arr.ndim == 2:
        out = np.full((out_h, out_w), pad_value, dtype=cropped.dtype)
        out[pad_top:pad_top+ch, pad_left:pad_left+cw] = cropped
    else:
        out = np.full((out_h, out_w, c), pad_value, dtype=cropped.dtype)
        out[pad_top:pad_top+ch, pad_left:pad_left+cw, :] = cropped
    return out

def resize_array(arr: np.ndarray, out_hw: Tuple[int, int], is_mask: bool, resample_img: int = Image.BILINEAR) -> np.ndarray:
    """Resize using PIL, ensuring nearest-neighbor interpolation for masks to prevent artifacting."""
    out_h, out_w = out_hw
    pil = Image.fromarray(arr)
    resample = Image.NEAREST if is_mask else resample_img
    pil2 = pil.resize((out_w, out_h), resample=resample)
    return np.array(pil2)

def ensure_dir(p: Path) -> None:
    """Helper to ensure the parent directory of a path exists."""
    p.parent.mkdir(parents=True, exist_ok=True)

def build_output_paths(
    pr: Pair, images_dir: Path, masks_dir: Path, out_dir: Path, organization: str, image_out_format: str
) -> Tuple[Path, Path]:
    """Generates the correct output paths based on the user's requested folder organization."""
    stem = pr.stem
    if organization == "patient":
        img_out = out_dir / pr.pid / f"{stem}.{image_out_format}"
        msk_out = out_dir / pr.pid / f"{stem}.png"
        if img_out == msk_out:
            img_out = out_dir / pr.pid / f"{stem}_image.png"
            msk_out = out_dir / pr.pid / f"{stem}_mask.png"
        return (img_out, msk_out)
    if organization == "patient_separate":
        return (out_dir / "images" / pr.pid / f"{stem}.{image_out_format}",
                out_dir / "masks" / pr.pid / f"{stem}.png")
    if organization == "flat_separate":
        return (out_dir / "images" / f"{stem}.{image_out_format}",
                out_dir / "masks" / f"{stem}.png")
    if organization == "flat_together":
        img_out = out_dir / "all" / f"{stem}.{image_out_format}"
        msk_out = out_dir / "all" / f"{stem}.png"
        if img_out == msk_out:
            img_out = out_dir / "all" / f"{stem}_image.png"
            msk_out = out_dir / "all" / f"{stem}_mask.png"
        return (img_out, msk_out)
    # keep relative structure
    rel_img = pr.image_path.relative_to(images_dir).with_suffix(f".{image_out_format}")
    rel_msk = pr.mask_path.relative_to(masks_dir).with_suffix(".png")
    return (out_dir / "images" / rel_img, out_dir / "masks" / rel_msk)

def write_reports(report_dir: Path, report: dict, pairs: List[Pair], prefix: str) -> Tuple[Path, Path, Path]:
    """Writes detailed scan/preprocess logs to JSON, CSV, and TXT files for record-keeping."""
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"{prefix}_report_{ts}.json"
    csv_path = report_dir / f"{prefix}_pairs_{ts}.csv"
    txt_path = report_dir / f"{prefix}_summary_{ts}.txt"

    report_full = dict(report)
    report_full["pairs"] = [{"pid": pr.pid, "sid": pr.sid, "stem": pr.stem,
                             "image_path": str(pr.image_path), "mask_path": str(pr.mask_path)} for pr in pairs]

    with open(json_path, "w") as f:
        json.dump(report_full, f, indent=2)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "sid", "stem", "image_path", "mask_path"])
        for pr in pairs:
            w.writerow([pr.pid, pr.sid, pr.stem, str(pr.image_path), str(pr.mask_path)])

    with open(txt_path, "w") as f:
        f.write(f"{prefix.upper()} SUMMARY\n")
        f.write("=" * 80 + "\n")
        for k, v in report.items():
            f.write(f"{k}: {v}\n")

    return json_path, csv_path, txt_path

def mask_sanity_stats(
    pairs: List[Pair], n_check: int, mask_mode: str, red_thresh: int, g_max: int, b_max: int
) -> dict:
    """Computes empty mask rates and foreground pixel ratios to warn of potential data errors."""
    n = len(pairs)
    idxs = list(range(n))
    if n_check and n_check < n:
        idxs = idxs[:n_check]

    fg_ratios: List[float] = []
    empty = 0
    for i in tqdm(idxs, desc="Mask sanity (sample)", leave=False):
        pr = pairs[i]
        with Image.open(pr.mask_path) as mk:
            mk = mk.copy()
        m = binarize_mask(mk, mode=mask_mode, red_thresh=red_thresh, g_max=g_max, b_max=b_max)
        fg = int(m.sum())
        total = int(m.size)
        if fg == 0:
            empty += 1
        fg_ratios.append(fg / max(total, 1))

    if not fg_ratios:
        return {"checked": 0}

    fg_ratios_sorted = sorted(fg_ratios)
    mid = fg_ratios_sorted[len(fg_ratios_sorted)//2]
    return {
        "checked": len(fg_ratios),
        "empty_masks": empty,
        "empty_rate": empty / len(fg_ratios),
        "fg_ratio_min": float(fg_ratios_sorted[0]),
        "fg_ratio_median": float(mid),
        "fg_ratio_max": float(fg_ratios_sorted[-1]),
    }

def _process_one_pair(
    pr: Pair, images_dir: Path, masks_dir: Path, out_dir: Path, organization: str, overwrite: bool,
    convert_grayscale: bool, mask_mode: str, red_thresh: int, g_max: int, b_max: int, mask_out_values: str,
    pad_hw: Optional[Tuple[int, int]], pad_only_smaller: bool, resize_hw: Optional[Tuple[int, int]],
    image_out_format: str, jpg_quality: int, dry_run: bool, img_upscale: str = "bilinear", img_downscale: str = "box",
    pad_to_own_square: bool = False
) -> Tuple[str, int, int, bool]:
    """The core pipeline for processing a single image/mask pair."""
    with Image.open(pr.image_path) as im: im = im.copy()
    with Image.open(pr.mask_path) as mk: mk = mk.copy()

    size_mismatch = 1 if im.size != mk.size else 0
    if convert_grayscale: im = to_grayscale(im)
    m_bin = binarize_mask(mk, mode=mask_mode, red_thresh=red_thresh, g_max=g_max, b_max=b_max)

    # Optional: minimally pad each image to its own max(H,W) square. Applied
    # before any fixed pad_hw target, and before resize. Image and mask get
    # the same zero-padding so they stay pixel-aligned.
    if pad_to_own_square:
        im_np = np.array(im)
        h, w = im_np.shape[:2]
        if h != w:
            side = max(h, w)
            im_np = center_pad_or_crop(im_np, (side, side), pad_value=0)
            m_bin = center_pad_or_crop(m_bin, (side, side), pad_value=0).astype(np.uint8)
            im = Image.fromarray(im_np)

    if pad_hw is not None:
        im_np = np.array(im)
        im_np = center_pad_or_crop(im_np, pad_hw, pad_value=0, pad_only_smaller=pad_only_smaller)
        m_bin = center_pad_or_crop(m_bin, pad_hw, pad_value=0, pad_only_smaller=pad_only_smaller).astype(np.uint8)
        im = Image.fromarray(im_np)

    if resize_hw is not None:
        im_np = np.array(im)
        in_h, in_w = im_np.shape[:2]
        out_h, out_w = resize_hw
        if out_h > in_h or out_w > in_w:
            resample_img = RESAMPLE_MAP.get(img_upscale, Image.BILINEAR)
        else:
            resample_img = RESAMPLE_MAP.get(img_downscale, Image.BOX)
        im_np = resize_array(im_np, resize_hw, is_mask=False, resample_img=resample_img)
        im = Image.fromarray(im_np)
        
        m255 = (m_bin * 255).astype(np.uint8)
        m255 = resize_array(m255, resize_hw, is_mask=True)
        m_bin = (m255 > 0).astype(np.uint8)

    m_out = (m_bin * 255).astype(np.uint8) if mask_out_values == "0255" else m_bin.astype(np.uint8)
    img_out, msk_out = build_output_paths(pr, images_dir, masks_dir, out_dir, organization, image_out_format)

    if img_out.exists() and msk_out.exists() and not overwrite:
        return (pr.stem, 0, size_mismatch, True)

    if not dry_run:
        ensure_dir(img_out)
        ensure_dir(msk_out)
        if image_out_format == "png":
            im.save(img_out, format="PNG")
        else:
            im.save(img_out, format="JPEG", quality=int(jpg_quality), subsampling=0)
        Image.fromarray(m_out).save(msk_out, format="PNG")

    return (pr.stem, 1, size_mismatch, False)

def manager_interactive() -> int:
    """Interactive flow for the Dataset Manager & Preprocessor."""
    print("=" * 80)
    print("DATASET MANAGER & PREPROCESSOR")
    print("=" * 80)

    images_dir = Path(os.path.abspath(os.path.expanduser(get_input("Path to IMAGES root folder"))))
    masks_dir = Path(os.path.abspath(os.path.expanduser(get_input("Path to MASKS root folder", default=str(images_dir)))))

    if not images_dir.exists() or not masks_dir.exists():
        print("ERROR: Folders do not exist.")
        return 2

    action = choose("What do you want to do?", ["Scan only (no changes)", "Preprocess and write output"], default_idx=0)
    pair_regex = get_input("Filename regex on STEM (extract pid/sid)", default=PAIR_REGEX_DEFAULT)
    allow_fallback = get_yes_no("Allow fallback patient-id extraction if regex doesn't match?", default="n")
    prefer_mask_ext = get_input("Preferred mask extension", default=".png")
    open_n = get_int("How many pairs to open for mode/size stats? (0 = open all)", default=200, min_val=0)

    pair_re = re.compile(pair_regex)
    print("\nSCANNING...")
    pairs, errors = match_pairs(images_dir, masks_dir, IMG_EXTS_DEFAULT, MSK_EXTS_DEFAULT, pair_re, prefer_mask_ext, allow_fallback)

    if not pairs:
        print("ERROR: No pairs found. Check folders.")
        return 1

    meta = summarize_pairs(pairs, open_n=open_n)
    report_dir = Path(os.path.abspath(os.path.expanduser(get_input("\nReport output folder", default=str(images_dir)))))

    print("\nMASK SANITY (optional)")
    sanity = None
    if get_yes_no("Estimate empty-mask rate & foreground ratio stats?", default="y"):
        sanity_n = int(get_int("How many masks to check? (0 = all)", default=200, min_val=0))
        sanity_mode = choose("Mask binarization method:", ["red", "nonzero"], default_idx=0)
        sanity_mode = "red" if sanity_mode.startswith("red") else "nonzero"
        rt = int(get_int("R threshold (R >= ?)", default=1, min_val=0))
        gm = int(get_int("G max (G <= ?)", default=20, min_val=0))
        bm = int(get_int("B max (B <= ?)", default=20, min_val=0))
        sanity = mask_sanity_stats(pairs, sanity_n, sanity_mode, rt, gm, bm)

    # Basic scan report logic
    scan_report = {"scan_info": {"images_dir": str(images_dir), "n_pairs": len(pairs)}, "meta": meta, "sanity": sanity}
    write_reports(report_dir, scan_report, pairs, prefix="scan")
    
    if action.startswith("Scan only"): return 0

    print("\nPREPROCESS CONFIG")
    out_dir = Path(os.path.abspath(os.path.expanduser(get_input("Output folder", default=str(images_dir) + "_processed"))))
    organization = choose("Organization:", ["keep", "patient", "patient_separate", "flat_separate", "flat_together"], default_idx=1).split()[0]
    overwrite = get_yes_no("Overwrite existing?", default="n")
    dry_run = get_yes_no("Dry-run?", default="n")
    
    convert_grayscale = get_yes_no("Convert images to grayscale?", default="y")
    mask_mode = choose("Convert masks via:", ["red", "nonzero"], default_idx=0).split()[0]
    rt_p, gm_p, bm_p = 1, 20, 20
    if mask_mode == "red":
        rt_p = int(get_int("Mask red-channel threshold (R >= ?)", default=1))
        gm_p = int(get_int("Mask max green (G <= ?)", default=20))
        bm_p = int(get_int("Mask max blue  (B <= ?)", default=20))
    mask_out_values = "01" if choose("Write masks as:", ["01", "0255"], default_idx=0).startswith("01") else "0255"

    pad_to = choose(
        "Padding/cropping:",
        [
            "none",
            "pad to own square (per image, minimal pad on short axis)",
            "pad/crop max square",
            "pad/crop specific",
        ],
        default_idx=1,
    )
    pad_to_own_square = pad_to.startswith("pad to own")
    pad_only_smaller = (
        False if pad_to_own_square
        else get_yes_no("Pad only smaller (never crop larger)?", default="y")
    )
    pad_hw = None
    if pad_to.startswith("pad/crop max"):
        s = meta["max_observed_square_size_in_sample"]
        if s > 0: pad_hw = (s, s)
    elif pad_to.startswith("pad/crop specific"):
        pad_hw = (int(get_int("Target height", default=512)), int(get_int("Target width", default=512)))

    resize_hw, img_upscale, img_downscale = None, "bilinear", "box"
    if get_yes_no("Also RESIZE after pad/crop?", default="n"):
        resize_hw = (int(get_int("Resize height", default=512)), int(get_int("Resize width", default=512)))
        img_upscale = choose("Upscaling:", ["bilinear", "bicubic", "lanczos"], default_idx=0).split()[0]
        img_downscale = choose("Downscaling:", ["box", "lanczos", "bicubic", "bilinear"], default_idx=0).split()[0]

    img_fmt = "png" if choose("Output format for images:", ["png", "jpg"], default_idx=0).startswith("png") else "jpg"
    jpg_quality = int(get_int("JPEG quality", default=95)) if img_fmt == "jpg" else 95

    use_parallel = _JOBLIB_AVAILABLE and get_yes_no("Use parallel processing (joblib)?", default="y")
    n_jobs = int(get_int("Parallel jobs (-1 = all)", default=-1)) if use_parallel else 1

    if not get_yes_no("Proceed with preprocessing?", default="y"): return 0

    if not dry_run: out_dir.mkdir(parents=True, exist_ok=True)
    if use_parallel:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_process_one_pair)(
                pr, images_dir, masks_dir, out_dir, organization, overwrite,
                convert_grayscale, mask_mode, rt_p, gm_p, bm_p, mask_out_values,
                pad_hw, pad_only_smaller, resize_hw, img_fmt, jpg_quality, dry_run, img_upscale, img_downscale,
                pad_to_own_square
            ) for pr in tqdm(pairs, desc="Preprocessing")
        )
        processed = sum(r[1] for r in results)
        skipped = sum(1 for r in results if r[3])
    else:
        processed = skipped = 0
        for pr in tqdm(pairs, desc="Preprocessing"):
            _, wrote, _, skip = _process_one_pair(
                pr, images_dir, masks_dir, out_dir, organization, overwrite,
                convert_grayscale, mask_mode, rt_p, gm_p, bm_p, mask_out_values,
                pad_hw, pad_only_smaller, resize_hw, img_fmt, jpg_quality, dry_run, img_upscale, img_downscale,
                pad_to_own_square
            )
            processed += wrote
            if skip: skipped += 1

    print(f"\nPreprocess complete. Written: {processed}, Skipped existing: {skipped}")
    return 0


# =============================================================================
# MODULE 2: INTERACTIVE OVERLAYS (from make_overlays_interactive.py)
# Creates a red visual overlay on top of grayscale images based on the mask.
# =============================================================================

def make_overlay(image_path: Path, mask_path: Path, out_path: Path, red_boost: int = 120) -> None:
    """Core function to combine an image and its mask into an RGB overlay."""
    img = Image.open(image_path).convert("L")
    msk = Image.open(mask_path).convert("L")
    I = np.array(img).astype(np.float32)
    M = (np.array(msk) > 0).astype(np.uint8)
    rgb = np.stack([I, I, I], axis=-1)
    rgb[..., 0] = np.clip(rgb[..., 0] + M * red_boost, 0, 255)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8)).save(out_path)

def collect_pairs_flat(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path, str]]:
    """Collects image/mask pairs from a flat directory structure."""
    pairs = []
    img_files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS_DEFAULT])
    for img_p in img_files:
        mask_p = masks_dir / img_p.name
        if mask_p.exists():
            pairs.append((img_p, mask_p, img_p.stem))
    return pairs

def collect_pairs_patient(images_root: Path, masks_root: Path) -> List[Tuple[Path, Path, str]]:
    """Collects image/mask pairs from patient-specific subfolders."""
    pairs = []
    for pid_dir in sorted([p for p in images_root.iterdir() if p.is_dir()]):
        pid = pid_dir.name
        mask_pid = masks_root / pid
        if not mask_pid.exists():
            continue
        for img_p in sorted([p for p in pid_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS_DEFAULT]):
            mask_p = mask_pid / img_p.name
            if mask_p.exists():
                pairs.append((img_p, mask_p, f"{pid}/{img_p.stem}"))
    return pairs

def overlays_interactive() -> int:
    """Interactive flow for generating red-mask overlays."""
    print("=" * 80)
    print("MAKE OVERLAYS — red mask overlay on grayscale image")
    print("=" * 80)

    mode = choose("Choose mode:", ["Single pair", "Batch (folder)"], default_idx=1)
    red_boost = get_int("Red boost intensity (higher = stronger red overlay)", default=120, min_val=1, max_val=255)

    if mode.startswith("Single"):
        img_p = Path(os.path.abspath(os.path.expanduser(get_input("Image path"))))
        msk_p = Path(os.path.abspath(os.path.expanduser(get_input("Mask path"))))
        out_p = Path(os.path.abspath(os.path.expanduser(get_input("Output path", default=str(img_p.with_name(img_p.stem + "_overlay.png"))))))
        if not img_p.exists() or not msk_p.exists():
            print("ERROR: Image or Mask not found.")
            return 2
        make_overlay(img_p, msk_p, out_p, red_boost=red_boost)
        print(f"Saved overlay: {out_p}")
        return 0

    images_root = Path(os.path.abspath(os.path.expanduser(get_input("Images root folder"))))
    masks_root = Path(os.path.abspath(os.path.expanduser(get_input("Masks root folder"))))
    out_root = Path(os.path.abspath(os.path.expanduser(get_input("Output folder", default=str(images_root.parent / "overlays")))))

    layout = detect_layout(images_root)
    pairs = collect_pairs_patient(images_root, masks_root) if layout == "patient_subfolders" else collect_pairs_flat(images_root, masks_root)
    
    if not pairs:
        print("ERROR: No pairs found.")
        return 2

    use_parallel = _JOBLIB_AVAILABLE and get_yes_no("Use parallel processing?", default="y")
    n_jobs = get_int("Number of jobs", default=-1) if use_parallel else 1
    keep_structure = get_yes_no("Preserve subfolder structure?", default="y")

    def out_for(stem: str) -> Path:
        return out_root / f"{stem if keep_structure else stem.replace('/', '__')}.png"

    if use_parallel:
        Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(make_overlay)(img_p, msk_p, out_for(stem), red_boost) for img_p, msk_p, stem in tqdm(pairs, desc="Processing")
        )
    else:
        for img_p, msk_p, stem in tqdm(pairs, desc="Processing"):
            make_overlay(img_p, msk_p, out_for(stem), red_boost=red_boost)

    print(f"Done. Overlays saved to: {out_root}")
    return 0


# =============================================================================
# MODULE 3: PATIENT SPLITS (from make_patient_splits.py)
# Generates patient-level or slice-balanced Train/Val/Test splits.json.
# =============================================================================

def extract_pid_from_stem(stem: str) -> str:
    """Helper to get patient ID specifically for splitting logic."""
    return stem.rsplit("_", 1)[0]

def count_slices_per_patient(images_dir: Path, layout: str) -> Dict[str, int]:
    """Counts how many image slices exist per patient to allow balancing."""
    counts: Dict[str, int] = defaultdict(int)
    if layout == "patient_subfolders":
        for pid_dir in sorted([p for p in images_dir.iterdir() if p.is_dir()]):
            pid = pid_dir.name
            n = sum(1 for f in pid_dir.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXTS_DEFAULT)
            if n > 0: counts[pid] = n
    else:
        for f in images_dir.glob("*"):
            if f.is_file() and f.suffix.lower() in IMG_EXTS_DEFAULT:
                counts[extract_pid_from_stem(f.stem)] += 1
    return dict(counts)

def make_patient_split(pids: List[str], train_frac: float, val_frac: float, rng: random.Random):
    """Splits patients equally regardless of how many slices each patient has."""
    n = len(pids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    return pids[:n_train], pids[n_train:n_train + n_val], pids[n_train + n_val:]

def make_slice_balanced_split(pids: List[str], slices_per_pid: Dict[str, int], train_frac: float, val_frac: float, rng: random.Random):
    """Splits patients such that the *total number of slices* in each group matches the fractions."""
    total_slices = sum(slices_per_pid.values())
    targets = {"train": int(round(total_slices * train_frac)), "val": int(round(total_slices * val_frac))}
    targets["test"] = total_slices - targets["train"] - targets["val"]

    # Greedy allocation: assign the largest patients first
    pids_sorted = sorted(pids, key=lambda pid: slices_per_pid[pid], reverse=True)
    splits = {"train": [], "val": [], "test": []}
    actual = {"train": 0, "val": 0, "test": 0}

    for pid in pids_sorted:
        remaining = {k: targets[k] - actual[k] for k in ("train", "val", "test")}
        best = max(remaining.values())
        if best > 0:
            chosen = rng.choice([k for k, v in remaining.items() if v == best])
        else:
            min_cur = min(actual.values())
            chosen = rng.choice([k for k, v in actual.items() if v == min_cur])

        splits[chosen].append(pid)
        actual[chosen] += slices_per_pid[pid]

    return splits["train"], splits["val"], splits["test"], targets, actual

def splits_interactive() -> int:
    """Interactive flow to create a splits.json file."""
    print("=" * 80)
    print("MAKE PATIENT SPLITS")
    print("=" * 80)

    dataset_root = Path(os.path.abspath(os.path.expanduser(get_input("Dataset root folder"))))
    images_dir = Path(os.path.abspath(os.path.expanduser(get_input("Images folder", default=str(dataset_root / "images")))))
    
    if not images_dir.exists():
        print(f"ERROR: Images folder not found: {images_dir}")
        return 2

    layout = detect_layout(images_dir)
    train_frac = get_float("Train fraction", default=0.70)
    val_frac = get_float("Val fraction", default=0.15)
    test_frac = 1.0 - train_frac - val_frac

    mode = choose("Split mode:", ["patient (each patient counts equally)", "slice_balanced (balance total slices)"], default_idx=1)
    mode_key = "slice_balanced" if mode.startswith("slice_balanced") else "patient"
    seed = get_int("Random seed", default=42)
    rng = random.Random(seed)

    slices_per_pid = count_slices_per_patient(images_dir, layout)
    pids = sorted(slices_per_pid.keys())
    rng.shuffle(pids)

    if mode_key == "patient":
        train, val, test = make_patient_split(pids, train_frac, val_frac, rng)
        targets = None
        actual = {k: sum(slices_per_pid[p] for p in lst) for k, lst in zip(["train", "val", "test"], [train, val, test])}
    else:
        train, val, test, targets, actual = make_slice_balanced_split(pids, slices_per_pid, train_frac, val_frac, rng)

    out_dir = Path(os.path.abspath(os.path.expanduser(get_input("Output folder", default=str(dataset_root)))))
    out_path = out_dir / "splits.json"

    obj = {
        "train": sorted(train), "val": sorted(val), "test": sorted(test),
        "seed": seed, "mode": mode_key, "fractions": {"train": train_frac, "val": val_frac, "test": test_frac},
        "diagnostics": {
            "total": {"n_patients": len(pids), "n_slices": sum(slices_per_pid.values())},
            "train": {"n_patients": len(train), "n_slices": actual["train"]},
            "val": {"n_patients": len(val), "n_slices": actual["val"]},
            "test": {"n_patients": len(test), "n_slices": actual["test"]},
            "slice_targets": targets,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)

    print(f"\nWrote {out_path}")
    print(f"Train: {len(train)} patients, {actual['train']} slices")
    print(f"Val:   {len(val)} patients, {actual['val']} slices")
    print(f"Test:  {len(test)} patients, {actual['test']} slices")
    return 0


# =============================================================================
# MODULE 4: MERGE DATASETS (from merge_datasets.py)
# Safely combines two datasets by adding a prefix to their patient IDs.
# =============================================================================

def merge_dataset_fn(source_dir: Path, dest_dir: Path, prefix: str, combined_splits: dict):
    """Copies images and masks from a source to a destination, appending a prefix."""
    print(f"\n--- Processing {source_dir.name} ---")
    splits_file = source_dir / "splits.json"
    if splits_file.exists():
        with open(splits_file, "r") as f:
            splits = json.load(f)
            for key in ["train", "val", "test"]:
                for patient_id in splits.get(key, []):
                    combined_splits[key].append(f"{prefix}_{patient_id}")
    else:
        print(f"Warning: No splits.json found in {source_dir}")

    for folder_name in ["images", "masks"]:
        src_folder = source_dir / folder_name
        dest_folder = dest_dir / folder_name
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        if src_folder.exists():
            patient_folders = [f for f in src_folder.iterdir() if f.is_dir()]
            for patient_folder in tqdm(patient_folders, desc=f"Copying {folder_name}"):
                new_folder_name = f"{prefix}_{patient_folder.name}"
                shutil.copytree(patient_folder, dest_folder / new_folder_name, dirs_exist_ok=True)

def merge_interactive() -> int:
    """Interactive flow to merge datasets."""
    print("=" * 80)
    print("MERGE DATASETS")
    print("=" * 80)
    
    btfe_path = Path(os.path.abspath(os.path.expanduser(get_input("Path to dataset 1 (e.g., BTFE)"))))
    prefix1 = get_input("Prefix for dataset 1", default="btfe")
    
    tse_path = Path(os.path.abspath(os.path.expanduser(get_input("Path to dataset 2 (e.g., SSH_TSE)"))))
    prefix2 = get_input("Prefix for dataset 2", default="tse")
    
    dest_path = Path(os.path.abspath(os.path.expanduser(get_input("Output path for combined dataset"))))
    
    if not get_yes_no("Proceed with merging?", default="y"):
        return 0
        
    master_splits = {"train": [], "val": [], "test": []}
    merge_dataset_fn(btfe_path, dest_path, prefix1, master_splits)
    merge_dataset_fn(tse_path, dest_path, prefix2, master_splits)
    
    with open(dest_path / "splits.json", "w") as f:
        json.dump(master_splits, f, indent=4)
        
    print(f"\nSuccess! Datasets merged into: {dest_path}")
    print(f"Total Train: {len(master_splits['train'])} | Total Val: {len(master_splits['val'])} | Total Test: {len(master_splits['test'])}")
    return 0


# =============================================================================
# MAIN MENU
# Entry point for the toolkit, allowing the user to choose the module.
# =============================================================================

def main() -> int:
    """Main application loop."""
    while True:
        print("\n" + "=" * 80)
        print("PLACENTA ACCRETA SPECTRUM (PAS) TOOLKIT")
        print("=" * 80)
        options = [
            "Manage/Preprocess Dataset (scan, pad/crop, resize, binarize)",
            "Create Patient Splits (train/val/test with slice balancing)",
            "Generate Overlays (red mask over grayscale)",
            "Merge Datasets (combine BTFE and SSH_TSE)",
            "Exit"
        ]
        choice = choose("What would you like to do?", options, default_idx=0)
        
        if choice.startswith("Manage"):
            manager_interactive()
        elif choice.startswith("Create Patient"):
            splits_interactive()
        elif choice.startswith("Generate Overlays"):
            overlays_interactive()
        elif choice.startswith("Merge Datasets"):
            merge_interactive()
        else:
            print("Goodbye!")
            break
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nCancelled by user (Ctrl+C).")
        sys.exit(1)
