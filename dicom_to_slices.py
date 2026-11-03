#!/usr/bin/env python3
"""
dicom_to_slices.py

Generic DICOM → NIfTI + PNG converter for the placenta-MRI cohorts.

Cohorts handled (default scan root: ./dataset/dicom):
    kbc_osijek/  → patient dirs that contain a DICOM/ subdir (Sectra CDs).
                   Already-converted PNG folders (patient01/, patient02/) are
                   skipped automatically because they have no DICOM/ subtree.
    kbc_rebro/   → FERIT/<id> patient dirs.
                   Extras carried through:
                     - any <id>.nii.gz at the patient root is copied as
                       mask.nii.gz (radiologist export, contents not assumed).
                     - any UZV_*.png in the patient root is copied to
                       <out>/<pid>/ultrasound/.

Output (default: ./dataset/dicom_converted/):
    <out>/<cohort_short>/<patient_id>/
        nifti/<NN>_<series_safe>.nii.gz     full-res 3D volume + affine
        png/<NN>_<series_safe>/<KKK>.png    uint8 PNG per slice, native size
        mask.nii.gz                          (rebro only, if present)
        ultrasound/UZV_*.png                 (rebro only, if present)
    <out>/manifest.csv                       one row per slice, all metadata
    <out>/conversion_summary.csv             one row per series

Notes
-----
- We DO NOT resize here. PNG slices are saved at their native resolution.
  Run pas_preprocessing_toolkit.py afterwards to produce 512x512 datasets.
- We DO NOT flip horizontally by default. Add --flip-horizontal if your
  cohort's Sectra exports come out spine-on-the-wrong-side (the old
  convert_to_png.py defaulted to flipping).
- All sequences are kept (no series filter). Filter later via manifest.csv.

Usage
-----
    conda run -n monai_placenta python dicom_to_slices.py
    conda run -n monai_placenta python dicom_to_slices.py --cohorts kbc_rebro
    conda run -n monai_placenta python dicom_to_slices.py --dry-run

Requirements (already in monai_placenta): pydicom, SimpleITK, nibabel,
                                          numpy, Pillow, tqdm, joblib.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

try:
    import nibabel as nib
    _NIB = True
except ImportError:
    _NIB = False

try:
    from joblib import Parallel, delayed
    _JOBLIB = True
except ImportError:
    _JOBLIB = False


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "dataset" / "dicom"
DEFAULT_OUT = HERE / "dataset" / "dicom_converted"
DEFAULT_COHORTS = ["kbc_osijek", "kbc_rebro"]
COHORT_SHORT = {"kbc_osijek": "osi", "kbc_rebro": "reb"}

# Files we know are NOT DICOM in Sectra CDs — fast-path skip.
SKIP_FILENAME_PATTERNS = (
    "autorun.inf", "DICOMDIR", "README.TXT", "license_", "translation_",
    "run_cdviewer.exe", "viewer.chk", "viewer.xml",
)
SKIP_EXTENSIONS = {".exe", ".dll", ".rtf", ".msg", ".chk", ".xml", ".inf",
                   ".zip", ".sh", ".bat", ".cmd", ".lnk", ".dat",
                   ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff",
                   ".ds_store", ".txt"}

# Path components to skip — Siemens "Viewer/scanograms" contains reformatted
# views and screenshots that share Study/Patient UIDs with the real imaging
# series. ImageSeriesReader groups by SeriesInstanceUID, so without this filter
# the reformats get mixed with the actual scan and ITK raises size-mismatch /
# wrong-region errors.
SKIP_PATH_COMPONENTS = {"Viewer", "scanograms", "scanogram"}

# Series descriptions that are NOT imaging (Siemens phoenix structured reports,
# screen saves, etc.). Matched case-insensitively as substrings.
SKIP_SERIES_DESC = ("phoenix", "screen save", "screensave", "report", "scanogram")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def slugify(s: str, maxlen: int = 80) -> str:
    if not s:
        return "unknown"
    s = re.sub(r"[^\w\-]+", "_", str(s).lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] or "unknown"


def is_dicom_file(path: Path) -> bool:
    """Detect DICOM by 'DICM' magic at offset 128 — works on extension-less files."""
    if not path.is_file() or path.stat().st_size < 132:
        return False
    if path.name in SKIP_FILENAME_PATTERNS or path.name.startswith("."):
        return False
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return False
    # Skip Sectra/Siemens reformat folders (Viewer/, scanograms/) — those files
    # share SeriesInstanceUID with the real imaging series and would otherwise
    # be merged into it by ImageSeriesReader, triggering size-mismatch errors.
    if any(part in SKIP_PATH_COMPONENTS for part in path.parts):
        return False
    try:
        with open(path, "rb") as fp:
            fp.seek(128)
            return fp.read(4) == b"DICM"
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Patient discovery
# ---------------------------------------------------------------------------
def find_osijek_patients(cohort_root: Path) -> list[Path]:
    """Return dirs anywhere under kbc_osijek/ that contain a DICOM/ subdir."""
    if not cohort_root.is_dir():
        return []
    out = []
    for d in cohort_root.rglob("*"):
        if d.is_dir() and (d / "DICOM").is_dir():
            out.append(d)
    return sorted(out)


def find_rebro_patients(cohort_root: Path) -> list[Path]:
    """Return immediate dirs under kbc_rebro/FERIT/."""
    ferit = cohort_root / "FERIT"
    if not ferit.is_dir():
        return []
    return sorted(d for d in ferit.iterdir() if d.is_dir())


COHORT_FINDERS = {
    "kbc_osijek": find_osijek_patients,
    "kbc_rebro": find_rebro_patients,
}


# ---------------------------------------------------------------------------
# Series grouping (pydicom)
# ---------------------------------------------------------------------------
@dataclass
class SeriesMeta:
    series_uid: str
    series_number: int
    series_description: str
    modality: str
    manufacturer: str
    tr: Optional[float]
    te: Optional[float]
    magnetic_field_strength: Optional[float]
    pixel_spacing_x: Optional[float]
    pixel_spacing_y: Optional[float]
    slice_thickness: Optional[float]
    native_height: Optional[int]
    native_width: Optional[int]
    n_files: int
    files: list[Path] = field(default_factory=list)


_TAGS = [
    "SeriesInstanceUID", "SeriesNumber", "SeriesDescription",
    "Modality", "Manufacturer",
    "RepetitionTime", "EchoTime", "MagneticFieldStrength",
    "PixelSpacing", "SliceThickness",
    "Rows", "Columns",
    "InstanceNumber", "ImagePositionPatient",
]


def _read_header(path: Path):
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True,
                             specific_tags=_TAGS, force=True)
        return ds
    except Exception:
        return None


def group_series(patient_dir: Path, show_progress: bool = True) -> dict[str, SeriesMeta]:
    """Walk patient_dir, group every DICOM file by SeriesInstanceUID.

    Sectra CDs can have hundreds of files per patient. Header reads are I/O-bound
    and serial here — patient-level parallelism (joblib in main()) is the
    primary throughput mechanism. tqdm gives per-patient visibility.
    """
    series: dict[str, SeriesMeta] = {}
    files = [p for p in patient_dir.rglob("*") if is_dicom_file(p)]
    iterator = tqdm(files, desc=f"  reading {patient_dir.name}", leave=False,
                    unit="file", disable=not show_progress)
    for f in iterator:
        ds = _read_header(f)
        if ds is None:
            continue
        sid = getattr(ds, "SeriesInstanceUID", None)
        if not sid:
            continue
        if sid not in series:
            ps = getattr(ds, "PixelSpacing", None)
            try:
                ps_x = float(ps[1]) if ps else None
                ps_y = float(ps[0]) if ps else None
            except (TypeError, IndexError, ValueError):
                ps_x = ps_y = None
            series[sid] = SeriesMeta(
                series_uid=str(sid),
                series_number=int(getattr(ds, "SeriesNumber", 0) or 0),
                series_description=str(getattr(ds, "SeriesDescription", "") or "").strip(),
                modality=str(getattr(ds, "Modality", "") or ""),
                manufacturer=str(getattr(ds, "Manufacturer", "") or ""),
                tr=_to_float(getattr(ds, "RepetitionTime", None)),
                te=_to_float(getattr(ds, "EchoTime", None)),
                magnetic_field_strength=_to_float(getattr(ds, "MagneticFieldStrength", None)),
                pixel_spacing_x=ps_x,
                pixel_spacing_y=ps_y,
                slice_thickness=_to_float(getattr(ds, "SliceThickness", None)),
                native_height=_to_int(getattr(ds, "Rows", None)),
                native_width=_to_int(getattr(ds, "Columns", None)),
                n_files=0,
            )
        series[sid].files.append(f)
        series[sid].n_files += 1
    return series


def _to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v):
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _sort_files(files: list[Path]) -> list[Path]:
    """Sort series files by InstanceNumber (fallback ImagePositionPatient[2])."""
    def key(p: Path):
        ds = _read_header(p)
        if ds is None:
            return (10**9, str(p))
        inst = getattr(ds, "InstanceNumber", None)
        if inst is not None:
            try:
                return (int(inst), str(p))
            except (TypeError, ValueError):
                pass
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            try:
                return (float(ipp[2]), str(p))
            except (TypeError, ValueError):
                pass
        return (10**9, str(p))
    return sorted(files, key=key)


# ---------------------------------------------------------------------------
# Per-series conversion (SimpleITK does the heavy lifting)
# ---------------------------------------------------------------------------
def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Per-slice normalization to uint8 (max-based, NaN-safe)."""
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    m = float(arr.max())
    if m <= 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.uint8(np.round((arr / m) * 255.0))


def convert_series(meta: SeriesMeta, out_nii: Path, out_png_dir: Path,
                   flip_horizontal: bool = False) -> dict:
    """Write NIfTI + per-slice PNGs for one DICOM series. Returns summary dict."""
    sorted_files = _sort_files(meta.files)
    file_strs = [str(p) for p in sorted_files]

    out_nii.parent.mkdir(parents=True, exist_ok=True)
    out_png_dir.mkdir(parents=True, exist_ok=True)

    def _fail(error: str, n_saved: int = 0) -> dict:
        """Return a failure dict and tidy up the empty PNG dir so we don't
        leave orphans like reb006/png/02_t2_haste_sag_p2_mbh_.752/."""
        try:
            if out_png_dir.is_dir() and not any(out_png_dir.iterdir()):
                out_png_dir.rmdir()
        except OSError:
            pass
        return {"ok": False, "error": error, "n_slices_saved": n_saved}

    # 1. Read full 3D volume with SimpleITK (handles affine/orientation)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_strs)
    try:
        image = reader.Execute()
    except Exception as e:
        return _fail(f"sitk read failed: {e}")

    # 2. Write NIfTI (lossless)
    try:
        sitk.WriteImage(image, str(out_nii))
    except Exception as e:
        return _fail(f"nifti write failed: {e}")

    # 3. Per-slice PNG export (uint8, native resolution)
    arr = sitk.GetArrayFromImage(image)  # (z, y, x), in pixel-space order

    # SimpleITK loads multi-frame DICOMs and 2D singletons both correctly here.
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    # Some Siemens multi-frame DICOMs come back as (1, z, y, x) with a leading
    # singleton "frame" axis — squeeze it so the per-slice loop works normally.
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        return _fail(f"unexpected array ndim={arr.ndim} shape={arr.shape}")
    n_slices = arr.shape[0]
    h, w = arr.shape[1], arr.shape[2]

    # Sanity: real imaging slices are ≥ 16x16. Anything tinier is structured-report
    # leakage (phoenix vectors, embedded LUTs) that survived the description filter.
    if h < 16 or w < 16:
        return _fail(f"degenerate slice shape ({h}x{w}) — not an imaging series")

    # PNG save loop is serial inside one series — patient-level parallelism is
    # already running on the outer joblib pool, so nested joblib here would
    # oversubscribe. tqdm gives per-series visibility.
    for z in tqdm(range(n_slices), desc=f"    PNG {out_png_dir.name}",
                  leave=False, unit="slice"):
        slc = arr[z]
        img8 = normalize_to_uint8(slc)
        try:
            pil = Image.fromarray(img8)
        except (TypeError, ValueError) as e:
            return _fail(
                f"PIL rejected slice {z} (shape={img8.shape} dtype={img8.dtype}): {e}",
                n_saved=z,
            )
        if flip_horizontal:
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
        pil.save(out_png_dir / f"{z:03d}.png")

    return {
        "ok": True,
        "n_slices_saved": n_slices,
        "native_h": h,
        "native_w": w,
        "files_in_series": len(sorted_files),
    }


# ---------------------------------------------------------------------------
# Per-patient conversion
# ---------------------------------------------------------------------------
def convert_patient(patient_dir: Path, patient_id: str, cohort: str,
                    out_root: Path, flip_horizontal: bool,
                    skip_existing: bool) -> list[dict]:
    """Run the full per-patient pipeline; return one row per slice for the manifest."""
    cohort_short = COHORT_SHORT[cohort]
    pdir = out_root / cohort_short / patient_id

    rows: list[dict] = []
    nifti_root = pdir / "nifti"
    png_root = pdir / "png"

    if skip_existing and nifti_root.exists() and any(nifti_root.iterdir()):
        return rows  # already done

    # --- 1. Group DICOMs by series ---
    series_map = group_series(patient_dir)
    if not series_map:
        print(f"  WARNING: {patient_id}: no DICOM files found in {patient_dir}")
        return rows

    # --- 2. Drop non-image series (phoenix reports, screen saves, etc.) ---
    kept = []
    for m in series_map.values():
        desc = (m.series_description or "").lower()
        if any(s in desc for s in SKIP_SERIES_DESC):
            print(f"  {patient_id}: skipping non-image series '{m.series_description}' ({m.n_files} files)")
            continue
        kept.append(m)

    # Sort series by SeriesNumber (then UID for stability)
    sorted_series = sorted(kept, key=lambda m: (m.series_number, m.series_uid))

    # --- 3. Convert each series ---
    for idx, meta in enumerate(sorted_series, start=1):
        safe = slugify(meta.series_description) if meta.series_description else f"series{meta.series_number}"
        # Append a 4-char suffix from the UID to disambiguate same-description series.
        safe_with_uid = f"{idx:02d}_{safe}_{meta.series_uid[-4:]}"

        out_nii = nifti_root / f"{safe_with_uid}.nii.gz"
        out_png_dir = png_root / safe_with_uid

        info = convert_series(meta, out_nii, out_png_dir, flip_horizontal)
        if not info["ok"]:
            print(f"  WARNING: {patient_id} / {safe_with_uid}: {info.get('error')}")
            continue

        # One manifest row per slice
        for z in range(info["n_slices_saved"]):
            rows.append({
                "cohort": cohort,
                "patient_id": patient_id,
                "original_patient_dir": str(patient_dir),
                "series_index": idx,
                "series_safe_name": safe_with_uid,
                "series_uid": meta.series_uid,
                "series_number": meta.series_number,
                "series_description": meta.series_description,
                "modality": meta.modality,
                "manufacturer": meta.manufacturer,
                "tr_ms": meta.tr,
                "te_ms": meta.te,
                "magnetic_field_T": meta.magnetic_field_strength,
                "pixel_spacing_x_mm": meta.pixel_spacing_x,
                "pixel_spacing_y_mm": meta.pixel_spacing_y,
                "slice_thickness_mm": meta.slice_thickness,
                "native_h": info["native_h"],
                "native_w": info["native_w"],
                "slice_index": z,
                "nifti_path": str(out_nii.relative_to(out_root)),
                "png_path": str((out_png_dir / f"{z:03d}.png").relative_to(out_root)),
                "n_slices_in_series": info["n_slices_saved"],
                "files_in_series": info["files_in_series"],
            })

    # --- 4. Rebro extras: mask NIfTI + per-slice mask PNGs + ultrasound ---
    if cohort == "kbc_rebro":
        rebro_extras = _carry_rebro_extras(patient_dir, pdir, flip_horizontal)
        # Stamp on every row so the manifest reflects per-patient extras.
        for r in rows:
            r["has_mask"] = bool(rebro_extras["mask_path"])
            r["has_ultrasound"] = bool(rebro_extras["us_files"])
            r["mask_path"] = rebro_extras["mask_path"] or ""
            r["mask_png_dir"] = rebro_extras["mask_png_dir"] or ""
            r["ultrasound_files"] = ";".join(rebro_extras["us_files"])

    return rows


def _make_mask_alignment_overlays(out_root: Path) -> tuple[int, int]:
    """Per-patient sanity check: write one RGB overlay (image + red mask) for
    every slice that contains positive mask pixels. Each patient gets their own
    subfolder under mask_alignment_check/<patient_id>/. Use this to eyeball
    image↔mask flip alignment across the whole annotated volume.

    Returns (n_patients_with_overlays, n_total_overlays_written).
    """
    overlay_root = out_root / "mask_alignment_check"
    rebro_root = out_root / COHORT_SHORT["kbc_rebro"]
    if not rebro_root.is_dir():
        return (0, 0)

    n_patients = 0
    n_total = 0
    for pdir in sorted(rebro_root.iterdir()):
        mask_dir = pdir / "png_mask"
        if not mask_dir.is_dir():
            continue
        img_series_dirs = sorted((pdir / "png").iterdir()) if (pdir / "png").is_dir() else []
        if not img_series_dirs:
            continue
        # Pick the image series whose slice count matches the mask — the one
        # the annotator drew over. Falls back to the first series if no match.
        masks = sorted(mask_dir.glob("*.png"))
        img_series = next(
            (d for d in img_series_dirs if sum(1 for _ in d.glob("*.png")) == len(masks)),
            img_series_dirs[0],
        )

        patient_overlay_dir = overlay_root / pdir.name
        n_this_patient = 0
        for m_path in masks:
            mk = np.asarray(Image.open(m_path)) > 0
            if not mk.any():
                continue   # skip slices with empty masks
            z = m_path.stem   # "019"
            img_path = img_series / f"{z}.png"
            if not img_path.is_file():
                continue
            im = np.asarray(Image.open(img_path))
            if im.shape[:2] != mk.shape:
                continue   # series-mask size mismatch — skip this slice
            rgb = np.stack([im, im, im], -1).astype(np.uint8)
            rgb[mk] = [255, 0, 0]
            patient_overlay_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(patient_overlay_dir / f"{z}.png")
            n_this_patient += 1
        if n_this_patient:
            n_patients += 1
            n_total += n_this_patient
    return (n_patients, n_total)


def _carry_rebro_extras(patient_dir: Path, out_pdir: Path,
                        flip_horizontal: bool) -> dict:
    """Copy <id>.nii.gz → mask.nii.gz, export per-slice mask PNGs (flipped to
    match image PNGs), and copy UZV_*.png → ultrasound/. Return relative paths."""
    mask_rel = None
    mask_png_rel = None
    nii_candidates = sorted(patient_dir.glob("*.nii.gz"))
    if nii_candidates:
        src = nii_candidates[0]   # rebro convention: 1.X.nii.gz at patient root
        dst = out_pdir / "mask.nii.gz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        mask_rel = str(dst.relative_to(out_pdir.parents[1]))

        # Per-slice mask PNGs — flipped consistently with image PNGs so that
        # spine-left images line up pixel-for-pixel with spine-left masks.
        # Also auto-detect z-axis reversal: the rebro annotation tool saves
        # some masks with the slice order reversed relative to the DICOM-
        # derived image, leaving the array data in mirrored order while the
        # affine origin is shifted by (Z-1)*z_step (with opposite sign of
        # what a proper z-flip would have written). We detect this and flip
        # the array along axis 0 so mask[k] aligns with image[k] in PNG space.
        mask_png_dir = out_pdir / "png_mask"
        try:
            sitk_mask = sitk.ReadImage(str(dst))
            arr = sitk.GetArrayFromImage(sitk_mask)  # (z, y, x)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 3:
                # --- Z-axis alignment check vs the matched image series ---
                # Pick the image NIfTI whose Z matches the mask's Z.
                z_flip_applied = False
                img_nii_candidates = sorted((out_pdir / "nifti").glob("*.nii.gz"))
                matched_img = next(
                    (
                        f for f in img_nii_candidates
                        if sitk.ReadImage(str(f)).GetSize()[2] == arr.shape[0]
                    ),
                    None,
                )
                if matched_img is not None:
                    sitk_img = sitk.ReadImage(str(matched_img))
                    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)
                    spacing = sitk_img.GetSpacing()
                    Z = sitk_img.GetSize()[2]
                    delta = (np.array(sitk_mask.GetOrigin())
                             - np.array(sitk_img.GetOrigin()))
                    expected_zflip = (Z - 1) * direction[:, 2] * spacing[2]
                    # Affected masks have delta ≈ -expected_zflip (opposite sign)
                    if np.allclose(delta, -expected_zflip, atol=2.0):
                        arr = arr[::-1]
                        z_flip_applied = True
                        print(f"  {out_pdir.name}: detected z-axis reversed "
                              f"mask — applied flip on axis 0")

                mask_png_dir.mkdir(parents=True, exist_ok=True)
                # Binarize defensively: mask NIfTIs are nominally {0,1} but some
                # tools save uint16 with rounding noise.
                bin_arr = (arr > 0.5).astype(np.uint8) * 255
                for z in range(bin_arr.shape[0]):
                    slc = bin_arr[z]
                    pil = Image.fromarray(slc)
                    if flip_horizontal:
                        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
                    pil.save(mask_png_dir / f"{z:03d}.png")
                mask_png_rel = str(mask_png_dir.relative_to(out_pdir.parents[1]))

                # If we flipped the data, also write a corrected mask NIfTI so
                # the copied mask.nii.gz matches what's in png_mask/.
                if z_flip_applied:
                    corrected = sitk.GetImageFromArray(
                        (arr > 0.5).astype(np.uint8)
                    )
                    corrected.CopyInformation(sitk_img)
                    sitk.WriteImage(corrected, str(dst))
        except Exception as e:
            print(f"  WARNING: mask PNG export failed for {out_pdir.name}: {e}")

    us_rel = []
    for us in sorted(patient_dir.glob("UZV_*.png")):
        us_out = out_pdir / "ultrasound" / us.name
        us_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(us, us_out)
        us_rel.append(str(us_out.relative_to(out_pdir.parents[1])))

    return {"mask_path": mask_rel, "mask_png_dir": mask_png_rel, "us_files": us_rel}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def assign_ids(cohort: str, patient_dirs: list[Path]) -> list[tuple[str, Path]]:
    """Return [(patient_id, patient_dir)] with sequential IDs based on sort order."""
    short = COHORT_SHORT[cohort]
    return [(f"{short}{i:03d}", d) for i, d in enumerate(patient_dirs, start=1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=str(DEFAULT_ROOT),
                   help="Top-level scan root containing kbc_* cohorts.")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                   help="Output root.")
    p.add_argument("--cohorts", type=str, default=",".join(DEFAULT_COHORTS),
                   help="Comma-separated cohort names to process.")
    p.add_argument("--flip-horizontal", action="store_true",
                   help="Flip every PNG left-right. Use for Sectra exports where "
                        "spine appears on the wrong side. Default OFF.")
    p.add_argument("--workers", type=int, default=4,
                   help="Patient-level parallelism (joblib). Set 1 to debug.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip patients whose nifti/ dir is already non-empty.")
    p.add_argument("--dry-run", action="store_true",
                   help="List patients and assigned IDs, then exit.")
    p.add_argument("--mask-overlays", action="store_true",
                   help="After conversion, write per-slice image+mask overlays "
                        "(red mask on grayscale image) for every slice with a "
                        "non-empty mask. Output goes to "
                        "<out>/mask_alignment_check/<patient_id>/<KKK>.png. "
                        "Use this to visually verify image-mask alignment "
                        "(e.g. after toggling --flip-horizontal). Default OFF.")
    args = p.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    unknown = [c for c in cohorts if c not in COHORT_FINDERS]
    if unknown:
        sys.exit(f"Unknown cohort(s): {unknown}. Known: {list(COHORT_FINDERS)}")

    # --- 1. Build the work list ---
    work = []
    for cohort in cohorts:
        crt = root / cohort
        if not crt.is_dir():
            print(f"WARNING: {cohort}: {crt} does not exist — skipping.")
            continue
        patients = COHORT_FINDERS[cohort](crt)
        assignments = assign_ids(cohort, patients)
        print(f"{cohort}: {len(patients)} patient(s):")
        for pid, pdir in assignments:
            extras = ""
            if cohort == "kbc_rebro":
                if list(pdir.glob("UZV_*.png")):
                    extras += "  [+US]"
                if list(pdir.glob("*.nii.gz")):
                    extras += "  [+mask.nii.gz]"
            print(f"    {pid:<8} ← {pdir.name}{extras}")
            work.append((cohort, pid, pdir))

    if args.dry_run:
        print(f"\nDry-run: {len(work)} patients would be converted to {out_root}.")
        return

    if not work:
        print("Nothing to convert.")
        return

    # --- 2. Convert each patient ---
    print(f"\nConverting {len(work)} patient(s) → {out_root}  "
          f"(workers={args.workers}, flip={args.flip_horizontal})")

    def _job(cohort, pid, pdir):
        return convert_patient(pdir, pid, cohort, out_root,
                               flip_horizontal=args.flip_horizontal,
                               skip_existing=args.skip_existing)

    all_rows: list[dict] = []
    if _JOBLIB and args.workers > 1:
        results = Parallel(n_jobs=args.workers, prefer="processes")(
            delayed(_job)(c, p, d) for (c, p, d) in tqdm(work, desc="Patients")
        )
        for rows in results:
            all_rows.extend(rows)
    else:
        for (c, p, d) in tqdm(work, desc="Patients"):
            all_rows.extend(_job(c, p, d))

    # --- 3. Write manifest.csv ---
    if not all_rows:
        print("WARNING: No slices written.")
        return

    # Unify column set (rebro rows have extra columns)
    all_keys = []
    seen = set()
    for r in all_rows:
        for k in r:
            if k not in seen:
                all_keys.append(k); seen.add(k)

    manifest = out_root / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nManifest: {manifest}  ({len(all_rows)} slices)")

    # --- 4. Quick series-level summary CSV ---
    by_series: dict[tuple, dict] = {}
    for r in all_rows:
        key = (r["patient_id"], r["series_safe_name"])
        if key not in by_series:
            base = {k: r[k] for k in (
                "cohort", "patient_id", "series_index", "series_safe_name",
                "series_uid", "series_number", "series_description",
                "modality", "manufacturer", "tr_ms", "te_ms",
                "magnetic_field_T", "pixel_spacing_x_mm", "pixel_spacing_y_mm",
                "slice_thickness_mm", "native_h", "native_w",
                "n_slices_in_series", "files_in_series",
                "nifti_path",
            )}
            by_series[key] = base

    summary_csv = out_root / "conversion_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(by_series.values())).keys()))
        w.writeheader()
        w.writerows(by_series.values())
    print(f"Series summary: {summary_csv}  ({len(by_series)} series)")

    # --- 5. Mask-alignment overlays (opt-in via --mask-overlays) ---
    if args.mask_overlays:
        n_patients, n_total = _make_mask_alignment_overlays(out_root)
        if n_patients:
            print(f"Mask overlays: {out_root / 'mask_alignment_check'}/  "
                  f"({n_patients} patient(s), {n_total} slice(s) total)")
        else:
            print("WARNING: --mask-overlays requested but no overlays produced "
                  "(no rebro masks found or no non-empty mask slices).")

    # --- 6. Brief stats ---
    print("\n── Native resolution histogram (h × w) ──")
    res_count: dict[tuple, int] = {}
    for r in by_series.values():
        k = (r["native_h"], r["native_w"])
        res_count[k] = res_count.get(k, 0) + 1
    for (h, w), n in sorted(res_count.items(), key=lambda kv: -kv[1]):
        print(f"   {h}x{w}: {n} series")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
    main()
