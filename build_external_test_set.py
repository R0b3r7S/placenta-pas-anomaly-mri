#!/usr/bin/env python3
"""
Stage the converted KBC Rebro cohort into a flat paired image+mask layout that
pas_preprocessing_toolkit.py can read. The toolkit then handles the 512×512
resize the same way it did for DATASET_BTFE / DATASET_SSH_TSE, so the external
test set goes through the exact same preprocessing pipeline as the training
data — this is essential for fair external-validation numbers.

Input:
    dataset/dicom_converted/reb/<reb_id>/
        png/<series>/<KKK>.png      ← flipped, native-resolution image slices
        png_mask/<KKK>.png          ← flipped, native-resolution binary masks

Output (toolkit-compatible staging layout):
    dataset/staging/rebro_for_toolkit/
        images/<reb_id>/<reb_id>_<NNN>.png   native resolution, grayscale uint8
        masks/<reb_id>/<reb_id>_<NNN>.png    native resolution, binary {0, 255}

Filename convention: `<reb_id>_<NNN>.png` matches the toolkit's default
filename regex `^(?P<pid>.+)_(?P<sid>\\d+)$`, where pid="reb001", sid="010".

Filtering:
    - Patients without a mask (reb002 = "Nije segmentirano") are skipped.
    - reb012 is excluded — DICOM header says sagittal but acquisition is not.
    - Slices with empty masks are skipped (matches BTFE convention).

Next steps after running this script:
    1. Run pas_preprocessing_toolkit.py (interactive) — settings shown in the
       printout at the end of this script.
    2. Run write_external_splits.py to generate splits.json (all in `test`).

Usage:
    conda run --no-capture-output -n monai_placenta python build_external_test_set.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SRC = HERE / "dataset" / "dicom_converted" / "reb"
STAGE = HERE / "dataset" / "staging" / "rebro_for_toolkit"

# Hard exclusions — these patients are intentionally NOT in the external set.
EXCLUDE = {"reb012"}   # DICOM header says sagittal but is not actually sagittal


def _pick_image_series(patient_dir: Path, n_mask_slices: int) -> Path | None:
    """Return the image series whose slice count matches the mask."""
    png_root = patient_dir / "png"
    if not png_root.is_dir():
        return None
    for s in sorted(png_root.iterdir()):
        if sum(1 for _ in s.glob("*.png")) == n_mask_slices:
            return s
    return None


def main() -> None:
    if not SRC.is_dir():
        raise FileNotFoundError(
            f"{SRC} missing — run dicom_to_slices.py first."
        )

    # Wipe + recreate the staging dir so re-runs are deterministic.
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "images").mkdir(parents=True)
    (STAGE / "masks").mkdir(parents=True)

    kept_patients: list[str] = []
    total_slices = 0
    skipped_no_mask: list[str] = []
    skipped_no_series_match: list[str] = []

    for pdir in sorted(SRC.iterdir()):
        pid = pdir.name
        if pid in EXCLUDE:
            print(f"  {pid}: hard-excluded")
            continue

        mask_dir = pdir / "png_mask"
        if not mask_dir.is_dir():
            skipped_no_mask.append(pid)
            print(f"  {pid}: no mask — skipping")
            continue

        mask_files = sorted(mask_dir.glob("*.png"))
        image_series = _pick_image_series(pdir, len(mask_files))
        if image_series is None:
            skipped_no_series_match.append(pid)
            print(f"  WARNING: {pid}: no image series matches mask slice count "
                  f"({len(mask_files)}) — skipping")
            continue

        out_img_dir = STAGE / "images" / pid
        out_msk_dir = STAGE / "masks" / pid
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_msk_dir.mkdir(parents=True, exist_ok=True)

        n_this = 0
        for mpath in mask_files:
            mk = np.asarray(Image.open(mpath))
            if (mk > 0).sum() == 0:
                continue   # empty-mask slice
            z = mpath.stem   # "019"
            ipath = image_series / f"{z}.png"
            if not ipath.is_file():
                continue

            im = np.asarray(Image.open(ipath))
            if im.shape[:2] != mk.shape[:2]:
                print(f"    WARNING: {pid}/{z}: image/mask shape mismatch "
                      f"({im.shape} vs {mk.shape}) — skipping slice")
                continue

            # Toolkit-compatible filename: <pid>_<NNN>.png
            stem = f"{pid}_{int(z):03d}.png"
            shutil.copy2(ipath, out_img_dir / stem)
            shutil.copy2(mpath, out_msk_dir / stem)
            n_this += 1

        if n_this == 0:
            out_img_dir.rmdir()
            out_msk_dir.rmdir()
            print(f"  {pid}: no non-empty mask slices — skipping")
            continue

        kept_patients.append(pid)
        total_slices += n_this
        print(f"  {pid}: {n_this} slice(s) staged @ native resolution")

    print(f"\nStaging complete at {STAGE}")
    print(f"   patients staged: {len(kept_patients)}")
    print(f"   slices staged:   {total_slices}")
    if skipped_no_mask:
        print(f"   skipped (no mask):   {skipped_no_mask}")
    if skipped_no_series_match:
        print(f"   skipped (no series): {skipped_no_series_match}")

    # ---- next-step instructions printed inline so they don't get lost -----
    print("\n" + "=" * 78)
    print("NEXT — run pas_preprocessing_toolkit.py with these EXACT prompts:")
    print("=" * 78)
    print(f"""
  python pas_preprocessing_toolkit.py
  → 1) Manage/Preprocess Dataset

    Path to IMAGES root folder    : {STAGE / 'images'}
    Path to MASKS root folder     : {STAGE / 'masks'}   ← TYPE THIS EXPLICITLY
                                                          (don't accept the default — it
                                                          duplicates the images path)
    What do you want to do?       : Preprocess and write output
    Filename regex on STEM        : ^(?P<{'pid'}>.+)_(?P<{'sid'}>\\d+)$
    Allow fallback patient-id?    : n
    Preferred mask extension      : .png
    Pairs to open for stats       : 0
    Estimate empty-mask rate?     : y
    Mask binarisation method      : nonzero      ← rebro masks are binary grayscale
    Output folder                 : dataset/mri_png/DATASET_EXTERNAL_REBRO
    Organisation                  : patient_separate   ← gives images/<pid>/...
                                                          + masks/<pid>/... layout
    Overwrite existing?           : y            ← so re-runs replace the broken output
    Dry-run?                      : n
    Convert images to grayscale?  : y
    Convert masks via             : nonzero      ← same as above
    Write masks as                : 01
    Padding/cropping              : none         ← staging already padded each image
                                                   to its own square at max(H,W)
    Also RESIZE after pad/crop?   : y
    Resize height                 : 512
    Resize width                  : 512
    Upscaling                     : bicubic
    Downscaling                   : bicubic
    Output format for images      : png
    Use parallel processing?      : y

  Pipeline result per patient:
    - 512×512 patients (reb001, 003, 007, 008, 009, 010): no-op staging, identity resize.
    - reb014 (640×640): no staging pad, bicubic downsample 640 → 512×512.
    - reb005 (384×384): no staging pad, bicubic upsample 384 → 512×512.
    - reb004/011 (320×320): no staging pad, bicubic upsample 320 → 512×512.
    - reb006 (256×208): staged pad W to 256×256, then bicubic upsample to 512×512.
    - reb013 (320×270): staged pad W to 320×320, then bicubic upsample to 512×512.

  This matches the BTFE/SSH_TSE training recipe (`padding=none` + bicubic
  resize) exactly — same preprocessing path as training data, so the
  external-validation numbers are directly comparable.

  Mask binarisation differs from training recipe (rebro masks are already
  binary grayscale, training used red-channel JPGs).

  After the toolkit finishes:
    python write_external_splits.py
  """)


if __name__ == "__main__":
    main()
