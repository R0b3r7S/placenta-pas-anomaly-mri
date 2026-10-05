#!/usr/bin/env python3
"""
Write splits.json for the external Rebro test set after the toolkit finishes.

All patients land in `test` — this dataset is held-out external validation
only. No training or validation on it.

Default dataset root is dataset/resized/DATASET_EXTERNAL_REBRO.
Override via --root if the toolkit wrote somewhere else.

Usage:
    conda run --no-capture-output -n monai_placenta python write_external_splits.py
    conda run --no-capture-output -n monai_placenta python write_external_splits.py \\
        --root dataset/resized/DATASET_EXTERNAL_REBRO
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="dataset/mri_png/DATASET_EXTERNAL_REBRO",
        help="Dataset root (must contain images/ subdir).",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    images = root / "images"
    if not images.is_dir():
        raise SystemExit(
            f"images/ not found under {root}\n"
            f"  → ran the toolkit yet? Check the output folder you gave it."
        )

    patients = sorted(p.name for p in images.iterdir() if p.is_dir())
    if not patients:
        raise SystemExit(f"No patient subfolders under {images}")

    splits = {
        "train": [],
        "val":   [],
        "test":  patients,
        "seed":  None,
        "mode":  "external",
        "fractions": {"train": 0.0, "val": 0.0, "test": 1.0},
        "diagnostics": {
            "source":     "kbc_rebro via dicom_to_slices.py + pas_preprocessing_toolkit.py",
            "n_patients": len(patients),
            "note":       "all patients held out -- external validation only",
        },
    }

    out = root / "splits.json"
    with open(out, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote {out}")
    print(f"  test patients: {len(patients)}")
    print(f"  {patients}")


if __name__ == "__main__":
    main()
