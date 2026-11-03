#!/usr/bin/env python3
"""
Build a unified patient_labels.csv covering both cohorts.

Sources:
    - Rebro: dataset/dicom/kbc_rebro/clinical_labels.csv  (parsed from HRZZ_FERIT.xlsx)
    - Osijek: every patient folder under dataset/dicom/kbc_osijek/DICOM placenta accreta/
              All Osijek patients are placenta accreta (no clinical sheet — confirmed by clinician).

Output:
    dataset/dicom_converted/patient_labels.csv  with columns:
        patient_id, cohort, folder_name, dx, is_pas, uzv, has_mask

Re-run any time the clinical sheet changes or new patients are added.

Usage:
    conda run --no-capture-output -n monai_placenta python build_patient_labels.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REBRO_LABELS = HERE / "dataset" / "dicom" / "kbc_rebro" / "clinical_labels.csv"
OSIJEK_ROOT = HERE / "dataset" / "dicom" / "kbc_osijek" / "DICOM placenta accreta"
OUT = HERE / "dataset" / "dicom_converted" / "patient_labels.csv"

COLS = ["patient_id", "cohort", "folder_name", "dx", "is_pas", "uzv", "has_mask"]


def build_rebro() -> pd.DataFrame:
    if not REBRO_LABELS.exists():
        raise FileNotFoundError(
            f"{REBRO_LABELS} missing — run parse_rebro_sheet.py first or "
            f"verify dataset/dicom/kbc_rebro/HRZZ_FERIT.xlsx exists."
        )
    df = pd.read_csv(REBRO_LABELS)
    df = df[df["reb_id"].notna() & (df["reb_id"] != "")].copy()
    df = df.rename(columns={"reb_id": "patient_id"})
    df["cohort"] = "kbc_rebro"
    df["has_mask"] = df["delivered"]
    return df[COLS]


def build_osijek() -> pd.DataFrame:
    if not OSIJEK_ROOT.is_dir():
        raise FileNotFoundError(f"{OSIJEK_ROOT} missing")
    dirs = sorted(p.name for p in OSIJEK_ROOT.iterdir() if p.is_dir())
    return pd.DataFrame({
        "patient_id":  [f"osi{i:03d}" for i in range(1, len(dirs) + 1)],
        "cohort":      ["kbc_osijek"] * len(dirs),
        "folder_name": dirs,
        "dx":          ["Placenta accreta"] * len(dirs),
        "is_pas":      [1] * len(dirs),
        "uzv":         [0] * len(dirs),
        "has_mask":    [False] * len(dirs),   # Osijek = MRI only, no GT masks
    })[COLS]


def main() -> None:
    osi = build_osijek()
    reb = build_rebro()
    out = pd.concat([osi, reb], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(out.to_string(index=False))
    print(f"\nSaved {OUT}")
    print(f"N total: {len(out)}  |  with_mask: {int(out.has_mask.sum())}  "
          f"|  PAS+: {int(out.is_pas.fillna(0).sum())}  "
          f"|  PAS-: {int((out.is_pas == 0).sum())}  "
          f"|  PAS unknown: {int(out.is_pas.isna().sum())}")


if __name__ == "__main__":
    main()
