#!/usr/bin/env python3
"""
Parse the KBC Rebro clinical sheet (HRZZ_FERIT.xlsx) into clinical_labels.csv.

Maps each sheet row (1.1 … 1.16) to the corresponding reb*** patient ID
assigned by dicom_to_slices.py (lex-sorted folder order under FERIT/), and
derives the binary is_pas label from the diagnosis text.

Output:
    dataset/dicom/kbc_rebro/clinical_labels.csv

Columns:
    sheet_id, ga, dx, uzv, comment, note, is_pas, delivered, reb_id, folder_name

Re-run any time HRZZ_FERIT.xlsx is updated or new patient folders arrive.

Usage:
    conda run --no-capture-output -n monai_placenta python parse_rebro_sheet.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SHEET = HERE / "dataset" / "dicom" / "kbc_rebro" / "HRZZ_FERIT.xlsx"
FERIT = HERE / "dataset" / "dicom" / "kbc_rebro" / "FERIT"
OUT = HERE / "dataset" / "dicom" / "kbc_rebro" / "clinical_labels.csv"

PAS_KEYS = ("acreta", "accreta", "increta", "percreta")


def is_pas(dx: str) -> int | None:
    if pd.isna(dx):
        return None
    s = str(dx).lower()
    return int(any(k in s for k in PAS_KEYS))


def main() -> None:
    if not SHEET.exists():
        raise FileNotFoundError(f"{SHEET} not found")
    if not FERIT.is_dir():
        raise FileNotFoundError(f"{FERIT} not found")

    df = pd.read_excel(SHEET, header=0)
    df.columns = ["sheet_id", "ga", "dx", "uzv", "_4", "comment", "_6", "note"]
    df = df[["sheet_id", "ga", "dx", "uzv", "comment", "note"]].copy()
    df["sheet_id"] = df["sheet_id"].astype(str).str.rstrip(".")
    df["is_pas"] = df["dx"].apply(is_pas)
    df["delivered"] = ~df["note"].astype(str).str.contains(
        "Nije segmentirano", na=False
    )

    # Lex-sort the FERIT folders — this matches dicom_to_slices.py assignment.
    dirs = sorted(d for d in FERIT.iterdir() if d.is_dir())
    # Folder names like "1.2 U" map to sheet_id "1.2" after stripping the " U"
    # suffix (the " U" marks paired ultrasound).
    mapping = {
        d.name.replace(" U", "").strip(): (f"reb{i:03d}", d.name)
        for i, d in enumerate(dirs, start=1)
    }
    df["reb_id"] = df["sheet_id"].map(lambda s: mapping.get(s, ("", ""))[0])
    df["folder_name"] = df["sheet_id"].map(lambda s: mapping.get(s, ("", ""))[1])

    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {OUT}")
    print(
        f"rows: {len(df)} | with reb_id: {int((df.reb_id != '').sum())} | "
        f"PAS+: {int(df.is_pas.fillna(0).sum())} | "
        f"PAS-: {int((df.is_pas == 0).sum())} | "
        f"PAS unknown: {int(df.is_pas.isna().sum())} | "
        f"delivered: {int(df.delivered.sum())}"
    )


if __name__ == "__main__":
    main()
