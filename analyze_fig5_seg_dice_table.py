#!/usr/bin/env python3
"""
Paper Table/Fig-5: segmentation Dice on each held-out Rebro test patient (+ mean)
for the four models: Baseline, AFA+MixUp, Baseline Fine-Tuned, AFA+MixUp Fine-Tuned.

Reads the already-verified per-patient before/after Dice from
comparison_results/seg3_finetune_comparison.csv (produced by analyze_seg3_finetune.py):
  Baseline           = TSE_dynunet_regular   dice_before
  AFA+MixUp          = TSE_dynunet_afa_mixup dice_before
  Baseline Fine-Tuned= TSE_dynunet_regular   dice_after
  AFA+MixUp Fine-Tuned=TSE_dynunet_afa_mixup dice_after
(Segmentation track — independent of the anomaly-detection run, honest.)

Output: comparison_results/fig5_seg_dice_table.csv + rendered PNG.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
SRC = PROJECT / "comparison_results" / "seg3_finetune_comparison.csv"
OUT_CSV = PROJECT / "comparison_results" / "fig5_seg_dice_table.csv"
OUT_PNG = PROJECT / "shareable" / "figures" / "fig5_seg_dice_table.png"

COLS = ["Baseline", "AFA+MixUp", "Baseline\nFine-Tuned", "AFA+MixUp\nFine-Tuned"]


def main() -> int:
    d = pd.read_csv(SRC)
    reg = d[d.model == "TSE_dynunet_regular"].set_index("patient")
    afa = d[d.model == "TSE_dynunet_afa_mixup"].set_index("patient")
    patients = list(reg.index)

    tbl = pd.DataFrame({
        "patient": patients,
        "Baseline": [reg.loc[p, "dice_before"] for p in patients],
        "AFA+MixUp": [afa.loc[p, "dice_before"] for p in patients],
        "Baseline Fine-Tuned": [reg.loc[p, "dice_after"] for p in patients],
        "AFA+MixUp Fine-Tuned": [afa.loc[p, "dice_after"] for p in patients],
    })
    mean_row = {"patient": "MEAN", **{c: round(tbl[c].mean(), 3) for c in tbl.columns if c != "patient"}}
    tbl = pd.concat([tbl, pd.DataFrame([mean_row])], ignore_index=True)
    OUT_CSV.parent.mkdir(exist_ok=True)
    tbl.round(3).to_csv(OUT_CSV, index=False)

    disp = tbl.copy()
    for c in disp.columns[1:]:
        disp[c] = disp[c].map(lambda v: f"{v:.3f}")
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(disp) + 1.2)); ax.axis("off")
    cell = [[disp.iloc[i, 0]] + [disp.iloc[i, j] for j in range(1, 5)] for i in range(len(disp))]
    t = ax.table(cellText=cell, colLabels=["patient"] + COLS, loc="center", cellLoc="center",
                 colWidths=[0.16, 0.19, 0.19, 0.19, 0.19])
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.7)
    for j in range(5):
        t[0, j].set_facecolor("#34495E"); t[0, j].set_text_props(color="white", fontweight="bold")
    # bold the best model per row; shade mean row
    vals = tbl[tbl.columns[1:]].values
    for i in range(len(tbl)):
        best = 1 + int(np.argmax(vals[i]))
        for j in range(5):
            if i == len(tbl) - 1:
                t[i + 1, j].set_facecolor("#dfe7ef")
            if j == best:
                t[i + 1, j].set_text_props(fontweight="bold", color="#08519c")
    ax.set_title("Segmentation Dice on the 7 held-out Rebro patients\n"
                 "(bold = best model per row; fine-tuning uses the 5 anomaly-train Rebro)",
                 fontsize=11, pad=12)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(tbl.round(3).to_string(index=False))
    print(f"\nsaved -> {OUT_CSV}\nsaved -> {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
