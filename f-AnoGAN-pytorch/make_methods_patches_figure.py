#!/usr/bin/env python3
"""
Methods figure: how boundary patches are extracted and rotation-normalized.
Panel A: an MRI slice (reb007, slice 39) with the placenta outline, the sampled
patch centres along the lower placental boundary, and the oriented sampling boxes
(each aligned to the local boundary tangent). Panel B: the resulting 64x64 patches
after rotation normalization -> the boundary runs horizontally through each patch.

Uses the manifest (center_x, center_y, angle_rad) from the honest run and
extract_oriented_patch() from the pipeline, so the illustration matches exactly.

Output: shareable/figures/methods_boundary_patches.png
--journal: JMBE version (paper Figure 1) -> shareable/figures/journal/Fig1.png,
           without the two panel titles (JMBE: no titles inside illustrations; the
           text lives in the caption), lettering 8 pt at print size, RGB, 600 dpi.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
from extract_boundary_patches import extract_oriented_patch   # exact pipeline fn
from journal_figure_style import TEXTWIDTH_IN, set_final_lettering, printed_size_pt, save_rgb_png

MANIFEST = PROJECT / "boundary_patches_loo" / "phase3_150ep_lf00" / "manifest.csv"
IMG = PROJECT / "dataset" / "mri_png" / "DATASET_EXTERNAL_REBRO" / "images" / "reb007" / "reb007_039.png"
MASK = PROJECT / "dataset" / "mri_png" / "DATASET_EXTERNAL_REBRO" / "masks" / "reb007" / "reb007_039.png"
OUT = PROJECT / "shareable" / "figures" / "methods_boundary_patches.png"
JOURNAL_PNG = PROJECT / "shareable" / "figures" / "journal" / "Fig1.png"
JOURNAL_WIDTH_IN = TEXTWIDTH_IN        # \includegraphics[width=\textwidth]
PID, SID, S = "reb007", 39, 64


def box_corners(cx, cy, angle, size):
    h = size / 2.0
    rx = np.array([-h, h, h, -h, -h]); ry = np.array([-h, -h, h, h, -h])
    ca, sa = np.cos(angle), np.sin(angle)
    return cx + ca * rx - sa * ry, cy + sa * rx + ca * ry


def main() -> int:
    ap = argparse.ArgumentParser(description="Methods figure: boundary-patch extraction.")
    ap.add_argument("--journal", action="store_true",
                    help="write the JMBE version: no panel titles, RGB PNG at 600 dpi")
    args = ap.parse_args()

    m = pd.read_csv(MANIFEST)
    rows = m[(m.patient_id == PID) & (m.slice_id == SID)].sort_values("boundary_index").reset_index(drop=True)
    img = np.asarray(Image.open(IMG).convert("L"), dtype=np.float64)
    mask = np.asarray(Image.open(MASK).convert("L"), dtype=np.uint8) > 0
    n_total = len(rows)
    if args.journal:   # 8 distinct patches evenly spaced around the CLOSED outline
        sel = np.unique(np.linspace(0, n_total, 8, endpoint=False).round().astype(int))
    else:              # (first and last index are neighbours on a closed contour)
        sel = np.unique(np.linspace(0, n_total - 1, 8).round().astype(int))   # 8 representative
    print(f"  {PID} slice {SID}: {n_total} patches total, showing {len(sel)}")

    fig = plt.figure(figsize=(15, 6.2))
    # Panel A — slice + boundary + a representative subset of oriented boxes (numbered)
    axA = fig.add_axes([0.02, 0.02, 0.44, 0.84])
    axA.imshow(img, cmap="gray")
    axA.contour(mask, levels=[0.5], colors="cyan", linewidths=1.2)
    for j, idx in enumerate(sel, start=1):
        r = rows.iloc[idx]
        bx, by = box_corners(r.center_x, r.center_y, r.angle_rad, S)
        axA.plot(bx, by, color="#F4D03F", lw=1.4)
        axA.plot(r.center_x, r.center_y, ".", color="red", ms=5)
        axA.text(r.center_x, r.center_y, str(j), color="red", fontsize=8, fontweight="bold",
                 ha="center", va="center")
    ys, xs = np.where(mask)
    axA.set_xlim(xs.min() - 70, xs.max() + 70); axA.set_ylim(ys.max() + 70, ys.min() - 70)
    axA.axis("off")
    if not args.journal:   # JMBE: no titles inside illustrations (text lives in the caption)
        fig.text(0.24, 0.93, f"A. Boundary-patch sampling ({n_total} patches on this slice; {len(sel)} shown)\n"
                 "cyan = placenta outline · yellow = oriented 64×64 sampling boxes along the placental boundary",
                 ha="center", fontsize=11)

    # Panel B — the rotation-normalized patches (2x4), numbered to match A
    if not args.journal:
        fig.text(0.74, 0.93, "B. Rotation-normalized 64×64 patches\n(boundary tangent aligned horizontally — yellow line)",
                 ha="center", fontsize=11)
    cols = 4
    for i, idx in enumerate(sel):
        r = rows.iloc[idx]
        patch = extract_oriented_patch(img, r.center_x, r.center_y, r.angle_rad, S)
        ax = fig.add_axes([0.525 + (i % cols) * 0.118, 0.44 - (i // cols) * 0.40, 0.105, 0.36])
        ax.imshow(patch, cmap="gray"); ax.axhline(S / 2, color="#F4D03F", lw=0.7, alpha=0.8)
        ax.set_title(str(i + 1), fontsize=9, color="red", pad=2)
        ax.set_xticks([]); ax.set_yticks([])
    if args.journal:
        JOURNAL_PNG.parent.mkdir(parents=True, exist_ok=True)
        fs = set_final_lettering(fig, JOURNAL_WIDTH_IN)
        print(f"  lettering: {fs:.1f} pt in the figure = "
              f"{printed_size_pt(fig, fs, JOURNAL_WIDTH_IN):.1f} pt at print size")
        save_rgb_png(fig, JOURNAL_PNG)
        out = JOURNAL_PNG
    else:
        fig.savefig(OUT, dpi=150, bbox_inches="tight")
        out = OUT
    plt.close(fig)
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
