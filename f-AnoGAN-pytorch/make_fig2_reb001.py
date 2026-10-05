#!/usr/bin/env python3
"""
Paper Fig-2: continuous anomaly map of the strongest PAS patient (reb001, placenta
percreta, patient-level score 0.950), five consecutive slices, from the HONEST
leave-healthy-out run (phase3_150ep, full placental boundary lf=0.0, Gaussian
smoothing sigma=6). Colour = continuous anomaly (jet), cyan = placenta outline.
Orientation matches the QC/DATASET (spine-left upright).

Clean, paper-appropriate title (no internal code parameters): states that the
complete placental boundary is analysed and that sigma=6 smoothing is used.

Output: shareable/figures/fig2_reb001_continuous.png
--journal: JMBE version (paper Figure 2) -> shareable/figures/journal/Fig2.png,
           lettering 8 pt at print size, RGB (no alpha), 600 dpi.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from journal_figure_style import TEXTWIDTH_IN, set_final_lettering, printed_size_pt, save_rgb_png

PROJECT = Path(__file__).resolve().parent.parent
NIFTI = PROJECT / "shareable" / "anomaly_nifti_rebro" / "lf00_sigma6" / "reb001"
OUT = PROJECT / "shareable" / "figures" / "fig2_reb001_continuous.png"
JOURNAL_PNG = PROJECT / "shareable" / "figures" / "journal" / "Fig2.png"
JOURNAL_WIDTH_IN = TEXTWIDTH_IN        # \includegraphics[width=\textwidth]
SLICES = [44, 45, 46, 47, 48]          # consecutive slices over the invaded region

disp = lambda a: np.fliplr(a.T)        # spine-left upright (matches DATASET/QC)


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper anomaly-map figure (reb001, five slices).")
    ap.add_argument("--journal", action="store_true",
                    help="write the JMBE version: RGB PNG at 600 dpi")
    args = ap.parse_args()

    mri = nib.load(str(NIFTI / "mri.nii.gz")).get_fdata()
    cont = nib.load(str(NIFTI / "anomaly_continuous.nii.gz")).get_fdata()
    pl = nib.load(str(NIFTI / "placenta_mask.nii.gz")).get_fdata()
    # scale the colour range to the shown slices so the anomaly is visible
    vmax = float(cont[:, :, SLICES].max())

    fig, ax = plt.subplots(1, len(SLICES), figsize=(3.0 * len(SLICES), 3.4))
    for i, k in enumerate(SLICES):
        ax[i].imshow(disp(mri[:, :, k]), cmap="gray")
        h = disp(cont[:, :, k])
        im = ax[i].imshow(np.ma.masked_where(h <= 0.05 * vmax, h), cmap="jet",
                          alpha=0.6, vmin=0, vmax=vmax)
        ax[i].contour(disp(pl[:, :, k]), levels=[0.5], colors="cyan", linewidths=0.7)
        ax[i].set_title(f"slice {k}", fontsize=11)
        ax[i].axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    # journal: at 8 pt the long label no longer fits the colour-bar height
    cbar.set_label("anomaly score" if args.journal else "continuous anomaly score", fontsize=9)
    # no embedded title/subtitle -- all descriptive text lives in the figure caption
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
    print(f"  reb001 volume {mri.shape}, shown-slice anomaly vmax={vmax:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
