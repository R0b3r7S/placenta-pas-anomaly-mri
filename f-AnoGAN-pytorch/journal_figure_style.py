#!/usr/bin/env python3
"""
Shared settings for the JMBE (journal) versions of the paper figures.

JMBE artwork rules (submission guidelines, pp. 15-19): RGB colour, >= 600 dpi for
halftone/combination art (vector for plots), no titles inside illustrations, and
lettering of about 8-12 pt at the final printed size.

Used by the `--journal` option of make_methods_patches_figure.py (Fig 1),
make_fig2_reb001.py (Fig 2, anomaly map) and make_fig3_group_scatter.py (Fig 3, scatter).
"""
from __future__ import annotations
import io

from PIL import Image
from matplotlib.text import Text

TEXTWIDTH_IN = 372.0 / 72.27   # sn-jnl text width (372 pt) in inches
FINAL_PT = 8.0                 # lettering size at the printed size
DPI = 600                      # JMBE minimum for combination art
PAD_IN = 0.1                   # padding used with bbox_inches="tight"


def set_final_lettering(fig, printed_width_in, final_pt=FINAL_PT, iters=4):
    """Set every text element so it prints at `final_pt` once LaTeX scales the
    saved (tight) figure to `printed_width_in`. Iterates because larger text
    slightly changes the tight bounding box. Returns the font size used."""
    fs = final_pt
    for _ in range(iters):
        fig.canvas.draw()
        width_in = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * PAD_IN
        fs = final_pt * width_in / printed_width_in
        for ax in fig.axes:
            ax.tick_params(labelsize=fs)
        for t in fig.findobj(Text):
            if t.get_text():
                t.set_fontsize(fs)
    return fs


def printed_size_pt(fig, fs, printed_width_in):
    """Font size `fs` as it will appear once the tight figure is scaled to print."""
    fig.canvas.draw()
    width_in = fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * PAD_IN
    return fs * printed_width_in / width_in


def save_rgb_png(fig, path, dpi=DPI):
    """Tight PNG at `dpi`, converted to RGB (no alpha channel)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=PAD_IN,
                facecolor="white")
    Image.open(buf).convert("RGB").save(path, dpi=(dpi, dpi))
