#!/usr/bin/env python3
"""
Paper Fig-3: anomaly-score scatter coloured by clinical group, built ONLY from
the HONEST leave-healthy-out run (phase3_150ep_lf00) — the model trained on 5
normal placentas (reb003/006/009/013/014) and NEVER on previa/PAS. This replaces
the earlier `fanogan_cross_modality_scatter.png`, which came from the OPTIMISTIC
run that trained on all 9 non-PAS Rebro (leakage).

Axes (like the referenced cross-modality plot): x = BTFE mean anomaly score,
y = SSH_TSE mean anomaly score. Mendeley patients have both modalities; Rebro is
single-modality (T2 HASTE/TSE) so it sits on the diagonal (x=y=score).

Groups: Rebro healthy, Rebro previa (marked), Rebro PAS,
Mendeley PAS. Only INDEPENDENT (held-out) Rebro test patients are shown — the 5
training patients are NOT plotted (their scores are low by construction).
No patient-id labels on the plot.

HONEST NOTE: reb010 (appendicitis, a non-PAS *abnormal* abdomen) is held out here
and scores high — it appears as a high non-PAS point. This is disclosed, not hidden.

Output: shareable/figures/fig3_anomaly_scatter_by_group.png  + a data CSV.
--journal: JMBE version (paper Figure 3) -> shareable/figures/journal/Fig3.pdf,
           vector PDF, no embedded title (JMBE: no titles inside illustrations),
           lettering 8 pt at print size.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from journal_figure_style import PAD_IN, TEXTWIDTH_IN, set_final_lettering, printed_size_pt

PROJECT = Path(__file__).resolve().parent.parent
SCORES = PROJECT / "runs_fanogan_loo" / "phase3_150ep_lf00" / "scores_kappa1.0" / "scores.csv"
OUT_PNG = PROJECT / "shareable" / "figures" / "fig3_anomaly_scatter_by_group.png"
OUT_CSV = PROJECT / "comparison_results" / "fig3_anomaly_scatter_data.csv"
JOURNAL_PDF = PROJECT / "shareable" / "figures" / "journal" / "Fig3.pdf"
JOURNAL_WIDTH_IN = 0.78 * TEXTWIDTH_IN  # \includegraphics[width=0.78\textwidth]

REBRO_PAS = ["reb001", "reb007", "reb008"]
REBRO_PREVIA = ["reb004"]
REBRO_HEALTHY = ["reb005", "reb010", "reb011"]   # held-out non-PAS non-previa
# (reb003/006/009/013/014 = training set -> excluded, not independent)


def parse(row):
    s = Path(str(row)).stem.split("_")
    if s[0] == "mendeley":
        return pd.Series({"cohort": s[1], "pid": s[2]})   # btfe/tse, subID
    return pd.Series({"cohort": "rebro", "pid": s[0]})


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper scatter figure (BTFE vs ssh-TSE anomaly scores).")
    ap.add_argument("--journal", action="store_true",
                    help="write the JMBE version: vector PDF without embedded title")
    args = ap.parse_args()
    if args.journal:
        plt.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts (editable text)

    df = pd.read_csv(SCORES)
    df[["cohort", "pid"]] = df.patch_path.apply(parse)
    per = df.groupby(["cohort", "pid"]).score.mean().reset_index()

    men = per[per.cohort.isin(["btfe", "tse"])].pivot(index="pid", columns="cohort", values="score").dropna()
    reb = per[per.cohort == "rebro"].set_index("pid").score

    fig, ax = plt.subplots(figsize=(8.2, 8))
    hi = max(men.max().max(), reb.max()) * 1.05
    ax.plot([0, hi], [0, hi], "--", color="#999", lw=1, alpha=0.7, label="y = x")

    # Mendeley PAS (both modalities)
    ax.scatter(men["btfe"], men["tse"], s=40, alpha=0.7, c="#E8A33D",
               edgecolor="white", lw=0.4, label=f"External PAS (n={len(men)})")

    # Rebro on the diagonal (single modality)
    def diag(pids, **kw):
        v = [reb[p] for p in pids if p in reb.index]
        ax.scatter(v, v, **kw)
        return v
    diag(REBRO_HEALTHY, s=140, marker="o", c="#2E8B57", edgecolor="black", lw=0.8,
         label=f"Internal non-PAS (n={len(REBRO_HEALTHY)})", zorder=5)
    diag(REBRO_PREVIA, s=140, marker="D", c="#5B9BD5", edgecolor="black", lw=1.0,
         label="Internal previa (n=1)", zorder=6)
    diag(REBRO_PAS, s=200, marker="*", c="#C0392B", edgecolor="black", lw=0.8,
         label=f"Internal PAS (n={len(REBRO_PAS)})", zorder=6)

    ax.set_xlabel("BTFE mean anomaly score (per patient)", fontsize=12)
    ax.set_ylabel("ssh-TSE mean anomaly score (per patient)", fontsize=12)
    if not args.journal:   # JMBE: no titles inside illustrations (text lives in the caption)
        ax.set_title("Patient-level anomaly scores by group\n"
                     "(leave-healthy-out model; trained only on 5 normal placentas)",
                     fontsize=12)
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.grid(alpha=0.3)
    # journal: the 8-pt legend is larger -> empty upper-left corner (no data there)
    ax.legend(fontsize=9, loc="upper left" if args.journal else "lower right",
              labelspacing=1.2, handletextpad=1.0, borderpad=1.0)
    if args.journal:
        JOURNAL_PDF.parent.mkdir(parents=True, exist_ok=True)
        fs = set_final_lettering(fig, JOURNAL_WIDTH_IN)
        print(f"  lettering: {fs:.1f} pt in the figure = "
              f"{printed_size_pt(fig, fs, JOURNAL_WIDTH_IN):.1f} pt at print size")
        fig.savefig(JOURNAL_PDF, bbox_inches="tight", pad_inches=PAD_IN)
        OUT = JOURNAL_PDF
    else:
        fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
        OUT = OUT_PNG
    plt.close(fig)

    # data table for the record
    rows = []
    for p in REBRO_PAS + REBRO_PREVIA + REBRO_HEALTHY:
        if p in reb.index:
            grp = ("PAS" if p in REBRO_PAS else "previa" if p in REBRO_PREVIA else "non-PAS")
            rows.append({"patient": p, "group": f"Rebro {grp}", "btfe": None, "ssh_tse": round(reb[p], 4)})
    for p in men.index:
        rows.append({"patient": p, "group": "Mendeley PAS",
                     "btfe": round(men.loc[p, "btfe"], 4), "ssh_tse": round(men.loc[p, "tse"], 4)})
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"saved -> {OUT}\nsaved -> {OUT_CSV}")
    print("\nRebro held-out scores (single modality):")
    for p in REBRO_PAS + REBRO_PREVIA + REBRO_HEALTHY:
        if p in reb.index:
            print(f"  {p}: {reb[p]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
