#!/usr/bin/env python3
"""
Paper Table/Fig-6: effect of manual (GT) vs algorithmic (predicted) placenta
segmentation on the anomaly score, for Rebro and Mendeley — mean |Δ anomaly
score| and the AUC difference.

Rebro  = 7 held-out test patients (comparison_results/impact_predmask_comparison.csv);
         predicted masks from the fine-tuned AFA+MixUp model.
Mendeley = 19 TEST patients only, unseen by the seg model
         (comparison_results/impact_mendeley_comparison.csv); base AFA+MixUp masks.
AUC difference is Rebro-only (Mendeley is positive-only -> AUC undefined).

Output: comparison_results/fig6_impact_table.csv + rendered PNG.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
REB = PROJECT / "comparison_results" / "impact_predmask_comparison.csv"
MEN = PROJECT / "comparison_results" / "impact_mendeley_comparison.csv"
OUT_CSV = PROJECT / "comparison_results" / "fig6_impact_table.csv"
OUT_PNG = PROJECT / "shareable" / "figures" / "fig6_impact_table.png"


def main() -> int:
    r = pd.read_csv(REB)
    mad_r = (r.gt_mean_score - r.pred_mean_score).abs().mean()
    y = r.is_PAS.astype(int)
    auc_gt, auc_pr = roc_auc_score(y, r.gt_mean_score), roc_auc_score(y, r.pred_mean_score)
    r6 = r[r.pid != "reb010"]; y6 = r6.is_PAS.astype(int)
    auc_gt6, auc_pr6 = roc_auc_score(y6, r6.gt_mean_score), roc_auc_score(y6, r6.pred_mean_score)

    m = pd.read_csv(MEN)
    mad_m = (m.gt_mean_score - m.pred_mean_score).abs().mean()

    rows = [
        {"cohort": "Rebro test (all 7)", "n": len(r),
         "mean |Δ anomaly score|": f"{mad_r:.3f}",
         "AUC (GT mask)": f"{auc_gt:.3f}", "AUC (predicted mask)": f"{auc_pr:.3f}",
         "Δ AUC": f"{auc_gt - auc_pr:+.3f}"},
        {"cohort": "Rebro test (excl. appendicitis)", "n": len(r6),
         "mean |Δ anomaly score|": "—",
         "AUC (GT mask)": f"{auc_gt6:.3f}", "AUC (predicted mask)": f"{auc_pr6:.3f}",
         "Δ AUC": f"{auc_gt6 - auc_pr6:+.3f}"},
        {"cohort": "Mendeley test (19, unseen by seg)", "n": len(m),
         "mean |Δ anomaly score|": f"{mad_m:.3f}",
         "AUC (GT mask)": "n/a", "AUC (predicted mask)": "n/a", "Δ AUC": "n/a"},
    ]
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    fig, ax = plt.subplots(figsize=(14.5, 0.55 * len(df) + 1.4)); ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center",
                 colWidths=[0.27, 0.05, 0.21, 0.15, 0.19, 0.10])
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.8)
    for j in range(len(df.columns)):
        t[0, j].set_facecolor("#34495E"); t[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Impact of manual vs algorithmic segmentation on the anomaly score\n"
                 "(small |Δ| = automated masks preserve the anomaly signal; AUC diff = Rebro only, "
                 "Mendeley is positive-only)", fontsize=10.5, pad=12)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(df.to_string(index=False))
    print(f"\nsaved -> {OUT_CSV}\nsaved -> {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
