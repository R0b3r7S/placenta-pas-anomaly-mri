#!/usr/bin/env python3
"""
Paper Table/Fig-4: PAS classification via the patient-level anomaly score, from
the HONEST leave-healthy-out run (phase3_150ep_lf00). Single in-domain score =
SSH_TSE (Rebro single-modality T2; Mendeley ssh_TSE).

- Rebro has both classes (3 PAS vs 4 held-out non-PAS) -> full metrics:
  AUC (threshold-free) + sensitivity/specificity at the Youden-optimal threshold.
  Reported all-7 and excluding reb010 (appendicitis, non-PAS confounder).
- Mendeley is POSITIVE-ONLY (130 PAS, no normals) -> AUC/specificity are
  UNDEFINED there. We report SENSITIVITY (detection rate) at the Rebro-derived
  threshold: fraction of external PAS flagged as anomalous. Honest caveat printed.

Output: comparison_results/fig4_classification_metrics.csv + rendered PNG.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
SCORES = PROJECT / "runs_fanogan_loo" / "phase3_150ep_lf00" / "scores_kappa1.0" / "scores.csv"
OUT_CSV = PROJECT / "comparison_results" / "fig4_classification_metrics.csv"
OUT_PNG = PROJECT / "shareable" / "figures" / "fig4_classification_metrics.png"

REBRO_PAS = ["reb001", "reb007", "reb008"]
REBRO_NONPAS = ["reb004", "reb005", "reb010", "reb011"]   # held-out (incl. reb010 appendicitis)


def per_patient():
    df = pd.read_csv(SCORES)
    parts = df.patch_path.apply(lambda p: Path(str(p)).stem.split("_"))
    df["cohort"] = parts.apply(lambda s: s[1] if s[0] == "mendeley" else "rebro")
    df["pid"] = parts.apply(lambda s: s[2] if s[0] == "mendeley" else s[0])
    per = df.groupby(["cohort", "pid"]).score.mean()
    return per["rebro"], per["tse"]


def youden_threshold(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def bootstrap_auc_ci(y, s, n_boot=10000, seed=42):
    """Stratified bootstrap 95% CI for AUC (resample within each class so both
    classes are always present; appropriate for the small n=7 Rebro test set)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        aucs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    reb, men = per_patient()
    rows = []

    for label, nonpas in [("Rebro (all 7 held-out)", REBRO_NONPAS),
                          ("Rebro (excl. reb010 appendicitis)", [p for p in REBRO_NONPAS if p != "reb010"])]:
        pids = REBRO_PAS + nonpas
        y = np.array([1] * len(REBRO_PAS) + [0] * len(nonpas))
        s = np.array([reb[p] for p in pids])
        auc = roc_auc_score(y, s)
        lo, hi = bootstrap_auc_ci(y, s)
        thr = youden_threshold(y, s)
        pred = (s >= thr).astype(int)
        sens = pred[y == 1].mean()
        spec = 1 - pred[y == 0].mean()
        rows.append({"cohort": label, "n_PAS": int(y.sum()), "n_nonPAS": int((y == 0).sum()),
                     "AUC": round(auc, 3), "AUC_CI95": f"[{lo:.2f}, {hi:.2f}]",
                     "threshold": round(thr, 3),
                     "sensitivity": round(sens, 3), "specificity": round(spec, 3)})

    # Mendeley: positive-only -> sensitivity at the Rebro (all-7) Youden threshold
    thr_all = rows[0]["threshold"]
    men_sens = (men.values >= thr_all).mean()
    rows.append({"cohort": "Mendeley (external, 130 PAS)", "n_PAS": len(men), "n_nonPAS": 0,
                 "AUC": "n/a (no normals)", "AUC_CI95": "n/a", "threshold": round(thr_all, 3),
                 "sensitivity": round(men_sens, 3), "specificity": "n/a (no normals)"})

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    fig, ax = plt.subplots(figsize=(12, 0.5 * len(df) + 1.4)); ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center",
                 colWidths=[0.30, 0.08, 0.09, 0.09, 0.13, 0.10, 0.10, 0.11])
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.7)
    for j in range(len(df.columns)):
        t[0, j].set_facecolor("#34495E"); t[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("PAS classification by patient-level anomaly score (leave-healthy-out model)\n"
                 "Mendeley is positive-only → sensitivity/detection-rate only (AUC & specificity undefined without normals)",
                 fontsize=10.5, pad=12)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(df.to_string(index=False))
    print(f"\nsaved -> {OUT_CSV}\nsaved -> {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
