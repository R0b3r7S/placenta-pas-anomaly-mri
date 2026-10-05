#!/usr/bin/env python3
"""
Cross-epoch comparison for the leave-healthy-out f-AnoGAN benchmarks.

Reads the per-tag `auc_table.csv` files already produced by
analyze_loo_results.py (one tag per epoch budget) and produces a single
combined view comparing 75 vs 150 vs 300 epochs at matched lower_fraction /
kappa. This is the single source of truth for the epoch-comparison numbers and
figures that go into the paper — do NOT eyeball it inline.

Tag -> epochs is parsed from the tag (e.g. 'phase4_75ep' -> 75). A tag whose
analysis dir is missing (e.g. the 75ep run still training) is skipped with a
warning, so this can be run early for a partial preview and re-run later to
complete the table.

Outputs (under <runs-dir>/epoch_comparison/ by default):
  epoch_comparison.csv          all rows: epochs x lower_fraction x kappa + every metric
  epoch_comparison_k1.csv       kappa=1.0 slice (the headline rows)
  best_config_per_epoch.csv     the top patient-AUC config for each epoch budget
  epoch_comparison.json         everything above, structured
  patient_auc_vs_lf.png         patient AUC vs lower_fraction, one line per epoch budget
  test_normal_max_vs_lf.png     worst held-out-healthy score (overfitting/false-positive
                                 blow-up indicator) vs lower_fraction, one line per epoch
  patch_auc_vs_lf.png           patch-level AUC vs lower_fraction, one line per epoch

Usage:
  conda run -n monai_placenta python f-AnoGAN-pytorch/compare_epoch_benchmarks.py
  conda run -n monai_placenta python f-AnoGAN-pytorch/compare_epoch_benchmarks.py \\
      --tags phase4_75ep phase3_150ep phase2_300ep
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    from tqdm import tqdm
    _JOBLIB = True
except ImportError:
    _JOBLIB = False
    def tqdm(x, **k):
        return x

# Default: the three benchmarks that share an identical grid (4 lower_fractions
# x 3 kappas), differing only in WGAN epochs. (The one-off phase1_150ep, which
# ran lower_fraction=0.5 only, and the original 9-patient all-in-training
# proof-of-concept are deliberately NOT included.)
DEFAULT_TAGS = ["phase4_75ep", "phase3_150ep", "phase2_300ep"]

# Clinical identity of each held-out Rebro patient (for the per-patient figure).
# (label, category) ; category drives colour/marker.
REBRO_CLINICAL = {
    "reb001": ("PAS (percreta)", "PAS"),
    "reb007": ("PAS (acreta)", "PAS"),
    "reb008": ("PAS (percreta+previa)", "PAS"),
    "reb004": ("previa, no accreta", "previa"),
    "reb005": ("healthy", "healthy"),
    "reb010": ("appendicitis", "confounder"),
    "reb011": ("healthy", "healthy"),
}
CAT_COLOR = {"PAS": "#27AE60", "previa": "#D4AC0D",
             "healthy": "#2E86C1", "confounder": "#C0392B"}


def epochs_of(tag: str) -> int:
    """Parse the epoch budget from a tag like 'phase4_75ep' -> 75."""
    m = re.search(r"(\d+)ep\b", tag)
    return int(m.group(1)) if m else -1


def _load_tag(runs_dir: Path, tag: str):
    """Read one tag's auc_table.csv and stamp epochs + tag. Returns a DataFrame
    or None if the analysis dir does not exist yet (run still in progress)."""
    csv = runs_dir / f"{tag}_analysis" / "auc_table.csv"
    if not csv.is_file():
        return None
    df = pd.read_csv(csv)
    df.insert(0, "epochs", epochs_of(tag))
    df.insert(1, "tag", tag)
    return df


# --------------------------------------------------------------------------
# Figures: one line per epoch budget, x = lower_fraction, at kappa=1.0
# --------------------------------------------------------------------------
def _line_by_epoch(df_k1, ycol, ylabel, title, out_path, ylim=None,
                   hline=None, hline_label=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for ep in sorted(df_k1.epochs.unique()):
        sub = df_k1[df_k1.epochs == ep].sort_values("lower_fraction")
        ax.plot(sub.lower_fraction, sub[ycol], "o-", lw=2, label=f"{ep} epochs")
    if hline is not None:
        ax.axhline(hline, color="#999", ls=":", label=hline_label)
    ax.set_xlabel("lower_fraction (0.0=whole contour, 0.5=lower half, 0.7=bottom 30%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _load_per_patient(runs_dir, tag, lf=0.0, kappa=1.0):
    """Per-patient scores for one tag at a fixed (lf, kappa). Module-level for
    joblib. Returns a DataFrame stamped with epochs, or None if absent."""
    csv = runs_dir / f"{tag}_analysis" / "per_patient_scores.csv"
    if not csv.is_file():
        return None
    df = pd.read_csv(csv)
    df = df[np.isclose(df.lower_fraction, lf) & np.isclose(df.kappa, kappa)].copy()
    df["epochs"] = epochs_of(tag)
    return df


def _overfitting_vs_epochs(df, out_path, lf=0.0):
    """Group-level overfitting signature at a fixed lower_fraction: held-out
    healthy scores inflate with epochs while PAS scores stay flat."""
    sub = df[np.isclose(df.lower_fraction, lf) & np.isclose(df.kappa, 1.0)].sort_values("epochs")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub.epochs, sub.test_normal_max, "o-", color="#C0392B", lw=2,
            label="worst held-out healthy")
    ax.plot(sub.epochs, sub.test_normal_mean, "o--", color="#E67E22", lw=2,
            label="mean held-out healthy")
    ax.plot(sub.epochs, sub.pas_mean, "s-", color="#27AE60", lw=2, label="mean PAS")
    ax.plot(sub.epochs, sub.pas_min, "s--", color="#16A085", lw=2, label="lowest PAS")
    ax.set_xticks(sub.epochs)
    ax.set_xlabel("WGAN epochs")
    ax.set_ylabel("per-patient anomaly score (kappa=1.0)")
    ax.set_title(f"Overfitting signature at lower_fraction={lf}\n"
                 "held-out healthy scores inflate with epochs; PAS scores stay flat")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _per_patient_trajectory(pp_all, out_path):
    """Per held-out Rebro patient: mean anomaly score vs epoch budget (lf=0.0,
    kappa=1.0). Shows previa staying low, appendicitis climbing, PAS high."""
    reb = pp_all[pp_all.cohort == "rebro"].copy()
    if reb.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for pid, g in reb.groupby("pid"):
        g = g.sort_values("epochs")
        label_txt, cat = REBRO_CLINICAL.get(pid, (pid, "healthy"))
        color = CAT_COLOR.get(cat, "#888888")
        style = "-" if g.is_anom.iloc[0] == 1 else "--"
        marker = {"previa": "*", "confounder": "X", "PAS": "s"}.get(cat, "o")
        ms = 14 if cat in ("previa", "confounder") else 8
        ax.plot(g.epochs, g.mean_score, style, marker=marker, ms=ms, color=color,
                lw=1.8, label=f"{pid}: {label_txt}")
    ax.set_xticks(sorted(reb.epochs.unique()))
    ax.set_xlabel("WGAN epochs")
    ax.set_ylabel("per-patient mean anomaly score (lf=0.0, kappa=1.0)")
    ax.set_title("Per-patient anomaly score vs epoch budget (held-out Rebro)", wrap=True)
    # marker key as a compact caption under the axes (keeps it out of the title)
    fig.text(0.5, -0.02,
             "blue circle = healthy   |   gold star = previa, no accreta   |   "
             "red X = appendicitis (confounder)   |   green square = PAS",
             ha="center", va="top", fontsize=8)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _auc_vs_kappa(df, out_path):
    """patient + patch AUC vs kappa, one line per epoch budget, at each epoch's
    OWN best lower_fraction (chosen on the kappa=1.0 slice). Shows how little
    kappa moves the ranking."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ep in sorted(df.epochs.unique()):
        sub_ep = df[df.epochs == ep]
        k1 = sub_ep[np.isclose(sub_ep.kappa, 1.0)]
        if k1.empty:
            continue
        best_lf = k1.loc[k1.patient_auc.idxmax(), "lower_fraction"]
        line = sub_ep[np.isclose(sub_ep.lower_fraction, best_lf)].sort_values("kappa")
        axes[0].plot(line.kappa, line.patient_auc, "o-", lw=2,
                     label=f"{ep} ep (lf={best_lf})")
        axes[1].plot(line.kappa, line.patch_auc, "s--", lw=2,
                     label=f"{ep} ep (lf={best_lf})")
    for ax, t in zip(axes, ["patient-level AUC", "patch-level AUC"]):
        ax.axhline(0.5, color="#999", ls=":", label="chance")
        ax.set_xlabel("kappa (weight of discriminator-feature residual A_D)")
        ax.set_title(t)
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("AUC vs kappa by epoch budget\n(each at its own best lower_fraction)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs_fanogan_loo", type=Path)
    p.add_argument("--tags", nargs="+", default=DEFAULT_TAGS,
                   help="Run tags to compare (one per epoch budget).")
    p.add_argument("--out", default=None, type=Path,
                   help="Output dir (default: <runs-dir>/epoch_comparison).")
    p.add_argument("--workers", type=int, default=20,
                   help="joblib workers for reading the per-tag tables (default 20).")
    args = p.parse_args()

    out = args.out or (args.runs_dir / "epoch_comparison")
    out.mkdir(parents=True, exist_ok=True)

    # read each tag's auc_table.csv (independent per tag -> parallel)
    if _JOBLIB and args.workers > 1 and len(args.tags) > 1:
        frames = Parallel(n_jobs=args.workers)(
            delayed(_load_tag)(args.runs_dir, t) for t in tqdm(args.tags, desc="tags"))
    else:
        frames = [_load_tag(args.runs_dir, t) for t in tqdm(args.tags, desc="tags")]

    found, missing = [], []
    for tag, fr in zip(args.tags, frames):
        (missing if fr is None else found).append(tag)
    if missing:
        print(f"  WARNING: no analysis dir yet for {missing} "
              f"(run analyze_loo_results.py for those tags first / still training) "
              f"-- producing a PARTIAL comparison.")
    if not found:
        print(f"ERROR: none of the tags {args.tags} have an auc_table.csv under "
              f"{args.runs_dir}. Run analyze_loo_results.py first.")
        return 1

    df = pd.concat([f for f in frames if f is not None], ignore_index=True)
    df = df.sort_values(["lower_fraction", "kappa", "epochs"])
    df.to_csv(out / "epoch_comparison.csv", index=False)

    k1 = df[np.isclose(df.kappa, 1.0)].copy()
    k1.to_csv(out / "epoch_comparison_k1.csv", index=False)

    # best config (max patient AUC) per epoch budget, over the kappa=1.0 slice
    best_rows = []
    for ep in sorted(k1.epochs.unique()):
        sub = k1[k1.epochs == ep]
        best_rows.append(sub.loc[sub.patient_auc.idxmax()])
    best = pd.DataFrame(best_rows)
    best.to_csv(out / "best_config_per_epoch.csv", index=False)

    with open(out / "epoch_comparison.json", "w") as f:
        json.dump({"all": df.to_dict("records"),
                   "kappa1": k1.to_dict("records"),
                   "best_per_epoch": best.to_dict("records"),
                   "tags_found": found, "tags_missing": missing},
                  f, indent=2, default=str)

    # figures (kappa=1.0)
    _line_by_epoch(k1, "patient_auc", "patient-level AUC",
                   "Leave-healthy-out patient AUC vs lower_fraction by epoch budget (kappa=1.0)",
                   out / "patient_auc_vs_lf.png", ylim=(0.4, 1.02),
                   hline=0.5, hline_label="chance")
    _line_by_epoch(k1, "test_normal_max",
                   "worst held-out healthy score (lower = better)",
                   "Held-out-healthy false-positive blow-up vs lower_fraction by epoch (kappa=1.0)\n"
                   "(this is the overfitting indicator)",
                   out / "test_normal_max_vs_lf.png")
    _line_by_epoch(k1, "patch_auc", "patch-level AUC",
                   "Patch-level AUC vs lower_fraction by epoch budget (kappa=1.0)",
                   out / "patch_auc_vs_lf.png", ylim=(0.4, 1.02),
                   hline=0.5, hline_label="chance")
    _auc_vs_kappa(df, out / "auc_vs_kappa_by_epoch.png")

    # clinical figures at the winning lower_fraction=0.0
    _overfitting_vs_epochs(df, out / "overfitting_vs_epochs.png", lf=0.0)
    if _JOBLIB and args.workers > 1 and len(found) > 1:
        pps = Parallel(n_jobs=args.workers)(
            delayed(_load_per_patient)(args.runs_dir, t) for t in tqdm(found, desc="per-patient"))
    else:
        pps = [_load_per_patient(args.runs_dir, t) for t in found]
    pps = [p for p in pps if p is not None]
    if pps:
        _per_patient_trajectory(pd.concat(pps, ignore_index=True),
                                out / "per_patient_trajectory_lf0.png")

    # console summary
    show = ["epochs", "lower_fraction", "patch_auc", "patient_auc",
            "test_normal_max", "pas_min",
            "clean_separation_rebro_excl_appendicitis", "previa_below_rebro_pas"]
    show = [c for c in show if c in k1.columns]
    print("\n=== Epoch comparison (kappa=1.0) ===")
    print(k1[show].round(3).to_string(index=False))
    print("\n=== Best config per epoch budget (by patient AUC, kappa=1.0) ===")
    bshow = [c for c in ["epochs", "lower_fraction", "patient_auc", "patch_auc",
                         "test_normal_max"] if c in best.columns]
    print(best[bshow].round(3).to_string(index=False))

    # FULL table: every lower_fraction x kappa x epoch combination
    full_show = [c for c in ["epochs", "lower_fraction", "kappa", "patch_auc",
                             "patient_auc", "test_normal_max", "pas_min"]
                 if c in df.columns]
    print("\n=== FULL grid (every lower_fraction x kappa x epoch) ===")
    print(df.sort_values(["lower_fraction", "epochs", "kappa"])[full_show]
          .round(3).to_string(index=False))

    # how much does kappa move the patient AUC? (spread across kappas, per lf x epoch)
    spread = (df.groupby(["epochs", "lower_fraction"])
                .patient_auc.agg(auc_min="min", auc_max="max").reset_index())
    spread["kappa_auc_spread"] = (spread.auc_max - spread.auc_min).round(4)
    spread.to_csv(out / "kappa_sensitivity.csv", index=False)
    print("\n=== Kappa sensitivity: patient-AUC spread across kappa "
          "{0.5,1.0,2.0} per (epochs, lower_fraction) ===")
    print(spread[["epochs", "lower_fraction", "kappa_auc_spread"]]
          .sort_values("kappa_auc_spread", ascending=False).round(4).to_string(index=False))
    print(f"  -> max kappa-induced AUC spread anywhere = "
          f"{spread.kappa_auc_spread.max():.4f}")

    print(f"\nDONE. CSV/JSON/figures saved under {out}/")
    if missing:
        print(f"PARTIAL: re-run after {missing} finish + their analyze_loo_results.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
