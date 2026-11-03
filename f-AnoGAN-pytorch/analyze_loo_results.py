#!/usr/bin/env python3
"""
Analysis + reporting for the leave-healthy-out f-AnoGAN experiment.

Reads the scores.csv files produced by run_leave_healthy_out.py and produces
paper-ready outputs:

  CSV:
    <out>/auc_table.csv            one row per (lower_fraction, kappa):
                                   patch_auc, patient_auc, test_normal_mean/max,
                                   pas_mean/min, reb004_score, clean_separation,
                                   clean_separation_excl_appendicitis (reb010 removed),
                                   previa_below_pas (reb004 below every PAS)
    <out>/per_patient_scores.csv   one row per (lower_fraction, kappa, patient):
                                   cohort, is_anom, mean_score, n_patches
    <out>/per_patient_ranked.csv   Rebro held-out patients ranked by mean score per
                                   (lower_fraction, kappa) with clinical_note — the
                                   clinical read (which patients sit above which)
  JSON:
    <out>/loo_report.json          everything above, structured, for reproducibility
  FIGURES (PNG):
    <out>/roc_patient_by_lf.png    patient-level ROC, one curve per lower_fraction (kappa=1.0)
    <out>/auc_vs_lower_fraction.png  patch + patient AUC vs lower_fraction
    <out>/auc_vs_kappa.png         patch + patient AUC vs kappa (at the best lower_fraction)
    <out>/per_patient_strip.png    per-patient mean score by group, reb004 highlighted
                                   (the key clinical figure)

This script does NOT compute anything on the fly elsewhere — it is the single
source of truth for the experiment's numbers and figures. Upgrade here if new
metrics/plots are needed.

Usage:
  conda run -n monai_placenta python f-AnoGAN-pytorch/analyze_loo_results.py \\
      --runs-dir runs_fanogan_loo --tag phase1_150ep
  conda run -n monai_placenta python f-AnoGAN-pytorch/analyze_loo_results.py \\
      --runs-dir runs_fanogan_loo --tag phase2_300ep
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
    def tqdm(x, **k):  # no-op fallback
        return x

KEY_NEGATIVE = "reb004"   # placenta previa WITHOUT accreta — must score LOW
REBRO_PAS = {"reb001", "reb007", "reb008"}
CONFOUNDER = "reb010"     # acute appendicitis in pregnancy — NOT a clean normal control;
                          # the recurring false-positive that caps held-out AUC.

# Clinical notes for the held-out Rebro test patients (for the ranked table). Keys are pids; anything not listed is left blank.
REBRO_CLINICAL = {
    "reb001": "PAS (percreta)",
    "reb007": "PAS (acreta, mildest)",
    "reb008": "PAS (percreta + previa)",
    "reb004": "previa, NO accreta",
    "reb005": "healthy",
    "reb010": "acute appendicitis (confounder)",
    "reb011": "healthy",
}


def patient_of(patch_path: str) -> str:
    stem = Path(patch_path).stem
    parts = stem.split("_")
    return parts[2] if parts[0] == "mendeley" else parts[0]


def cohort_of(patch_path: str) -> str:
    s = Path(patch_path).stem
    if s.startswith("mendeley_btfe"):
        return "mendeley_btfe"
    if s.startswith("mendeley_tse"):
        return "mendeley_tse"
    return "rebro"


def group_of(pid: str, is_anom: int, cohort: str) -> str:
    """Coarse group label for the strip plot."""
    if is_anom == 0:
        return "held-out healthy"
    if cohort == "rebro":
        return "Rebro PAS"
    if cohort == "mendeley_btfe":
        return "Mendeley BTFE PAS"
    return "Mendeley SSH_TSE PAS"


def load_scores(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["pid"] = df.patch_path.apply(patient_of)
    df["cohort"] = df.patch_path.apply(cohort_of)
    return df


def discover_runs(runs_dir: Path, tag: str):
    """Yield (lower_fraction, kappa, scores_csv) for every run found."""
    out = []
    for run_dir in sorted(runs_dir.glob(f"{tag}_lf*")):
        m = re.search(r"_lf(\d+)$", run_dir.name)
        if not m:
            continue
        # lf05 -> 0.5, lf00 -> 0.0, lf03 -> 0.3, lf07 -> 0.7
        digits = m.group(1)
        lf = float(f"{digits[0]}.{digits[1:]}") if len(digits) >= 2 else float(digits)
        for sc in sorted(run_dir.glob("scores_kappa*/scores.csv")):
            km = re.search(r"scores_kappa([0-9.]+)", sc.parent.name)
            kappa = float(km.group(1)) if km else float("nan")
            out.append((lf, kappa, sc))
    return out


def per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient summary of the per-patch scores under several aggregations.
    f-AnoGAN itself never aggregates (it decides per patch); the patient-level
    aggregation is OUR extension, so we expose mean / max / 90th-percentile /
    median and compare them. `mean_score` stays the primary column used by the
    figures and the ranked table."""
    g = df.groupby(["pid", "cohort", "is_anom"]).score
    pp = g.agg(mean_score="mean", max_score="max", median_score="median",
               p90_score=lambda s: s.quantile(0.90), n_patches="count").reset_index()
    pp["group"] = pp.apply(lambda r: group_of(r.pid, r.is_anom, r.cohort), axis=1)
    return pp


AGG_COLS = {"mean": "mean_score", "max": "max_score",
            "p90": "p90_score", "median": "median_score"}


def auc_safe(y_true, y_score):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def summarize(df: pd.DataFrame, lf: float, kappa: float) -> dict:
    pp = per_patient(df)
    neg = pp[pp.is_anom == 0]
    pos = pp[pp.is_anom == 1]
    key = pp[pp.pid == KEY_NEGATIVE]
    # held-out healthy with the appendicitis confounder (reb010) removed
    neg_clean = neg[neg.pid != CONFOUNDER]
    # Rebro-only PAS (same scanner/cohort as the held-out healthy) — the
    # controlled comparison, free of the easier/harder Mendeley distribution.
    pos_rebro = pos[pos.cohort == "rebro"]
    return {
        "lower_fraction": lf,
        "kappa": kappa,
        "n_patches": int(len(df)),
        "patch_auc": auc_safe(df.is_anom, df.score),
        # patient-level AUC under each aggregation of the per-patch scores
        # (mean dilutes focal anomalies; max / p90 are more focal-sensitive)
        "patient_auc": auc_safe(pp.is_anom, pp.mean_score),       # = mean (primary)
        "patient_auc_mean": auc_safe(pp.is_anom, pp.mean_score),
        "patient_auc_max": auc_safe(pp.is_anom, pp.max_score),
        "patient_auc_p90": auc_safe(pp.is_anom, pp.p90_score),
        "patient_auc_median": auc_safe(pp.is_anom, pp.median_score),
        "test_normal_mean": float(neg.mean_score.mean()) if len(neg) else float("nan"),
        "test_normal_max": float(neg.mean_score.max()) if len(neg) else float("nan"),
        "pas_mean": float(pos.mean_score.mean()) if len(pos) else float("nan"),
        "pas_min": float(pos.mean_score.min()) if len(pos) else float("nan"),
        "reb004_score": float(key.mean_score.iloc[0]) if len(key) else float("nan"),
        "clean_separation": bool(len(neg) and len(pos)
                                 and pos.mean_score.min() > neg.mean_score.max()),
        # does ANY PAS (incl. Mendeley) separate from healthy+previa once the
        # appendicitis case is excluded? (full test set — the honest global view)
        "clean_separation_excl_appendicitis": bool(
            len(neg_clean) and len(pos)
            and pos.mean_score.min() > neg_clean.mean_score.max()),
        # same-cohort controlled view: do the 3 Rebro PAS separate from the
        # Rebro healthy+previa once the appendicitis case is excluded?
        "clean_separation_rebro_excl_appendicitis": bool(
            len(neg_clean) and len(pos_rebro)
            and pos_rebro.mean_score.min() > neg_clean.mean_score.max()),
        # smallest Mendeley PAS mean (the false-negative tail that caps global AUC)
        "pas_min_mendeley": (float(pos[pos.cohort != "rebro"].mean_score.min())
                             if len(pos[pos.cohort != "rebro"]) else float("nan")),
        # is the previa-without-accreta control (reb004) below every PAS patient?
        "previa_below_pas": bool(
            len(key) and len(pos)
            and key.mean_score.iloc[0] < pos.mean_score.min()),
        # ... and below every Rebro PAS (same-cohort controlled view)?
        "previa_below_rebro_pas": bool(
            len(key) and len(pos_rebro)
            and key.mean_score.iloc[0] < pos_rebro.mean_score.min()),
    }


def _process_run(run):
    """Worker for one (lower_fraction, kappa) score file: read it, summarize,
    and build the per-patient frame. Module-level so joblib can pickle it.
    Returns (summary_dict, per_patient_df_with_lf_kappa_cols)."""
    lf, kappa, sc = run
    df = load_scores(sc)
    summary = summarize(df, lf, kappa)
    pp = per_patient(df)
    pp.insert(0, "kappa", kappa)
    pp.insert(0, "lower_fraction", lf)
    return summary, pp


def ranked_rebro_table(pp_df: pd.DataFrame) -> pd.DataFrame:
    """Per (lower_fraction, kappa), the Rebro held-out patients ranked by mean
    score, annotated with their clinical note. This is the clinical read of the
    experiment (which patients sit above which) and is paper facing,
    so it lives here rather than in an ad-hoc snippet."""
    reb = pp_df[pp_df.cohort == "rebro"].copy()
    reb["clinical_note"] = reb.pid.map(REBRO_CLINICAL).fillna("")
    reb = reb.sort_values(["lower_fraction", "kappa", "mean_score"])
    cols = ["lower_fraction", "kappa", "pid", "is_anom",
            "mean_score", "n_patches", "clinical_note"]
    return reb[cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_roc_by_lf(runs, out_path):
    """Patient-level ROC, one curve per lower_fraction at kappa=1.0."""
    try:
        from sklearn.metrics import roc_curve
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    for lf, kappa, sc in runs:
        if abs(kappa - 1.0) > 1e-9:
            continue
        df = load_scores(sc)
        pp = per_patient(df)
        fpr, tpr, _ = roc_curve(pp.is_anom, pp.mean_score)
        a = auc_safe(pp.is_anom, pp.mean_score)
        ax.plot(fpr, tpr, lw=2, label=f"lower_fraction={lf}  (AUC={a:.3f})")
    ax.set_xlabel("False positive rate (held-out healthy flagged)")
    ax.set_ylabel("True positive rate (PAS detected)")
    ax.set_title("Patient-level ROC by lower_fraction  (kappa=1.0)\n"
                 "leave-healthy-out: 4 held-out healthy vs all PAS")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_auc_vs_lf(summ_df, out_path):
    sub = summ_df[np.isclose(summ_df.kappa, 1.0)].sort_values("lower_fraction")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub.lower_fraction, sub.patient_auc, "o-", label="patient-level AUC", lw=2)
    ax.plot(sub.lower_fraction, sub.patch_auc, "s--", label="patch-level AUC", lw=2)
    ax.axhline(0.5, color="#999", ls=":", label="chance")
    ax.set_xlabel("lower_fraction (0.0=whole contour, 0.5=lower half, 0.7=bottom 30%)")
    ax.set_ylabel("AUC")
    ax.set_title("Leave-healthy-out AUC vs lower_fraction  (kappa=1.0)")
    ax.set_ylim(0.4, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_auc_vs_kappa(summ_df, out_path):
    # pick best lower_fraction by patient_auc at kappa=1.0
    at1 = summ_df[np.isclose(summ_df.kappa, 1.0)]
    if at1.empty:
        return
    best_lf = at1.loc[at1.patient_auc.idxmax(), "lower_fraction"]
    sub = summ_df[np.isclose(summ_df.lower_fraction, best_lf)].sort_values("kappa")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub.kappa, sub.patient_auc, "o-", label="patient-level AUC", lw=2)
    ax.plot(sub.kappa, sub.patch_auc, "s--", label="patch-level AUC", lw=2)
    ax.axhline(0.5, color="#999", ls=":", label="chance")
    ax.set_xlabel("kappa (weight of discriminator-feature residual A_D)")
    ax.set_ylabel("AUC")
    ax.set_title(f"AUC vs kappa  (best lower_fraction={best_lf})")
    ax.set_ylim(0.4, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_per_patient_strip(runs, out_path):
    """Per-patient mean score by group, at the best lower_fraction (kappa=1.0).
    reb004 (previa, no accreta) highlighted."""
    # choose lower_fraction with highest patient AUC at kappa=1.0
    best = None
    best_auc = -1
    for lf, kappa, sc in runs:
        if abs(kappa - 1.0) > 1e-9:
            continue
        df = load_scores(sc)
        pp = per_patient(df)
        a = auc_safe(pp.is_anom, pp.mean_score)
        if a > best_auc:
            best_auc, best = a, (lf, sc, pp)
    if best is None:
        return
    lf, sc, pp = best

    groups = ["held-out healthy", "Rebro PAS", "Mendeley BTFE PAS", "Mendeley SSH_TSE PAS"]
    colors = {"held-out healthy": "#3498DB", "Rebro PAS": "#27AE60",
              "Mendeley BTFE PAS": "#E67E22", "Mendeley SSH_TSE PAS": "#C0392B"}
    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        sub = pp[pp.group == g]
        if sub.empty:
            continue
        x = i + (rng.random(len(sub)) - 0.5) * 0.3
        ax.scatter(x, sub.mean_score, s=40, alpha=0.6, color=colors[g],
                   edgecolor="white", lw=0.4, label=f"{g} (n={len(sub)})")
        # group mean line
        ax.plot([i - 0.25, i + 0.25], [sub.mean_score.mean()] * 2,
                color="black", lw=2)
    # highlight reb004
    key = pp[pp.pid == KEY_NEGATIVE]
    if len(key):
        ax.scatter([0], [key.mean_score.iloc[0]], s=220, marker="*",
                   color="yellow", edgecolor="black", lw=1.2, zorder=5,
                   label=f"reb004 (previa, NO accreta) = {key.mean_score.iloc[0]:.3f}")
    # threshold = max held-out healthy
    neg_max = pp[pp.is_anom == 0].mean_score.max()
    ax.axhline(neg_max, color="#3498DB", ls="--", alpha=0.6,
               label=f"max held-out healthy = {neg_max:.3f}")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.set_ylabel("per-patient mean anomaly score")
    ax.set_title(f"Leave-healthy-out per-patient scores  "
                 f"(lower_fraction={lf}, kappa=1.0, patient AUC={best_auc:.3f})\n"
                 f"black bar = group mean; star = reb004 (the key previa-without-accreta control)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs_fanogan_loo", type=Path)
    p.add_argument("--tag", required=True,
                   help="Run tag, e.g. phase1_150ep or phase2_300ep.")
    p.add_argument("--out", default=None, type=Path,
                   help="Output dir (default: <runs-dir>/<tag>_analysis).")
    p.add_argument("--workers", type=int, default=20,
                   help="joblib workers for reading/summarizing the per-run "
                        "score files in parallel (default 20).")
    args = p.parse_args()

    runs = discover_runs(args.runs_dir, args.tag)
    if not runs:
        print(f"ERROR: no runs found for tag '{args.tag}' under {args.runs_dir}")
        return 1
    out = args.out or (args.runs_dir / f"{args.tag}_analysis")
    out.mkdir(parents=True, exist_ok=True)

    print(f"  found {len(runs)} (lower_fraction, kappa) score files for tag '{args.tag}'")

    # --- tables (one independent job per (lower_fraction, kappa) score file) ---
    if _JOBLIB and args.workers > 1 and len(runs) > 1:
        results = Parallel(n_jobs=args.workers)(
            delayed(_process_run)(r) for r in tqdm(runs, desc="runs"))
    else:
        results = [_process_run(r) for r in tqdm(runs, desc="runs")]
    summaries = [s for s, _ in results]
    pp_rows = [p for _, p in results]

    summ_df = pd.DataFrame(summaries).sort_values(["lower_fraction", "kappa"])
    pp_df = pd.concat(pp_rows, ignore_index=True)
    ranked = ranked_rebro_table(pp_df)
    summ_df.to_csv(out / "auc_table.csv", index=False)
    pp_df.to_csv(out / "per_patient_scores.csv", index=False)
    ranked.to_csv(out / "per_patient_ranked.csv", index=False)
    with open(out / "loo_report.json", "w") as f:
        json.dump({"summary": summaries,
                   "per_patient": pp_df.to_dict("records"),
                   "ranked_rebro": ranked.to_dict("records")}, f, indent=2, default=str)

    # --- figures ---
    fig_roc_by_lf(runs, out / "roc_patient_by_lf.png")
    fig_auc_vs_lf(summ_df, out / "auc_vs_lower_fraction.png")
    fig_auc_vs_kappa(summ_df, out / "auc_vs_kappa.png")
    fig_per_patient_strip(runs, out / "per_patient_strip.png")

    print("\n=== AUC table (kappa=1.0 rows) ===")
    print(summ_df[np.isclose(summ_df.kappa, 1.0)][
        ["lower_fraction", "patch_auc", "patient_auc", "test_normal_max",
         "pas_min", "pas_min_mendeley", "reb004_score",
         "clean_separation_rebro_excl_appendicitis",
         "previa_below_rebro_pas"]].round(3).to_string(index=False))

    print("\n=== Patient-level AUC by AGGREGATION of per-patch scores (kappa=1.0) ===")
    print("    (f-AnoGAN decides per-patch; mean/max/p90/median are OUR aggregations)")
    agg = summ_df[np.isclose(summ_df.kappa, 1.0)][
        ["lower_fraction", "patch_auc", "patient_auc_mean", "patient_auc_max",
         "patient_auc_p90", "patient_auc_median"]]
    print(agg.round(3).to_string(index=False))

    print("\n=== Rebro held-out patients ranked by mean score (kappa=1.0) ===")
    print(f"    (confounder = {CONFOUNDER}; clean_separation_excl_appendicitis "
          f"ignores it)")
    r1 = ranked[np.isclose(ranked.kappa, 1.0)]
    for lf in sorted(r1.lower_fraction.unique()):
        sub = r1[r1.lower_fraction == lf]
        print(f"\n  --- lower_fraction={lf} ---")
        print(sub[["pid", "is_anom", "mean_score", "clinical_note"]]
              .round(3).to_string(index=False))
    print(f"\nDONE. CSV/JSON/figures saved under {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
