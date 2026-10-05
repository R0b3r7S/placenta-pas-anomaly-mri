#!/usr/bin/env python3
"""
==============================================================================
 Placenta-MRI segmentation — Model Comparison Dashboard
 Auto-discovers all test_metrics.json files under ./runs/ and produces:
   1. Console comparison tables (per (network × train_modality), same/cross)
   2. CSV exports of every metric, every cell
   3. LaTeX tables for the paper (compact metric subset)
   4. Per-network bar charts (Dice, same-domain vs cross-domain)
   5. Cross-domain heatmap across all 16 trained models
   6. "Δ over baseline" plot per network (baseline = the 'regular' cell)

 Layout assumption (see run_augmentation_training.sh):
   runs/<TRAIN_MOD>_<NETWORK>_<CELL>/fold_0/test_metrics.json   (after training)
   runs/<TRAIN_MOD>_<NETWORK>_<CELL>/test_on_<TEST_MOD>/test_metrics.json
   runs/<TRAIN_MOD>_<NETWORK>_<CELL>/test_on_<TEST_MOD>/fold_0/test_metrics.json
==============================================================================
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

# scipy is required for paired Wilcoxon tests; it's already in monai_placenta.
try:
    from scipy.stats import wilcoxon
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# joblib is used to render plots in parallel processes (matplotlib isn't
# thread-safe, but separate processes are fine because each one re-imports
# matplotlib with the 'Agg' backend).
try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

# ============================================================================
# 1. BENCHMARK CONFIGURATION  (matches run_augmentation_*.sh)
# ============================================================================
RUNS_DIR = Path(__file__).parent / "runs"
OUTPUT_DIR = Path(__file__).parent / "comparison_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NETWORKS = ["unetplusplus", "dynunet"]
TRAIN_MODS = ["BTFE", "TSE"]
CELLS = ["none", "regular", "afa_mixup", "afa_cutmix_mixup"]
BASELINE_CELL = "regular"   # baseline for "Δ over baseline" plots

# Short, human-readable label per (network, cell)
NETWORK_SHORT = {"unetplusplus": "U++", "dynunet": "DynU"}
CELL_LABEL = {
    "none":             "No-Aug",
    "regular":          "Regular",
    "afa_mixup":        "AFA+MixUp",
    "afa_cutmix_mixup": "AFA+CutMix+MixUp",
}

# ----- All 16 model names, programmatically -----
def model_name(mod, net, cell):
    return f"{mod}_{net}_{cell}"

ALL_MODELS = [model_name(m, n, c) for m in TRAIN_MODS for n in NETWORKS for c in CELLS]

# Helper: build the per-(network, train_mod) group used everywhere
def models_for(net, mod):
    return [model_name(mod, net, c) for c in CELLS]

# Display name for a model (used in tables, plots, LaTeX rows)
def display_name(model):
    parts = model.split("_")
    if len(parts) < 3:
        return model
    mod = parts[0]
    # everything after the modality up to the last cell-token forms the network name
    # but our cells are simple slugs, so split as: mod, net, *cell_parts
    net = parts[1]
    cell = "_".join(parts[2:])
    return f"{mod} / {NETWORK_SHORT.get(net, net)} / {CELL_LABEL.get(cell, cell)}"

# Legacy names from the previous paper still kept for backward-compatible display
LEGACY_DISPLAY_NAMES = {
    "BTFE_unetplusplus_1Fold":        "BTFE / U++ / legacy-1Fold",
    "BTFE_unetplusplus_mixup":        "BTFE / U++ / legacy-MixUp",
    "BTFE_unetplusplus_cutmix":       "BTFE / U++ / legacy-CutMix",
    "BTFE_unetplusplus_afa":          "BTFE / U++ / legacy-AFA",
    "BTFE_unetplusplus_mixup_afa":    "BTFE / U++ / legacy-MixUp+AFA",
    "BTFE_unetplusplus_cutmix_afa":   "BTFE / U++ / legacy-CutMix+AFA",
    "TSE_unetplusplus_1Fold":         "TSE / U++ / legacy-1Fold",
    "TSE_unetplusplus_mixup":         "TSE / U++ / legacy-MixUp",
    "TSE_unetplusplus_cutmix":        "TSE / U++ / legacy-CutMix",
    "TSE_unetplusplus_afa":           "TSE / U++ / legacy-AFA",
    "TSE_unetplusplus_mixup_afa":     "TSE / U++ / legacy-MixUp+AFA",
    "TSE_unetplusplus_cutmix_afa":    "TSE / U++ / legacy-CutMix+AFA",
    "COMBINED_unetplusplus_1Fold":    "COMBINED / U++ / legacy",
    "COMBINED_unetplusplus_mixup_afa":  "COMBINED / U++ / legacy-MixUp+AFA",
    "COMBINED_unetplusplus_cutmix_afa": "COMBINED / U++ / legacy-CutMix+AFA",
    "Transfer_BTFE_to_TSE_1Fold":     "Transfer BTFE→TSE (legacy)",
    "Transfer_TSE_to_BTFE_1Fold":     "Transfer TSE→BTFE (legacy)",
}

def pretty_name(model):
    return LEGACY_DISPLAY_NAMES.get(model, display_name(model))


# ============================================================================
# 2. METRICS
# ============================================================================
# Console table — kept tight enough to fit on a wide terminal.
CONSOLE_METRICS = ["dice", "iou", "sens", "prec", "hd95", "msd", "nsd", "biou", "mcc"]
# CSV/LaTeX — all of them.
ALL_METRICS = ["dice", "iou", "sens", "prec", "spec", "hd95", "msd", "nsd", "vs", "mcc", "biou"]

METRICS_LABELS = {
    "dice": "Dice ↑",
    "iou":  "IoU ↑",
    "sens": "Sens ↑",
    "prec": "Prec ↑",
    "spec": "Spec ↑",
    "hd95": "HD95 (px) ↓",
    "msd":  "MSD (px) ↓",
    "nsd":  "NSD ↑",
    "vs":   "VolSim ↑",
    "mcc":  "MCC ↑",
    "biou": "B-IoU ↑",
}
HIGHER_IS_BETTER = {"dice", "iou", "sens", "prec", "spec", "nsd", "vs", "mcc", "biou"}
LOWER_IS_BETTER = {"hd95", "msd"}

# Sentinel for missing metrics — distinguishes "not computed" from 0.0
def _format_val(val, metric, width=10):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return f"{'—':>{width}}"
    if metric in ("hd95", "msd"):
        return f"{val:>{width}.2f}"
    return f"{val:>{width}.4f}"


# ============================================================================
# 3. DATA LOADING
# ============================================================================
def discover_metrics(runs_dir):
    """Find every test_metrics.json + per_patient_metrics.json under runs/.

    Returns:
        results      — aggregated metrics, indexed [model][test_domain][metric]
        per_patient  — per-patient values, indexed [model][test_domain][pid][metric]
                       (empty dict if per_patient_metrics.json is missing for a cell)
    """
    results = {}
    per_patient = {}

    if not runs_dir.exists():
        return results, per_patient

    for metrics_file in sorted(runs_dir.rglob("test_metrics.json")):
        parts = metrics_file.relative_to(runs_dir).parts
        # Expected: MODEL_NAME / test_on_DOMAIN / [fold_X/] test_metrics.json
        model_name = parts[0]
        test_domain = None
        for p in parts:
            if p.startswith("test_on_"):
                test_domain = p.replace("test_on_", "")
                break
        if test_domain is None:
            continue

        with open(metrics_file) as f:
            results.setdefault(model_name, {})[test_domain] = json.load(f)

        # Sibling per_patient_metrics.json — new in v8 (saved by test mode).
        # Old runs that predate this dump simply won't have one.
        pp_file = metrics_file.with_name("per_patient_metrics.json")
        if pp_file.exists():
            with open(pp_file) as f:
                per_patient.setdefault(model_name, {})[test_domain] = json.load(f)

    return results, per_patient


def metric_value(results, model, test_domain, metric):
    """Return metric or None when absent."""
    try:
        v = results[model][test_domain].get(metric, None)
    except (KeyError, AttributeError):
        return None
    if v is None:
        return None
    try:
        v = float(v)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


# ============================================================================
# 3b. SIGNIFICANCE (paired Wilcoxon vs the 'regular' baseline)
# ============================================================================
def per_patient_metric_vector(per_patient, model, test_domain, metric):
    """Return a SORTED-by-patient-id list of (pid, value) tuples for one cell.

    Sorting by patient ID guarantees the same patient ordering across cells,
    which is required for a *paired* Wilcoxon test.
    """
    try:
        pdict = per_patient[model][test_domain]
    except KeyError:
        return []
    rows = []
    for pid in sorted(pdict.keys()):
        v = pdict[pid].get(metric, None)
        if v is None:
            continue
        try:
            v = float(v)
            if not (np.isnan(v) or np.isinf(v)):
                rows.append((pid, v))
        except (TypeError, ValueError):
            continue
    return rows


def paired_wilcoxon(per_patient, baseline_model, test_model, test_domain, metric):
    """Paired Wilcoxon signed-rank test, baseline vs test, on the patients
    they have in common. Returns (statistic, p_value, n_pairs) or (None, None, 0)
    when scipy is missing / too little data / all-zero differences."""
    if not _SCIPY_AVAILABLE:
        return None, None, 0
    base = dict(per_patient_metric_vector(per_patient, baseline_model, test_domain, metric))
    test = dict(per_patient_metric_vector(per_patient, test_model, test_domain, metric))
    common = sorted(set(base.keys()) & set(test.keys()))
    if len(common) < 3:
        return None, None, len(common)
    a = np.array([base[p] for p in common])
    b = np.array([test[p] for p in common])
    # If every patient is identical, wilcoxon raises "all differences are zero".
    if np.allclose(a, b):
        return 0.0, 1.0, len(common)
    try:
        res = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        return None, None, len(common)
    return float(res.statistic), float(res.pvalue), len(common)


def stars_for_p(p):
    """LaTeX/console convention: * p<0.05, ** p<0.01, *** p<0.001."""
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ============================================================================
# 4. CONSOLE TABLES
# ============================================================================
def _best_value(results, available, test_domain, metric):
    vals = [metric_value(results, m, test_domain, metric) for m in available]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return max(vals) if metric in HIGHER_IS_BETTER else min(vals)


def print_table(results, test_domain, model_list, title):
    available = [m for m in model_list if m in results and test_domain in results[m]]
    if not available:
        print(f"  (no results for: {title})")
        return

    width = 30 + len(CONSOLE_METRICS) * 12
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")

    # Header
    header = f"{'Model':<30}"
    for m in CONSOLE_METRICS:
        header += f" {METRICS_LABELS[m]:>11}"
    print(header)
    print("-" * width)

    # Best per metric for * marker
    best_vals = {m: _best_value(results, available, test_domain, m) for m in CONSOLE_METRICS}

    for model in available:
        name = pretty_name(model)[:29]
        row = f"{name:<30}"
        for metric in CONSOLE_METRICS:
            val = metric_value(results, model, test_domain, metric)
            is_best = (val is not None
                       and best_vals[metric] is not None
                       and abs(val - best_vals[metric]) < 1e-6)
            cell = _format_val(val, metric, width=9)
            row += f" {cell}{' *' if is_best else '  '}"
        print(row)

    print("-" * width)
    print("  * = best in column   '—' = metric not in test_metrics.json (older run)")


def print_cross_domain_gap(results, model_list, train_domain, title):
    other_domain = "TSE" if train_domain == "BTFE" else "BTFE"

    available = [m for m in model_list
                 if m in results
                 and train_domain in results[m]
                 and other_domain in results[m]]
    if not available:
        return

    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    print(f"{'Model':<30} {'Same Dice':>12} {'Cross Dice':>12} {'Δ Gap':>10} {'Δ vs Baseline':>15}")
    print("-" * 100)

    # Baseline = the 'regular' cell of THIS train_domain, picking the same network as each row
    for model in available:
        same = metric_value(results, model, train_domain, "dice")
        cross = metric_value(results, model, other_domain, "dice")
        if same is None or cross is None:
            continue
        gap = same - cross

        # Network of this row determines the baseline lookup
        net = model.split("_")[1] if len(model.split("_")) >= 2 else NETWORKS[0]
        baseline_key = model_name(train_domain, net, BASELINE_CELL)
        baseline_cross = metric_value(results, baseline_key, other_domain, "dice")
        if baseline_cross is None:
            delta_str = f"{'—':>13}"
        else:
            delta = cross - baseline_cross
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:>13.4f}"

        name = pretty_name(model)[:29]
        print(f"{name:<30} {same:>12.4f} {cross:>12.4f} {gap:>10.4f} {delta_str}")

    print("-" * 100)


# ============================================================================
# 5. CSV EXPORT
# ============================================================================
def discover_test_domains(results) -> list[str]:
    """Return all unique test domains found across discovered models.

    Includes the two training modalities (BTFE, TSE) as same/cross-domain
    test sets AND any external cohorts (e.g. EXTERNAL_REBRO). Order is
    deterministic: training modalities first (TRAIN_MODS order), then
    externals sorted alphabetically.
    """
    seen = set()
    for m in results.values():
        seen.update(m.keys())
    train = [d for d in TRAIN_MODS if d in seen]
    external = sorted([d for d in seen if d not in TRAIN_MODS])
    return train + external


def export_csv(results, output_dir):
    for test_domain in discover_test_domains(results):
        csv_path = output_dir / f"comparison_test_on_{test_domain}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Model"] + [METRICS_LABELS[m] for m in ALL_METRICS])

            for model in sorted(results.keys()):
                if test_domain not in results[model]:
                    continue
                row = [pretty_name(model)]
                for metric in ALL_METRICS:
                    val = metric_value(results, model, test_domain, metric)
                    row.append(f"{val:.6f}" if val is not None else "")
                w.writerow(row)
        print(f"  Saved: {csv_path}")


# ============================================================================
# 6. LATEX EXPORT
# ============================================================================
PAPER_METRICS = ["dice", "iou", "sens", "prec", "hd95", "msd", "nsd", "biou", "mcc"]

def _latex_escape(s):
    return s.replace("→", "$\\rightarrow$").replace("_", "\\_")

def export_latex(results, per_patient, output_dir):
    """Per-test-domain LaTeX table.

    Significance markers (*) come from a paired Wilcoxon vs the same-network,
    same-train-modality 'regular' baseline; baseline row is unmarked. n.s. is
    blank. Stars only appear when per_patient_metrics.json is present for
    both baseline and the row's model.
    """
    for test_domain in discover_test_domains(results):
        latex_path = output_dir / f"latex_table_test_on_{test_domain}.tex"
        available = sorted([m for m in results if test_domain in results[m]])
        best_vals = {m: _best_value(results, available, test_domain, m) for m in PAPER_METRICS}

        with open(latex_path, "w") as f:
            col_spec = "l" + "c" * len(PAPER_METRICS)
            f.write(f"% Auto-generated LaTeX table — Test on {test_domain}\n")
            f.write("\\begin{table}[htbp]\n\\centering\n")
            f.write(f"\\caption{{Segmentation performance on {test_domain} test set "
                    f"(patient-averaged). Significance vs the same-network "
                    f"'{BASELINE_CELL}' baseline by paired Wilcoxon "
                    f"signed-rank test on per-patient values: "
                    f"$^{{*}} p<0.05,\\ ^{{**}} p<0.01,\\ ^{{***}} p<0.001$.}}\n")
            f.write(f"\\label{{tab:test_{test_domain.lower()}}}\n")
            f.write("\\resizebox{\\textwidth}{!}{\n")
            f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")

            header_parts = ["Model"]
            for m in PAPER_METRICS:
                arrow = "$\\uparrow$" if m in HIGHER_IS_BETTER else "$\\downarrow$"
                short = {
                    "dice": "Dice", "iou": "IoU",
                    "sens": "Sens", "prec": "Prec",
                    "hd95": "HD95", "msd": "MSD",
                    "nsd": "NSD", "biou": "B-IoU", "mcc": "MCC",
                }[m]
                header_parts.append(f"{short} {arrow}")
            f.write(" & ".join(header_parts) + " \\\\\n\\midrule\n")

            for model in available:
                name = _latex_escape(pretty_name(model))
                # Identify this row's baseline (same network, same train mod, 'regular' cell).
                row_parts = [name]
                parts = model.split("_")
                row_baseline_key = None
                if len(parts) >= 2:
                    row_baseline_key = model_name(parts[0], parts[1], BASELINE_CELL)
                is_baseline = (model == row_baseline_key)

                for metric in PAPER_METRICS:
                    val = metric_value(results, model, test_domain, metric)
                    if val is None:
                        row_parts.append("---")
                        continue
                    formatted = (f"{val:.2f}" if metric in ("hd95", "msd")
                                 else f"{val:.4f}")
                    is_best = (best_vals[metric] is not None
                               and abs(val - best_vals[metric]) < 1e-6)
                    if is_best:
                        formatted = f"\\textbf{{{formatted}}}"
                    # Decorate non-baseline rows with significance star(s)
                    if not is_baseline and row_baseline_key is not None:
                        _, p, n = paired_wilcoxon(per_patient, row_baseline_key, model,
                                                  test_domain, metric)
                        s = stars_for_p(p)
                        if s:
                            formatted += f"$^{{{s}}}$"
                    row_parts.append(formatted)
                f.write(" & ".join(row_parts) + " \\\\\n")

            f.write("\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n")
        print(f"  Saved: {latex_path}")


def export_significance_csv(per_patient, output_dir):
    """Per-test-domain CSV with the raw Wilcoxon p-value table."""
    if not _SCIPY_AVAILABLE:
        print("  scipy not installed — skipping significance CSV")
        return
    if not per_patient:
        print("  no per_patient_metrics.json files found — skipping significance CSV")
        return

    metrics_for_sig = ["dice", "iou", "hd95", "msd", "nsd", "biou", "mcc"]
    # discover_test_domains lives below the export_csv definition;
    # results' keys here are model→{domain:metrics}, so we derive locally.
    seen_domains = set()
    for m in per_patient.values():
        seen_domains.update(m.keys())
    domain_order = [d for d in TRAIN_MODS if d in seen_domains] + sorted(
        d for d in seen_domains if d not in TRAIN_MODS
    )
    for test_domain in domain_order:
        rows = []
        for net in NETWORKS:
            for train_mod in TRAIN_MODS:
                baseline_key = model_name(train_mod, net, BASELINE_CELL)
                for cell in CELLS:
                    if cell == BASELINE_CELL:
                        continue
                    test_key = model_name(train_mod, net, cell)
                    if test_key not in per_patient or baseline_key not in per_patient:
                        continue
                    row = {
                        "network": net,
                        "train_modality": train_mod,
                        "cell": cell,
                        "test_domain": test_domain,
                        "vs_baseline": BASELINE_CELL,
                    }
                    for m in metrics_for_sig:
                        _, p, n = paired_wilcoxon(per_patient, baseline_key, test_key,
                                                  test_domain, m)
                        row[f"{m}_p"] = f"{p:.4g}" if p is not None else ""
                        row[f"{m}_n"] = n
                    rows.append(row)

        if not rows:
            continue

        csv_path = output_dir / f"wilcoxon_test_on_{test_domain}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved: {csv_path}")


# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
CELL_COLORS = {
    "none":             "#7F8C8D",   # gray
    "regular":          "#4A90D9",   # blue (baseline)
    "afa_mixup":        "#9B59B6",   # purple
    "afa_cutmix_mixup": "#E67E22",   # orange — the 3-way combo
}


def _bar_chart_one(net, train_domain, results, output_dir):
    """Render ONE bar chart (same-domain vs cross-domain) for a (network, train_mod) cell.

    Designed to be called from a joblib worker process so all 4 bar charts run
    in parallel. Each process re-imports matplotlib under the 'Agg' backend so
    no GUI state is shared. Returns the saved path or None.
    """
    other_domain = "TSE" if train_domain == "BTFE" else "BTFE"
    group = models_for(net, train_domain)
    available = [m for m in group if m in results]
    if not available:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{train_domain} / {NETWORK_SHORT[net]}  — Dice comparison",
                 fontsize=15, fontweight='bold')

    for ax_idx, (test_domain, ax_title) in enumerate([
        (train_domain, f"Same-Domain ({train_domain} → {train_domain})"),
        (other_domain, f"Cross-Domain ({train_domain} → {other_domain})"),
    ]):
        ax = axes[ax_idx]
        labels = [CELL_LABEL[m.split("_", 2)[2]] for m in available]
        vals = [metric_value(results, m, test_domain, "dice") for m in available]
        colors = [CELL_COLORS[m.split("_", 2)[2]] for m in available]

        plotted_vals = [v if v is not None else 0 for v in vals]
        bars = ax.bar(labels, plotted_vals, color=colors,
                      edgecolor='white', linewidth=1.5, width=0.6)
        for bar, val in zip(bars, vals):
            text = "—" if val is None else f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    text, ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_title(ax_title, fontsize=12, fontweight='bold')
        ax.set_ylabel("Dice", fontsize=11)
        clean = [v for v in vals if v is not None]
        if clean:
            lo = max(0.0, min(clean) - 0.05)
            hi = min(1.0, max(clean) + 0.05)
            ax.set_ylim(lo, hi)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        baseline_key = model_name(train_domain, net, BASELINE_CELL)
        base_val = metric_value(results, baseline_key, test_domain, "dice")
        if base_val is not None:
            ax.axhline(y=base_val, color='gray', linestyle='--', alpha=0.6,
                       label=f'Regular baseline ({base_val:.3f})')
            ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    save = output_dir / f"bar_chart_{train_domain}_{net}.png"
    plt.savefig(save, dpi=200, bbox_inches='tight')
    plt.close()
    return str(save)


def plot_bar_charts(results, output_dir):
    """Dispatcher kept for back-compat: produces every (net, train_mod) bar chart."""
    saved = []
    for net in NETWORKS:
        for train_domain in TRAIN_MODS:
            s = _bar_chart_one(net, train_domain, results, output_dir)
            if s:
                saved.append(s)
    for s in saved:
        print(f"  Saved: {s}")


def _boxplot_one(net, train_domain, per_patient, metric, output_dir):
    """Render ONE boxplot for a (network, train_mod, metric) cell.

    Returns the saved path (str) or None when there's no data to plot.
    """
    if not per_patient:
        return None
    other_domain = "TSE" if train_domain == "BTFE" else "BTFE"
    metric_label = METRICS_LABELS.get(metric, metric)
    baseline_key = model_name(train_domain, net, BASELINE_CELL)

    any_data = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{train_domain} / {NETWORK_SHORT[net]} — per-patient {metric_label} distribution",
                 fontsize=14, fontweight='bold')

    for ax_idx, (test_domain, ax_title) in enumerate([
        (train_domain, f"Same-Domain ({train_domain} → {train_domain})"),
        (other_domain, f"Cross-Domain ({train_domain} → {other_domain})"),
    ]):
        ax = axes[ax_idx]
        box_data, labels, colors, p_stars = [], [], [], []

        for c in CELLS:
            m = model_name(train_domain, net, c)
            vals = [v for _, v in
                    per_patient_metric_vector(per_patient, m, test_domain, metric)]
            if not vals:
                continue
            box_data.append(vals)
            labels.append(CELL_LABEL[c])
            colors.append(CELL_COLORS[c])
            if c == BASELINE_CELL:
                p_stars.append("")
            else:
                _, p, _ = paired_wilcoxon(per_patient, baseline_key, m,
                                          test_domain, metric)
                p_stars.append(stars_for_p(p))

        if not box_data:
            ax.set_title(f"{ax_title}: no data", fontsize=11)
            ax.axis("off")
            continue
        any_data = True

        bp = ax.boxplot(box_data, patch_artist=True, widths=0.55,
                        medianprops=dict(color='black', linewidth=2),
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                       markeredgecolor='black', markersize=6))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        for i, vals in enumerate(box_data, start=1):
            jitter = np.random.default_rng(42).normal(0, 0.04, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=14, color='black', alpha=0.5, zorder=3)

        ax.set_xticklabels(labels, fontsize=10)
        ax.set_title(ax_title, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ymax = max(max(v) for v in box_data)
        ymin = min(min(v) for v in box_data)
        pad = 0.04 * (ymax - ymin + 1e-6)
        for i, s in enumerate(p_stars, start=1):
            if s:
                ax.text(i, ymax + pad, s, ha='center', va='bottom',
                        fontsize=14, fontweight='bold', color='black')

        base_vals = [v for _, v in
                     per_patient_metric_vector(per_patient, baseline_key,
                                               test_domain, metric)]
        if base_vals:
            ax.axhline(y=np.median(base_vals), color='gray', linestyle='--',
                       alpha=0.5, linewidth=1,
                       label=f'{BASELINE_CELL} median ({np.median(base_vals):.3f})')
            ax.legend(loc='lower right', fontsize=9)

    if not any_data:
        plt.close()
        return None

    plt.tight_layout()
    save = output_dir / f"boxplot_{train_domain}_{net}_{metric}.png"
    plt.savefig(save, dpi=200, bbox_inches='tight')
    plt.close()
    return str(save)


def plot_boxplots(per_patient, output_dir, metric="dice"):
    """Back-compat dispatcher: produces all (net × train_mod) boxplots for one metric."""
    if not per_patient:
        print(f"  No per_patient_metrics.json available → skipping boxplots for '{metric}'.")
        return
    saved = []
    for net in NETWORKS:
        for train_domain in TRAIN_MODS:
            s = _boxplot_one(net, train_domain, per_patient, metric, output_dir)
            if s:
                saved.append(s)
    for s in saved:
        print(f"  Saved: {s}")


def plot_cross_domain_heatmap(results, output_dir):
    """16 rows × 2 cols (Test on BTFE / Test on TSE) heatmap of Dice."""
    available = [m for m in ALL_MODELS if m in results]
    if not available:
        return

    n = len(available)
    matrix = np.full((n, 2), np.nan)
    for i, model in enumerate(available):
        for j, td in enumerate(TRAIN_MODS):
            v = metric_value(results, model, td, "dice")
            matrix[i, j] = v if v is not None else np.nan

    height = max(6, 0.45 * n)
    fig, ax = plt.subplots(figsize=(8, height))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.50, vmax=0.95)

    ax.set_xticks(range(2))
    ax.set_xticklabels([f"Test on {d}" for d in TRAIN_MODS], fontsize=12, fontweight='bold')
    ax.set_yticks(range(n))
    ax.set_yticklabels([pretty_name(m) for m in available], fontsize=9)

    for i in range(n):
        for j in range(2):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha='center', va='center', fontsize=10, color='black')
            else:
                text_color = 'white' if v < 0.75 else 'black'
                ax.text(j, i, f"{v:.3f}", ha='center', va='center',
                        fontsize=9, fontweight='bold', color=text_color)

    ax.set_title("Cross-Domain Generalization (Dice)", fontsize=14, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Dice")
    plt.tight_layout()
    save = output_dir / "heatmap_cross_domain.png"
    plt.savefig(save, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save}")


def _improvement_one(net, results, output_dir):
    """Render ONE 'improvement over baseline' figure (two panels) for a given network."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{NETWORK_SHORT[net]} — Cross-domain Dice Δ over '{BASELINE_CELL}' baseline",
                 fontsize=14, fontweight='bold')

    any_data = False
    for ax_idx, train_domain in enumerate(TRAIN_MODS):
        ax = axes[ax_idx]
        other_domain = "TSE" if train_domain == "BTFE" else "BTFE"

        baseline_key = model_name(train_domain, net, BASELINE_CELL)
        base_cross = metric_value(results, baseline_key, other_domain, "dice")
        if base_cross is None:
            ax.set_title(f"{train_domain} → {other_domain}: no baseline yet", fontsize=11)
            ax.axis("off")
            continue

        cells_to_compare = [c for c in CELLS if c != BASELINE_CELL]
        labels, improvements, colors = [], [], []
        for c in cells_to_compare:
            m = model_name(train_domain, net, c)
            v = metric_value(results, m, other_domain, "dice")
            if v is None:
                continue
            labels.append(CELL_LABEL[c])
            improvements.append(v - base_cross)
            colors.append(CELL_COLORS[c])

        if not labels:
            ax.set_title(f"{train_domain} → {other_domain}: no comparable cells", fontsize=11)
            ax.axis("off")
            continue
        any_data = True

        bars = ax.bar(labels, [v * 100 for v in improvements], color=colors,
                      edgecolor='white', width=0.6)
        for bar, val in zip(bars, improvements):
            y = bar.get_height() + 0.2 if val >= 0 else bar.get_height() - 0.8
            ax.text(bar.get_x() + bar.get_width()/2, y, f"{val*100:+.2f}%",
                    ha='center', va='bottom' if val >= 0 else 'top',
                    fontweight='bold', fontsize=10)

        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_title(f"{train_domain} → {other_domain}  (baseline cross = {base_cross:.4f})",
                     fontsize=11, fontweight='bold')
        ax.set_ylabel("Δ Dice (% points)", fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if not any_data:
        plt.close()
        return None

    plt.tight_layout()
    save = output_dir / f"improvement_over_baseline_{net}.png"
    plt.savefig(save, dpi=200, bbox_inches='tight')
    plt.close()
    return str(save)


def plot_improvement_over_baseline(results, output_dir):
    """Back-compat dispatcher: produces all per-network improvement plots."""
    saved = []
    for net in NETWORKS:
        s = _improvement_one(net, results, output_dir)
        if s:
            saved.append(s)
    for s in saved:
        print(f"  Saved: {s}")


# ============================================================================
# 7b. Parallel plot dispatcher (matplotlib-safe via separate processes)
# ============================================================================
def render_all_plots(results, per_patient, output_dir, n_jobs=None):
    """Build the flat list of every plot task and render them in parallel.

    matplotlib is not thread-safe, so we use processes via joblib (loky backend).
    Each plot function does its own savefig + close inside the worker.
    """
    boxplot_metrics = ("dice", "iou", "nsd", "biou")

    tasks = []
    # 1. Bar charts: one per (network, train_mod) = 4
    for net in NETWORKS:
        for td in TRAIN_MODS:
            tasks.append(("bar", (_bar_chart_one, (net, td, results, output_dir))))
    # 2. Boxplots: one per (network, train_mod, metric) = up to 16
    if per_patient:
        for net in NETWORKS:
            for td in TRAIN_MODS:
                for m in boxplot_metrics:
                    tasks.append(("box", (_boxplot_one, (net, td, per_patient, m, output_dir))))
    # 3. Improvement plots: one per network = 2
    for net in NETWORKS:
        tasks.append(("imp", (_improvement_one, (net, results, output_dir))))
    # 4. Heatmap: 1 figure (cheap, just run inline)
    plot_cross_domain_heatmap(results, output_dir)

    if not tasks:
        return

    if _JOBLIB_AVAILABLE:
        n_workers = n_jobs or min(8, (os.cpu_count() or 4))
        print(f"  Rendering {len(tasks)} plots in parallel ({n_workers} workers)...")
        results_list = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(fn)(*args) for _, (fn, args) in tqdm(tasks, desc="Plots")
        )
    else:
        print(f"  Rendering {len(tasks)} plots sequentially (no joblib)...")
        results_list = []
        for _, (fn, args) in tqdm(tasks, desc="Plots"):
            results_list.append(fn(*args))

    for path in results_list:
        if path:
            print(f"  {path}")


# ============================================================================
# 8. ANALYSIS: what experiments are missing in the grid?
# ============================================================================
def print_recommendations(results):
    print(f"\n{'='*80}")
    print("  GRID COVERAGE")
    print(f"{'='*80}")

    missing = []
    for net in NETWORKS:
        for mod in TRAIN_MODS:
            for cell in CELLS:
                m = model_name(mod, net, cell)
                if m not in results:
                    missing.append(m)
                    continue
                domains = set(results[m].keys())
                expected = {"BTFE", "TSE"}
                if not expected.issubset(domains):
                    print(f"  WARNING: {pretty_name(m)} — missing test_on_{expected - domains}")

    if missing:
        print(f"\n  WARNING: {len(missing)} model(s) have no test_metrics.json yet:")
        for m in missing:
            print(f"     - {pretty_name(m)}")
    else:
        n_domains = len(discover_test_domains(results))
        print(f"  All 16 cells × {n_domains} test set(s) are covered.")
        externals = [d for d in discover_test_domains(results) if d not in TRAIN_MODS]
        if externals:
            print(f"     External test sets present: {externals}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("  PLACENTA SEGMENTATION — MODEL COMPARISON DASHBOARD")
    print("=" * 80)

    results, per_patient = discover_metrics(RUNS_DIR)
    print(f"\n  Discovered {len(results)} models with test metrics under {RUNS_DIR}:")
    for model in sorted(results.keys()):
        domains = ", ".join(sorted(results[model].keys()))
        has_pp = "yes" if model in per_patient else "no"
        print(f"    - {pretty_name(model):<45} tested on: {domains:<14}  [per-patient: {has_pp}]")
    if not _SCIPY_AVAILABLE:
        print("\n  WARNING: scipy not installed → Wilcoxon significance + boxplots disabled. "
              "`pip install scipy` to enable.")

    # --- Console Tables: per-network, both same- and cross-domain ---
    for net in NETWORKS:
        for train_domain in TRAIN_MODS:
            other = "TSE" if train_domain == "BTFE" else "BTFE"
            group = models_for(net, train_domain)
            print_table(results, train_domain, group,
                        f"{NETWORK_SHORT[net]}  {train_domain}-TRAINED → Tested on {train_domain} (Same-Domain)")
            print_table(results, other, group,
                        f"{NETWORK_SHORT[net]}  {train_domain}-TRAINED → Tested on {other} (Cross-Domain)")

    # --- Console Tables for any external test domains (e.g. EXTERNAL_REBRO) ---
    external_domains = [d for d in discover_test_domains(results) if d not in TRAIN_MODS]
    for ext in external_domains:
        for net in NETWORKS:
            for train_domain in TRAIN_MODS:
                group = models_for(net, train_domain)
                print_table(results, ext, group,
                            f"{NETWORK_SHORT[net]}  {train_domain}-TRAINED → Tested on {ext} (External)")

    # --- Cross-Domain gap, per network, per train modality ---
    for net in NETWORKS:
        for train_domain in TRAIN_MODS:
            group = models_for(net, train_domain)
            print_cross_domain_gap(results, group, train_domain,
                                   f"{NETWORK_SHORT[net]} / {train_domain}-trained: cross-domain gap (Dice)")

    # --- Grid coverage ---
    print_recommendations(results)

    # --- Exports ---
    print(f"\n{'='*80}")
    print(f"  EXPORTING TO: {OUTPUT_DIR}")
    print(f"{'='*80}")
    export_csv(results, OUTPUT_DIR)
    export_latex(results, per_patient, OUTPUT_DIR)
    export_significance_csv(per_patient, OUTPUT_DIR)
    # All plots — bar charts, boxplots (4 metrics), improvement plots, heatmap —
    # are dispatched in parallel via joblib.
    render_all_plots(results, per_patient, OUTPUT_DIR)

    print(f"\nDone! All artifacts saved to: {OUTPUT_DIR}")
