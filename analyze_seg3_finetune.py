#!/usr/bin/env python3
"""
Seg-3 analysis: side-by-side BEFORE (Mendeley-only) vs AFTER (fine-tuned on the
5 anomaly-train Rebro) segmentation Dice on the 7 held-out Rebro patients, for
both TSE_dynunet_afa_mixup and TSE_dynunet_regular. Single source of truth for
the Seg-3 numbers (no on-the-fly).

BEFORE = runs/<model>/test_on_EXTERNAL_REBRO/fold_0/per_patient_metrics.json
AFTER  = runs/<model>_rebroFT/test_on_heldout/per_patient_metrics.json

Output: comparison_results/seg3_finetune_comparison.csv  + printed table.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent
MODELS = ["TSE_dynunet_afa_mixup", "TSE_dynunet_regular"]
TEST_7 = ["reb001", "reb004", "reb005", "reb007", "reb008", "reb010", "reb011"]
DIAG = {"reb001": "PAS percreta", "reb007": "PAS acreta", "reb008": "PAS percreta+previa",
        "reb004": "previa", "reb005": "myoma", "reb010": "appendicitis", "reb011": "healthy"}


def load_dice(path: Path) -> dict:
    """Return {pid: dice} from a per_patient_metrics.json (dict or list form)."""
    if not path.is_file():
        return {}
    d = json.load(open(path))
    out = {}
    if isinstance(d, dict):
        for pid, m in d.items():
            if isinstance(m, dict):
                out[pid] = m.get("dice", m.get("Dice"))
    elif isinstance(d, list):
        for r in d:
            pid = r.get("patient") or r.get("patient_id") or r.get("id")
            if pid is not None:
                out[pid] = r.get("dice", r.get("Dice"))
    return out


def make_figure(df: pd.DataFrame, path: Path) -> None:
    """Grouped before/after Dice bars per patient, one panel per model."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(13, 5.0), sharey=True,
                             constrained_layout=True)
    for ax, model in zip(np.atleast_1d(axes), MODELS):
        sub = df[df.model == model].reset_index(drop=True)
        x = np.arange(len(sub)); w = 0.38
        ax.bar(x - w / 2, sub.dice_before, w, label="before (Mendeley-only)", color="#9ecae1")
        ax.bar(x + w / 2, sub.dice_after, w, label="after (Rebro fine-tuned)", color="#08519c")
        mb, ma = sub.dice_before.mean(), sub.dice_after.mean()
        ax.axhline(mb, ls="--", lw=1, color="#9ecae1")
        ax.axhline(ma, ls="--", lw=1, color="#08519c")
        ax.set_title(f"{model.replace('TSE_dynunet_', '')}: mean "
                     f"{mb:.3f} -> {ma:.3f} ({ma - mb:+.3f})", fontsize=10, pad=8)
        ax.set_xticks(x); ax.set_xticklabels(sub.patient, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Dice")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Seg-3: segmentation Dice before vs after fine-tuning on 5 Rebro (7 held-out)",
                 fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {path}")


def paired_after_stats(df: pd.DataFrame, out: Path, n_boot: int = 10000,
                       seed: int = 42) -> dict:
    """Small-sample stats for the AFTER-fine-tuning AFA+MixUp vs baseline Dice
    comparison on the 7 held-out Rebro patients (Option 1):
      - Wilcoxon signed-rank test (exact for n=7, paired)
      - bootstrap 95% CI on the mean Dice difference (afa+mixup − baseline)
    Reusable single-source-of-truth; numbers land in seg3_finetune_stats.csv.
    """
    piv = df.pivot(index="patient", columns="model", values="dice_after").loc[TEST_7]
    a = piv["TSE_dynunet_afa_mixup"].to_numpy(dtype=float)   # AFA+MixUp, fine-tuned
    b = piv["TSE_dynunet_regular"].to_numpy(dtype=float)     # baseline, fine-tuned
    diff = a - b
    n = int(len(diff))

    # Wilcoxon signed-rank (paired). Default method auto-selects the exact test
    # for small n with no zero differences — which is our case (n=7).
    try:
        wres = stats.wilcoxon(a, b, alternative="two-sided")
        w_stat, w_p = float(wres.statistic), float(wres.pvalue)
    except Exception as exc:  # pragma: no cover
        w_stat, w_p = float("nan"), float("nan")
        print(f"  [warn] Wilcoxon failed: {exc}")

    # Bootstrap 95% CI on the mean difference (vectorized, fixed seed = reproducible).
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[boot_idx].mean(axis=1)
    ci_lo, ci_hi = (float(x) for x in np.percentile(boot_means, [2.5, 97.5]))

    res = {
        "comparison": "AFA+MixUp(FT) - Baseline(FT), Dice on 7 held-out Rebro",
        "n_patients": n,
        "mean_dice_afa_mixup_ft": round(float(a.mean()), 4),
        "mean_dice_baseline_ft": round(float(b.mean()), 4),
        "mean_diff": round(float(diff.mean()), 4),
        "median_diff": round(float(np.median(diff)), 4),
        "n_afa_better": int((diff > 0).sum()),
        "n_baseline_better": int((diff < 0).sum()),
        "wilcoxon_stat": round(w_stat, 4),
        "wilcoxon_p_two_sided": round(w_p, 4),
        "boot_ci95_lo": round(ci_lo, 4),
        "boot_ci95_hi": round(ci_hi, 4),
        "n_boot": n_boot,
        "boot_seed": seed,
    }
    pd.DataFrame([res]).to_csv(out / "seg3_finetune_stats.csv", index=False)

    print("\n=== Seg-3 SMALL-SAMPLE STATS (Option 1): AFA+MixUp(FT) vs Baseline(FT), 7 held-out ===")
    print(f"  mean Dice:  AFA+MixUp-FT {res['mean_dice_afa_mixup_ft']:.3f}  "
          f"vs  Baseline-FT {res['mean_dice_baseline_ft']:.3f}")
    print(f"  mean difference (AFA+MixUp − Baseline): {res['mean_diff']:+.3f}  "
          f"(AFA+MixUp better on {res['n_afa_better']}/{n} patients)")
    print(f"  Wilcoxon signed-rank (exact, two-sided): W={res['wilcoxon_stat']:.1f}, "
          f"p={res['wilcoxon_p_two_sided']:.4f}")
    print(f"  bootstrap 95% CI on mean difference: "
          f"[{res['boot_ci95_lo']:+.3f}, {res['boot_ci95_hi']:+.3f}]  "
          f"({n_boot} resamples, seed {seed})")
    verdict = ("CI excludes 0 -> improvement is statistically supported"
               if res["boot_ci95_lo"] > 0 else
               "CI includes 0 -> improvement NOT statistically established at n=7")
    print(f"  -> {verdict}")
    print(f"  saved -> {out / 'seg3_finetune_stats.csv'}")
    return res


def main() -> int:
    rows = []
    for model in MODELS:
        before = load_dice(PROJECT / "runs" / model / "test_on_EXTERNAL_REBRO" /
                           "fold_0" / "per_patient_metrics.json")
        after = load_dice(PROJECT / "runs" / f"{model}_rebroFT" /
                          "test_on_heldout" / "per_patient_metrics.json")
        for pid in TEST_7:
            b, a = before.get(pid), after.get(pid)
            rows.append({"model": model, "patient": pid, "diagnosis": DIAG.get(pid, ""),
                         "dice_before": round(b, 3) if b is not None else None,
                         "dice_after": round(a, 3) if a is not None else None,
                         "delta": round(a - b, 3) if (a is not None and b is not None) else None})
    df = pd.DataFrame(rows)
    out = PROJECT / "comparison_results"; out.mkdir(exist_ok=True)
    df.to_csv(out / "seg3_finetune_comparison.csv", index=False)
    make_figure(df, out / "seg3_finetune_comparison.png")
    paired_after_stats(df, out)

    print("=== Seg-3: BEFORE (Mendeley-only) vs AFTER (fine-tuned on 5 Rebro), "
          "Dice on the 7 held-out Rebro ===")
    for model in MODELS:
        sub = df[df.model == model]
        print(f"\n--- {model} ---")
        print(sub[["patient", "diagnosis", "dice_before", "dice_after", "delta"]]
              .to_string(index=False))
        b = sub.dice_before.dropna(); a = sub.dice_after.dropna()
        improved = int((sub.delta.dropna() > 0).sum())
        print(f"  MEAN: before={b.mean():.3f}  after={a.mean():.3f}  "
              f"delta={a.mean() - b.mean():+.3f}   ({improved}/{len(sub)} patients improved)")
    print(f"\nsaved -> {out / 'seg3_finetune_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
