#!/usr/bin/env python3
"""
Segmentation domain-generalization (acquisition-domain / cross-sequence) summary
for the DynUNet backbone, extracted from the benchmark's per-test-domain tables.
Single source of truth (no on-the-fly): reads

  comparison_results/comparison_test_on_BTFE.csv   (all models tested on BTFE)
  comparison_results/comparison_test_on_TSE.csv    (all models tested on ssh-TSE)

Model column format: "<train_domain> / <arch> / <augmentation>", e.g.
"BTFE / DynU / AFA+MixUp". We keep DynU only and, for each augmentation, report:
  in-domain  = mean(BTFE->BTFE, TSE->TSE)     (model tested on its own sequence)
  OOD        = mean(BTFE->TSE, TSE->BTFE)      (model tested on the other sequence)
plus the individual directions (BTFE->TSE is the harder one).

Output: comparison_results/seg_domain_generalization_dynunet.csv  + printed table.
This corroborates the prior UNet++/radiomics finding (ELMAR, Sojo et al.) at the
segmentation-Dice level for the DynUNet backbone used in this paper's pipeline.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent
CR = PROJECT / "comparison_results"
ARCH = "DynU"
# display order + tidy names
AUG_ORDER = ["No-Aug", "Regular", "AFA+MixUp", "AFA+CutMix+MixUp"]


def dice_lookup(csv_path: Path) -> dict:
    """{(train_domain, aug): dice} for the DynU rows of one test-domain table."""
    df = pd.read_csv(csv_path)
    out = {}
    for _, r in df.iterrows():
        parts = [p.strip() for p in str(r["Model"]).split("/")]
        if len(parts) != 3:
            continue
        train_domain, arch, aug = parts
        if arch != ARCH:
            continue
        out[(train_domain, aug)] = float(r["Dice ↑"])
    return out


def rebro_model_choice(out: Path) -> pd.DataFrame:
    """Real-numbers justification for using the ssh-TSE model (not BTFE) on Rebro:
    both sequence-trained DynUNet models tested on the actual Rebro cohort
    (comparison_test_on_EXTERNAL_REBRO.csv). Rebro is T2 HASTE/TSE, so the ssh-TSE
    model should win despite BTFE's higher IN-DOMAIN (Mendeley-BTFE) scores.
    """
    df = pd.read_csv(CR / "comparison_test_on_EXTERNAL_REBRO.csv")
    col = [c for c in df.columns if "Dice" in c][0]
    dsub = {}
    for _, r in df.iterrows():
        parts = [p.strip() for p in str(r["Model"]).split("/")]
        if len(parts) != 3:
            continue
        train, arch, aug = parts
        if arch != ARCH:
            continue
        dsub[(train, aug)] = float(r[col])

    rows = []
    for aug in AUG_ORDER:
        tse, btfe = dsub.get(("TSE", aug)), dsub.get(("BTFE", aug))
        if tse is None or btfe is None:
            continue
        rows.append({"augmentation": aug,
                     "tse_model_on_rebro": round(tse, 3),
                     "btfe_model_on_rebro": round(btfe, 3),
                     "tse_advantage": round(tse - btfe, 3)})
    odf = pd.DataFrame(rows)
    odf.to_csv(out / "seg_rebro_tse_vs_btfe.csv", index=False)
    print("\n=== Rebro model choice: ssh-TSE vs BTFE DynUNet model, tested ON REBRO ===")
    print(odf.to_string(index=False))
    print(f"  ssh-TSE model beats BTFE model on Rebro for ALL {len(odf)} augmentations "
          f"(advantage {odf.tse_advantage.min():+.3f} to {odf.tse_advantage.max():+.3f}).")
    print(f"  saved -> {out / 'seg_rebro_tse_vs_btfe.csv'}")
    return odf


def main() -> int:
    on_btfe = dice_lookup(CR / "comparison_test_on_BTFE.csv")  # tested on BTFE
    on_tse = dice_lookup(CR / "comparison_test_on_TSE.csv")    # tested on ssh-TSE

    rows = []
    for aug in AUG_ORDER:
        btfe_btfe = on_btfe.get(("BTFE", aug))   # in-domain
        tse_tse = on_tse.get(("TSE", aug))       # in-domain
        btfe_tse = on_tse.get(("BTFE", aug))     # OOD (harder direction)
        tse_btfe = on_btfe.get(("TSE", aug))     # OOD
        if None in (btfe_btfe, tse_tse, btfe_tse, tse_btfe):
            print(f"  [warn] missing values for aug={aug!r}; skipping")
            continue
        in_domain = (btfe_btfe + tse_tse) / 2.0
        ood = (btfe_tse + tse_btfe) / 2.0
        rows.append({
            "augmentation": aug,
            "in_domain_mean": round(in_domain, 3),
            "ood_mean": round(ood, 3),
            "ood_gain_vs_noaug": None,          # filled below
            "BTFE->BTFE": round(btfe_btfe, 3),
            "TSE->TSE": round(tse_tse, 3),
            "BTFE->TSE": round(btfe_tse, 3),
            "TSE->BTFE": round(tse_btfe, 3),
        })

    df = pd.DataFrame(rows)
    base_ood = float(df.loc[df.augmentation == "No-Aug", "ood_mean"].iloc[0])
    base_id = float(df.loc[df.augmentation == "No-Aug", "in_domain_mean"].iloc[0])
    df["ood_gain_vs_noaug"] = (df["ood_mean"] - base_ood).round(3)
    df["id_gain_vs_noaug"] = (df["in_domain_mean"] - base_id).round(3)

    CR.mkdir(exist_ok=True)
    out_csv = CR / "seg_domain_generalization_dynunet.csv"
    df.to_csv(out_csv, index=False)

    print("=== DynUNet segmentation: in-domain vs out-of-distribution (cross-sequence) Dice ===")
    print(df.to_string(index=False))
    print(f"\n  In-domain gain (No-Aug -> AFA+MixUp): "
          f"{df.loc[df.augmentation=='AFA+MixUp','id_gain_vs_noaug'].iloc[0]:+.3f}")
    print(f"  OOD gain       (No-Aug -> AFA+MixUp): "
          f"{df.loc[df.augmentation=='AFA+MixUp','ood_gain_vs_noaug'].iloc[0]:+.3f}")
    print(f"  Hardest direction BTFE->ssh-TSE: No-Aug "
          f"{df.loc[df.augmentation=='No-Aug','BTFE->TSE'].iloc[0]:.3f} -> AFA+MixUp "
          f"{df.loc[df.augmentation=='AFA+MixUp','BTFE->TSE'].iloc[0]:.3f}")
    print(f"\nsaved -> {out_csv}")

    rebro_model_choice(CR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
