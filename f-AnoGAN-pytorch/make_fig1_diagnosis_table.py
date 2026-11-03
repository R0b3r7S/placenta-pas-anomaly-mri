#!/usr/bin/env python3
"""
Paper Fig/Table-1: diagnosis for every Rebro patient, split by role in the
anomaly model (train vs test). For TEST patients, also the honest patient-level
anomaly score (leave-healthy-out run phase3_150ep_lf00, SSH_TSE/T2 in-domain).

Diagnoses come from dataset/dicom_converted/patient_labels.csv (raw Croatian dx),
mapped to English below (verified, not guessed). Anomaly scores from the HONEST
run only. Training patients have NO independent score (in training) -> shown as
"in training".

Output: comparison_results/fig1_diagnosis_table.csv + a rendered PNG.
"""
from __future__ import annotations
import csv
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent.parent
LABELS = PROJECT / "dataset" / "dicom_converted" / "patient_labels.csv"
SCORES = PROJECT / "runs_fanogan_loo" / "phase3_150ep_lf00" / "scores_kappa1.0" / "scores.csv"
OUT_CSV = PROJECT / "comparison_results" / "fig1_diagnosis_table.csv"
OUT_PNG = PROJECT / "shareable" / "figures" / "fig1_diagnosis_table.png"

TRAIN = ["reb003", "reb006", "reb009", "reb013", "reb014"]
TEST = ["reb001", "reb004", "reb005", "reb007", "reb008", "reb010", "reb011"]

# verified English mapping of the raw Croatian dx in patient_labels.csv
DX_EN = {
    "Placenta percreta": "PAS – percreta",
    "Placenta acreta": "PAS – accreta",
    "Marginalna placenta previja, percreta": "PAS – percreta + previa",
    "Uredna posteljica": "Normal placenta",
    "Placenta uredna": "Normal placenta",
    "Uredna placenta": "Normal placenta",
    "Uredan nalaz": "Normal placenta",
    "Placenta previja": "Placenta previa",
    "Miom": "Uterine myoma",
    "apendicitis": "Appendicitis",
    "": "Normal (unspecified)",
}


def rebro_score():
    df = pd.read_csv(SCORES)
    df["pid"] = df.patch_path.apply(lambda p: Path(str(p)).stem.split("_")[0])
    df = df[df.pid.str.startswith("reb")]
    return df.groupby("pid").score.mean()


def main() -> int:
    raw = {r["patient_id"]: (r["dx"].strip(), r["is_pas"])
           for r in csv.DictReader(open(LABELS)) if r["cohort"] == "kbc_rebro"}
    sc = rebro_score()

    rows = []
    for role, pids in [("train (model)", TRAIN), ("test (held-out)", TEST)]:
        for p in pids:
            dx_raw, is_pas = raw[p]
            pas = "PAS" if str(is_pas).startswith("1") else "non-PAS"
            score = f"{sc[p]:.3f}" if (role.startswith("test") and p in sc.index) else "in training"
            rows.append({"patient": p, "role": role,
                         "diagnosis": DX_EN.get(dx_raw, dx_raw or "?"),
                         "PAS status": pas, "anomaly score": score})
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # rendered table
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(df) + 1.1)); ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="left",
                   colWidths=[0.13, 0.17, 0.34, 0.15, 0.17])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.45)
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor("#34495E"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, r in enumerate(rows, start=1):
        base = "#eaf3ea" if r["PAS status"] == "non-PAS" else "#fdeaea"
        if r["role"].startswith("train"):
            base = "#eef2f7"
        for j in range(len(df.columns)):
            tbl[i, j].set_facecolor(base)
    ax.set_title("Rebro cohort: diagnosis and role in the anomaly model\n"
                 "(anomaly score = honest leave-healthy-out run; training patients not scored)",
                 fontsize=11, pad=12)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(df.to_string(index=False))
    print(f"\nsaved -> {OUT_CSV}\nsaved -> {OUT_PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
