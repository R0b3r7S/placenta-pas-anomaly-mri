# placenta-pas-anomaly-mri

**Detecting Placenta Accreta Spectrum (PAS) on MRI by learning only normal placental appearance** — a reconstruction-based anomaly-detection model (f-AnoGAN) trained on **no PAS cases**, plus the DynUNet placenta-segmentation pipeline that feeds it.

This repository contains the **code** accompanying the paper:

> R. Šojo, M. Benčević, L. Kovačević, A. Šimić, M. Prutki, I. Galić. *Learning Normal Placental Appearance for MRI Detection of Placenta Accreta Spectrum.* (under review, Journal of Medical and Biological Engineering)
> Faculty of Electrical Engineering, Computer Science and Information Technology Osijek (FERIT), Josip Juraj Strossmayer University of Osijek.

> **Code only — no data.** No MRI, DICOM, masks, patient labels, trained weights, or derived outputs are included (see *Data availability*). Scripts expect you to supply data in the layout below.

---

## Method in brief
1. **Segmentation.** A 2D MONAI **DynUNet** (nnU-Net-style) segments the placenta, trained per MRI sequence with standard augmentation + **AFA** (Auxiliary Fourier-basis Augmentation) + **MixUp**, and fine-tuned on the internal cohort.
2. **Boundary patches.** Because PAS manifests at the uteroplacental interface, each placenta is represented by small oriented `64×64` patches sampled along its (manual or predicted) boundary.
3. **Normal-appearance model.** **f-AnoGAN** (WGAN-GP generator/discriminator + `izi_f` encoder) is trained **only on normal placental boundary patches**. PAS is detected purely as a departure from the learned normal manifold; the anomaly score is `A(x) = A_R(x) + κ·A_D(x)`.
4. **Evaluation.** A **leave-healthy-out** protocol; patient-level score = mean over that patient's boundary patches.

> **Full method, exact configuration, augmentation sources, dataset details, and attribution:** see **[docs/METHODS.md](docs/METHODS.md)**.

## Repository layout
```
.
├── dicom_to_slices.py                DICOM → NIfTI + PNG slices (+ per-series metadata table)
├── parse_rebro_sheet.py              internal clinical sheet → clinical_labels.csv
├── build_patient_labels.py           unified patient_labels.csv (both cohorts)
├── build_external_test_set.py        stage the converted internal cohort for the toolkit
├── pas_preprocessing_toolkit.py      grayscale + pad/crop + resize to 512×512 + mask binarisation + splits
├── write_external_splits.py          splits.json for the internal cohort
├── rebro_acquisition_params.py       Table 1: internal-cohort acquisition parameters from DICOM headers
├── train_placenta_2d_monai_v8.py     DynUNet (or UNet++) training / testing
├── run_augmentation_training.sh      train per sequence × augmentation setting
├── run_augmentation_inference.sh     in-/cross-sequence inference on the external test sets
├── run_external_inference.sh         inference on the internal cohort
├── run_seg3_finetune.sh              fine-tune on the internal cohort (both models)
├── run_seg3_afa_only.sh              re-run only the AFA+MixUp fine-tuning
├── compare_test_metrics.py           collect segmentation test metrics into tables
├── analyze_domain_generalization.py  Table 4 (+ internal ssh-TSE vs BTFE model choice)
├── analyze_seg3_finetune.py          Table 3 statistics (Wilcoxon, bootstrap CI)
├── analyze_fig5_seg_dice_table.py    Table 3 values
├── augmentations/                    afa.py, mixup_binary.py, cutmix_binary.py, dual_norm.py
├── f-AnoGAN-pytorch/                 anomaly detection (see its own README.md)
│   ├── models.py, data.py, utils.py              f-AnoGAN networks and data loading
│   ├── train_wgan.py, train_encoder.py, score.py WGAN-GP → izi_f encoder → anomaly score
│   ├── smoke_test.py                             end-to-end check on synthetic data
│   ├── extract_boundary_patches.py               boundary → arc-length resampling → oriented 64×64 patches
│   ├── run_leave_healthy_out.py                  leave-healthy-out experiment (patches, training, scoring)
│   ├── analyze_loo_results.py, compare_epoch_benchmarks.py   configuration sweep summaries
│   ├── make_fig1_diagnosis_table.py              Table 2
│   ├── make_fig4_classification.py               AUC, 95% CI, threshold, sensitivity/specificity, external detection rate
│   ├── run_impact_predmask.py, run_impact_mendeley.py, make_fig6_impact_table.py   impact of automated segmentation
│   ├── dataset_slice_summary.py                  slice counts (Table 1)
│   ├── make_anomaly_nifti.py                     continuous anomaly maps (input to Fig. 2)
│   ├── make_methods_patches_figure.py            Fig. 1
│   ├── make_fig2_reb001.py                       Fig. 2
│   ├── make_fig3_group_scatter.py                Fig. 3
│   └── journal_figure_style.py                   shared settings for the journal figures (--journal)
├── docs/METHODS.md
├── requirements.txt, CITATION.cff, LICENSE, NOTICE, .gitignore
```
Some script names follow an earlier figure numbering (e.g. `make_fig1_diagnosis_table.py` produces Table 2); the table in *Reproducing the paper* gives the current mapping. In the code, the internal cohort is called `rebro` (patients `reb001`–`reb014`, written I01–I14 in the paper) and the external cohort `mendeley`.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # or a conda env (the study used conda)
pip install -r requirements.txt
# Install a PyTorch build matching your CUDA/CPU: https://pytorch.org
```

## Expected data layout (you provide this)
Scripts read from a local `dataset/` folder (git-ignored):
```
dataset/
├── dicom/                     raw DICOM (internal cohort) — input of dicom_to_slices.py
├── dicom_converted/           output of dicom_to_slices.py (incl. conversion_summary.csv)
└── mri_png/
    ├── DATASET_BTFE/          {images,masks}/<patient_id>/<patient_id>_<slice>.png + splits.json
    ├── DATASET_SSH_TSE/       (same structure)
    ├── DATASET_EXTERNAL_REBRO/   internal cohort (same structure; all patients in "test")
    └── DATASET_REBRO_FT/      internal cohort for fine-tuning (see below)
```
All slices are `512×512` 8-bit grayscale PNGs; `splits.json` defines patient-level train/val/test lists. `DATASET_REBRO_FT` reuses the internal images and masks with the fine-tuning split (train = the five normal placentas used to train f-AnoGAN, test = the seven held-out patients):
```bash
mkdir -p dataset/mri_png/DATASET_REBRO_FT && cd dataset/mri_png/DATASET_REBRO_FT
ln -s ../DATASET_EXTERNAL_REBRO/images images && ln -s ../DATASET_EXTERNAL_REBRO/masks masks
cat > splits.json << 'EOF'
{"train": ["reb003", "reb006", "reb009", "reb013", "reb014"], "val": [],
 "test": ["reb001", "reb004", "reb005", "reb007", "reb008", "reb010", "reb011"],
 "seed": 42, "mode": "finetune_rebro"}
EOF
```

## Reproducing the paper
Run from the repository root. Outputs go to git-ignored folders (`runs/`, `runs_fanogan_loo/`, `boundary_patches_loo/`, `comparison_results/`, `shareable/`).

```bash
# 1. Data preparation
python dicom_to_slices.py --cohorts kbc_rebro   # internal cohort: DICOM → NIfTI/PNG + conversion_summary.csv
python parse_rebro_sheet.py                # internal clinical sheet → clinical_labels.csv
python build_patient_labels.py             # unified patient_labels.csv
python build_external_test_set.py          # stage the internal cohort for the toolkit
python pas_preprocessing_toolkit.py        # all cohorts → 512×512 PNG slices + masks (interactive)
python write_external_splits.py            # splits.json for the internal cohort

# 2. Segmentation (the paper reports DynUNet)
NETWORKS=dynunet bash run_augmentation_training.sh
NETWORKS=dynunet bash run_augmentation_inference.sh
NETWORKS=dynunet bash run_external_inference.sh
bash run_seg3_finetune.sh
python compare_test_metrics.py
python analyze_domain_generalization.py
python analyze_seg3_finetune.py && python analyze_fig5_seg_dice_table.py

# 3. f-AnoGAN, leave-healthy-out — final configuration (150 epochs, full boundary, κ = 1.0)
python f-AnoGAN-pytorch/run_leave_healthy_out.py --tag phase3_150ep --epochs 150 --lower-fractions 0.0 --kappas 1.0
#    configuration sweep reported in the Methods: repeat with
#    --tag phase4_75ep --epochs 75 | --tag phase3_150ep --epochs 150 | --tag phase2_300ep --epochs 300,
#    each with --lower-fractions 0.0 0.3 0.5 0.7 --kappas 0.5 1.0 2.0, then:
python f-AnoGAN-pytorch/analyze_loo_results.py --tag phase3_150ep
python f-AnoGAN-pytorch/compare_epoch_benchmarks.py

# 4. Tables and numbers
python rebro_acquisition_params.py                   # Table 1 (internal acquisition parameters)
python f-AnoGAN-pytorch/dataset_slice_summary.py     # Table 1 (slice counts)
python f-AnoGAN-pytorch/make_fig1_diagnosis_table.py # Table 2
python f-AnoGAN-pytorch/make_fig4_classification.py  # AUC, 95% CI, threshold, sensitivity/specificity, external detection
python train_placenta_2d_monai_v8.py --mode test --network dynunet \
    --dataset_root dataset/mri_png/DATASET_SSH_TSE \
    --checkpoint runs/TSE_dynunet_afa_mixup/fold_0/best_model.pth --use_afa --amp \
    --out_dir runs/TSE_dynunet_afa_mixup_mendeleyTSE_pred   # predicted masks, 19 external test patients
python f-AnoGAN-pytorch/run_impact_predmask.py       # internal: uses the fine-tuned model's predictions
python f-AnoGAN-pytorch/run_impact_mendeley.py       # external: uses the predictions above
python f-AnoGAN-pytorch/make_fig6_impact_table.py    # impact of automated segmentation

# 5. Figures (journal versions → shareable/figures/journal/Fig1.png, Fig2.png, Fig3.pdf)
python f-AnoGAN-pytorch/make_methods_patches_figure.py --journal
python f-AnoGAN-pytorch/make_anomaly_nifti.py --run-tag phase3_150ep --lf 0.0 --sigma 6 --patients reb001
python f-AnoGAN-pytorch/make_fig2_reb001.py --journal
python f-AnoGAN-pytorch/make_fig3_group_scatter.py --journal
```

| Result in the paper | Script(s) |
|---|---|
| Table 1 — acquisition parameters, slice counts | `rebro_acquisition_params.py` (after `dicom_to_slices.py`), `f-AnoGAN-pytorch/dataset_slice_summary.py` |
| Table 2 — per-patient findings and anomaly scores | `f-AnoGAN-pytorch/run_leave_healthy_out.py` → `make_fig1_diagnosis_table.py` |
| Table 3 — Dice before/after fine-tuning; Wilcoxon test, bootstrap CI | `run_external_inference.sh`, `run_seg3_finetune.sh`, `analyze_seg3_finetune.py`, `analyze_fig5_seg_dice_table.py` |
| Table 4 — Dice in-domain / out-of-domain; ssh-TSE vs BTFE on the internal cohort | `run_augmentation_training.sh`, `run_augmentation_inference.sh`, `run_external_inference.sh`, `compare_test_metrics.py`, `analyze_domain_generalization.py` |
| Configuration selection (epochs, κ, boundary fraction) | `f-AnoGAN-pytorch/run_leave_healthy_out.py`, `analyze_loo_results.py`, `compare_epoch_benchmarks.py` |
| AUC, 95% CI, Youden threshold, sensitivity/specificity, external detection rate | `f-AnoGAN-pytorch/make_fig4_classification.py` |
| Impact of automated segmentation | `run_seg3_finetune.sh` (internal predictions), `train_placenta_2d_monai_v8.py --mode test` (external predictions), `f-AnoGAN-pytorch/run_impact_predmask.py`, `run_impact_mendeley.py`, `make_fig6_impact_table.py` |
| Fig. 1 — boundary-patch extraction | `f-AnoGAN-pytorch/make_methods_patches_figure.py --journal` |
| Fig. 2 — continuous anomaly map (I01) | `f-AnoGAN-pytorch/make_anomaly_nifti.py`, then `make_fig2_reb001.py --journal` |
| Fig. 3 — BTFE vs ssh-TSE anomaly scores | `f-AnoGAN-pytorch/make_fig3_group_scatter.py --journal` |

Shell scripts use relative paths (they `cd` to the repository root); run them inside the activated Python environment.

## Data availability
- **Public (External) cohort — Mendeley "Placenta Accreta Spectrum Disorders":** DOI **10.17632/284gwmf5bh.1** (Yang, 2023; CC BY 4.0). All patients are PAS-positive, provided as paired BTFE + ssh-TSE slices with expert placental masks.
- **Private (Internal) clinical cohort:** not publicly available due to patient-privacy constraints; available from the authors on reasonable request and with the appropriate institutional approvals.

No dataset is redistributed in this repository.

## Credits
- **f-AnoGAN:** Schlegl et al., *f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks*, Medical Image Analysis 54 (2019) 30–44, DOI 10.1016/j.media.2019.01.010. The code in `f-AnoGAN-pytorch/` is our PyTorch re-implementation faithful to the original.
- **MONAI**, **nnU-Net** design principles, and **AFA** (Vaish et al., MIDL 2025) underpin the segmentation.

## License
**Apache License 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The `augmentations/` modules are derived and modified from Apache-2.0 upstream code (AFA — MIA Group-UT; MixUp/CutMix — nnU-Net), so the project is distributed under the same permissive license; the NOTICE file records the required attributions.
