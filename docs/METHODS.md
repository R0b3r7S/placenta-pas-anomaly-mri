# Methods & project overview

A detailed description of **what this project does and how**, so anyone landing on the
repository understands the approach, the external work it builds on, and the data it
uses (and does *not* redistribute).

> **Code only.** This repository contains no MRI, DICOM, masks, patient labels, trained
> weights, or derived outputs. See [Datasets](#2-datasets) for availability.

**Contents**
1. [Motivation & approach](#1-motivation--approach)
2. [Datasets](#2-datasets)
3. [Placenta segmentation](#3-placenta-segmentation)
4. [Augmentations and where they come from](#4-augmentations-and-where-they-come-from)
5. [Boundary-patch representation](#5-boundary-patch-representation)
6. [Normal-appearance model (f-AnoGAN)](#6-normal-appearance-model-f-anogan)
7. [Anomaly score & configuration selection](#7-anomaly-score--configuration-selection)
8. [Impact of automated segmentation](#8-impact-of-automated-segmentation)
9. [End-to-end pipeline (which script does what)](#9-end-to-end-pipeline-which-script-does-what)
10. [Credits & references](#10-credits--references)

---

## 1. Motivation & approach
Placenta Accreta Spectrum (PAS) is a rare but dangerous pregnancy complication. Supervised
deep learning needs many labelled PAS examples, which are scarce. We therefore frame PAS
assessment as **anomaly detection**: a model learns the distribution of **normal** placental
appearance and flags departures from it — **without ever seeing a PAS case during training**.

Because PAS manifests at the **uteroplacental interface**, we represent each placenta by small
patches sampled along its **boundary** (not the whole organ), learn what a *normal* boundary
looks like with a reconstruction model (f-AnoGAN), and score how far each patient's boundary
deviates from that learned normal manifold.

## 2. Datasets
Two cohorts are used. **Neither is included in this repository.**

### External — public (Mendeley "Placenta Accreta Spectrum Disorders")
- **Source:** Yang, T. (2023), *Placenta Accreta Spectrum Disorders (PASDs)*, Mendeley Data v1,
  **DOI [10.17632/284gwmf5bh.1](https://doi.org/10.17632/284gwmf5bh.1)**, licensed **CC BY 4.0**.
- **Content:** all patients are **PAS-positive**, imaged after 28 weeks of gestation, provided
  as paired **BTFE** (balanced turbo field echo, i.e. balanced SSFP) and **ssh-TSE** (single-shot
  turbo spin echo) slices, each with an expert placental mask. We use the **130 patients imaged
  in both sequences** (paired). Slices are used at 512×512 (a small subset of lower-resolution
  cases is resized during preprocessing). Acquisition protocol (Philips Achieva 1.5 T) is
  described in the associated paper (Huang et al., 2024).
- **Role:** external validation for both segmentation and anomaly detection. In the code this
  cohort is referred to as `mendeley`.

### Internal — private clinical cohort
- **Content:** a retrospective, multi-scanner set of abdominopelvic MRI examinations of 12
  pregnant women on **T2-weighted sagittal** series — predominantly half-Fourier acquisition
  single-shot turbo spin echo (HASTE), with a few standard turbo spin echo (TSE) series — with
  expert placental segmentations and clinical diagnoses. The acquisition parameters reported in
  the paper (Table 1) are read from the DICOM headers by `rebro_acquisition_params.py`.
- **Availability:** **not publicly redistributed** because of patient-privacy constraints;
  available from the authors on reasonable request and with the appropriate institutional
  approvals. In the code this cohort is referred to as `rebro`.

Because the internal examinations are T2-weighted TSE acquisitions (predominantly HASTE), the
Mendeley **ssh-TSE** sequence is the sequence-matched (*in-domain*) acquisition for anomaly
scoring, while **BTFE** is *out-of-domain*.

## 3. Placenta segmentation
Segmentation produces the placental masks that define the boundary.

- **Network:** **DynUNet**, the dynamic U-Net from **MONAI** (Cardoso et al., 2022), which
  instantiates the **nnU-Net** design principles (Isensee et al., 2021) on the **U-Net**
  backbone (Ronneberger et al., 2015). For 512×512 inputs it uses 7 resolution levels; the
  channel schedule `{32,64,128,256,512,1024,1024}` follows the BraTS 2021 configuration of
  **Futrega et al. (2022)**. Blocks use 3×3 convolutions, residual connections, instance
  normalisation, and leaky-ReLU.
- **Training:** combined Dice + cross-entropy loss, AdamW (lr 1e-3, weight decay 1e-5), plateau
  LR schedule, mixed precision, fixed seed (42), early stopping (patience 25), up to 150 epochs,
  batch size 8 (fits the 16 GB training GPU).
- **Per-sequence models:** separate DynUNet models are trained on each Mendeley sequence (BTFE,
  ssh-TSE) with a patient-level 70/15/15 split (91/20/19). The ssh-TSE model (matching the
  internal T2 sequence) is applied to the internal cohort; both sequence models are kept for the
  cross-sequence generalisation analysis.
- **Fine-tuning:** an internal-cohort variant is obtained by fine-tuning the ssh-TSE model on the
  five internal patients used to train f-AnoGAN, disjoint from the seven held-out test patients
  (lr 1e-4, batch 16, up to 50 epochs, early stopping).

Code: `train_placenta_2d_monai_v8.py`, `run_augmentation_training.sh`, `run_augmentation_inference.sh`,
`run_external_inference.sh`, `run_seg3_finetune.sh`, `compare_test_metrics.py`,
`analyze_domain_generalization.py`, `analyze_seg3_finetune.py`, `analyze_fig5_seg_dice_table.py`.

## 4. Augmentations and where they come from
Three augmentation configurations are compared — *no augmentation*, *regular* (baseline), and
*AFA+MixUp* — built from:

| Augmentation | Source | Notes |
|---|---|---|
| Flips, affine, elastic, zoom, contrast, Gaussian noise/smoothing | standard **MONAI** transforms | the "regular"/baseline recipe |
| **AFA** — Auxiliary Fourier-basis Augmentation | **Vaish et al., MIDL 2025** (arXiv 2505.10223) | data-agnostic augmentation for out-of-distribution generalisation |
| **MixUp** (α = 0.2) | **Zhang et al., ICLR 2018** | convex combinations of inputs/labels |
| **CutMix** | **Yun et al., ICCV 2019** | implemented in `augmentations/cutmix_binary.py`; **not used in the paper's reported configurations** (kept as an option) |
| **Dual normalisation** (main/auxiliary route) | AFA line of work | used for AFA-enabled runs: AFA-perturbed images pass through a separate auxiliary normalisation path. DynUNet's instance-norm layers become two instance-norm paths (`DualInstanceNorm2d`); batch-norm networks use DuBIN (half instance norm, half dual batch norm). Only the main path is used at inference |

Code: `augmentations/` (`afa.py`, `mixup_binary.py`, `cutmix_binary.py`, `dual_norm.py`). This is
the same augmentation methodology validated in our prior placental-MRI segmentation study.

## 5. Boundary-patch representation
For every slice, the placental contour is extracted from the (manual or predicted) mask and
optionally restricted to its lower portion via a **lower-fraction** parameter `f ∈ [0,1]`
(`f = 0` keeps the entire contour; higher `f` keeps less). The retained contour is resampled at a
constant arc length (stride 32 px), and at each sample an oriented **64×64** patch is extracted
and **rotation-normalised** so the local boundary tangent runs horizontally through the patch
centre. Patches are scaled from `[0,255]` to `[-1,1]` to match the generator's tanh output.

Code: `f-AnoGAN-pytorch/extract_boundary_patches.py`.

## 6. Normal-appearance model (f-AnoGAN)
The normal-appearance model is **f-AnoGAN** (Schlegl et al., 2019). The code in
`f-AnoGAN-pytorch/` is our **PyTorch re-implementation**, faithful to the original TensorFlow
release and to the paper's equations/hyperparameters.

- **Architecture:** generator, discriminator, and encoder are residual convolutional networks on
  64×64 grayscale patches with a 128-dimensional latent space. As required by the gradient-penalty
  objective, the **discriminator uses LayerNorm** (not BatchNorm); the generator uses a tanh output.
- **Stage 1 — WGAN-GP** (Gulrajani et al., 2017), trained **only on normal boundary patches**:
  150 epochs, 5 critic updates per generator update, gradient-penalty weight λ = 10,
  `z ∼ N(0, I)`, Adam (β1 = 0, β2 = 0.9, lr 1e-4), batch size 64.
- **Stage 2 — encoder** (`izi_f` objective): the generator/discriminator are frozen and an encoder
  is trained for 50,000 iterations (RMSprop, lr 5e-5), combining an image-space reconstruction
  term and a discriminator-feature term.
- **Crucially**, the model sees only normal placental patches and **never a PAS example**; PAS is
  detected purely as a departure from the learned normal manifold.

Code: `f-AnoGAN-pytorch/train_wgan.py`, `train_encoder.py`, `score.py`, `models.py`.

## 7. Anomaly score & configuration selection
Each patch `x` receives the f-AnoGAN anomaly score

```
A(x) = A_R(x) + κ · A_D(x)
```

where `A_R` is the mean-squared image-reconstruction residual `‖x − G(E(x))‖² / n`, `A_D` is the
discriminator-feature residual `‖f(x) − f(G(E(x)))‖² / n_d`, and `κ = 1.0`. A **patient-level**
score is the mean over all of that patient's boundary patches.

**Configuration selection (leave-healthy-out).** The normal-appearance model is trained on a
subset of normal internal placentas and evaluated on the held-out internal test set (PAS and
non-PAS) plus the external cohort — with training/validation/test disjoint at every stage. Using
the held-out patient-level AUC to guide selection, we swept:

- training length ∈ {75, 150, 300} epochs,
- boundary lower-fraction `f` ∈ {0, 0.3, 0.5, 0.7},
- score weighting `κ` ∈ {0.5, 1.0, 2.0},

and adopted **`f = 0` (full boundary), 150 epochs, κ = 1.0** (the f-AnoGAN default) for all
reported results. Classification is summarised by AUC (with a stratified-bootstrap CI),
sensitivity and specificity; paired Dice comparisons use the Wilcoxon signed-rank test with a
bootstrap CI.

Code: `f-AnoGAN-pytorch/run_leave_healthy_out.py`, `analyze_loo_results.py`,
`make_fig4_classification.py`, `analyze_seg3_finetune.py`.

## 8. Impact of automated segmentation
To test whether the pipeline works **fully automatically** (no manual placental segmentation),
the entire boundary-patch + scoring pipeline is recomputed using **predicted** masks instead of
manual masks, and the change in patient-level score and classification AUC is reported. Predicted
masks come from patients disjoint from segmentation training (leakage-free).

Code: `f-AnoGAN-pytorch/run_impact_predmask.py`, `run_impact_mendeley.py`, `make_fig6_impact_table.py`.

## 9. End-to-end pipeline (which script does what)
```
raw MRI slices / DICOM
      │  dicom_to_slices.py           (DICOM → NIfTI + PNG slices + mask)
      │  pas_preprocessing_toolkit.py (grayscale + pad/crop + resize 512² + binarise masks + splits)
      ▼
segmentation
      │  run_augmentation_training.sh → train_placenta_2d_monai_v8.py   (DynUNet, per sequence)
      │  run_seg3_finetune.sh                                           (fine-tune on internal cohort)
      ▼
boundary patches
      │  f-AnoGAN-pytorch/extract_boundary_patches.py   (64×64 oriented, arc-length resampled)
      ▼
normal-appearance model + scoring
      │  f-AnoGAN-pytorch/run_leave_healthy_out.py
      │    → train_wgan.py → train_encoder.py → score.py   (A = A_R + κ·A_D)
      ▼
evaluation / tables / figures
         f-AnoGAN-pytorch/analyze_loo_results.py, compare_epoch_benchmarks.py, make_*.py,
         run_impact_*.py, dataset_slice_summary.py, top-level analyze_*.py, rebro_acquisition_params.py
```
The README's *Reproducing the paper* section lists the exact commands and which script produces
each table and figure.
Absolute paths are **not** baked into the scripts (shell scripts `cd` to their own directory);
adjust dataset/output paths to your own layout. See the main [README](../README.md) for the
expected `dataset/` layout and quick-start commands.

## 10. Credits & references
This work builds directly on the following. Please cite them if you use this code.

- **f-AnoGAN** — Schlegl, T., Seeböck, P., Waldstein, S. M., Langs, G., Schmidt-Erfurth, U.
  *f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks.*
  Medical Image Analysis 54 (2019) 30–44. DOI 10.1016/j.media.2019.01.010.
- **WGAN-GP** — Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., Courville, A.
  *Improved Training of Wasserstein GANs.* NeurIPS 2017.
- **MONAI** — Cardoso, M. J., et al. *MONAI: An open-source framework for deep learning in
  healthcare.* arXiv:2211.02701 (2022).
- **nnU-Net** — Isensee, F., et al. *nnU-Net: a self-configuring method for deep learning-based
  biomedical image segmentation.* Nature Methods 18 (2021) 203–211.
- **U-Net** — Ronneberger, O., Fischer, P., Brox, T. *U-Net: Convolutional Networks for
  Biomedical Image Segmentation.* MICCAI 2015.
- **UNet++** — Zhou, Z., et al. *UNet++: A Nested U-Net Architecture for Medical Image
  Segmentation.* DLMIA 2018.
- **Optimized U-Net (BraTS 2021)** — Futrega, M., Milesi, A., Marcinkiewicz, M., Ribalta, P.
  *Optimized U-Net for Brain Tumor Segmentation.* In: Brainlesion: Glioma, Multiple Sclerosis,
  Stroke and Traumatic Brain Injuries (BrainLes 2021), LNCS 12963, pp. 15–29. Springer (2022).
  DOI 10.1007/978-3-031-09002-8_2.
- **AFA** — Vaish, P., Meister, F., Heimann, T., Brune, C., Wolterink, J. M.
  *Data-Agnostic Augmentations for Unknown Variations: Out-of-Distribution Generalisation in MRI
  Segmentation.* MIDL 2025 (arXiv:2505.10223).
- **MixUp** — Zhang, H., Cisse, M., Dauphin, Y. N., Lopez-Paz, D. *mixup: Beyond Empirical Risk
  Minimization.* ICLR 2018.
- **CutMix** — Yun, S., et al. *CutMix: Regularization Strategy to Train Strong Classifiers with
  Localizable Features.* ICCV 2019.
- **Public dataset** — Yang, T. *Placenta Accreta Spectrum Disorders (PASDs).* Mendeley Data v1
  (2023). DOI 10.17632/284gwmf5bh.1 (CC BY 4.0).
- **Dataset imaging protocol** — Huang, F., Lyu, G.-R., Lai, Q.-Q., Li, Y.-Z. *Nomogram model for
  predicting invasive placenta in patients with placenta previa…* Scientific Reports 14 (2024)
  200. DOI 10.1038/s41598-023-50900-z.
- **Prior augmentation study (this group)** — Šojo, R., Benčević, M., Galić, I., Košuta Petrović, M.,
  Kopačin, V. *Radiomics-based evaluation of data-agnostic augmentations for placental MRI
  segmentation under image corruptions.* 2026 International Symposium ELMAR, Zadar, Croatia, IEEE,
  pp. 71–74. DOI 10.1109/ELMAR71231.2026.11712250.
  This is the "prior placental-MRI segmentation study" referred to in §4.

If you use this repository, please also cite the accompanying paper (see the main README).
