import argparse
import csv
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents GUI errors on headless servers
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import CacheDataset, Dataset
from monai.losses import DiceCELoss
import gc

# --- Optional parallel saving for inference (used in Phase 2 of test mode) ---
try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

# --- Data-Agnostic Augmentations (Paper: arXiv 2505.10223) ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augmentations import (
    RandomMixUpBinary, RandomCutMixBinary, AFA,
    convert_to_dual_norm, set_dual_norm_route
)

from monai.metrics import (
    DiceMetric,
    compute_hausdorff_distance,
    compute_average_surface_distance,
    compute_surface_dice,
)
# DiceMetric is imported for completeness/future use. PatientMetricsTracker
# computes Dice directly from aggregated TP/FP/FN counts across all slices of a
# patient ("global/volume Dice"), which is more meaningful at patient level than
# averaging per-slice Dice. DiceMetric would give the per-slice-mean variant.

from scipy.ndimage import distance_transform_edt
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    Lambdad, EnsureTyped, RandFlipd, RandAffined, RandZoomd,
    RandAdjustContrastd, RandGaussianNoised, RandGaussianSmoothd
)
from monai.utils import set_determinism

# ---------------------------------------------------------------------------
# 1. Dataset Parsing
# ---------------------------------------------------------------------------
def read_splits(splits_path):
    with open(splits_path, "r") as f:
        return json.load(f)

def build_items(images_dir, masks_dir, patient_ids):
    """Build image/mask pair items.

    Works with two on-disk layouts:
      • patient subfolders:  images/<pid>/<pid>_<sid>.png   (placenta dataset)
      • flat layout:         images/<pid>_<sid>.png          (fetal-head etc.)
    """
    items = []
    pid_set = set(patient_ids)
    images_path = Path(images_dir)

    subdirs = [p for p in images_path.iterdir() if p.is_dir()]
    is_flat = len(subdirs) == 0

    for img_path in images_path.rglob("*.png"):
        if is_flat:
            stem = img_path.stem
            if stem in pid_set:
                patient_id = stem
            elif '_' in stem:
                patient_id = stem.rsplit('_', 1)[0]
                while patient_id not in pid_set and '_' in patient_id:
                    patient_id = patient_id.rsplit('_', 1)[0]
            else:
                patient_id = stem
        else:
            patient_id = img_path.parent.name

        if patient_id in pid_set:
            rel_path = img_path.relative_to(images_dir)
            mask_path = Path(masks_dir) / rel_path
            if mask_path.exists():
                items.append({
                    "image": str(img_path),
                    "label": str(mask_path),
                    "patient_id": patient_id,
                    "slice_id": img_path.stem
                })
    return items

# ---------------------------------------------------------------------------
# 2. Network Switcher
# ---------------------------------------------------------------------------
def get_network(name, dropout: float = 0.0):
    """Return a 2D segmentation network with optional dropout.

    Dropout is OFF by default (dropout=0.0). When dropout > 0, the value is
    applied to every network that supports it.
    """
    # ==========================================
    # 1. THE U-NET FAMILY (CNNs)
    # ==========================================
    if name == "unet":
        from monai.networks.nets import UNet
        return UNet(spatial_dims=2, in_channels=1, out_channels=1,
                    channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
                    num_res_units=2, dropout=dropout)

    elif name == "attentionunet":
        from monai.networks.nets import AttentionUnet
        return AttentionUnet(spatial_dims=2, in_channels=1, out_channels=1,
                             channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
                             dropout=dropout)

    elif name == "basicunet":
        from monai.networks.nets import BasicUNet
        # A lightweight, ultra-fast baseline U-Net
        return BasicUNet(spatial_dims=2, in_channels=1, out_channels=1, dropout=dropout)

    elif name == "unetplusplus":
        from monai.networks.nets import BasicUNetPlusPlus
        # Dense, nested skip connections for complex boundaries
        return BasicUNetPlusPlus(spatial_dims=2, in_channels=1, out_channels=1,
                                 deep_supervision=False, dropout=dropout)

    elif name == "flexunet":
        from monai.networks.nets import FlexibleUNet
        # Uses an EfficientNet backbone (highly memory efficient and accurate)
        return FlexibleUNet(in_channels=1, out_channels=1, backbone="efficientnet-b2",
                            spatial_dims=2, dropout=dropout)

    elif name == "dynunet":
        from monai.networks.nets import DynUNet

        # 1. THE MATH FOR 512x512 IMAGES:
        # We need 7 levels. The first level stays at 512.
        # The next 6 levels halve the size: 256 -> 128 -> 64 -> 32 -> 16 -> 8

        # Kernel size is always 3x3 for every block
        kernels = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]

        # Stride of 1 keeps the size the same. Stride of 2 cuts it in half.
        strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

        # Upsampling kernels must perfectly match the downsampling strides
        upsample_kernels = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

        # The number of channels learned at each depth level
        filters = [32, 64, 128, 256, 512, 1024, 1024]

        # Dropout uses the same --dropout setting as every other network
        # (default 0.0 = OFF). The 7-level / filters layout above is the tuned
        # config and stays unchanged.
        return DynUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=upsample_kernels,
            filters=filters,
            dropout=dropout,
            res_block=True,          # nnU-Net style residual blocks
            deep_supervision=False   # ensure a single final mask output
        )

    # ==========================================
    # 2. THE RESIDUAL & HIGH-RES FAMILY
    # ==========================================
    elif name == "segresnet":
        from monai.networks.nets import SegResNet
        return SegResNet(spatial_dims=2, in_channels=1, out_channels=1,
                         init_filters=32, blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1),
                         dropout_prob=dropout)

    elif name == "vnet":
        from monai.networks.nets import VNet
        return VNet(spatial_dims=2, in_channels=1, out_channels=1)

    elif name == "highresnet":
        from monai.networks.nets import HighResNet
        return HighResNet(spatial_dims=2, in_channels=1, out_channels=1)

    # ==========================================
    # 3. THE VISION TRANSFORMERS (ViTs)
    # ==========================================
    elif name == "unetr":
        from monai.networks.nets import UNETR
        return UNETR(in_channels=1, out_channels=1, img_size=(512, 512), spatial_dims=2)

    elif name == "swinunetr":
        from monai.networks.nets import SwinUNETR
        return SwinUNETR(img_size=(512, 512), in_channels=1, out_channels=1,
                         feature_size=24, spatial_dims=2)

    else:
        raise ValueError(f"Unknown network: {name}")

# ---------------------------------------------------------------------------
# 3. Boundary IoU helper (Cheng et al. 2021)
# ---------------------------------------------------------------------------
def _boundary_iou(pred_bin: np.ndarray, gt_bin: np.ndarray, dilation_ratio: float = 0.02) -> float:
    """Boundary IoU: IoU restricted to a thin band along each mask's boundary.

    Paper: "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation",
    Cheng et al. CVPR 2021. The dilation_ratio is multiplied by the image diagonal
    to set the band thickness `d` in pixels (paper default: 0.02 → ~14 px at 512x512).

    Args:
        pred_bin, gt_bin: 2D HxW boolean arrays.
        dilation_ratio:   band thickness as a fraction of the image diagonal.

    Returns:
        Boundary IoU in [0, 1]. Returns 1.0 when both masks are empty (degenerate).
    """
    if not pred_bin.any() and not gt_bin.any():
        return 1.0
    h, w = pred_bin.shape
    d = max(1, int(round(dilation_ratio * np.sqrt(h * h + w * w))))
    # distance_transform_edt(mask) → distance from each FG pixel to nearest BG pixel,
    # so pixels with dist<=d are the boundary band of the mask.
    pred_band = pred_bin & (distance_transform_edt(pred_bin) <= d)
    gt_band = gt_bin & (distance_transform_edt(gt_bin) <= d)
    inter = np.logical_and(pred_band, gt_band).sum()
    union = np.logical_or(pred_band, gt_band).sum()
    return float(inter / union) if union > 0 else 1.0


# ---------------------------------------------------------------------------
# 4. Patient-Total Comprehensive Metrics Tracker
# ---------------------------------------------------------------------------
class PatientMetricsTracker:
    """Patient-averaged segmentation metrics for a 2D-slice MRI dataset.

    HOW IT AGGREGATES (this is the rule the whole pipeline reports under):

      • Within a patient, for region metrics that decompose into pixel counts
        (dice, iou, sens, prec, spec, vs, mcc):
            sum TP/FP/FN/TN across all of the patient's slices,
            then apply the formula ONCE on the totals.
        → "patient-level volume Dice", treats the patient's slice stack as
          one 3D volume.

      • Within a patient, for spatial metrics that need per-image geometry
        (hd95, msd, nsd, biou):
            compute the metric per slice, then average those numbers within
            the patient (skipping NaN/Inf).

      • Across patients: simple mean of the per-patient numbers.
        → Each of the 19 test patients (or 20 val patients) counts EQUALLY,
          regardless of how many slices they have.

    Reports: dice, iou, sens, prec, spec, hd95, msd, nsd, vs, mcc, biou.
    """

    def __init__(self, nsd_tolerance: float = 2.0, biou_ratio: float = 0.02):
        self.tp, self.fp, self.fn, self.tn = (
            defaultdict(float), defaultdict(float),
            defaultdict(float), defaultdict(float),
        )
        self.hd95_vals = defaultdict(list)
        self.msd_vals = defaultdict(list)
        self.nsd_vals = defaultdict(list)
        self.biou_vals = defaultdict(list)
        # NSD tolerance in pixels (2 px default at 512x512).
        # Boundary-IoU dilation ratio (Cheng 2021 default = 0.02 of image diagonal).
        self.nsd_tolerance = nsd_tolerance
        self.biou_ratio = biou_ratio

    def reset(self):
        self.tp.clear(); self.fp.clear(); self.fn.clear(); self.tn.clear()
        self.hd95_vals.clear(); self.msd_vals.clear(); self.nsd_vals.clear()
        self.biou_vals.clear()
        self._last_per_patient = None  # cleared so a fresh compute() can repopulate

    @torch.no_grad()
    def update(self, preds, labels, patient_ids):
        # 1. Stateless functional boundary distances (no memory buffers)
        hd = compute_hausdorff_distance(y_pred=preds, y=labels, include_background=True, percentile=95)
        msd = compute_average_surface_distance(y_pred=preds, y=labels, include_background=True)
        nsd = compute_surface_dice(y_pred=preds, y=labels,
                                   class_thresholds=[self.nsd_tolerance],
                                   include_background=True)

        # 2. Binarize for pixel-level counting
        preds_bin = (preds > 0.5).int()
        labels_bin = (labels > 0.5).int()

        for i, pid in enumerate(patient_ids):
            p = preds_bin[i].flatten()
            l = labels_bin[i].flatten()

            self.tp[pid] += torch.sum((p == 1) & (l == 1)).item()
            self.fp[pid] += torch.sum((p == 1) & (l == 0)).item()
            self.fn[pid] += torch.sum((p == 0) & (l == 1)).item()
            self.tn[pid] += torch.sum((p == 0) & (l == 0)).item()

            if not torch.isnan(hd[i][0]) and not torch.isinf(hd[i][0]):
                self.hd95_vals[pid].append(hd[i][0].item())
            if not torch.isnan(msd[i][0]) and not torch.isinf(msd[i][0]):
                self.msd_vals[pid].append(msd[i][0].item())
            if not torch.isnan(nsd[i][0]) and not torch.isinf(nsd[i][0]):
                self.nsd_vals[pid].append(nsd[i][0].item())

            # Boundary IoU is per-slice (depends on the local boundary band);
            # we accumulate per-slice scores then average per patient in compute().
            p_np = preds_bin[i, 0].cpu().numpy().astype(bool)
            l_np = labels_bin[i, 0].cpu().numpy().astype(bool)
            self.biou_vals[pid].append(_boundary_iou(p_np, l_np, dilation_ratio=self.biou_ratio))

    def compute(self):
        if not self.tp:
            return {}
        res = {}
        for pid in self.tp.keys():
            tp, fp, fn, tn = self.tp[pid], self.fp[pid], self.fn[pid], self.tn[pid]

            denom_dice = (2.0 * tp + fp + fn)
            denom_iou = (tp + fp + fn)
            denom_sens = (tp + fn)
            denom_prec = (tp + fp)
            denom_spec = (tn + fp)

            dice = (2.0 * tp / denom_dice) if denom_dice > 0 else 1.0
            iou = (tp / denom_iou) if denom_iou > 0 else 1.0
            sens = (tp / denom_sens) if denom_sens > 0 else 1.0
            prec = (tp / denom_prec) if denom_prec > 0 else 1.0
            spec = (tn / denom_spec) if denom_spec > 0 else 1.0

            # Volumetric Similarity: 1 - |Vp - Vg| / (Vp + Vg)
            vp = tp + fp  # predicted volume
            vg = tp + fn  # ground-truth volume
            vs = 1.0 - abs(vp - vg) / (vp + vg) if (vp + vg) > 0 else 1.0

            # Matthews Correlation Coefficient (binary)
            mcc_num = float(tp) * float(tn) - float(fp) * float(fn)
            mcc_den = math.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
            mcc = (mcc_num / mcc_den) if mcc_den > 0 else 0.0

            hd95_mean = float(np.mean(self.hd95_vals[pid])) if self.hd95_vals[pid] else 0.0
            msd_mean = float(np.mean(self.msd_vals[pid])) if self.msd_vals[pid] else 0.0
            nsd_mean = float(np.mean(self.nsd_vals[pid])) if self.nsd_vals[pid] else 0.0
            biou_mean = float(np.mean(self.biou_vals[pid])) if self.biou_vals[pid] else 0.0

            res[pid] = {
                "dice": dice, "iou": iou, "sens": sens, "prec": prec, "spec": spec,
                "hd95": hd95_mean, "msd": msd_mean, "nsd": nsd_mean,
                "vs": vs, "mcc": mcc, "biou": biou_mean,
            }

        metrics = ["dice", "iou", "sens", "prec", "spec", "hd95", "msd", "nsd", "vs", "mcc", "biou"]
        avg_res = {k: float(np.mean([res[pid][k] for pid in res.keys()])) for k in metrics}
        # Stash per-patient values so callers can save them as per_patient_metrics.json
        # (used in test mode for downstream paired-Wilcoxon tests and boxplots).
        self._last_per_patient = res
        return avg_res

    def per_patient(self):
        """Return the per-patient metrics dict from the most recent compute() call.

        Format: {patient_id: {metric_name: value, ...}, ...}
        Returns None if compute() was never called since the last reset().
        """
        return getattr(self, "_last_per_patient", None)

# ---------------------------------------------------------------------------
# 4. Inference Savers (Visual Overlay + Raw Mask + optional uncertainty)
# ---------------------------------------------------------------------------
def save_inference_outputs(image, label, pred, patient_id, slice_id, out_dir,
                           var_map=None, mean_prob_map=None,
                           aleatoric_map=None, epistemic_map=None):
    """Saves visual overlays, error maps, raw binary masks, and (optionally)
    MC-dropout uncertainty maps + the per-pixel mean MC probability + the
    aleatoric/epistemic uncertainty decomposition.
    """
    out_dir = Path(out_dir)

    img_np = image[0, 0].cpu().numpy().T
    lbl_np = label[0, 0].cpu().numpy().T
    pred_np = pred[0, 0].cpu().numpy().T

    # --- 1. Save Raw Binary Mask ---
    raw_dir = out_dir / "inference_raw_masks" / patient_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_img = (pred_np * 255).astype(np.uint8)
    Image.fromarray(mask_img).save(raw_dir / f"{slice_id}_pred.png")

    # --- 1b. Optional per-pixel mean MC probability (for ECE/Brier) ---
    if mean_prob_map is not None:
        prob_dir = out_dir / "inference_mean_prob" / patient_id
        prob_dir.mkdir(parents=True, exist_ok=True)
        np.save(prob_dir / f"{slice_id}_prob.npy",
                mean_prob_map[0, 0].cpu().numpy().astype(np.float32))

    # --- 1c. Optional aleatoric / epistemic uncertainty maps ---
    if aleatoric_map is not None:
        aleat_dir = out_dir / "inference_aleatoric" / patient_id
        aleat_dir.mkdir(parents=True, exist_ok=True)
        np.save(aleat_dir / f"{slice_id}_aleat.npy",
                aleatoric_map[0, 0].cpu().numpy().astype(np.float32))
    if epistemic_map is not None:
        epist_dir = out_dir / "inference_epistemic" / patient_id
        epist_dir.mkdir(parents=True, exist_ok=True)
        np.save(epist_dir / f"{slice_id}_epist.npy",
                epistemic_map[0, 0].cpu().numpy().astype(np.float32))

    # --- 2. Color-Coded Error Map ---
    H, W = pred_np.shape
    error_map = np.zeros((H, W, 3), dtype=np.uint8)
    error_map[(pred_np == 1) & (lbl_np == 1)] = [0, 255, 0]   # TP -> Green
    error_map[(pred_np == 1) & (lbl_np == 0)] = [255, 0, 0]   # FP -> Red
    error_map[(pred_np == 0) & (lbl_np == 1)] = [0, 0, 255]   # FN -> Blue

    # --- 3. Slice-Level Metric Stamp ---
    tp = np.sum((pred_np == 1) & (lbl_np == 1))
    fp = np.sum((pred_np == 1) & (lbl_np == 0))
    fn = np.sum((pred_np == 0) & (lbl_np == 1))
    s_dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    s_sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    s_prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    title_stamp = (f"Patient: {patient_id} | Slice: {slice_id}\n"
                   f"Dice: {s_dice:.2f} | Sens: {s_sens:.2f} | Prec: {s_prec:.2f}")

    # --- 4. Plot 4-Panel (or 5-Panel) Figure ---
    overlay_dir = out_dir / "inference_overlays" / patient_id
    overlay_dir.mkdir(parents=True, exist_ok=True)

    num_panels = 5 if var_map is not None else 4
    fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4))
    fig.suptitle(title_stamp, fontsize=14, fontweight='bold')

    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("1. Input MRI")
    axes[0].axis("off")

    axes[1].imshow(img_np, cmap="gray")
    axes[1].imshow(lbl_np, cmap="Greens", alpha=0.4)
    axes[1].set_title("2. Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(img_np, cmap="gray")
    axes[2].imshow(pred_np, cmap="Reds", alpha=0.4)
    axes[2].set_title("3. Predicted Mask")
    axes[2].axis("off")

    axes[3].imshow(error_map)
    axes[3].set_title("4. Error Map (G=TP, R=FP, B=FN)")
    axes[3].axis("off")

    if var_map is not None:
        var_np = var_map[0, 0].numpy().T
        im = axes[4].imshow(var_np, cmap="jet")
        axes[4].set_title("5. Uncertainty (Variance)")
        axes[4].axis("off")
        fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(overlay_dir / f"{slice_id}_overlay.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Main Script
# ---------------------------------------------------------------------------
def main(args):
    set_determinism(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # ---- Compile auto-skip rules: AFA and DynUNet ----
    compile_active = args.compile
    if compile_active and (args.use_afa or args.network == "dynunet"):
        reason = []
        if args.use_afa:
            reason.append("--use_afa (dual-norm route switching)")
        if args.network == "dynunet":
            reason.append("--network dynunet (heavy graph specialization)")
        print(f"WARNING: --compile requested but skipped because of: {', '.join(reason)}")
        compile_active = False

    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    splits = read_splits(args.dataset_root + "/splits.json")

    pre_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader="PILReader"),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        Lambdad(keys=["label"], func=lambda x: (x > 0).astype(np.float32)),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])

    # --- Augmentation level switch ---
    # Two levels:
    #   none    → no augmentation (raw 512x512)
    #   regular → original v8 MRI recipe (flip / affine / elastic / zoom / contrast / noise / smooth)
    #
    # The "advanced" tier emerges from regular + at least one of
    # --use_mixup / --use_cutmix / --use_afa.
    #
    # Legacy --use_augmentation: maps to 'regular' if --aug_level is unset.
    regular_transforms = [
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        # ~17 degrees of rotation, modest translate + scale
        RandAffined(keys=["image", "label"], prob=0.5,
                    rotate_range=0.3, translate_range=15, scale_range=0.15),
        # Elastic deformation (gold standard for soft tissue like placentas)
        monai.transforms.Rand2DElasticd(keys=["image", "label"], prob=0.3,
                                        spacing=(20, 20), magnitude_range=(1, 2)),
        RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=0.90, max_zoom=1.10),
        RandAdjustContrastd(keys=["image"], prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.15),
        RandGaussianSmoothd(keys=["image"], prob=0.1),
    ]

    if args.aug_level == "none":
        aug_transforms = Compose([])
        aug_level_used = "none"
    elif args.aug_level == "regular":
        aug_transforms = Compose(regular_transforms)
        aug_level_used = "regular"
    else:
        # No explicit --aug_level: fall back to legacy --use_augmentation.
        if args.use_augmentation:
            aug_transforms = Compose(regular_transforms)
            aug_level_used = "regular (legacy --use_augmentation)"
        else:
            aug_transforms = Compose([])
            aug_level_used = "none (legacy, --use_augmentation not set)"

    n_transforms = len(aug_transforms.transforms) if hasattr(aug_transforms, "transforms") else 0
    print(f"Augmentation level: {aug_level_used} — {n_transforms} transforms applied")

    # --- AFA sanity check ---
    if args.use_afa and n_transforms == 0:
        print("WARNING: --use_afa is on but the main-route augmentation pipeline is empty. "
              "The AFA paper composes AFA in the auxiliary branch on top of a strong "
              "main-branch augmentation recipe. Consider --aug_level regular.")

    # --- CROSS-VALIDATION LOGIC ---
    if args.cv_folds > 0:
        all_patients = sorted(splits["train"] + splits["val"])
        rng = np.random.default_rng(args.seed)
        rng.shuffle(all_patients)
        # If cv_folds==1 we still slice into 5 pieces to keep a ~20% val portion.
        split_pieces = 5 if args.cv_folds == 1 else args.cv_folds
        patient_folds = np.array_split(all_patients, split_pieces)
        num_runs = args.cv_folds
    else:
        num_runs = 1
        patient_folds = None

    # ==========================================
    # MODE: TRAIN
    # ==========================================
    if args.mode == "train":
        for fold in range(num_runs):
            if args.cv_folds > 0:
                print(f"\n{'='*50}\nSTARTING RANDOM SPLIT: FOLD {fold}/{max(0, args.cv_folds - 1)}\n{'='*50}")
                current_out_dir = base_out_dir / f"fold_{fold}"
                val_ids = patient_folds[fold].tolist()
                train_ids = [p for i, f in enumerate(patient_folds) if i != fold for p in f.tolist()]
            else:
                print(f"\n{'='*50}\nSTARTING CLASSIC TRAIN/VAL SPLIT\n{'='*50}")
                current_out_dir = base_out_dir
                train_ids = splits["train"]
                val_ids = splits["val"]

            current_out_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(current_out_dir / "logs"))

            # --- 1. INITIALIZE NETWORK ---
            net = get_network(args.network, dropout=args.dropout).to(device)

            # --- 2. DUAL NORMALIZATION FOR AFA ---
            # Must happen BEFORE any transfer-learning load: an AFA checkpoint
            # carries dual-norm keys (norm1/2/3), so the architecture has to be
            # converted first or load_state_dict fails with "Missing key(s)".
            if args.use_afa:
                print("Converting normalization layers to dual-path for AFA...")
                net = convert_to_dual_norm(net)
                net = net.to(device)

            # Transfer-learning warm start (AFTER dual-norm conversion so an AFA
            # checkpoint's dual-norm weights match the network exactly)
            if args.checkpoint and Path(args.checkpoint).exists():
                print(f"TRANSFER LEARNING: Loading pre-trained weights from {args.checkpoint}")
                state_dict = torch.load(args.checkpoint, map_location=device)
                clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
                net.load_state_dict(clean_state_dict)

            if compile_active:
                print("Compiling network with torch.compile() (this may take 1-3 minutes)...")
                net = torch.compile(net)

            loss_function = DiceCELoss(sigmoid=True)
            optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

            # --- 3. DATA-AGNOSTIC AUGMENTATIONS ---
            mixup_aug = RandomMixUpBinary(p=1.0, alpha=args.mixup_alpha) if args.use_mixup else None
            cutmix_aug = RandomCutMixBinary(p=1.0, alpha=args.cutmix_alpha) if args.use_cutmix else None
            afa_aug = AFA(min_str=args.afa_min_str, mean_str=args.afa_mean_str) if args.use_afa else None
            if mixup_aug is not None:
                print(f"MixUp enabled (alpha={args.mixup_alpha})")
            if cutmix_aug is not None:
                print(f"CutMix enabled (alpha={args.cutmix_alpha})")
            if afa_aug is not None:
                print(f"AFA enabled (min_str={args.afa_min_str}, mean_str={args.afa_mean_str})")
            if mixup_aug is not None and cutmix_aug is not None:
                print(f"MixUp + CutMix both enabled: 50/50 one-of per batch "
                      f"(p_mixup={args.mixup_prob}).")

            if args.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            elif args.scheduler == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
            else:
                scheduler = None

            metrics_tracker = PatientMetricsTracker(nsd_tolerance=args.nsd_tolerance,
                                                    biou_ratio=args.biou_ratio)

            train_items = build_items(args.dataset_root + "/images", args.dataset_root + "/masks", train_ids)
            val_items = build_items(args.dataset_root + "/images", args.dataset_root + "/masks", val_ids)

            print("Caching datasets (this makes training super fast!)...")
            train_cached_ds = CacheDataset(data=train_items, transform=pre_transforms, cache_rate=1.0)
            val_cached_ds = CacheDataset(data=val_items, transform=pre_transforms, cache_rate=1.0)
            train_ds = Dataset(data=train_cached_ds, transform=aug_transforms)

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_cached_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            best_val_patient_dice = 0.0
            epochs_no_improve = 0
            start_epoch = 1   # default: train from scratch

            # --- RESUME from last_checkpoint.pth if --resume is set ---
            # last_checkpoint.pth is written at the END of every epoch and
            # contains: model + optimizer + scheduler + scaler + epoch counter
            # + best_val_dice + epochs_no_improve. It's separate from
            # best_model.pth (which only saves on improvement).
            last_ckpt_path = current_out_dir / "last_checkpoint.pth"
            if args.resume and last_ckpt_path.exists():
                print(f"RESUME: loading {last_ckpt_path}")
                ck = torch.load(str(last_ckpt_path), map_location=device, weights_only=False)
                # Model: strip the torch.compile wrapper prefix if present
                msd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
                net.load_state_dict(msd)
                optimizer.load_state_dict(ck["optimizer"])
                if scheduler is not None and ck.get("scheduler") is not None:
                    scheduler.load_state_dict(ck["scheduler"])
                if "scaler" in ck and ck["scaler"] is not None:
                    try:
                        scaler.load_state_dict(ck["scaler"])
                    except Exception:
                        pass  # scaler shape can differ if --amp toggled mid-run
                start_epoch = int(ck.get("epoch", 0)) + 1
                best_val_patient_dice = float(ck.get("best_val_patient_dice", 0.0))
                epochs_no_improve = int(ck.get("epochs_no_improve", 0))
                print(f"   → resuming at epoch {start_epoch}  (best Dice so far: "
                      f"{best_val_patient_dice:.4f}, epochs_no_improve={epochs_no_improve})")
            elif args.resume:
                print(f"RESUME requested but no checkpoint at {last_ckpt_path} — training from scratch.")

            for epoch in range(start_epoch, args.epochs + 1):
                print(f"\n--- Fold {fold} | Epoch {epoch}/{args.epochs} ---")
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current Learning Rate: {current_lr}")

                # --- TRAIN ---
                net.train()
                if args.use_afa:
                    set_dual_norm_route(net, 'M')
                epoch_loss = 0
                for batch in tqdm(train_loader, desc="Training", leave=False):
                    images, labels = batch["image"].to(device), batch["label"].to(device)

                    # --- Batch-level label-mixing augmentations ---
                    # When BOTH MixUp and CutMix are enabled, flip a coin per batch
                    # and apply exactly one (50/50 by default). Matches the AFA paper's
                    # one-of-per-batch composition (Appx. G, 3-way table).
                    if mixup_aug is not None and cutmix_aug is not None:
                        if torch.rand(1).item() < args.mixup_prob:
                            images, labels = mixup_aug(images, labels)
                        else:
                            images, labels = cutmix_aug(images, labels)
                    elif mixup_aug is not None:
                        images, labels = mixup_aug(images, labels)
                    elif cutmix_aug is not None:
                        images, labels = cutmix_aug(images, labels)

                    optimizer.zero_grad()

                    if afa_aug is not None:
                        # === AFA DUAL-PATH (memory-efficient sequential backward) ===
                        # Backward each path immediately so the activations are freed.
                        with torch.amp.autocast('cuda', enabled=args.amp):
                            set_dual_norm_route(net, 'M')
                            outputs = net(images)
                            if isinstance(outputs, (list, tuple)):
                                outputs = outputs[0]
                            loss_clean = loss_function(outputs, labels)
                        scaler.scale(loss_clean * 0.5).backward()

                        with torch.amp.autocast('cuda', enabled=args.amp):
                            set_dual_norm_route(net, 'A')
                            outputs_afa = net(afa_aug(images))
                            if isinstance(outputs_afa, (list, tuple)):
                                outputs_afa = outputs_afa[0]
                            loss_afa = loss_function(outputs_afa, labels)
                        scaler.scale(loss_afa * 0.5).backward()

                        set_dual_norm_route(net, 'M')
                        loss = 0.5 * (loss_clean.detach() + loss_afa.detach())
                    else:
                        with torch.amp.autocast('cuda', enabled=args.amp):
                            outputs = net(images)
                            if isinstance(outputs, (list, tuple)):
                                outputs = outputs[0]
                            loss = loss_function(outputs, labels)
                        scaler.scale(loss).backward()

                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += loss.item()

                epoch_loss /= len(train_loader)
                writer.add_scalar("Train/Loss", epoch_loss, epoch)

                # --- VALIDATION ---
                net.eval()
                if args.use_afa:
                    set_dual_norm_route(net, 'M')
                metrics_tracker.reset()
                val_loss = 0

                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validating", leave=False):
                        images, labels, pids = batch["image"].to(device), batch["label"].to(device), batch["patient_id"]

                        with torch.amp.autocast('cuda', enabled=args.amp):
                            outputs = net(images)
                            if isinstance(outputs, (list, tuple)):
                                outputs = outputs[0]
                            loss = loss_function(outputs, labels)
                        val_loss += loss.item()

                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        metrics_tracker.update(preds, labels, pids)

                        del images, labels, outputs, loss, preds

                val_loss /= len(val_loader)
                val_metrics = metrics_tracker.compute()
                val_patient_dice = val_metrics.get("dice", 0.0)

                print(f"Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"Metrics -> Dice: {val_patient_dice:.4f} | IoU: {val_metrics.get('iou', 0.0):.4f} | "
                      f"Sens: {val_metrics.get('sens', 0.0):.4f} | Prec: {val_metrics.get('prec', 0.0):.4f}")
                print(f"Boundary-> HD95: {val_metrics.get('hd95', 0.0):.2f} px | "
                      f"MSD: {val_metrics.get('msd', 0.0):.2f} px | "
                      f"NSD: {val_metrics.get('nsd', 0.0):.4f} | "
                      f"B-IoU: {val_metrics.get('biou', 0.0):.4f}")
                print(f"Other   -> VS: {val_metrics.get('vs', 0.0):.4f} | MCC: {val_metrics.get('mcc', 0.0):.4f}")

                writer.add_scalar("Val_Loss/Loss", val_loss, epoch)
                writer.add_scalar("Val_Metrics/Patient_Dice", val_patient_dice, epoch)
                writer.add_scalar("Val_Metrics/Patient_IoU", val_metrics.get("iou", 0.0), epoch)
                writer.add_scalar("Val_Metrics/Patient_Sensitivity", val_metrics.get("sens", 0.0), epoch)
                writer.add_scalar("Val_Metrics/Patient_Precision", val_metrics.get("prec", 0.0), epoch)
                writer.add_scalar("Val_Metrics/Patient_Specificity", val_metrics.get("spec", 0.0), epoch)
                writer.add_scalar("Val_Metrics/Patient_NSD", val_metrics.get("nsd", 0.0), epoch)
                writer.add_scalar("Val_Metrics/Patient_VS", val_metrics.get("vs", 0.0), epoch)
                writer.add_scalar("Val_Metrics/Patient_MCC", val_metrics.get("mcc", 0.0), epoch)
                writer.add_scalar("Val_Metrics/Patient_Boundary_IoU", val_metrics.get("biou", 0.0), epoch)
                writer.add_scalar("Val_Distances/Patient_HD95_px", val_metrics.get("hd95", 0.0), epoch)
                writer.add_scalar("Val_Distances/Patient_MSD_px", val_metrics.get("msd", 0.0), epoch)

                # --- CHECKPOINTING & EARLY STOPPING ---
                if val_patient_dice > best_val_patient_dice:
                    best_val_patient_dice = val_patient_dice
                    epochs_no_improve = 0
                    torch.save(net.state_dict(), current_out_dir / "best_model.pth")
                    print("New best model saved!")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve} epochs.")

                if args.scheduler == "cosine":
                    scheduler.step()
                elif args.scheduler == "plateau":
                    scheduler.step(val_patient_dice)

                # --- RESUME-ABLE CHECKPOINT (overwritten every epoch) ---
                # Saves AFTER the scheduler step so resuming continues with the
                # correct LR schedule. Distinct from best_model.pth (which only
                # saves on val-dice improvement); this one saves every epoch.
                torch.save({
                    "epoch": epoch,
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict(),
                    "best_val_patient_dice": best_val_patient_dice,
                    "epochs_no_improve": epochs_no_improve,
                    "args": vars(args),
                }, current_out_dir / "last_checkpoint.pth")

                if args.early_stopping and epochs_no_improve >= args.patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

                gc.collect()
                torch.cuda.empty_cache()

            print(f"\nFold {fold} training complete!")

            del net, optimizer, scheduler, train_loader, val_loader, train_cached_ds, val_cached_ds
            gc.collect()
            torch.cuda.empty_cache()

    # ==========================================
    # MODE: TEST / INFERENCE
    # ==========================================
    elif args.mode == "test":
        print("\n--- Inference / Testing Phase ---")

        for fold in range(num_runs):
            if args.cv_folds > 0:
                print(f"\n{'='*40}\nTESTING FOLD {fold}\n{'='*40}")
                current_out_dir = base_out_dir / f"fold_{fold}"
            else:
                current_out_dir = base_out_dir

            net = get_network(args.network, dropout=args.dropout).to(device)

            if args.use_afa:
                print("Converting normalization layers to dual-path for AFA-trained model...")
                net = convert_to_dual_norm(net)
                net = net.to(device)

            ckpt_path = args.checkpoint if args.checkpoint else current_out_dir / "best_model.pth"
            if not Path(ckpt_path).exists():
                print(f"WARNING: Cannot find checkpoint at {ckpt_path}. Skipping Fold {fold}.")
                continue

            state_dict = torch.load(ckpt_path, map_location=device)
            clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            net.load_state_dict(clean_state_dict)
            net.eval()
            if args.use_afa:
                set_dual_norm_route(net, 'M')

            # --- MC Dropout: keep dropout layers in train() mode for stochastic inference ---
            if args.mc_dropout:
                n_dropout = 0
                n_active = 0  # layers with p > 0 — only these produce real stochasticity
                for m in net.modules():
                    if m.__class__.__name__.startswith('Dropout'):
                        m.train()
                        n_dropout += 1
                        p = getattr(m, 'p', 0.0)
                        if p and p > 0:
                            n_active += 1
                print(f"MC Dropout: {args.mc_passes} passes per image, "
                      f"{n_dropout} dropout layer(s) found, {n_active} with p>0.")
                if n_active == 0:
                    print("WARNING: --mc_dropout requested but every Dropout layer has p=0 — "
                          "all MC passes will be identical. Retrain with --dropout > 0 "
                          "(or use --network dynunet which has baked-in dropout=0.1).")

            test_items = build_items(args.dataset_root + "/images", args.dataset_root + "/masks", splits["test"])
            test_cached_ds = CacheDataset(data=test_items, transform=pre_transforms, cache_rate=1.0)
            test_loader = DataLoader(test_cached_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            metrics_tracker = PatientMetricsTracker(nsd_tolerance=args.nsd_tolerance,
                                                    biou_ratio=args.biou_ratio)

            # === PHASE 1: GPU INFERENCE (batched) ===
            print(f"\nPhase 1: GPU inference for {len(test_items)} slices (batch_size={args.batch_size})...")
            all_results = []
            slice_metrics_list = []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc="GPU Inference"):
                    images, labels = batch["image"].to(device), batch["label"].to(device)
                    pids, slice_ids = batch["patient_id"], batch["slice_id"]

                    if args.mc_dropout:
                        preds_list = []
                        with torch.amp.autocast('cuda', enabled=args.amp):
                            for _ in range(args.mc_passes):
                                outputs = net(images)
                                if isinstance(outputs, (list, tuple)):
                                    outputs = outputs[0]
                                preds_list.append(torch.sigmoid(outputs))

                        preds_tensor = torch.stack(preds_list)  # [T, B, C, H, W]
                        mean_preds = preds_tensor.mean(dim=0)
                        var_preds = preds_tensor.var(dim=0)

                        # Kendall & Gal 2017 — aleatoric/epistemic decomposition.
                        eps = 1e-7
                        per_pass_entropy = -(
                            preds_tensor * torch.log(preds_tensor.clamp(min=eps)) +
                            (1.0 - preds_tensor) * torch.log((1.0 - preds_tensor).clamp(min=eps))
                        )
                        aleatoric_map = per_pass_entropy.mean(dim=0)
                        total_entropy = -(
                            mean_preds * torch.log(mean_preds.clamp(min=eps)) +
                            (1.0 - mean_preds) * torch.log((1.0 - mean_preds).clamp(min=eps))
                        )
                        epistemic_map = (total_entropy - aleatoric_map).clamp(min=0.0)

                        preds = (mean_preds > 0.5).float()
                    else:
                        with torch.amp.autocast('cuda', enabled=args.amp):
                            outputs = net(images)
                            if isinstance(outputs, (list, tuple)):
                                outputs = outputs[0]
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        mean_preds = None
                        var_preds = None
                        aleatoric_map = None
                        epistemic_map = None

                    metrics_tracker.update(preds, labels, pids)

                    # Slice-level metrics for CSV
                    hd = compute_hausdorff_distance(y_pred=preds, y=labels, include_background=True, percentile=95)
                    msd = compute_average_surface_distance(y_pred=preds, y=labels, include_background=True)
                    nsd = compute_surface_dice(y_pred=preds, y=labels,
                                               class_thresholds=[args.nsd_tolerance],
                                               include_background=True)
                    preds_bin = (preds > 0.5).int()
                    labels_bin = (labels > 0.5).int()

                    for i in range(images.shape[0]):
                        pid = pids[i]
                        sid = slice_ids[i]
                        p = preds_bin[i].flatten()
                        l = labels_bin[i].flatten()

                        tp_v = torch.sum((p == 1) & (l == 1)).item()
                        fp_v = torch.sum((p == 1) & (l == 0)).item()
                        fn_v = torch.sum((p == 0) & (l == 1)).item()
                        tn_v = torch.sum((p == 0) & (l == 0)).item()

                        dice_v = (2.0 * tp_v) / (2.0 * tp_v + fp_v + fn_v) if (2.0 * tp_v + fp_v + fn_v) > 0 else 1.0
                        iou_v = tp_v / (tp_v + fp_v + fn_v) if (tp_v + fp_v + fn_v) > 0 else 1.0
                        sens_v = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 1.0
                        prec_v = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 1.0
                        spec_v = tn_v / (tn_v + fp_v) if (tn_v + fp_v) > 0 else 1.0

                        vp_v = tp_v + fp_v
                        vg_v = tp_v + fn_v
                        vs_v = 1.0 - abs(vp_v - vg_v) / (vp_v + vg_v) if (vp_v + vg_v) > 0 else 1.0

                        mcc_num = float(tp_v) * float(tn_v) - float(fp_v) * float(fn_v)
                        mcc_den = math.sqrt(float(tp_v + fp_v) * float(tp_v + fn_v) * float(tn_v + fp_v) * float(tn_v + fn_v))
                        mcc_v = (mcc_num / mcc_den) if mcc_den > 0 else 0.0

                        hd_val = hd[i][0].item() if not torch.isnan(hd[i][0]) and not torch.isinf(hd[i][0]) else np.nan
                        msd_val = msd[i][0].item() if not torch.isnan(msd[i][0]) and not torch.isinf(msd[i][0]) else np.nan
                        nsd_val = nsd[i][0].item() if not torch.isnan(nsd[i][0]) and not torch.isinf(nsd[i][0]) else np.nan

                        # Slice-level Boundary IoU (Cheng 2021)
                        p_np = preds_bin[i, 0].cpu().numpy().astype(bool)
                        l_np = labels_bin[i, 0].cpu().numpy().astype(bool)
                        biou_val = _boundary_iou(p_np, l_np, dilation_ratio=args.biou_ratio)

                        unc_val = var_preds[i].mean().item() if var_preds is not None else np.nan
                        aleat_val = aleatoric_map[i].mean().item() if aleatoric_map is not None else np.nan
                        epist_val = epistemic_map[i].mean().item() if epistemic_map is not None else np.nan

                        slice_metrics_list.append({
                            "patient_id": pid,
                            "slice_id": sid,
                            "dice": dice_v,
                            "iou": iou_v,
                            "sens": sens_v,
                            "prec": prec_v,
                            "spec": spec_v,
                            "hd95": hd_val,
                            "msd": msd_val,
                            "nsd": nsd_val,
                            "vs": vs_v,
                            "mcc": mcc_v,
                            "biou": biou_val,
                            "mean_uncertainty": unc_val,
                            "mean_aleatoric": aleat_val,
                            "mean_epistemic": epist_val,
                        })

                        save_prob = args.save_mc_prob and (var_preds is not None)

                        all_results.append({
                            "image": images[i:i+1].cpu(),
                            "label": labels[i:i+1].cpu(),
                            "pred": preds[i:i+1].cpu(),
                            "var_map": var_preds[i:i+1].cpu() if var_preds is not None else None,
                            "mean_prob": mean_preds[i:i+1].cpu() if save_prob else None,
                            "aleatoric_map": aleatoric_map[i:i+1].cpu() if (save_prob and aleatoric_map is not None) else None,
                            "epistemic_map": epistemic_map[i:i+1].cpu() if (save_prob and epistemic_map is not None) else None,
                            "pid": pid,
                            "sid": sid,
                        })

            # === PHASE 2: SAVE OVERLAYS (parallel CPU) ===
            n_save_workers = min(8, os.cpu_count() or 4)
            if _JOBLIB_AVAILABLE and len(all_results) > 1:
                print(f"Phase 2: Saving {len(all_results)} overlays in parallel ({n_save_workers} workers)...")
                Parallel(n_jobs=n_save_workers)(
                    delayed(save_inference_outputs)(
                        r["image"], r["label"], r["pred"], r["pid"], r["sid"], current_out_dir,
                        r["var_map"], r.get("mean_prob"),
                        r.get("aleatoric_map"), r.get("epistemic_map")
                    ) for r in tqdm(all_results, desc="Saving Overlays")
                )
            else:
                print(f"Phase 2: Saving {len(all_results)} overlays (sequential)...")
                for r in tqdm(all_results, desc="Saving Overlays"):
                    save_inference_outputs(r["image"], r["label"], r["pred"], r["pid"], r["sid"], current_out_dir,
                                           r["var_map"], r.get("mean_prob"),
                                           r.get("aleatoric_map"), r.get("epistemic_map"))

            final_metrics = metrics_tracker.compute()

            print(f"\nFinal Test Metrics for Fold {fold} (Patient-Averaged):")
            for k, v in final_metrics.items():
                print(f" - {k.upper()}: {v:.4f}")

            with open(current_out_dir / "test_metrics.json", "w") as f:
                json.dump(final_metrics, f, indent=4)

            # --- Per-patient JSON (consumed by compare_test_metrics.py for
            #     paired Wilcoxon tests and boxplots) ---
            per_patient = metrics_tracker.per_patient() or {}
            with open(current_out_dir / "per_patient_metrics.json", "w") as f:
                json.dump(per_patient, f, indent=4)

            csv_path = current_out_dir / "slice_metrics.csv"
            with open(csv_path, mode='w', newline='') as f:
                fieldnames = [
                    "patient_id", "slice_id",
                    "dice", "iou", "sens", "prec", "spec",
                    "hd95", "msd", "nsd", "vs", "mcc", "biou",
                    "mean_uncertainty", "mean_aleatoric", "mean_epistemic",
                ]
                writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
                writer_csv.writeheader()
                for row in slice_metrics_list:
                    writer_csv.writerow(row)

            print(f"\nAll metrics saved to {current_out_dir / 'test_metrics.json'}")
            print(f"Detailed slice metrics saved to {current_out_dir / 'slice_metrics.csv'}")
            print(f"Raw binary masks saved in {current_out_dir / 'inference_raw_masks'}")
            print(f"Visual overlays saved in {current_out_dir / 'inference_overlays'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to your dataset folder")
    parser.add_argument("--out_dir", type=str, default="./runs_placenta", help="Where to save logs/models")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Run mode")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model weights (for test mode)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from <out_dir>/last_checkpoint.pth if present. "
                             "Restores model, optimizer, scheduler, scaler, epoch, best Dice and "
                             "early-stopping counter. If no checkpoint exists, trains from scratch.")
    parser.add_argument("--network", type=str, default="unet", choices=[
        "unet", "attentionunet", "basicunet", "unetplusplus", "flexunet",
        "segresnet", "vnet", "highresnet", "unetr", "swinunetr", "dynunet"
    ])

    parser.add_argument("--cv_folds", type=int, default=0,
                        help="0=Classic Split, 1=Single Random 80/20 Split, >1=Full CV")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile() for ~20% faster training. "
                             "Auto-skipped for --network dynunet and for --use_afa.")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"])
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")

    # --- Augmentation level (Change C) ---
    parser.add_argument("--use_augmentation", action="store_true",
                        help="Legacy flag — enables the regular augmentation pipeline. "
                             "Ignored when --aug_level is set.")
    parser.add_argument("--aug_level", type=str, default=None,
                        choices=[None, "none", "regular"],
                        help="Explicit augmentation level. "
                             "'none' = no augmentation. 'regular' = flip/affine/elastic/zoom/contrast/noise/smooth (v8 values). "
                             "'advanced' is implicit: 'regular' + at least one of --use_mixup / --use_cutmix / --use_afa.")

    # --- Dropout (Change B) ---
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout probability for networks that support it. "
                             "Default 0.0 = OFF. DynUNet keeps its tuned 0.1 unless overridden.")

    # --- MC Dropout (Change F) ---
    parser.add_argument("--mc_dropout", action="store_true",
                        help="Enable MC Dropout during testing (forces Dropout layers into train mode).")
    parser.add_argument("--mc_passes", type=int, default=20,
                        help="Number of forward passes for MC Dropout.")
    parser.add_argument("--save_mc_prob", action="store_true",
                        help="Save per-pixel mean MC probability + aleatoric/epistemic .npy maps "
                             "(only effective when --mc_dropout is enabled).")

    # --- NSD tolerance + Boundary-IoU dilation (Change E) ---
    parser.add_argument("--nsd_tolerance", type=float, default=2.0,
                        help="Surface tolerance (in pixels) for Normalized Surface Dice (NSD).")
    parser.add_argument("--biou_ratio", type=float, default=0.02,
                        help="Boundary IoU band thickness as a fraction of the image diagonal "
                             "(Cheng et al. CVPR 2021, default 0.02 ~= 14 px at 512x512).")

    # --- Data-Agnostic Augmentations (Paper: arXiv 2505.10223) ---
    # MixUp and CutMix can be used independently OR together (50/50 one-of per batch).
    parser.add_argument("--use_mixup", action="store_true",
                        help="Enable MixUp augmentation (blends pairs of images and masks).")
    parser.add_argument("--use_cutmix", action="store_true",
                        help="Enable CutMix augmentation (pastes rectangular patches between samples).")
    parser.add_argument("--use_afa", action="store_true",
                        help="Enable Auxiliary Fourier Augmentation with dual batch norm. "
                             "Can be combined with --use_mixup and/or --use_cutmix. "
                             "IMPORTANT: If you trained with --use_afa, you MUST also pass it during --mode test.")
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
                        help="MixUp Beta distribution parameter. Smaller=less mixing. Paper default: 0.2.")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0,
                        help="CutMix Beta distribution parameter. Controls cut region size. Default: 1.0.")
    parser.add_argument("--mixup_prob", type=float, default=0.5,
                        help="When BOTH --use_mixup and --use_cutmix are on, probability per batch "
                             "of choosing MixUp over CutMix. Default: 0.5 (50/50 one-of).")
    parser.add_argument("--afa_min_str", type=float, default=10.0,
                        help="AFA minimum perturbation strength. Paper default: 10.")
    parser.add_argument("--afa_mean_str", type=float, default=20.0,
                        help="AFA mean perturbation strength (exponential distribution). Paper default: 20.")

    args = parser.parse_args()
    main(args)

    '''
    python train_placenta_2d_monai_v8.py \
    --dataset_root "dataset/mri_png/DATASET_BTFE/" \
    --out_dir "./runs/BTFE_unetplusplus_1Fold" \
    --mode train \
    --network unetplusplus \
    --cv_folds 1 \
    --epochs 150 \
    --batch_size 8 \
    --amp \
    --compile \
    --scheduler plateau \
    --early_stopping \
    --patience 25 \
    --num_workers 8 \
    --aug_level regular
    '''

    '''
    python train_placenta_2d_monai_v8.py \
    --dataset_root "dataset/mri_png/DATASET_BTFE/" \
    --out_dir "./runs/BTFE_unetplusplus_1Fold" \
    --mode test \
    --network unetplusplus \
    --cv_folds 1 \
    --batch_size 8 \
    --amp \
    --num_workers 8
    '''
