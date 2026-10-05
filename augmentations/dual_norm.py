"""
Dual-path normalization layers for Auxiliary Fourier Augmentation (AFA).

From: "Augmentations for the Unknown" (arXiv: 2505.10223)

The key idea: when AFA perturbs images, the perturbation shifts the feature
statistics (mean/variance). If we run both clean and perturbed images through
the same batch norm, the statistics get contaminated. Instead, we maintain
TWO sets of normalization statistics:
    - Route 'M' (Main):      for clean / MixUp / CutMix images
    - Route 'A' (Auxiliary):  for AFA-perturbed images

During inference, only route 'M' is used.

Three normalization variants are provided:
    - DuBIN:                 Splits channels — half through InstanceNorm,
                             half through DualBatchNorm. Replaces BatchNorm2d.
    - DualBatchNorm2d:       Two separate BatchNorm2d (main + aux route).
    - DualInstanceNorm2d:    Two separate InstanceNorm2d (main + aux route).
                             For MONAI networks that use InstanceNorm (e.g.,
                             DynUNet, SegResNet, UNETR, SwinUNETR).

Network compatibility with MONAI:
    - Works well (has BatchNorm2d):
       UNet, AttentionUnet, BasicUNet, BasicUNetPlusPlus, FlexibleUNet,
       HighResNet, VNet

    - Uses InstanceNorm2d (converted via DualInstanceNorm2d):
       DynUNet, SegResNet

    - Uses LayerNorm / other (no conversion, AFA still runs but without
       dual statistics — less effective):
       UNETR, SwinUNETR
"""

import torch
import torch.nn as nn


class DualBatchNorm2d(nn.Module):
    """BatchNorm2d with two separate sets of statistics (main + auxiliary).

    Routing is controlled by the `route` attribute:
        'M' → main statistics (clean images, inference)
        'A' → auxiliary statistics (AFA-perturbed images)
    """

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(num_features),  # Route 'M' (Main)
            nn.BatchNorm2d(num_features),  # Route 'A' (Auxiliary)
        ])
        self.num_features = num_features
        self.route = 'M'

    def forward(self, x):
        idx = 0 if self.route == 'M' else 1
        return self.bn[idx](x)


class DualInstanceNorm2d(nn.Module):
    """InstanceNorm2d with two separate sets of parameters (main + auxiliary).

    For MONAI networks that use InstanceNorm instead of BatchNorm (e.g., DynUNet).
    """

    def __init__(self, num_features, affine=True):
        super().__init__()
        self.in_norm = nn.ModuleList([
            nn.InstanceNorm2d(num_features, affine=affine),  # Route 'M'
            nn.InstanceNorm2d(num_features, affine=affine),  # Route 'A'
        ])
        self.num_features = num_features
        self.route = 'M'

    def forward(self, x):
        idx = 0 if self.route == 'M' else 1
        return self.in_norm[idx](x)


class DuBIN(nn.Module):
    """Dual Instance-Batch Normalization (DuBIN).

    From "Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net".
    Splits channels: first half → InstanceNorm, second half → DualBatchNorm.

    This replaces standard BatchNorm2d layers in the network.
    """

    def __init__(self, planes):
        super().__init__()
        self.half = int(planes * 0.5)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = DualBatchNorm2d(planes - self.half)

    @property
    def route(self):
        return self.BN.route

    @route.setter
    def route(self, value):
        self.BN.route = value

    def forward(self, x):
        split = x.split(self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat((out1, out2), 1)


def convert_to_dual_norm(model):
    """Convert all normalization layers in a model to dual-path versions.

    This enables the AFA dual-statistics approach:
      - BatchNorm2d    → DuBIN (half InstanceNorm + half DualBatchNorm)
      - InstanceNorm2d → DualInstanceNorm2d (two separate InstanceNorm paths)
      - Other norm layers (LayerNorm, GroupNorm) are left unchanged.

    Args:
        model: nn.Module to convert (modified in-place and returned).

    Returns:
        The converted model.
    """
    converted_count = 0
    for name, module in dict(model.named_children()).items():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, DuBIN(module.num_features))
            converted_count += 1
        elif isinstance(module, nn.InstanceNorm2d):
            setattr(model, name, DualInstanceNorm2d(
                module.num_features, affine=module.affine
            ))
            converted_count += 1
        else:
            # Recurse into child modules
            child_count = convert_to_dual_norm(module)
            if isinstance(child_count, int):
                converted_count += child_count
    return model


def set_dual_norm_route(model, route: str):
    """Set the normalization route for all dual-norm layers in the model.

    Args:
        model: nn.Module (the network).
        route: 'M' for main (clean images / inference) or 'A' for auxiliary (AFA).
    """
    for m in model.modules():
        if hasattr(m, 'route'):
            m.route = route
