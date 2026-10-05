# Data-Agnostic Augmentations for OOD Generalization
# Adapted from: "Augmentations for the Unknown" (arXiv: 2505.10223)
# Simplified for binary (1-channel sigmoid) segmentation with MONAI

from .mixup_binary import RandomMixUpBinary
from .cutmix_binary import RandomCutMixBinary
from .afa import AFA
from .dual_norm import (
    DualBatchNorm2d,
    DualInstanceNorm2d,
    DuBIN,
    convert_to_dual_norm,
    set_dual_norm_route,
)

__all__ = [
    'RandomMixUpBinary',
    'RandomCutMixBinary',
    'AFA',
    'DualBatchNorm2d',
    'DualInstanceNorm2d',
    'DuBIN',
    'convert_to_dual_norm',
    'set_dual_norm_route',
]
