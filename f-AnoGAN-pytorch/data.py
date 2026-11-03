"""
Patch datasets for f-AnoGAN training.

The original code expects a flat folder of 64×64 grayscale PNGs (`tflib/img_loader.py`).
We keep the same convention. For the placenta project, you'll generate patches
along the lower placental boundary from the segmentation pipeline (track 2),
then point this loader at that folder.

Two datasets:
    NormalPatchDataset  — folder of "normal" (no-PAS) patches, used for both
                          WGAN-GP training and Encoder training.
    AnomalyPatchDataset — folder of "anomalous" (PAS) patches, used only at
                          inference time to compute test anomaly scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _scan_patches(root: Path, exts: Iterable[str] = _IMG_EXTS) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in exts)


def _load_gray_minus_one_to_one(path: Path, size: int = 64) -> torch.Tensor:
    """Load a grayscale PNG, resize to size×size if needed, scale to [-1, 1]."""
    img = Image.open(path).convert("L")
    if img.size != (size, size):
        # Use bicubic for resize — patches are usually already at target size,
        # this is a safety net. Mask patches should never be passed here.
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0     # [0, 1]
    arr = arr * 2.0 - 1.0                                # [-1, 1]
    return torch.from_numpy(arr).unsqueeze(0)            # (1, H, W)


class NormalPatchDataset(Dataset):
    """Flat folder of 64×64 grayscale "normal" patches.

    Optionally applies horizontal flip with p=0.5 (matches the original
    `make_generator` in tflib/img_loader.py).
    """

    def __init__(self, root: str | Path, size: int = 64, hflip: bool = True):
        self.root = Path(root)
        self.size = size
        self.hflip = hflip
        self.files = _scan_patches(self.root)
        if not self.files:
            raise FileNotFoundError(f"No image files under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = _load_gray_minus_one_to_one(self.files[idx], size=self.size)
        if self.hflip and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-1])
        return x


class AnomalyPatchDataset(Dataset):
    """Flat folder of test patches. Returns (patch, path_str).

    The path_str is kept so scores can be written back per file.
    """

    def __init__(self, root: str | Path, size: int = 64):
        self.root = Path(root)
        self.size = size
        self.files = _scan_patches(self.root)
        if not self.files:
            raise FileNotFoundError(f"No image files under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        p = self.files[idx]
        x = _load_gray_minus_one_to_one(p, size=self.size)
        return x, str(p)
