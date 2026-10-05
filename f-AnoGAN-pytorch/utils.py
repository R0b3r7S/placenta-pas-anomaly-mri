"""
Shared utilities for the f-AnoGAN PyTorch port.

  • Gradient penalty for WGAN-GP
  • Image-grid saving (sample sheets)
  • Checkpoint save/load
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ---------------------------------------------------------------------------
# WGAN-GP gradient penalty (Gulrajani et al. 2017)
# ---------------------------------------------------------------------------
def gradient_penalty(
    D: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    """∥∇_x̃ D(x̃)∥₂ pulled to 1 along the line between real and fake."""
    bs = real.size(0)
    alpha = torch.rand(bs, 1, 1, 1, device=device, dtype=real.dtype)
    interp = (alpha * real + (1.0 - alpha) * fake).detach().requires_grad_(True)
    d_interp = D(interp, return_features=False)
    # ∂D/∂x̃
    grads = torch.autograd.grad(
        outputs=d_interp.sum(),
        inputs=interp,
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.view(bs, -1)
    return ((grads.norm(2, dim=1) - 1.0) ** 2).mean()


# ---------------------------------------------------------------------------
# Image grid saving
# ---------------------------------------------------------------------------
def _denorm(t: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]."""
    return (t.clamp(-1.0, 1.0) + 1.0) * 0.5


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int | None = None) -> None:
    """Save a B×1×H×W (or B×3×H×W) tensor of in-[-1,1] images to a PNG grid."""
    images = _denorm(images).detach().cpu()
    b, c, h, w = images.shape
    if nrow is None:
        nrow = int(math.sqrt(b))
        while b % nrow != 0 and nrow > 1:
            nrow -= 1
    ncol = max(1, b // max(1, nrow))
    if c == 1:
        grid = np.zeros((h * nrow, w * ncol), dtype=np.uint8)
    else:
        grid = np.zeros((h * nrow, w * ncol, 3), dtype=np.uint8)
    for k in range(min(b, nrow * ncol)):
        i, j = k // ncol, k % ncol
        arr = images[k].numpy()
        if c == 1:
            grid[i * h:(i + 1) * h, j * w:(j + 1) * w] = (arr[0] * 255).astype(np.uint8)
        else:
            grid[i * h:(i + 1) * h, j * w:(j + 1) * w] = (
                arr.transpose(1, 2, 0) * 255
            ).astype(np.uint8)
    Image.fromarray(grid).save(str(path))


def save_pair_row(real: torch.Tensor, recon: torch.Tensor, path: str | Path) -> None:
    """Save a row of (real | reconstruction) pairs for the encoder samples."""
    real = _denorm(real).detach().cpu()
    recon = _denorm(recon).detach().cpu()
    b, c, h, w = real.shape
    assert recon.shape == real.shape, "real and recon shapes must match"
    pair_w = 2 * w + 4   # 4px gutter
    out = np.zeros((h, pair_w * b), dtype=np.uint8)
    for k in range(b):
        a = (real[k, 0].numpy() * 255).astype(np.uint8)
        rg = (recon[k, 0].numpy() * 255).astype(np.uint8)
        x0 = k * pair_w
        out[:, x0:x0 + w] = a
        out[:, x0 + w + 4:x0 + 2 * w + 4] = rg
    Image.fromarray(out).save(str(path))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    """Save a checkpoint atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, str(tmp))
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(str(path), map_location=map_location, weights_only=False)
