"""
f-AnoGAN architectures, PyTorch port.

Faithful to the original 64x64 grayscale WGAN-GP from Schlegl et al. 2019
(`wgangp_64x64.py`) and the izi_f encoder from `z_encoding_izif.py`. All
TensorFlow 1.x / `tflib` helpers are replaced with standard `torch.nn`
modules so this runs on modern PyTorch + Blackwell (RTX 5080).

Architecture summary:
    z ∈ R^128  ─G─→  x ∈ [-1, 1]^(1×64×64)  ─D─→  (logit, features ∈ R^8192)
                                  ↑                            │
                                  └──── G(E(x)) ←── z = E(x) ──┘

    Generator     : Linear → 4 ResBlock-up   → Conv → tanh         (uses BN)
    Discriminator : Conv   → 4 ResBlock-down → Linear              (uses LN)
    Encoder       : Conv   → 4 ResBlock-down → Linear → tanh-clip  (uses BN)

The discriminator uses LayerNorm (axes [C,H,W]) instead of BatchNorm — required
by the WGAN-GP gradient penalty (BN's running statistics would couple samples
inside the same batch and break the per-sample gradient norm). LayerNorm over
all spatial+channel dims per sample = nn.GroupNorm(1, C).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def layernorm_2d(num_channels: int) -> nn.Module:
    """LayerNorm over (C, H, W) per sample = GroupNorm with one group."""
    return nn.GroupNorm(num_groups=1, num_channels=num_channels, affine=True)


def upsample2x(x: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor 2× upsample, matching the original's depth_to_space trick."""
    return F.interpolate(x, scale_factor=2, mode="nearest")


# ---------------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------------
class ResBlockUp(nn.Module):
    """Residual block with 2× nearest-neighbor upsample.

    Path: BN1 → ReLU → up → 3×3 conv (in→out) → BN2 → ReLU → 3×3 conv (out→out)
    Shortcut: up → 1×1 conv (in→out)
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.shortcut_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shortcut
        sc = upsample2x(x)
        sc = self.shortcut_conv(sc)
        # main path
        h = F.relu(self.bn1(x))
        h = upsample2x(h)
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)
        return sc + h


class ResBlockDown(nn.Module):
    """Residual block with 2× average-pool downsample.

    Path: Norm1 → ReLU → 3×3 conv (in→in) → Norm2 → ReLU → 3×3 conv (in→out) → 2×2 avgpool
    Shortcut: 2×2 avgpool → 1×1 conv (in→out)

    Norm is BatchNorm in the Encoder, LayerNorm in the Discriminator (WGAN-GP
    requirement). Pass norm="bn" or norm="ln".
    """

    def __init__(self, in_ch: int, out_ch: int, norm: str = "bn"):
        super().__init__()
        if norm == "bn":
            self.norm1 = nn.BatchNorm2d(in_ch)
            self.norm2 = nn.BatchNorm2d(in_ch)
        elif norm == "ln":
            self.norm1 = layernorm_2d(in_ch)
            self.norm2 = layernorm_2d(in_ch)
        else:
            raise ValueError(f"norm must be 'bn' or 'ln', got {norm!r}")
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.pool = nn.AvgPool2d(2)
        self.shortcut_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = self.pool(x)
        sc = self.shortcut_conv(sc)
        h = F.relu(self.norm1(x))
        h = self.conv1(h)
        h = F.relu(self.norm2(h))
        h = self.conv2(h)
        h = self.pool(h)
        return sc + h


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    """G(z): z ∈ R^128 → x ∈ [-1, 1]^(1×64×64)."""

    def __init__(self, z_dim: int = 128, dim: int = 64, out_channels: int = 1):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim

        self.fc = nn.Linear(z_dim, 4 * 4 * 8 * dim)
        # 4 → 8 → 16 → 32 → 64 spatial
        self.res1 = ResBlockUp(8 * dim, 8 * dim)
        self.res2 = ResBlockUp(8 * dim, 4 * dim)
        self.res3 = ResBlockUp(4 * dim, 2 * dim)
        self.res4 = ResBlockUp(2 * dim, 1 * dim)
        self.bn_out = nn.BatchNorm2d(1 * dim)
        self.conv_out = nn.Conv2d(1 * dim, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), 8 * self.dim, 4, 4)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = F.relu(self.bn_out(h))
        h = self.conv_out(h)
        return torch.tanh(h)

    def sample_z(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Sample latent codes from N(0, I)."""
        return torch.randn(n, self.z_dim, device=device)


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------
class Discriminator(nn.Module):
    """D(x): x ∈ [-1, 1]^(1×64×64) → (logit, feature_vector ∈ R^(4·4·8·dim))."""

    def __init__(self, dim: int = 64, in_channels: int = 1):
        super().__init__()
        self.dim = dim

        self.conv_in = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        # 64 → 32 → 16 → 8 → 4
        self.res1 = ResBlockDown(1 * dim, 2 * dim, norm="ln")
        self.res2 = ResBlockDown(2 * dim, 4 * dim, norm="ln")
        self.res3 = ResBlockDown(4 * dim, 8 * dim, norm="ln")
        self.res4 = ResBlockDown(8 * dim, 8 * dim, norm="ln")
        self.fc = nn.Linear(4 * 4 * 8 * dim, 1)

    def forward(
        self, x: torch.Tensor, return_features: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        h = self.conv_in(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        feats = h.view(h.size(0), -1)        # (B, 4·4·8·dim) = (B, 8192) with dim=64
        logit = self.fc(feats).squeeze(-1)   # (B,)
        if return_features:
            return logit, feats
        return logit


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    """E(x): x ∈ [-1, 1]^(1×64×64) → z ∈ R^128 (regularized by tanh by default).

    z_reg options (from the paper / Appendix A):
        - 'tanh_fc'  : tanh(z)       (paper default — soft-clip to (-1, 1))
        - 'hard_clip': clamp(z, -1, 1)
        - None       : raw linear output (unconstrained)
    """

    def __init__(
        self,
        z_dim: int = 128,
        dim: int = 64,
        in_channels: int = 1,
        z_reg: str | None = "tanh_fc",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        self.z_reg = z_reg

        self.conv_in = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.res1 = ResBlockDown(1 * dim, 2 * dim, norm="bn")
        self.res2 = ResBlockDown(2 * dim, 4 * dim, norm="bn")
        self.res3 = ResBlockDown(4 * dim, 8 * dim, norm="bn")
        self.res4 = ResBlockDown(8 * dim, 8 * dim, norm="bn")
        self.fc = nn.Linear(4 * 4 * 8 * dim, z_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        if self.z_reg == "tanh_fc":
            z = torch.tanh(z)
        elif self.z_reg == "hard_clip":
            z = torch.clamp(z, -1.0, 1.0)
        elif self.z_reg is None:
            pass
        else:
            raise ValueError(f"Unknown z_reg: {self.z_reg!r}")
        return z


# ---------------------------------------------------------------------------
# Weight initialization (matches the original's He-init choice on convs)
# ---------------------------------------------------------------------------
def init_weights(model: nn.Module) -> None:
    """He-normal for Conv2d, default for Linear and norm layers."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
