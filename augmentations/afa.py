"""
Auxiliary Fourier Augmentation (AFA) for 2D images.

From: "Augmentations for the Unknown" (arXiv: 2505.10223)

How it works:
    AFA injects random planar sinusoidal waves into the image. These waves
    simulate unknown frequency-domain perturbations (like different MRI
    acquisition protocols, hardware differences, etc.) WITHOUT changing the
    segmentation labels — the perturbation is image-only.

    During training, the model processes each batch twice:
      1. Clean/mixed images → through MAIN batch norm statistics
      2. AFA-perturbed images → through AUXILIARY batch norm statistics

    This dual-path approach (via DuBIN normalization) teaches the model to
    produce the same segmentation for both clean and perturbed inputs,
    improving out-of-distribution robustness.

    The grid coordinates are lazy-initialized and cached, so the first forward
    pass creates them for the input spatial size.
"""

import torch


class AFA(torch.nn.Module):
    """Auxiliary Fourier Augmentation for 2D images.

    Injects random sinusoidal waves into images to simulate unknown
    frequency-domain perturbations. Applied only to images, not labels.

    Args:
        min_str (float): Minimum strength of perturbation. Default: 10.
        mean_str (float): Mean strength (controls exponential distribution).
            Higher = stronger perturbation on average. Default: 20.
    """

    def __init__(self, min_str: float = 10, mean_str: float = 20):
        super().__init__()
        self.min_str = min_str
        self.mean_str = mean_str

        # Lazy-initialized coordinate grids (cached for performance)
        self._cached_size = None
        self._x = None
        self._y = None
        self.eps_scale = 1.0

    def _ensure_grid(self, h: int, w: int, device: torch.device):
        """Create or reuse coordinate grids matching the input spatial size."""
        if self._cached_size != (h, w):
            _x = torch.linspace(-h / 2, h / 2, steps=h, device=device)
            _y = torch.linspace(-w / 2, w / 2, steps=w, device=device)
            self._x, self._y = torch.meshgrid(_x, _y, indexing='ij')
            self._cached_size = (h, w)
            self.eps_scale = max(h, w) / 32

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) float tensor — the image batch.

        Returns:
            Perturbed image tensor (same shape). Labels are NOT modified.
        """
        init_shape = x.shape
        if len(x.shape) == 3:  # (C, H, W) → add batch dim
            x = x.unsqueeze(0)

        b, c, h, w = x.shape
        self._ensure_grid(h, w, x.device)

        # Random frequency, phase, and strength for each sample × channel
        freqs = 1 - torch.rand((b, c, 1, 1), device=x.device)
        phases = -torch.pi * torch.rand((b, c, 1, 1), device=x.device)
        strengths = torch.empty_like(phases).exponential_(1 / self.mean_str) + self.min_str

        # Generate planar sinusoidal waves
        waves = self._gen_planar_waves(freqs, phases, x.device)

        result = x + strengths * waves
        return result.reshape(init_shape)

    def _gen_planar_waves(self, freqs, phases, device):
        """Generate normalized planar sinusoidal waves."""
        _x, _y = self._x.to(device), self._y.to(device)
        _waves = torch.sin(
            2 * torch.pi * freqs * (
                _x * torch.cos(phases) + _y * torch.sin(phases)
            ) - torch.rand(1, device=device) * torch.pi
        )
        _waves.div_(_waves.norm(dim=(-2, -1), keepdim=True))
        return self.eps_scale * _waves

    def __repr__(self):
        return f"{self.__class__.__name__}(min_str={self.min_str}, mean_str={self.mean_str})"
