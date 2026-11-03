"""
CutMix augmentation simplified for binary (1-channel) 2D segmentation.

Original paper: "CutMix: Regularization Strategy to Train Strong Classifiers
with Localizable Features" (https://arxiv.org/abs/1905.04899)
Adapted from the nnU-Net implementation in this repo for use with MONAI pipelines.

How it works:
    A random rectangular region is selected. That region in image_A is replaced
    with the corresponding region from image_B (rolled batch). The same swap
    is applied to the segmentation mask — this is clean spatial label swapping,
    no need for one-hot encoding in binary segmentation.
"""

import math

import torch


class RandomCutMixBinary(torch.nn.Module):
    """CutMix augmentation for binary 2D segmentation (1-channel masks).

    Applied at batch level inside the training loop, NOT as a MONAI transform.

    Args:
        p (float): Probability of applying CutMix to the batch. Default: 1.0.
        alpha (float): Hyperparameter of the Beta distribution controlling
            the size of the cut region. Default: 1.0.
    """

    def __init__(self, p: float = 1.0, alpha: float = 1.0):
        super().__init__()
        if alpha <= 0:
            raise ValueError("Alpha must be > 0")
        self.p = p
        self.alpha = alpha

    @torch.no_grad()
    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            images: (B, C, H, W) float tensor — the image batch.
            labels: (B, 1, H, W) float tensor — the binary mask batch.

        Returns:
            CutMixed images and labels (same shapes).
        """
        if torch.rand(1).item() >= self.p:
            return images, labels

        # Clone to avoid modifying the original cached data
        images = images.clone()
        labels = labels.clone()

        # Sample λ from Beta(α, α) via Dirichlet
        lam = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        _, _, H, W = images.shape

        # Random center point for the cut box
        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        # Box half-sizes proportional to (1 - λ)
        r = 0.5 * math.sqrt(1.0 - lam)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        # Clamp to image boundaries
        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        # Roll batch by 1 to create pairs
        images_rolled = images.roll(1, 0)
        labels_rolled = labels.roll(1, 0)

        # Paste the cut region from rolled samples
        images[:, :, y1:y2, x1:x2] = images_rolled[:, :, y1:y2, x1:x2]
        labels[:, :, y1:y2, x1:x2] = labels_rolled[:, :, y1:y2, x1:x2]

        return images, labels

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, alpha={self.alpha})"
