"""
MixUp augmentation simplified for binary (1-channel) segmentation.

Original paper: "mixup: Beyond Empirical Risk Minimization" (https://arxiv.org/abs/1710.09412)
Adapted from the nnU-Net implementation in this repo for use with MONAI pipelines.

How it works:
    For each batch, pairs of images are created by rolling the batch by 1 position.
    A mixing coefficient λ ~ Beta(α, α) is sampled.
    The mixed image  = λ * image_A + (1-λ) * image_B
    The mixed label  = λ * label_A + (1-λ) * label_B

    Since our labels are binary float masks (0.0 or 1.0, single channel),
    no one-hot encoding is needed. The resulting soft labels (e.g., 0.7)
    are handled natively by DiceCELoss with sigmoid=True.
"""

import torch


class RandomMixUpBinary(torch.nn.Module):
    """MixUp augmentation for binary segmentation (1-channel masks).

    Applied at batch level inside the training loop, NOT as a MONAI transform.

    Args:
        p (float): Probability of applying MixUp to the batch. Default: 1.0.
        alpha (float): Hyperparameter of the Beta distribution. Smaller alpha
            produces mixing coefficients closer to 0 or 1 (less mixing).
            Paper default for segmentation: 0.2. Default: 0.2.
    """

    def __init__(self, p: float = 1.0, alpha: float = 0.2):
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
            Mixed images and labels (same shapes).
        """
        if torch.rand(1).item() >= self.p:
            return images, labels

        # Sample λ from Beta(α, α) via Dirichlet
        lam = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        # Roll batch by 1 to create pairs (faster than random shuffling)
        images_rolled = images.roll(1, 0)
        labels_rolled = labels.roll(1, 0)

        # Linear interpolation of images and labels
        images = lam * images + (1.0 - lam) * images_rolled
        labels = lam * labels + (1.0 - lam) * labels_rolled

        return images, labels

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, alpha={self.alpha})"
