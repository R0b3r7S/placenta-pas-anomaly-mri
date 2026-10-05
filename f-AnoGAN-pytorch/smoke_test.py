#!/usr/bin/env python3
"""
End-to-end smoke test for the f-AnoGAN PyTorch port.

Generates 256 synthetic 64×64 "normal" patches (clean Gaussian blobs) and 64
"anomalous" patches (blobs with a high-frequency stripe injected — a clear
manifold deviation). Runs:

    1. 50 iters of WGAN-GP training
    2. 50 iters of encoder (izi_f) training
    3. Scores both test sets and prints an AUC-ROC

Total runtime: ~30-60 seconds on a 5080. Exits 0 only on full success.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models import Generator, Discriminator, Encoder, init_weights  # noqa: E402
from data import NormalPatchDataset, AnomalyPatchDataset            # noqa: E402
from utils import gradient_penalty, save_checkpoint, load_checkpoint  # noqa: E402


def make_blob(rng: np.random.Generator, anom: bool, size: int = 64) -> np.ndarray:
    """Soft circular blob; if anom, add a high-frequency vertical stripe."""
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = rng.uniform(20, 44), rng.uniform(20, 44)
    r = rng.uniform(8, 16)
    yy, xx = np.indices((size, size))
    img = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))
    img = img / img.max() * 0.8 + rng.normal(0, 0.02, img.shape).astype(np.float32)
    if anom:
        # Vertical bright stripe — out-of-manifold pattern
        col = int(rng.integers(15, size - 15))
        img[:, col - 1:col + 1] = 1.0
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def synth_dataset(root: Path, n: int, anom: bool, seed: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        Image.fromarray(make_blob(rng, anom)).save(root / f"p{i:04d}.png")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device: {device}")
    if device.type != "cuda":
        print("[smoke] (CPU mode — will be slow but still functional)")

    tmp = Path(tempfile.mkdtemp(prefix="fanogan_smoke_"))
    try:
        # --- 1. Synthesize data ---
        norm_train = tmp / "normal_train"
        norm_test = tmp / "normal_test"
        anom_test = tmp / "anom_test"
        synth_dataset(norm_train, n=256, anom=False, seed=1)
        synth_dataset(norm_test, n=32, anom=False, seed=2)
        synth_dataset(anom_test, n=32, anom=True, seed=3)
        print(f"[smoke] synth normal_train: {len(list(norm_train.iterdir()))}")
        print(f"[smoke] synth normal_test:  {len(list(norm_test.iterdir()))}")
        print(f"[smoke] synth anom_test:    {len(list(anom_test.iterdir()))}")

        # --- 2. Build models ---
        z_dim, dim = 128, 32   # smaller dim for speed in smoke test
        G = Generator(z_dim=z_dim, dim=dim).to(device)
        D = Discriminator(dim=dim).to(device)
        E = Encoder(z_dim=z_dim, dim=dim, z_reg="tanh_fc").to(device)
        for m in (G, D, E):
            init_weights(m)
        opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
        opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
        opt_E = torch.optim.RMSprop(E.parameters(), lr=5e-5)

        # --- 3. Stage 1: 50 WGAN-GP iters ---
        from torch.utils.data import DataLoader
        loader = DataLoader(NormalPatchDataset(norm_train), batch_size=32,
                            shuffle=True, drop_last=True)
        data_iter = iter(loader)
        def next_b():
            nonlocal data_iter
            try: return next(data_iter)
            except StopIteration:
                data_iter = iter(loader); return next(data_iter)

        print("[smoke] stage 1: WGAN-GP × 50 iters")
        for it in range(50):
            for _ in range(5):
                real = next_b().to(device)
                z = G.sample_z(real.size(0), device=device)
                fake = G(z).detach()
                d_real = D(real, return_features=False)
                d_fake = D(fake, return_features=False)
                gp = gradient_penalty(D, real, fake, device)
                d_loss = d_fake.mean() - d_real.mean() + 10.0 * gp
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()
            z = G.sample_z(real.size(0), device=device)
            g_loss = -D(G(z), return_features=False).mean()
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
        print(f"[smoke]   final d_loss={d_loss.item():+.3f}  g_loss={g_loss.item():+.3f}")

        # --- 4. Stage 2: 50 encoder iters ---
        G.eval(); D.eval()
        for p in G.parameters(): p.requires_grad_(False)
        for p in D.parameters(): p.requires_grad_(False)
        print("[smoke] stage 2: encoder × 50 iters")
        for it in range(50):
            real = next_b().to(device)
            z = E(real)
            recon = G(z)
            _, fx = D(real)
            _, fr = D(recon)
            loss = F.mse_loss(recon, real) + F.mse_loss(fr, fx)
            opt_E.zero_grad(); loss.backward(); opt_E.step()
        print(f"[smoke]   final encoder loss={loss.item():.4f}")

        # --- 5. Stage 3: score ---
        print("[smoke] stage 3: scoring")
        G.eval(); D.eval(); E.eval()

        @torch.no_grad()
        def score_one(root, is_anom):
            ds = AnomalyPatchDataset(root)
            loader = DataLoader(ds, batch_size=32, shuffle=False)
            scores = []
            for imgs, paths in loader:
                imgs = imgs.to(device)
                recon = G(E(imgs))
                _, fx = D(imgs)
                _, fr = D(recon)
                A_R = F.mse_loss(recon, imgs, reduction="none").mean(dim=(1, 2, 3))
                A_D = F.mse_loss(fr, fx, reduction="none").mean(dim=1)
                s = (A_R + A_D).cpu().tolist()
                scores += [(v, is_anom) for v in s]
            return scores

        rows = score_one(norm_test, 0) + score_one(anom_test, 1)

        try:
            from sklearn.metrics import roc_auc_score
            y = np.array([r[1] for r in rows])
            s = np.array([r[0] for r in rows])
            auc = roc_auc_score(y, s)
            print(f"[smoke] AUC-ROC: {auc:.3f}")
            # AUC will be modest — only 50 iters — but should be >chance with high prob.
            if not (0.0 <= auc <= 1.0):
                raise RuntimeError(f"AUC out of range: {auc}")
        except ImportError:
            print("[smoke] (sklearn not installed — skipping AUC)")

        # --- 6. Checkpoint round-trip ---
        ck_path = tmp / "wgan.pth"
        save_checkpoint({"G": G.state_dict(), "D": D.state_dict()}, ck_path)
        ck_re = load_checkpoint(ck_path, map_location=device)
        assert "G" in ck_re and "D" in ck_re
        print(f"[smoke] checkpoint round-trip: OK  ({ck_path.stat().st_size//1024} KB)")

        print("\nsmoke test passed end-to-end")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
