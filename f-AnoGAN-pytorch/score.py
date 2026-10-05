#!/usr/bin/env python3
"""
Stage 3 of f-AnoGAN: anomaly scoring + per-pixel heatmap on test patches.

  For each input patch x:
      z      = E(x)
      x_hat  = G(z)
      f_x    = D_features(x)
      f_xhat = D_features(x_hat)
      A_R(x) = mean( (x - x_hat)^2 )          # image-space residual
      A_D(x) = mean( (f_x - f_xhat)^2 )       # discriminator-feature residual
      score  = A_R + κ · A_D                  # dual-image-loss
      heatmap_pixel = (x - x_hat)^2           # for visualization

Outputs:
    out_dir/scores.csv            patch_path, A_R, A_D, score, is_anom (if labelled)
    out_dir/heatmaps/<stem>.png   per-patch heatmap PNG (residual normalized)
    out_dir/recon/<stem>.png      side-by-side: input | recon | residual

If you pass both --normal_root and --anom_root, the CSV gets `is_anom` filled
in (0 / 1). Useful for AUC-ROC on the test set.

Example:
    conda run -n monai_placenta python -m f_anogan_pytorch.score \\
        --encoder_ckpt ./runs_fanogan/encoder_v1/encoder_final.pth \\
        --normal_root  /path/to/normal_test_patches/ \\
        --anom_root    /path/to/pas_test_patches/ \\
        --out_dir      ./runs_fanogan/scores_v1 \\
        --kappa        1.0
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models import Generator, Discriminator, Encoder       # noqa: E402
from data import AnomalyPatchDataset                       # noqa: E402
from utils import load_checkpoint                          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_ckpt", type=str, required=True,
                   help="encoder_final.pth from train_encoder.py (also points to its WGAN).")
    p.add_argument("--wgan_ckpt", type=str, default=None,
                   help="Override the WGAN ckpt path. If omitted, reads from encoder_ckpt['wgan_ckpt'].")
    p.add_argument("--normal_root", type=str, default=None,
                   help="Folder of NORMAL test patches (is_anom=0).")
    p.add_argument("--anom_root", type=str, default=None,
                   help="Folder of ANOMALOUS test patches (is_anom=1).")
    p.add_argument("--out_dir", type=str, default="./runs_fanogan/scores")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--kappa", type=float, default=1.0,
                   help="Weight on A_D in the final score (paper default 1.0).")
    p.add_argument("--save_heatmaps", action="store_true",
                   help="Save per-patch residual heatmap PNG and reconstruction collage.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _denorm01(t: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]."""
    return (t.clamp(-1.0, 1.0) + 1.0) * 0.5


def save_heatmap_collage(input_t: torch.Tensor, recon_t: torch.Tensor,
                         path_out: Path) -> None:
    """Save a 3-panel PNG: input | reconstruction | per-pixel residual (jet)."""
    inp = _denorm01(input_t).cpu().numpy()[0]
    rec = _denorm01(recon_t).cpu().numpy()[0]
    res = ((input_t - recon_t) ** 2).cpu().numpy()[0]
    # Normalize residual to [0, 1] for visualization
    rmin, rmax = float(res.min()), float(res.max())
    if rmax > rmin:
        res_n = (res - rmin) / (rmax - rmin)
    else:
        res_n = np.zeros_like(res)

    # Build jet colormap inline (no matplotlib dep here)
    def jet(v: np.ndarray) -> np.ndarray:
        # piecewise linear approximation of MATLAB jet
        r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
        return np.stack([r, g, b], axis=-1)

    h, w = inp.shape
    gutter = 4
    out = np.zeros((h, 3 * w + 2 * gutter, 3), dtype=np.uint8)
    out[:, 0:w, :] = (np.stack([inp] * 3, axis=-1) * 255).astype(np.uint8)
    out[:, w + gutter:2 * w + gutter, :] = (np.stack([rec] * 3, axis=-1) * 255).astype(np.uint8)
    out[:, 2 * (w + gutter):, :] = (jet(res_n) * 255).astype(np.uint8)
    Image.fromarray(out).save(str(path_out))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.normal_root is None and args.anom_root is None:
        sys.exit("Pass --normal_root and/or --anom_root.")

    # --- Resolve encoder + matching WGAN ---
    enc_ck = load_checkpoint(args.encoder_ckpt, map_location=device)
    enc_args = enc_ck.get("args", {})
    wgan_path = Path(args.wgan_ckpt) if args.wgan_ckpt else Path(enc_ck.get("wgan_ckpt", ""))
    if not wgan_path.exists():
        sys.exit(f"Cannot find WGAN checkpoint at {wgan_path}. Pass --wgan_ckpt explicitly.")
    wgan_ck = load_checkpoint(wgan_path, map_location=device)

    z_dim = enc_args.get("z_dim", 128)
    dim = enc_args.get("dim", 64)
    z_reg = enc_args.get("z_reg", "tanh_fc")
    z_reg = None if z_reg == "none" else z_reg
    patch_size = enc_args.get("patch_size", 64)

    G = Generator(z_dim=z_dim, dim=dim, out_channels=1).to(device).eval()
    D = Discriminator(dim=dim, in_channels=1).to(device).eval()
    E = Encoder(z_dim=z_dim, dim=dim, in_channels=1, z_reg=z_reg).to(device).eval()
    G.load_state_dict(wgan_ck["G"])
    D.load_state_dict(wgan_ck["D"])
    E.load_state_dict(enc_ck["E"])
    for m in (G, D, E):
        for p in m.parameters():
            p.requires_grad_(False)

    print(f"[score] G/D from: {wgan_path}")
    print(f"[score] E   from: {args.encoder_ckpt}")
    print(f"[score] z_dim={z_dim}  dim={dim}  z_reg={z_reg}  patch={patch_size}")

    # --- Score one dataset at a time ---
    heatmap_dir = out_dir / "heatmaps"
    recon_dir = out_dir / "recon"
    if args.save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        recon_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "scores.csv"
    rows: list[dict] = []

    @torch.no_grad()
    def score_folder(root: str, is_anom: int) -> None:
        ds = AnomalyPatchDataset(root, size=patch_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        print(f"[score] {'anom ' if is_anom else 'norm '}folder: {root}  ({len(ds)} patches)")
        for imgs, paths in tqdm(loader, desc=f"is_anom={is_anom}", dynamic_ncols=True):
            imgs = imgs.to(device, non_blocking=True)
            z = E(imgs)
            recon = G(z)
            _, fx = D(imgs)
            _, fr = D(recon)
            # per-image residuals
            A_R = F.mse_loss(recon, imgs, reduction="none").mean(dim=(1, 2, 3))
            A_D = F.mse_loss(fr, fx, reduction="none").mean(dim=1)
            score = A_R + args.kappa * A_D

            for i in range(imgs.size(0)):
                p = Path(paths[i])
                rows.append({
                    "patch_path": str(p),
                    "is_anom": is_anom,
                    "A_R": float(A_R[i].item()),
                    "A_D": float(A_D[i].item()),
                    "score": float(score[i].item()),
                })
                if args.save_heatmaps:
                    save_heatmap_collage(imgs[i], recon[i], heatmap_dir / f"{p.stem}.png")

    if args.normal_root is not None:
        score_folder(args.normal_root, is_anom=0)
    if args.anom_root is not None:
        score_folder(args.anom_root, is_anom=1)

    # --- Write CSV ---
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patch_path", "is_anom", "A_R", "A_D", "score"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[score] wrote {csv_path}  ({len(rows)} patches)")

    # --- AUC-ROC if both labelled folders were provided ---
    if args.normal_root is not None and args.anom_root is not None:
        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            y_true = np.array([r["is_anom"] for r in rows], dtype=int)
            y_score = np.array([r["score"] for r in rows], dtype=float)
            auc = roc_auc_score(y_true, y_score)
            print(f"[score] AUC-ROC (score):     {auc:.4f}")
            auc_R = roc_auc_score(y_true, [r["A_R"] for r in rows])
            auc_D = roc_auc_score(y_true, [r["A_D"] for r in rows])
            print(f"[score] AUC-ROC (A_R only):  {auc_R:.4f}")
            print(f"[score] AUC-ROC (A_D only):  {auc_D:.4f}")
            # Save the ROC table
            fpr, tpr, thr = roc_curve(y_true, y_score)
            with open(out_dir / "roc.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["fpr", "tpr", "threshold"])
                for a, b, c in zip(fpr, tpr, thr):
                    w.writerow([a, b, c])
            print(f"[score] wrote {out_dir / 'roc.csv'}")
        except ImportError:
            print("[score] sklearn not installed — skipping AUC. `pip install scikit-learn` to enable.")


if __name__ == "__main__":
    main()
