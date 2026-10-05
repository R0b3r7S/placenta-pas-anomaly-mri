#!/usr/bin/env python3
"""
Stage 1 of f-AnoGAN: train the WGAN-GP (Generator + Discriminator) on the
"normal" patch manifold. After this stage finishes, run train_encoder.py.

Defaults match the paper: Adam(lr=1e-4, β1=0, β2=0.9), 5 critic updates per
generator update, gradient-penalty λ=10, batch=64, z~N(0, I), z_dim=128, 64×64
grayscale patches. Override anything via CLI.

Example:
    conda run -n monai_placenta python -m f_anogan_pytorch.train_wgan \\
        --data_root /path/to/normal_patches/ \\
        --out_dir   ./runs_fanogan/wgan_v1 \\
        --epochs    7 \\
        --batch_size 64
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models import Generator, Discriminator, init_weights  # noqa: E402
from data import NormalPatchDataset                         # noqa: E402
from utils import gradient_penalty, save_image_grid, save_checkpoint, load_checkpoint  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                   help="Folder of normal (no-PAS) 64x64 grayscale patches.")
    p.add_argument("--out_dir", type=str, default="./runs_fanogan/wgan",
                   help="Where checkpoints + samples land.")
    p.add_argument("--epochs", type=int, default=7,
                   help="Number of epochs over the dataset.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--dim", type=int, default=64,
                   help="Model dimensionality (channels of the first conv).")
    p.add_argument("--critic_iters", type=int, default=5,
                   help="D updates per G update.")
    p.add_argument("--gp_lambda", type=float, default=10.0,
                   help="Gradient-penalty coefficient.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.0)
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--amp", action="store_true",
                   help="Mixed-precision (fp16) for the forward + non-gradient-penalty losses.")
    p.add_argument("--sample_every", type=int, default=500,
                   help="Save a sample image grid every N iterations.")
    p.add_argument("--ckpt_every_epoch", action="store_true", default=True,
                   help="Save last_checkpoint.pth at end of each epoch.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from <out_dir>/last_checkpoint.pth if present.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    ds = NormalPatchDataset(args.data_root, size=args.patch_size, hflip=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)
    print(f"[wgan] dataset: {len(ds)} patches  iters/epoch: {len(loader)}  device: {device}")

    # --- Models ---
    G = Generator(z_dim=args.z_dim, dim=args.dim, out_channels=1).to(device)
    D = Discriminator(dim=args.dim, in_channels=1).to(device)
    init_weights(G); init_weights(D)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # --- Fixed noise for sampling sheets ---
    fixed_z = G.sample_z(64, device=device)

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    last_path = out_dir / "last_checkpoint.pth"
    if args.resume and last_path.exists():
        print(f"[wgan] resuming from {last_path}")
        ck = load_checkpoint(last_path, map_location=device)
        G.load_state_dict(ck["G"])
        D.load_state_dict(ck["D"])
        opt_G.load_state_dict(ck["opt_G"])
        opt_D.load_state_dict(ck["opt_D"])
        if "scaler" in ck:
            try: scaler.load_state_dict(ck["scaler"])
            except Exception: pass
        start_epoch = int(ck.get("epoch", 0))
        global_step = int(ck.get("global_step", 0))
        fixed_z = ck.get("fixed_z", fixed_z).to(device)
        print(f"[wgan]   → resuming at epoch {start_epoch + 1}, global_step {global_step}")
    elif args.resume:
        print(f"[wgan] resume requested but no checkpoint at {last_path} — training from scratch.")

    # --- Train loop ---
    print(f"[wgan] training {args.epochs} epoch(s), critic_iters={args.critic_iters}, λ_gp={args.gp_lambda}")
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(range(len(loader)), desc=f"epoch {epoch+1}/{args.epochs}",
                    dynamic_ncols=True)
        d_loss_ema = g_loss_ema = None

        for _ in pbar:
            # --- (1) Critic step ---
            for _crit in range(args.critic_iters):
                real = next_batch().to(device, non_blocking=True)
                z = G.sample_z(real.size(0), device=device)
                with torch.amp.autocast("cuda", enabled=args.amp):
                    fake = G(z).detach()
                    d_real = D(real, return_features=False)
                    d_fake = D(fake, return_features=False)
                    # GP needs fp32 grads w.r.t. inputs; compute it outside autocast
                # GP outside autocast for stable double-backward
                gp = gradient_penalty(D, real.float(), fake.float(), device)
                d_loss = d_fake.mean() - d_real.mean() + args.gp_lambda * gp
                opt_D.zero_grad(set_to_none=True)
                scaler.scale(d_loss).backward()
                scaler.step(opt_D)
                scaler.update()

            # --- (2) Generator step ---
            z = G.sample_z(args.batch_size, device=device)
            with torch.amp.autocast("cuda", enabled=args.amp):
                fake = G(z)
                d_fake = D(fake, return_features=False)
                g_loss = -d_fake.mean()
            opt_G.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(opt_G)
            scaler.update()

            global_step += 1
            d_loss_ema = d_loss.item() if d_loss_ema is None else 0.95 * d_loss_ema + 0.05 * d_loss.item()
            g_loss_ema = g_loss.item() if g_loss_ema is None else 0.95 * g_loss_ema + 0.05 * g_loss.item()
            pbar.set_postfix(d=f"{d_loss_ema:+.3f}", g=f"{g_loss_ema:+.3f}", gp=f"{gp.item():.3f}")

            # --- Sample sheet ---
            if global_step % args.sample_every == 0:
                G.eval()
                with torch.no_grad():
                    samples = G(fixed_z)
                save_image_grid(samples, samples_dir / f"samples_step{global_step:06d}.png")
                G.train()

        # --- End of epoch: checkpoint ---
        save_checkpoint({
            "G": G.state_dict(),
            "D": D.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "fixed_z": fixed_z.detach().cpu(),
            "args": vars(args),
        }, last_path)
        print(f"[wgan] epoch {epoch+1} done → {last_path}")

    # --- Final dump for stage 2 ---
    final = out_dir / "wgan_final.pth"
    save_checkpoint({
        "G": G.state_dict(), "D": D.state_dict(),
        "epoch": args.epochs, "global_step": global_step,
        "args": vars(args),
    }, final)
    print(f"[wgan] done → {final}")


if __name__ == "__main__":
    main()
