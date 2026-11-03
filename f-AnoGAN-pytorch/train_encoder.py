#!/usr/bin/env python3
"""
Stage 2 of f-AnoGAN: train the izi_f encoder E.

  Given a fixed Generator G and Discriminator D from stage 1, learn E such that
  E(x) ≈ z⋆ where G(z⋆) ≈ x AND f(G(z⋆)) ≈ f(x), where f(·) are the
  intermediate D features (the 4·4·8·dim vector before D's final linear layer).

  Loss = MSE(x, G(E(x))) + κ · MSE(f(x), f(G(E(x))))

  Default κ=1.0, RMSprop(lr=5e-5), 50 000 iterations.

Example:
    conda run -n monai_placenta python -m f_anogan_pytorch.train_encoder \\
        --data_root  /path/to/normal_patches/ \\
        --wgan_ckpt  ./runs_fanogan/wgan_v1/wgan_final.pth \\
        --out_dir    ./runs_fanogan/encoder_v1 \\
        --iters      50000 \\
        --kappa      1.0
"""

from __future__ import annotations

import argparse
import sys
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models import Generator, Discriminator, Encoder, init_weights  # noqa: E402
from data import NormalPatchDataset                                 # noqa: E402
from utils import save_pair_row, save_checkpoint, load_checkpoint   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--wgan_ckpt", type=str, required=True,
                   help="wgan_final.pth (or last_checkpoint.pth) from stage 1.")
    p.add_argument("--out_dir", type=str, default="./runs_fanogan/encoder")
    p.add_argument("--iters", type=int, default=50000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--z_reg", type=str, default="tanh_fc",
                   choices=["tanh_fc", "hard_clip", "none"])
    p.add_argument("--kappa", type=float, default=1.0,
                   help="Weight on the D-feature MSE term.")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--sample_every", type=int, default=500,
                   help="Save a (real | recon) pair-row + log every N iters.")
    p.add_argument("--ckpt_every", type=int, default=5000,
                   help="Snapshot every N iters (in addition to last_checkpoint.pth).")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_gd(path: Path, z_dim: int, dim: int, device: torch.device) -> tuple:
    """Build G, D with the given hyperparams and load weights from a WGAN checkpoint."""
    G = Generator(z_dim=z_dim, dim=dim, out_channels=1).to(device)
    D = Discriminator(dim=dim, in_channels=1).to(device)
    ck = load_checkpoint(path, map_location=device)
    G.load_state_dict(ck["G"])
    D.load_state_dict(ck["D"])
    return G, D, ck.get("args", {})


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
    print(f"[enc] dataset: {len(ds)} patches  device: {device}")

    # --- Frozen G, D from stage 1 ---
    G, D, wgan_args = load_gd(Path(args.wgan_ckpt), z_dim=args.z_dim,
                              dim=args.dim, device=device)
    G.eval(); D.eval()
    for p in G.parameters(): p.requires_grad_(False)
    for p in D.parameters(): p.requires_grad_(False)
    print(f"[enc] loaded WGAN from {args.wgan_ckpt}")

    # --- Encoder (trainable) ---
    z_reg = None if args.z_reg == "none" else args.z_reg
    E = Encoder(z_dim=args.z_dim, dim=args.dim, in_channels=1, z_reg=z_reg).to(device)
    init_weights(E)
    opt_E = torch.optim.RMSprop(E.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # --- Resume ---
    start_iter = 0
    last_path = out_dir / "last_checkpoint.pth"
    if args.resume and last_path.exists():
        ck = load_checkpoint(last_path, map_location=device)
        E.load_state_dict(ck["E"])
        opt_E.load_state_dict(ck["opt_E"])
        if "scaler" in ck:
            try: scaler.load_state_dict(ck["scaler"])
            except Exception: pass
        start_iter = int(ck.get("iter", 0))
        print(f"[enc] resumed from {last_path} at iter {start_iter}")
    elif args.resume:
        print(f"[enc] resume requested but no checkpoint at {last_path}.")

    # --- Pin a fixed validation batch for the sample row ---
    fixed_real = next(iter(loader)).to(device)[: min(8, args.batch_size)]

    # --- Infinite iterator ---
    def inf():
        while True:
            for batch in loader:
                yield batch

    gen = inf()
    pbar = tqdm(range(start_iter, args.iters), desc="encoder", dynamic_ncols=True)
    loss_ema = li_ema = lf_ema = None

    for it in pbar:
        real = next(gen).to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.amp):
            z = E(real)
            recon = G(z)
            _, feats_real = D(real)
            _, feats_recon = D(recon)
            loss_img = F.mse_loss(recon, real)
            loss_feat = F.mse_loss(feats_recon, feats_real)
            loss = loss_img + args.kappa * loss_feat

        opt_E.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt_E)
        scaler.update()

        loss_ema = loss.item() if loss_ema is None else 0.95 * loss_ema + 0.05 * loss.item()
        li_ema = loss_img.item() if li_ema is None else 0.95 * li_ema + 0.05 * loss_img.item()
        lf_ema = loss_feat.item() if lf_ema is None else 0.95 * lf_ema + 0.05 * loss_feat.item()
        pbar.set_postfix(loss=f"{loss_ema:.4f}", img=f"{li_ema:.4f}", feat=f"{lf_ema:.4f}")

        # --- Sample pair row ---
        if (it + 1) % args.sample_every == 0:
            E.eval()
            with torch.no_grad():
                recon_fixed = G(E(fixed_real))
            save_pair_row(fixed_real, recon_fixed,
                          samples_dir / f"pairs_iter{it+1:06d}.png")
            E.train()

        # --- Periodic checkpoints ---
        if (it + 1) % args.ckpt_every == 0:
            save_checkpoint({
                "E": E.state_dict(),
                "opt_E": opt_E.state_dict(),
                "scaler": scaler.state_dict(),
                "iter": it + 1,
                "args": vars(args),
                "wgan_args": wgan_args,
            }, last_path)
            # named snapshot too (so you can compare iters later)
            save_checkpoint({"E": E.state_dict(), "iter": it + 1},
                            out_dir / f"encoder_iter{it+1:06d}.pth")

    # --- Final dump for stage 3 ---
    final = out_dir / "encoder_final.pth"
    save_checkpoint({
        "E": E.state_dict(),
        "iter": args.iters,
        "args": vars(args),
        "wgan_args": wgan_args,
        "wgan_ckpt": str(args.wgan_ckpt),
    }, final)
    print(f"[enc] done → {final}")


if __name__ == "__main__":
    main()
