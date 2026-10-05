# f-AnoGAN (PyTorch port)

PyTorch re-implementation of Schlegl et al. 2019:
*"f-AnoGAN: Fast Unsupervised Anomaly Detection with Generative Adversarial Networks"* (Medical Image Analysis 54, 30–44).

Faithful to the authors' original TensorFlow 1.2 release
([github.com/tSchlegl/f-AnoGAN](https://github.com/tSchlegl/f-AnoGAN)), modernized for
PyTorch ≥ 2.0 (tested on an NVIDIA RTX 5080).

## Role in this project

f-AnoGAN is trained **only on boundary patches from normal placentas**; patches
from PAS patients (and all other test patients) are then scored as departures
from that learned normal appearance. For the experiment in the paper,
`run_leave_healthy_out.py` runs patch extraction, both training stages and
scoring; the stages below can also be run on their own.

## Architecture

```
   z ∈ R^128  ─G─→  x ∈ [-1, 1]^(1×64×64)  ─D─→  (logit, features ∈ R^8192)
                                  ↑                            │
                                  └──── G(E(x)) ←── z = E(x) ──┘
```

| Module | Structure | Norm |
|---|---|---|
| **Generator** | Linear → 4× ResBlockUp (4→8→16→32→64) → 3×3 Conv → tanh | BatchNorm2d |
| **Discriminator** | 3×3 Conv → 4× ResBlockDown (64→32→16→8→4) → Linear | LayerNorm (= GroupNorm(1)) — required for WGAN-GP |
| **Encoder** | 3×3 Conv → 4× ResBlockDown → Linear → tanh-regularization | BatchNorm2d |

The Discriminator returns both the scalar logit AND the 8192-dim feature
vector (the conv stack output before the final Linear). The encoder's
`izi_f` loss uses these features.

## Three-stage pipeline

### Stage 1 — Train WGAN-GP on normal patches

```bash
python f-AnoGAN-pytorch/train_wgan.py \
    --data_root /path/to/normal_patches/ \
    --out_dir   ./runs_fanogan/wgan_v1 \
    --epochs    150 \
    --batch_size 64 \
    --amp
```

Produces `./runs_fanogan/wgan_v1/wgan_final.pth` + sample sheets under `samples/`.
Resume any interrupted run with `--resume`.

### Stage 2 — Train the izi_f encoder

```bash
python f-AnoGAN-pytorch/train_encoder.py \
    --data_root /path/to/normal_patches/ \
    --wgan_ckpt ./runs_fanogan/wgan_v1/wgan_final.pth \
    --out_dir   ./runs_fanogan/encoder_v1 \
    --iters     50000 \
    --kappa     1.0 \
    --amp
```

Encoder loss is `MSE(x, G(E(x))) + κ · MSE(D_feat(x), D_feat(G(E(x))))`.
Periodic real-vs-recon pair rows are saved to `samples/pairs_iter*.png`.

### Stage 3 — Anomaly scoring on test patches

```bash
python f-AnoGAN-pytorch/score.py \
    --encoder_ckpt ./runs_fanogan/encoder_v1/encoder_final.pth \
    --normal_root  /path/to/normal_test_patches/ \
    --anom_root    /path/to/pas_test_patches/   \
    --out_dir      ./runs_fanogan/scores_v1 \
    --kappa        1.0 \
    --save_heatmaps
```

Outputs:
- `scores.csv` — per-patch `A_R`, `A_D`, `score`, `is_anom`
- `roc.csv` — ROC points + thresholds when both labelled folders are passed
- `heatmaps/<stem>.png` — 3-panel input | reconstruction | per-pixel residual (jet)
- AUC-ROC printed (overall + A_R-only + A_D-only)

The **anomaly score** per patch is `A_R + κ · A_D`:
- `A_R = mean( (x - G(E(x)))² )` — image-space residual
- `A_D = mean( (D_feat(x) - D_feat(G(E(x))))² )` — discriminator-feature residual

The **per-pixel heatmap** is `(x - G(E(x)))²`, a residual map for visual inspection.
(The continuous anomaly maps in the paper's Fig. 2 are built from the patch scores by
`make_anomaly_nifti.py`.)

## Data layout

All three stages expect a flat folder of 64×64 grayscale PNG patches:

```
normal_patches/
├── patch_000001.png
├── patch_000002.png
└── …
```

They are produced by `extract_boundary_patches.py` from the placental masks
(the paper uses the full boundary, `--lower_fraction 0.0`, stride 32 px, 64×64 patches).
`run_leave_healthy_out.py` calls it for every patient group.

## Key differences from the original TF 1.2 code

| Original (TF 1.2)             | This port (PyTorch ≥ 2.0)         |
|---|---|
| `tf.Session`, manual `feed_dict`              | `torch.utils.data.DataLoader` + train loops |
| `tflib.ops.Conv2D` with He init manually wired | `nn.Conv2d` + `init_weights()` (Kaiming) |
| `tflib.ops.batchnorm.Batchnorm`                | `nn.BatchNorm2d`                            |
| `tflib.ops.layernorm.Layernorm([1,2,3], x)`    | `nn.GroupNorm(1, C, affine=True)`           |
| `depth_to_space` upsample trick                | `F.interpolate(scale=2, mode='nearest')`    |
| `RMSPropOptimizer(lr=5e-5)` for encoder        | `torch.optim.RMSprop`                       |
| Adam(lr=1e-4, β1=0, β2=0.9) for WGAN           | `torch.optim.Adam(betas=(0, 0.9))`          |
| `tf.gradients(...)` for gradient penalty       | `torch.autograd.grad(create_graph=True)`    |
| Custom checkpoint logic with regex             | `torch.save / torch.load` + `last_checkpoint.pth` resume |

The model topology is **bit-identical** to the original: same channel
counts, same number of res-blocks, same kernel sizes, same z-dim.

## Smoke test

```bash
python f-AnoGAN-pytorch/smoke_test.py
```

(Generates 256 synthetic patches, trains 100 WGAN iters and 50 encoder iters,
and scores 32 patches. Exits cleanly only when everything works end-to-end.)

## Citation

If you build on this in a paper, cite the original:

```
Schlegl, T., Seeböck, P., Waldstein, S.M., Langs, G., Schmidt-Erfurth, U., 2019.
f-AnoGAN: Fast Unsupervised Anomaly Detection with Generative Adversarial Networks.
Medical Image Analysis 54, 30-44. DOI: 10.1016/j.media.2019.01.010
```

## File map

```
f-AnoGAN-pytorch/
├── README.md            this file
├── models.py            Generator / Discriminator / Encoder + ResBlocks
├── data.py              NormalPatchDataset + AnomalyPatchDataset
├── utils.py             gradient_penalty, image grid, checkpoint helpers
├── train_wgan.py        Stage 1
├── train_encoder.py     Stage 2 (izi_f encoder)
├── score.py             Stage 3 (anomaly scoring + per-patch heatmaps)
└── smoke_test.py        end-to-end synthetic-data check (≤1 min on a 5080)
```

The other scripts in this folder (boundary-patch extraction, the leave-healthy-out
experiment, and the scripts that produce the paper's tables and figures) are listed,
with the exact commands, in the main [README](../README.md#reproducing-the-paper).
