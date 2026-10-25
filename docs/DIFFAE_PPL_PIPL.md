# HOPE PPL / PIPL evaluation on FFHQ128 DiffAE

This note walks through the evaluation protocol used in the current code, together with a completed 50,000-step, ε=10⁻⁴ run. The method name is **HOPE**; in code, `--method hop` and directories named `hop` refer to the same method.

Evaluation uses 2,000 images from the FFHQ test split. This is a separate experiment from the 20-image re-scoring analysis. Lower PPL and PIPL are better; both measure perceptual change under a defined perturbation, and they are not intended as direct measures of attribute accuracy or disentanglement.

## 1. Code and checkpoints

Please use this working directory:

```text
models/DiffAE/
```

Main implementations:

| File | Role |
|---|---|
| `evaluate_ffhq128_metrics.py` | Load model and data; build pairs; extract directions; LPIPS and trimmed statistics |
| `generate_ffhq128_metric_samples.py` | Legacy pipeline: sequentially generate FID images, PPL pairs, PIPL pairs |
| `generate_ffhq128_path_samples.py` | Newer pipeline: generate only PPL/PIPL pairs; reset the RNG before each metric |
| `score_ffhq128_saved_samples.py` | Score saved float image pairs with LPIPS; includes `score_pairs` |
| `score_ffhq128_path_samples.py` | Validate newer path-sample metadata, then compute PPL/PIPL |
| `model/unet_autoenc.py` | Apply PIPL perturbations to the selected regional conditioning projector input |

Verified 50k checkpoint:

```text
checkpoints/ffhq128_householder_random_single4090/checkpoints/step_050000.ckpt
```

The model is built from `ffhq128_autoenc_130M()`; checkpoint loading uses a strict `state_dict` load. Inference uses `ema_model.eval()`, and encoding uses `ema_model.encoder` rather than the online training weights. HOPE enables regional conditioning projectors; if a checkpoint was trained with a time/style MLP, please pass `--use-time-style-mlp` so the architecture matches training.

`--steps 50000` is only a label in the scoring output; it does not switch weights or read the step from inside the checkpoint. The loaded weights are determined by `--checkpoint` at generation time; please check `source_checkpoint` in `metadata.json` when comparing runs.

## 2. Evaluation configuration

| Setting | This experiment |
|---|---|
| Data | FFHQ LMDB, `split='test'`, `do_augment=False` |
| Data path | `datasets/` (or your local FFHQ LMDB directory) |
| Sample order | Fixed `test[0:2000]`; no shuffle, no drop-last |
| Image size | 128 × 128 |
| PPL / PIPL pairs | 2,000 pairs each (4,000 generated images per metric) |
| Seed | 0; set Python, NumPy, PyTorch, and CUDA seeds |
| Generation batch size | 16 pairs; PPL encoding concatenates 32 input images |
| DDIM steps | 100 |
| ε | Main runs explicitly set `1e-4` |
| PPL sampling | `end`; interpolation location fixed at t=0 |
| PIPL regions | `enc`, `mid`, `dec` |
| Directions per region | 10 (30 region–direction combinations in total) |
| Perceptual net | `lpips.LPIPS(net='vgg').eval()` |
| Pair storage | CPU float32 `.pt`; no PNG/JPEG quantization |
| Statistics | Trim 1st/99th percentiles of ε²-normalized distances; report the mean |

Both images in a pair use the same initial Gaussian noise so that independent noise variation is not mixed into path change. Generation is conditioned from random noise; we do not run DDIM inversion on each real image.

Please pass ε explicitly: script defaults are not unified. The newer path scripts default to `1e-2`, so relying on defaults will not reproduce `1e-4` results.

## 3. PPL: local interpolation between randomly paired condition codes

### 3.1 Pairing

For the first 2,000 test images, draw a random permutation with `np.random.default_rng(0)`. Pair each index with the previous one in a cycle so every image appears once as a start and once as an end, with no self-pairs.

Newer metadata stores the full `ppl_partner_indices` so pairings can be checked across checkpoints.

### 3.2 Image generation

Let z₁, z₂ be encodings of two real images, n the common initial Gaussian noise, and G the 100-step conditional DDIM generator:

```text
z(t) = (1 − t) z₁ + t z₂
x₀ = G(n, z(t))
x₁ = G(n, z(t + ε))
```

Please use `torch.lerp` (linear interpolation). Spherical interpolation and unit-normalizing z₁, z₂ are not used here. With the current `end` setting, t is fixed at 0, so the comparison is:

```text
x₀ = G(n, z₁)
x₁ = G(n, z₁ + ε(z₂ − z₁))
```

Per-pair PPL value:

```text
d_PPL = LPIPS(x₀, x₁) / ε²
```

Trim and average the 2,000 values. This protocol evaluates smoothness near an interpolation endpoint in the encoded condition space; it is not a PPL with uniform sampling over the full path.

## 4. PIPL: local perturbation along regional directions

### 4.1 Direction source

Extract directions from the EMA model’s `style_enc`, `style_mid`, and `style_dec` regional projectors. The effective Householder matrix is reconstructed as in code:

```text
W = H(U) S H(V)
```

Run SVD of W on CPU float32. The third return of `torch.linalg.svd` is Vh; its leading rows are the right singular vectors used here. Each vector has unit L2 norm.

The effective-rank threshold is `max(singular_values) × 1e-6`, with at most the first 10 valid directions per region. The completed 50k run has 10 directions in every region. Directions come from decomposing the projector matrices; they are not the Pose / Age / Gender-selected directions from re-scoring.

### 4.2 Where the perturbation is applied

Let z be the real-image encoding, r the selected region, and v the region direction:

```text
x_base = G(n, z)
x_edit = G_r(n, z; εv)
d_PIPL = LPIPS(x_base, x_edit) / ε²
```

G_r denotes generation with a regional direction applied. The actual perturbation is added to the internal `cond_emb` before the selected `style_r` projector; the other two regions keep the unperturbed condition. For models with a time/style MLP, this is not the same as editing the raw code z and passing it to all regions.

Base and edited images use the same initial noise and keep identical DDIM steps / other settings. Please use a one-sided difference (0 vs +ε) rather than a centered −ε / +ε difference.

### 4.3 How samples are assigned to directions

The 2,000 images are split into 125 batches of size 16. Each batch uses one region–direction combination, cycling through:

```text
enc:0, ..., enc:9, mid:0, ..., mid:9, dec:0, ..., dec:9
```

**PIPL totals 2,000 pairs** (not 2,000 × 30, and not every direction on every image).

Current assignment:

- `enc:0`–`enc:4`: 80 pairs each
- Remaining 25 groups: 64 pairs each
- Total: 5 × 80 + 25 × 64 = 2,000 pairs

Overall PIPL merges all 2,000 distances, then trims and averages; it is not an equal-weight average of 30 group means. `per_group` stores separately trimmed stats per group. When comparing methods or checkpoints, please keep batch size, image order, and direction assignment the same.

## 5. LPIPS inputs and statistics

`model.render` maps diffusion outputs to image space. At scoring time, saved float images are mapped by ×2 − 1 into VGG-LPIPS as implemented; we do not apply extra face alignment, crop, resize, PNG round-trip, or an additional clamp on scored images.

Divide each pair’s LPIPS distance by ε², then pool all distances. Compute the 1st and 99th percentiles q₁, q₉₉ and keep values satisfying:

```text
q₁ ≤ dᵢ ≤ q₉₉
```

Report `mean` and `std` of the kept values; `std` is NumPy’s default population standard deviation, not a standard error or CI. Save `raw_count`, `count`, `trim_lower`, `trim_upper`. Because boundary values are included and duplicates may exist, the kept count need not be exactly 1,960.

At ε=10⁻⁴, ε²=10⁻⁸, so numerical differences are amplified. When comparing runs, please keep ε, image storage precision, LPIPS version, and generation protocol fixed. A fixed seed does not guarantee bit-identical results across GPUs or software versions.

## 6. Completed 50k / ε=10⁻⁴ results

Historical samples and results (example path on the original cluster):

```text
output/ffhq128_hop_50k_metric_samples
```

Numbers below were checked against that directory’s `metadata.json` and `metrics.json`:

| Metric | Raw mean | Raw std | Raw pairs | Trimmed pairs |
|---|---:|---:|---:|---:|
| PPL | 310.6407470703125 | 449.8682556152344 | 2,000 | 1,960 |
| PIPL | 21.142597198486328 | 39.47710037231445 | 2,000 | 1,980 |

Result files store raw scores with no extra ×/÷ 10. If a paper table uses scaling factor 10 (display = raw / 10), this run becomes **PPL=31.06, PIPL=2.11**. Display scaling does not change the metric definition; please avoid mixing scores from different ε, weights, or architectures.

## 7. Re-score historically saved samples

The following only scores existing float pairs (no new generation, no Slurm). Please make sure CUDA and LPIPS weights are available:

```bash
cd models/DiffAE
# activate your DiffAE conda env
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python - <<'PY'
import json
from pathlib import Path
import lpips
import torch
from evaluate_ffhq128_metrics import seed_everything
from score_ffhq128_saved_samples import score_pairs

sample_dir = Path('output/ffhq128_hop_50k_metric_samples')
metadata = json.loads((sample_dir / 'metadata.json').read_text())
assert metadata['epsilon'] == 1e-4
assert metadata['samples'] == 2000
assert metadata['method'] == 'hop'
seed_everything(0)
device = torch.device('cuda:0')
perceptual_model = lpips.LPIPS(net='vgg').to(device).eval()
results = {
    'source_checkpoint': metadata['source_checkpoint'],
    'sample_metadata': metadata,
}
for metric in ('ppl', 'pipl'):
    results[metric] = score_pairs(
        sample_dir, metadata, metric, device, perceptual_model)
(sample_dir / 'path_metrics_recomputed.json').write_text(
    json.dumps(results, indent=2))
print({k: results[k]['mean'] for k in ('ppl', 'pipl')})
PY
```

`score_pairs` writes or updates `ppl_distances.npy` and `pipl_distances.npy`; the snippet above saves a new `path_metrics_recomputed.json` and does not overwrite the historical `metrics.json`.

## 8. Newer path-only generation workflow

Please use an explicit ε=10⁻⁴ and a fresh output directory: generate samples, then score. The generation directory should be missing or empty; for repeats, please choose a new path.

```bash
cd models/DiffAE
# activate your DiffAE conda env
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_MODE=disabled
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python -u generate_ffhq128_path_samples.py \
  --mode checkpoint \
  --checkpoint checkpoints/ffhq128_householder_random_single4090/checkpoints/step_050000.ckpt \
  --data-path datasets \
  --output-dir output/c_hop_50k_path_eps1e4_reproduction \
  --method hop --use-time-style-mlp \
  --seed 0 --device cuda:0 --num-workers 0 --batch-size 16 \
  --ddim-steps 100 --samples 2000 --epsilon 1e-4 \
  --ppl-sampling end --pipl-directions 10

python -u score_ffhq128_path_samples.py \
  --sample-dirs output/c_hop_50k_path_eps1e4_reproduction \
  --steps 50000 --method hop --epsilon 1e-4 --device cuda:0 \
  --output-dir output/c_hop_50k_path_eps1e4_reproduction_metrics
```

**Please note that the newer pipeline and the legacy pipeline use different RNG order.** Legacy `generate_ffhq128_metric_samples.py` sets the seed once, then generates FID, then PPL, then PIPL, so later noise continues from the RNG state left by earlier stages. Newer `generate_ffhq128_path_samples.py` reseeds before PPL and before PIPL. The commands above match the metric configuration but are not guaranteed to reproduce Section 6’s historical numbers exactly. To recompute historical scores, please use Section 7 on existing samples; to compare many checkpoints, please stick to one generation pipeline throughout.

Historical metadata lacks newer required fields such as `mode`, `noise_seed_per_metric`, and `ppl_partner_indices`, so it cannot be passed directly to the newer strict-validation CLI. Please avoid inventing those fields just to force validation to pass.

## 9. Outputs and checks

A newer generation directory typically contains:

```text
metadata.json
ppl_pairs/0000.pt ... 0124.pt     # start, end
pipl_pairs/0000.pt ... 0124.pt    # base, edited, region, direction index
```

After scoring it also contains:

```text
ppl_distances.npy                # 2,000 ε²-normalized raw distances
pipl_distances.npy
metrics.json
```

When scores are written to their own folder, that folder includes `metrics_summary.json` and `metrics_table.md`. The newer CLI also supports `--validate-only` if you would like to check metadata and file layout without loading the model for perceptual scoring.

Before comparing results, please double-check: the actual checkpoint, EMA and model architecture, ε, fixed images and PPL pairing, noise RNG workflow, DDIM steps, batch size, PIPL directions and group counts, LPIPS architecture, and trimming rule. The ε used in PPL/PIPL is a local finite-difference scale; it is not the same parameter as α or the traversal range in semantic traversal figures.
