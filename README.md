# HOPE: Householder Orthogonal Projector for Editing

<p align="center">
  <a href="https://arxiv.org/abs/2307.08012"><img src="https://img.shields.io/badge/arXiv-2307.08012-b31b1b.svg" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license"></a>
</p>

Official code for **Householder Projector for Unsupervised Latent Semantics Discovery** ([ICCV 2023](https://arxiv.org/abs/2307.08012)), with an extended journal version **Householder Projector for Unsupervised Latent Semantics Discovery of GANs and Diffusion** (under review).

Projector-based semantic discovery reads salient directions of the map that predicts feature parameters from latent or conditioning codes. In a pretrained generator this map usually misses orthogonality, therefore a naive full-rank orthogonalization often underperforms: each direction carries only a weak signal, and semantic edits become difficult to perceive. HOPE instead uses a **compact Householder orthogonal** factorization (rank-k for perceptually meaningful factors) and adapts each architecture with a gentle fine-tune that keeps the pretrained generative prior safe. The same projector generalizes across style-based generators (StyleGAN2 / StyleGAN3 / StyleGAN-Human / EG3D) and semantically guided diffusion autoencoders (DiffAE), yielding clear progress on trustworthy latent semantics and interpretability, while preserving image quality.

## Highlights

- **Method**: compact (rank-k) Householder orthogonal projector; preserves linearity; avoids naive full-rank orthogonalization
- **Goal**: more interpretable, trustworthy latent directions
- **Generalization**: one projector design across 2D GANs, 3D-aware EG3D, and diffusion autoencoders (DiffAE)
- **Adaptation**: gentle, safe fine-tuning on StyleGAN2 / StyleGAN3 / StyleGAN-Human / DiffAE / EG3D
- **Metrics**: FID, PPL, PIPL (path smoothness vs. projector direction sensitivity)

## How Householder Projector is used in each architecture

Across all settings, HOPE reparameterizes the linear map that predicts feature modulation parameters from a latent or conditioning code as

$$
W = Q(U)\,S\,Q(V)
$$

where $Q(\cdot)$ is a product of Householder reflections and $S$ is a diagonal mask of width `diag_size` (the effective rank k). This design preserves linearity, so after a gentle fine-tune that respects the host architecture, semantic directions can be recovered via SVD of $W$ and used for closed-form traversal. Throughout, we keep the projector orthogonal and rank-k: a compact basis that stays interpretable, avoiding the drawbacks of a naive full-rank orthogonal basis.

Please replace bracketed placeholders with your own paths. Fuller flags and eval recipes are in the [Quick links](#quick-links) guides.

**StyleGAN2.**  
The projector stands on the style modulation path (affine layers from $w$ to per-layer scales/biases). Please run from `models/StyleGAN2/` with `hope` activated (`--ortho_id` / `--diag_size` enable Householder):

```bash
# fine-tune (FFHQ 1024 example)
python -m torch.distributed.launch train_1024.py \
  --nproc_per_node=4 --master_port=9032 \
  train_1024.py --batch 8 [dataset_path] \
  --ckpt [pretrained_model] --size 1024 --ortho_id -2 --diag_size 10 \
  --checkpoints_dir [save_model_path] --sample_dir [save_sample_path] \
  --loadd --training_FULL --iter 10000000

# closed-form factors, then traverse
python closed_form_factorization.py --out [factor_path] [save_model_path] --is_ortho
python apply_factor.py --output_dir [save_results_path] \
  --ckpt [save_model_path] --factor [factor_path] --ortho_id -2 --size 1024
```

**StyleGAN3.**  
Same conceptual slot as StyleGAN2; please run from `models/StyleGAN3/`:

```bash
# fine-tune (AFHQv2 example)
python train.py --outdir=[save_sample_path] --cfg=stylegan3-r \
  --data=[dataset_path] --gpus=3 --batch-gpu=2 --batch=6 --gamma=16.4 \
  --mbstd-group 2 --resume=[pretrained_model] \
  --diag_size 10 --is_ortho True --snap 5

python closed_form_factorization.py --out [factor_path] \
  --resume_pkl [save_model_path] --is_ortho
python apply_factor.py --outdir=[save_results_path] --cfg=stylegan3-r \
  --data=[dataset_path] --gpus=1 --batch-gpu=1 --batch=1 --gamma=16.4 \
  --mbstd-group 2 --resume=[save_model_path] \
  --diag_size 10 --is_ortho True --factor [factor_path]
```

**StyleGAN-Human.**  
Uses the StyleGAN3 training stack on full-body data. Please enter `models/StyleGANHuman/training_scripts/sg3/` first:

```bash
python train.py --outdir=[save_results_path] --cfg=stylegan3-r \
  --gpus=4 --batch=16 --gamma=12.4 --mbstd-group 4 \
  --mirror=1 --aug=noaug --data=[dataset_path] --square=False --snap=5 \
  --resume=[pretrained_model] --diag_size 10 --is_ortho True

python closed_form_factorization.py --out [factor_path] \
  --resume_pkl [save_model_path] --is_ortho
python apply_factor.py --outdir=[save_results_path] --cfg=stylegan3-r \
  --data=[dataset_path] --gpus=1 --batch-gpu=1 --batch=1 \
  --resume=[save_model_path] --diag_size 10 --is_ortho True --factor [factor_path]
```

**EG3D.**  
Builds on a StyleGAN2-style generator; the projector again stands on style modulation, while the neural renderer is left unchanged. Edits act on 3D-consistent volumetric features. Please run from `models/EG3D/`:

```bash
python closed_form_factorization.py --out [factor_path] [save_model_path] --is_ortho
python apply_factor.py --outdir [save_results_path] \
  --network [save_model_path] --factor [factor_path] --is_ortho
```

**DiffAE.**  
No StyleGAN $w$-space: projectors stand on the conditioning path (`style_enc` / `style_mid` / `style_dec`) that maps semantic code $c$ into the UNet. Please run from `models/DiffAE/`:

```bash
# fine-tune DiffAE + Householder (FFHQ128)
python run_ffhq128.py
# or: bash scripts/train/ffhq128_projector_train.sh

python closed_form_factorization.py --out [factor_path] [checkpoint_path] \
  --is_ortho --diag_size 10
python apply_factor.py --output_dir [output_path] \
  --ckpt [checkpoint_path] --factor [factor_path] --size 128

# PPL / PIPL (paper protocol); see docs/DIFFAE_PPL_PIPL.md
python evaluate_ffhq128_ppl_readme.py --arm householder10
python evaluate_ffhq128_pipl_readme.py --arm householder10
```

## Code organization

```text
HOPE/
├── README.md
├── HOPE.yaml
├── docs/
├── imgs/
├── pretrained/
├── nvdiffrast/
└── models/
    ├── StyleGAN2/
    ├── StyleGAN3/
    ├── StyleGANHuman/
    ├── DiffAE/
    └── EG3D/
```

Large artifacts (`checkpoints/`, `datasets/`, logs, factors) are **gitignored**. Please download weights from the tables below, or train your own models.

## Environment

Please create the conda environment and activate it:

```bash
conda env create -f HOPE.yaml
conda activate hope
```

A GPU with CUDA is required for training and evaluation.

## Quick links

Please see the guides below for architecture-specific setup and commands.

| Architecture | Guide | Code |
| --- | --- | --- |
| StyleGAN2 | [docs/STYLEGAN2.md](docs/STYLEGAN2.md) | [`models/StyleGAN2/`](models/StyleGAN2/) |
| StyleGAN3 | [docs/STYLEGAN3.md](docs/STYLEGAN3.md) | [`models/StyleGAN3/`](models/StyleGAN3/) |
| StyleGAN-Human | [docs/STYLEGAN_HUMAN.md](docs/STYLEGAN_HUMAN.md) | [`models/StyleGANHuman/`](models/StyleGANHuman/) |
| DiffAE | [models/DiffAE/README.md](models/DiffAE/README.md) | [`models/DiffAE/`](models/DiffAE/) |
| EG3D | [models/EG3D/README.md](models/EG3D/README.md) | [`models/EG3D/`](models/EG3D/) |
| DiffAE PPL / PIPL protocol | [docs/DIFFAE_PPL_PIPL.md](docs/DIFFAE_PPL_PIPL.md) |  |

## Datasets

Please download datasets from their official sources, then convert them with the scripts below:

- **StyleGAN2**: [`models/StyleGAN2/prepare_data.py`](models/StyleGAN2/prepare_data.py)
- **StyleGAN3**: [`models/StyleGAN3/prepare_data.py`](models/StyleGAN3/prepare_data.py)
- **DiffAE**: put LMDB files under `models/DiffAE/datasets/` (e.g. `ffhq256.lmdb`, `bedroom256.lmdb`, `horse256.lmdb`). Conversion helpers: `models/DiffAE/data_resize_*.py`

## Pretrained models

Please find pretrained StyleGANs and our fine-tuned checkpoints below:

| Dataset | Architecture | Resolution | Fine-tuned | Pretrained |
| --- | :---: | :---: | :---: | :---: |
| FFHQ | StyleGAN2 | 256 | to be updated | [link](https://drive.google.com/file/d/1mbmGRkrdZfWwCaRRV9OF_xW2AH1Cj0-H/view?usp=sharing) |
| FFHQ | StyleGAN2 | 1024 | [link](https://drive.google.com/file/d/1MEBk5Br2stbaSNl-4ImQQ1qUJnBBQPjM/view?usp=sharing) | [link](https://drive.google.com/file/d/1EqgBZKvqiC32yjGl-dSG8lqXcU927JmI/view?usp=sharing) |
| LSUN Church | StyleGAN2 | 256 | [link](https://drive.google.com/file/d/15Nzei6bMev3gJu3votONi9LqcqU1ihiN/view?usp=sharing) | [link](https://drive.google.com/file/d/1Be3fKnwHYnbHUBMTpWfjKzF5YD828IjB/view?usp=sharing) |
| LSUN Cat | StyleGAN2 | 256 | [link](https://drive.google.com/file/d/1bon039aXMfsY72ApRv_lJ3160HDsa353/view?usp=sharing) | [link](https://drive.google.com/file/d/1icL6XqN7-oLpcYBzODQnF2Pd2KtkT-R1/view?usp=sharing) |
| AFHQv2 | StyleGAN3 | 512 | [link](https://drive.google.com/file/d/1OZsu5RPeBbxk4mNfqEoq0e_Af5GZFpf9/view?usp=sharing) | [link](https://drive.google.com/file/d/1HJUlp-Yfe7bGLV5XktIVLBZxmY1Q2ep8/view?usp=sharing) |
| MetFaces | StyleGAN3 | 1024 | [link](https://drive.google.com/file/d/14_Qt4n5N1n45c0-BATXjZ-BpMEzWhbL1/view?usp=sharing) | [link](https://drive.google.com/file/d/1vPo-eojSbXt9FHsMCEv1jLc1pQhqmcdy/view?usp=sharing) |
| SHHQv1 | StyleGAN3 | 512×256 | [link](https://drive.google.com/file/d/1y40QPk_OAS-1pNcpajt-e_wuWlAJcMQo/view?usp=sharing) | [link](https://drive.google.com/file/d/1tyi4hVjfJiGVrcukVRbdJaMkmR8yIvbK/view?usp=sharing) |
| FFHQ | DiffAE | 128 | [link](https://drive.usercontent.google.com/download?id=1Fwc9hdgUWnYXnbhceUhZlYviUTYDBjdu&export=download&authuser=0) | [link](https://drive.google.com/drive/folders/11pdjMQ6NS8GFFiGOq3fziNJxzXU1Mw3l) |

## Citation

If you use this code, please cite our ICCV paper and the extended journal version (under review):

```bibtex
@inproceedings{song2023householder,
  title={Householder Projector for Unsupervised Latent Semantics Discovery},
  author={Song, Yue and Zhang, Jichao and Sebe, Nicu and Wang, Wei},
  booktitle={ICCV},
  year={2023}
}

@article{zhang2026householder,
  title={Householder Projector for Unsupervised Latent Semantics Discovery of GANs and Diffusion},
  author={Zhang, Jichao and Zhang, Chenyu and Wang, Wei and Sebe, Nicu and Song, Yue},
  note={Under review},
  year={2026}
}
```

## Acknowledgements

This repository builds on [StyleGAN2](https://github.com/NVlabs/stylegan2), [StyleGAN3](https://github.com/NVlabs/stylegan3), [StyleGAN-Human](https://github.com/stylegan-human/StyleGAN-Human), [DiffAE](https://github.com/phizaz/diffae), and [EG3D](https://github.com/NVlabs/eg3d). Please see the component folders for their original licenses.

## Contact

Please feel free to reach out with questions or feedback:

- Jichao Zhang: `jichao.zhang@ouc.edu.cn`
- Chenyu Zhang: `chenyu.zhang@unitn.it`
- Yue Song: `yue.song@unitn.it`
