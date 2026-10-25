# DiffAE + HOPE

Householder Orthogonal Projector for Editing (and LoRA baseline) on top of [Diffusion Autoencoders](https://github.com/phizaz/diffae).

Please run all commands from this directory (`models/DiffAE/`) with the `hope` conda env.

## Layout

```text
models/DiffAE/
├── model/                 # UNet autoencoder + projector layers
├── diffusion/             # DDIM / training diffusion
├── experiment.py          # Lightning training / eval loop
├── templates.py           # experiment configs
├── run_*.py               # Python entry points (train / ablation / eval)
├── ppl.py / pipl.py       # legacy PPL / PIPL evaluators
├── evaluate_ffhq128_*_readme.py  # README-protocol PPL / PIPL
├── closed_form_factorization.py / apply_factor.py
├── scripts/
│   ├── run.sh             # generic Slurm job (runs one Python entry point)
│   ├── lib.sh             # shared shell helpers
│   └── submit/            # one launcher per experiment (see scripts/README.md)
└── datasets/              # (gitignored) put *.lmdb here
```

## Setup

Please create the environment from the repository top level, then enter this folder:

```bash
# from the repository top level
conda env create -f HOPE.yaml
conda activate hope
cd models/DiffAE
```

Please place LMDB datasets under `datasets/` (e.g. `ffhq256.lmdb`). You can build them with `data_resize_*.py` if needed.

## Basic DiffAE + Householder (FFHQ128)

```bash
python run_ffhq128.py
```

Closed-form factors and latent traversal:

```bash
python closed_form_factorization.py --out [factor_path] [checkpoint_path] --is_ortho --diag_size 10
python apply_factor.py --output_dir [output_path] --ckpt [checkpoint_path] --factor [factor_path] --size 128
```

## Projector ablations (rank / LoRA)

Cold-start DiffAE architecture + random projector, with saves at 20k / 50k:

```bash
# Householder rank 5 / 10 / 512
python run_ffhq128_projector_ablation.py --method householder --rank 10
# LoRA rank 10
python run_ffhq128_projector_ablation.py --method lora --rank 10

# or on Slurm (all arms, with resume jobs); see scripts/README.md
bash scripts/submit/projector_ablation.sh train
```

FID:

```bash
python run_ffhq128_projector_ablation.py --mode eval --method householder --rank 10 --step 50000 --gpus 0
```

## PPL / PIPL

**Recommended (paper protocol):** fixed `test[0:2000]`, ε=`1e-4`, seed `0`. For full details, please see [`docs/DIFFAE_PPL_PIPL.md`](../../docs/DIFFAE_PPL_PIPL.md).

```bash
python evaluate_ffhq128_ppl_readme.py --arm householder10
python evaluate_ffhq128_pipl_readme.py --arm householder10
# arms: householder5 | householder10 | householder512 | lora10 | vanilla
```

Legacy evaluators (`ppl.py` / `pipl.py`) remain for older experiments; they shuffle and sample directions differently, so please avoid mixing those numbers with the README protocol.

## Weights

| Dataset | Architecture | Resolution | Fine-tuned | Pretrained DiffAE |
| --- | --- | :---: | :---: | :---: |
| FFHQ | DiffAE + HOPE | 128 | [download](https://drive.usercontent.google.com/download?id=1Fwc9hdgUWnYXnbhceUhZlYviUTYDBjdu&export=download&authuser=0) | [folder](https://drive.google.com/drive/folders/11pdjMQ6NS8GFFiGOq3fziNJxzXU1Mw3l) |

Please put checkpoints under `checkpoints/` (gitignored).

## Acknowledgement

DiffAE architecture: [Preechakul et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Preechakul_Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_Representation_CVPR_2022_paper.html).
