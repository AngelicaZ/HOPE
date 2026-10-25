#!/bin/bash
#SBATCH --gres gpu:1
#
# SeFa baseline: closed-form factors + traversal on the original (non-orthogonal) FFHQ-U 256 model.
# Usage (from models/StyleGANHuman/training_scripts/sg3/):
#   sbatch -p <partition> scripts/Visualize_256_ffhq_sefa.sh
# Env: NETWORK (./stylegan3-r-ffhqu-256x256.pkl), DATA (./ffhq-256x256.zip),
#      FACTOR (./factor_256_sefa.pt), OUTDIR (./training-runs_256)

set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-stylegan3}"

NETWORK=${NETWORK:-./stylegan3-r-ffhqu-256x256.pkl}
FACTOR=${FACTOR:-./factor_256_sefa.pt}

python closed_form_factorization.py --out "$FACTOR" --resume_pkl "$NETWORK"

python apply_factor.py --outdir="${OUTDIR:-./training-runs_256}" --cfg=stylegan3-r \
  --data="${DATA:-./ffhq-256x256.zip}" \
  --gpus=1 --batch=1 --gamma=1 --mirror=1 --aug=noaug --cbase=16384 --dlr=0.0025 --mbstd-group 1 \
  --resume="$NETWORK" --diag_size 10 --is_ortho False --factor "$FACTOR"
