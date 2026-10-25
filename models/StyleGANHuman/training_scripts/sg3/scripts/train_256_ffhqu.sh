#!/bin/bash
#SBATCH --gres gpu:3
#
# Fine-tune StyleGAN3-R on FFHQ-U 256 with the Householder projector.
# Usage (from models/StyleGANHuman/training_scripts/sg3/):
#   sbatch -p <partition> scripts/train_256_ffhqu.sh
# Env: DATA (./ffhq-256x256.zip), RESUME (./stylegan3-r-ffhqu-256x256.pkl), OUTDIR (./training-runs_256)

set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-stylegan3}"

python train.py --outdir="${OUTDIR:-./training-runs_256}" --cfg=stylegan3-r \
  --data="${DATA:-./ffhq-256x256.zip}" \
  --gpus=3 --batch=6 --gamma=1 --mirror=1 --aug=noaug --cbase=16384 --dlr=0.0025 --mbstd-group 2 \
  --resume="${RESUME:-./stylegan3-r-ffhqu-256x256.pkl}" --diag_size 10 --is_ortho True --snap 5
