#!/bin/bash
#SBATCH --gres gpu:4
#
# Fine-tune StyleGAN-Human (StyleGAN3-R, SHHQ) with the Householder projector.
# Usage (from models/StyleGANHuman/training_scripts/sg3/):
#   DATA=/path/to/SHHQ-1.0 sbatch -p <partition> scripts_human/train_human.sh
# Env: RESUME (./stylegan_human_v3_512.pkl), OUTDIR (./training-runs_human)

set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-stylegan3}"

python train.py --outdir="${OUTDIR:-./training-runs_human}" --cfg=stylegan3-r \
  --gpus=4 --batch=16 --gamma=12.4 --mbstd-group 4 \
  --mirror=1 --aug=noaug --data="${DATA:?set DATA to the SHHQ folder}" --square=False --snap=5 \
  --resume="${RESUME:-./stylegan_human_v3_512.pkl}" --diag_size 10 --is_ortho True
