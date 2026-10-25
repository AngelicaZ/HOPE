#!/bin/bash
#SBATCH --gres gpu:1
#
# Closed-form factors + traversal for a Householder-fine-tuned FFHQ-U 256 model.
# Usage (from models/StyleGANHuman/training_scripts/sg3/):
#   NETWORK=training-runs_256/<run>/network-snapshot-<kimg>.pkl sbatch -p <partition> scripts/Visualize_256_ffhqu.sh
# Env: DATA (./ffhq-256x256.zip), FACTOR (./factor_256.pt), OUTDIR (./training-runs_256)

set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-stylegan3}"

NETWORK=${NETWORK:?set NETWORK to the fine-tuned snapshot .pkl}
FACTOR=${FACTOR:-./factor_256.pt}

python closed_form_factorization.py --out "$FACTOR" --resume_pkl "$NETWORK" --is_ortho

python apply_factor.py --outdir="${OUTDIR:-./training-runs_256}" --cfg=stylegan3-r \
  --data="${DATA:-./ffhq-256x256.zip}" \
  --gpus=1 --batch=1 --gamma=1 --mirror=1 --aug=noaug --cbase=16384 --dlr=0.0025 --mbstd-group 1 \
  --resume="$NETWORK" --diag_size 10 --is_ortho True --factor "$FACTOR"
