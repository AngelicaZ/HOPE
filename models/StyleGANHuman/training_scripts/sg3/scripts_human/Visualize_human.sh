#!/bin/bash
#SBATCH --gres gpu:1
#
# Closed-form factors + traversal for a Householder-fine-tuned StyleGAN-Human model.
# Usage (from models/StyleGANHuman/training_scripts/sg3/):
#   NETWORK=training-runs_human/<run>/network-snapshot-<kimg>.pkl DATA=/path/to/SHHQ-1.0 \
#     sbatch -p <partition> scripts_human/Visualize_human.sh
# Env: FACTOR (./factor_human.pt), OUTDIR (./visual_human)

set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-stylegan3}"

NETWORK=${NETWORK:?set NETWORK to the fine-tuned snapshot .pkl}
FACTOR=${FACTOR:-./factor_human.pt}

python closed_form_factorization.py --out "$FACTOR" --resume_pkl "$NETWORK" --is_ortho

python apply_factor.py --outdir="${OUTDIR:-./visual_human}" --cfg=stylegan3-r \
  --data="${DATA:?set DATA to the SHHQ folder}" \
  --gpus=1 --batch-gpu=1 --batch=1 --gamma=16.4 --mbstd-group 1 \
  --resume="$NETWORK" --diag_size 10 --is_ortho True --factor "$FACTOR"
