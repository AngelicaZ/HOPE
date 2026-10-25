#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH --gres gpu:4
#
# Fine-tune StyleGAN2 (FFHQ 1024) with the Householder projector on 4 GPUs.
# Usage (from models/StyleGAN2/):
#   DATA=/path/to/ffhq_lmdb CKPT=/path/to/pretrained.pt sbatch -p <partition> train_styleGAN2_FFHQ.sh
# Optional: OUT (default ckpt/ffhq1024), SAMPLES (default sample/ffhq1024), CONDA_ENV (default hope).
# Training logs to Weights & Biases: run `wandb login` once, or set WANDB_MODE=offline.

set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-hope}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python -m torch.distributed.launch --nproc_per_node=4 --master_port=9032 \
  train_1024.py --batch 8 "${DATA:?set DATA to the FFHQ lmdb}" \
  --ckpt "${CKPT:?set CKPT to the pretrained checkpoint}" \
  --size 1024 --ortho_id -2 --diag_size 10 --iter 10000000 \
  --checkpoints_dir "${OUT:-ckpt/ffhq1024}/" --sample_dir "${SAMPLES:-sample/ffhq1024}/" \
  --loadd --training_FULL
