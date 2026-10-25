#!/bin/sh
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32GB
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --time=72:00:00
#SBATCH -o logs/%x-%j.log
#
# Multi-node FFHQ256 training (2 nodes x 4 GPUs). Run from models/DiffAE/:
#   sbatch -p <partition> scripts/run_ffhq256_multinode.sh

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python run_ffhq256.py
