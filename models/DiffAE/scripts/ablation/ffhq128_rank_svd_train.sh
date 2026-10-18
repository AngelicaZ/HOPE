#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:2
#SBATCH -o o_file/train/0225_ffhq128_rank_svd_%j.o
#SBATCH -e e_file/train/0225_ffhq128_rank_svd_%j.e

# Usage:
#   sbatch --export=RANK=5   scripts/ablation/ffhq128_rank_svd_train.sh
#   sbatch --export=RANK=10  scripts/ablation/ffhq128_rank_svd_train.sh
#   sbatch --export=RANK=512 scripts/ablation/ffhq128_rank_svd_train.sh

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}" 2>/dev/null || true
conda activate "${CONDA_ENV:-hope}"
export TMPDIR="${TMPDIR:-/tmp}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFAE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DIFFAE_ROOT}"
mkdir -p out_file/train o_file/train e_file/train "$TMPDIR"

RANK=${RANK:?RANK required}

python run_ffhq128_rank_svd_ablation.py --rank ${RANK} \
  >> out_file/train/0225_ffhq128_rank${RANK}_svd.out 2>&1
