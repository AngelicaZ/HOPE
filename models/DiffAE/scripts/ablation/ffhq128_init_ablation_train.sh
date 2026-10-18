#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:2
#SBATCH -o o_file/train/0222_ffhq128_init_ablation_%j.o
#SBATCH -e e_file/train/0222_ffhq128_init_ablation_%j.e

# Initialization ablation train.
# Usage:
#   sbatch --export=INIT=svd_identity scripts/ablation/ffhq128_init_ablation_train.sh
#   sbatch --export=INIT=random      scripts/ablation/ffhq128_init_ablation_train.sh

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}" 2>/dev/null || true
conda activate "${CONDA_ENV:-hope}"
export TMPDIR="${TMPDIR:-/tmp}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFAE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DIFFAE_ROOT}"
mkdir -p out_file/train o_file/train e_file/train "$TMPDIR"

INIT=${INIT:-svd_identity}

python run_ffhq128_init_ablation.py --init ${INIT} \
  >> out_file/train/0222_ffhq128_init_${INIT}.out 2>&1
