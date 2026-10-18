#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0222_ffhq128_init_eval_%j.o
#SBATCH -e e_file/eval/0222_ffhq128_init_eval_%j.e

# Usage:
#   sbatch --export=INIT=svd_identity,STEP=0 scripts/ablation/ffhq128_init_ablation_eval.sh

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}" 2>/dev/null || true
conda activate "${CONDA_ENV:-hope}"
export TMPDIR="${TMPDIR:-/tmp}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
mkdir -p "$TORCH_HOME/checkpoints" "$XDG_CACHE_HOME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFAE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DIFFAE_ROOT}"
mkdir -p out_file/eval o_file/eval e_file/eval

INIT=${INIT:?INIT required}
STEP=${STEP:?STEP required}

python run_ffhq128_init_ablation.py --mode eval --init ${INIT} --step ${STEP} --gpus 0 \
  >> out_file/eval/0222_ffhq128_init_${INIT}_step${STEP}.out 2>&1
