#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0927_proj_fid_%j.o
#SBATCH -e e_file/eval/0927_proj_fid_%j.e

# Usage:
#   sbatch --export=METHOD=householder,RANK=10,STEP=20000 scripts/ablation/ffhq128_projector_ablation_fid.sh

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
mkdir -p out_file/eval o_file/eval e_file/eval evals

METHOD=${METHOD:?}
RANK=${RANK:?}
STEP=${STEP:?}

python run_ffhq128_projector_ablation.py \
  --mode eval --method ${METHOD} --rank ${RANK} --step ${STEP} --gpus 0 \
  >> out_file/eval/0927_${METHOD}_rank${RANK}_step${STEP}_fid.out 2>&1
