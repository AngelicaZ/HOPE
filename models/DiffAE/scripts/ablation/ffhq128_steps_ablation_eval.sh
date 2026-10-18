#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0214_ffhq128_steps_eval_%j.o
#SBATCH -e e_file/eval/0214_ffhq128_steps_eval_%j.e

# Usage:
#   sbatch --export=STEP=0     scripts/ablation/ffhq128_steps_ablation_eval.sh
#   sbatch --export=STEP=10000 scripts/ablation/ffhq128_steps_ablation_eval.sh
#   sbatch --export=STEP=25000 scripts/ablation/ffhq128_steps_ablation_eval.sh

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}" 2>/dev/null || true
conda activate "${CONDA_ENV:-hope}"
export TMPDIR="${TMPDIR:-/tmp}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFAE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DIFFAE_ROOT}"
mkdir -p out_file/eval o_file/eval e_file/eval "$TMPDIR"

STEP=${STEP:?STEP env var required}

python run_ffhq128_steps_ablation.py --mode eval --step ${STEP} --gpus 0 \
  >> out_file/eval/0214_ffhq128_steps_step${STEP}.out 2>&1
