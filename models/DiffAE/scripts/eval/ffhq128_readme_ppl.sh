#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 12:00:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0927_readme_ppl_%j.o
#SBATCH -e e_file/eval/0927_readme_ppl_%j.e

# Usage:
#   sbatch --export=ARM=householder5 scripts/eval/ffhq128_readme_ppl.sh

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}" 2>/dev/null || true
conda activate "${CONDA_ENV:-hope}"
export TMPDIR="${TMPDIR:-/tmp}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$TORCH_HOME/checkpoints" "$XDG_CACHE_HOME"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFAE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DIFFAE_ROOT}"
mkdir -p evals/readme_ppl out_file/eval o_file/eval e_file/eval

ARM=${ARM:?}
EPS=${EPS:-1e-4}
SAMPLES=${SAMPLES:-2000}

python -u evaluate_ffhq128_ppl_readme.py \
  --arm ${ARM} \
  --samples ${SAMPLES} \
  --batch-size 16 \
  --epsilon ${EPS} \
  --ddim-steps 100 \
  --seed 0 \
  --ppl-sampling end \
  >> out_file/eval/readme_ppl_${ARM}_eps${EPS}_n${SAMPLES}.out 2>&1
