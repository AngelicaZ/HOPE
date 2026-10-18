#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/ffhq128_base_ppl_pipl_%j.o
#SBATCH -e e_file/eval/ffhq128_base_ppl_pipl_%j.e

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
mkdir -p factors evals out_file/eval o_file/eval e_file/eval

CKPT=checkpoints/ffhq128_autoenc_130M/last.ckpt
FACTOR=factors/ffhq128_autoenc_130M.pt
OUT_NAME=ffhq128_autoenc_130M
OUT=out_file/eval/ffhq128_autoenc_130M_ppl_pipl.out

{
  echo "=== closed_form_factorization (vanilla DiffAE) ==="
  python closed_form_factorization.py \
    --out ${FACTOR} \
    ${CKPT}

  echo "=== PPL vanilla DiffAE ==="
  python ppl.py \
    --ckpt ${CKPT} \
    --vanilla \
    --out_name ${OUT_NAME} \
    --size 128 \
    --batch 16 \
    --n_sample 5000 \
    --eps 1e-1 \
    --sampling full

  echo "=== PIPL vanilla DiffAE ==="
  python pipl.py \
    --ckpt ${CKPT} \
    --vanilla \
    --out_name ${OUT_NAME} \
    --factor ${FACTOR} \
    --size 128 \
    --batch 16 \
    --n_sample 5000 \
    --eps 1e-1 \
    --sampling full

  echo "=== done ==="
} > ${OUT} 2>&1
