#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0222_ffhq128_init_ppl_pipl_%j.o
#SBATCH -e e_file/eval/0222_ffhq128_init_ppl_pipl_%j.e

# Usage:
#   sbatch --export=INIT=svd_identity,STEP=0,EPS=0.01 scripts/ablation/ffhq128_init_ablation_ppl_pipl.sh

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

INIT=${INIT:?INIT required}
STEP=${STEP:?STEP required}
EPS=${EPS:-0.01}
RANK=10

EXP=0222_ffhq128_autoenc_130M_multi_projector_init_${INIT}
CKPT=checkpoints/${EXP}/checkpoints/step=${STEP}.ckpt
FACTOR=factors/${EXP}_step${STEP}.pt
EPS_TAG=$(python -c "print(f'{float(\"${EPS}\"):g}'.replace('.', 'p'))")
OUT_NAME=${EXP}_step${STEP}_eps${EPS_TAG}
OUT=out_file/eval/${EXP}_step${STEP}_ppl_pipl_eps${EPS_TAG}.out

if [[ ! -f "$CKPT" ]]; then
  echo "missing ckpt: $CKPT" >&2
  exit 1
fi

{
  if [[ ! -f "$FACTOR" ]]; then
    echo "=== closed_form_factorization init=${INIT} step=${STEP} ==="
    python closed_form_factorization.py \
      --out ${FACTOR} \
      ${CKPT} \
      --is_ortho \
      --diag_size ${RANK}
  else
    echo "=== reuse factor ${FACTOR} ==="
  fi

  echo "=== PPL init=${INIT} step=${STEP} eps=${EPS} ==="
  python ppl.py \
    --ckpt ${CKPT} \
    --rank ${RANK} \
    --out_name ${OUT_NAME} \
    --size 128 \
    --batch 16 \
    --n_sample 5000 \
    --eps ${EPS} \
    --sampling full

  echo "=== PIPL init=${INIT} step=${STEP} eps=${EPS} ==="
  python pipl.py \
    --ckpt ${CKPT} \
    --rank ${RANK} \
    --out_name ${OUT_NAME} \
    --factor ${FACTOR} \
    --size 128 \
    --batch 16 \
    --n_sample 5000 \
    --eps ${EPS} \
    --sampling full

  echo "=== done init=${INIT} step=${STEP} eps=${EPS} ==="
} > ${OUT} 2>&1
