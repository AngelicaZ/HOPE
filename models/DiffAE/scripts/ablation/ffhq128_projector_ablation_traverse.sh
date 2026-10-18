#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 04:00:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0927_proj_traverse_%j.o
#SBATCH -e e_file/eval/0927_proj_traverse_%j.e

# Usage:
#   sbatch --export=METHOD=householder,RANK=512,STEP=50000 \
#     scripts/ablation/ffhq128_projector_ablation_traverse.sh

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
mkdir -p factors out_file/eval o_file/eval e_file/eval

METHOD=${METHOD:?}
RANK=${RANK:?}
STEP=${STEP:?}
DEGREE=${DEGREE:-5}
N_SAMPLE=${N_SAMPLE:-7}
N_STEPS=${N_STEPS:-4}
T=${T:-20}

if [[ "$METHOD" == "householder" ]]; then
  EXP=0927_ffhq128_autoenc_rank${RANK}_svd_random
  FACTOR_FLAGS=(--is_ortho --diag_size ${RANK})
elif [[ "$METHOD" == "lora" ]]; then
  EXP=0927_ffhq128_autoenc_lora${RANK}_random
  FACTOR_FLAGS=(--is_lora --diag_size ${RANK})
else
  echo "unknown METHOD=$METHOD" >&2
  exit 1
fi

CKPT=checkpoints/${EXP}/checkpoints/step=${STEP}.ckpt
FACTOR=factors/${EXP}_step${STEP}.pt
OUT=out_file/eval/${EXP}_step${STEP}_traverse.out

if [[ ! -f "$CKPT" ]]; then
  echo "missing ckpt: $CKPT" >&2
  exit 1
fi

{
  if [[ ! -f "$FACTOR" ]]; then
    echo "=== closed_form_factorization ==="
    python closed_form_factorization.py \
      --out ${FACTOR} \
      ${CKPT} \
      "${FACTOR_FLAGS[@]}"
  else
    echo "=== reuse factor ${FACTOR} ==="
  fi

  echo "=== traverse method=${METHOD} rank=${RANK} step=${STEP} ==="
  python run_ffhq128_projector_ablation_traverse.py \
    --method ${METHOD} --rank ${RANK} --step ${STEP} \
    --factor ${FACTOR} \
    --n_sample ${N_SAMPLE} --n_steps ${N_STEPS} \
    --degree ${DEGREE} --T ${T}

  echo "=== done ==="
} > ${OUT} 2>&1
