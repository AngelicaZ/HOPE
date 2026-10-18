#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0927_proj_ppl_pipl_%j.o
#SBATCH -e e_file/eval/0927_proj_ppl_pipl_%j.e

# Usage:
#   sbatch --export=METHOD=householder,RANK=10,STEP=20000,EPS=0.01 \
#     scripts/ablation/ffhq128_projector_ablation_ppl_pipl.sh
#   # PIPL only (skip PPL):
#   sbatch --export=METHOD=householder,RANK=10,STEP=50000,EPS=1e-4,SKIP_PPL=1 \
#     scripts/ablation/ffhq128_projector_ablation_ppl_pipl.sh

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

METHOD=${METHOD:?}
RANK=${RANK:?}
STEP=${STEP:?}
EPS=${EPS:-0.01}
SKIP_PPL=${SKIP_PPL:-0}
SKIP_PIPL=${SKIP_PIPL:-0}
N_SAMPLE=${N_SAMPLE:-5000}

if [[ "$METHOD" == "householder" ]]; then
  EXP=0927_ffhq128_autoenc_rank${RANK}_svd_random
  LORA_FLAG=""
  FACTOR_FLAGS=(--is_ortho --diag_size ${RANK})
elif [[ "$METHOD" == "lora" ]]; then
  EXP=0927_ffhq128_autoenc_lora${RANK}_random
  LORA_FLAG="--lora"
  FACTOR_FLAGS=(--is_lora --diag_size ${RANK})
else
  echo "unknown METHOD=$METHOD" >&2
  exit 1
fi

CKPT=checkpoints/${EXP}/checkpoints/step=${STEP}.ckpt
FACTOR=factors/${EXP}_step${STEP}.pt
EPS_TAG=$(python -c "print(f'{float(\"${EPS}\"):g}'.replace('.', 'p'))")
OUT_NAME=${EXP}_step${STEP}_eps${EPS_TAG}_n${N_SAMPLE}
if [[ "$SKIP_PPL" == "1" && "$SKIP_PIPL" == "1" ]]; then
  echo "both SKIP_PPL and SKIP_PIPL set; nothing to do" >&2
  exit 1
elif [[ "$SKIP_PPL" == "1" ]]; then
  OUT=out_file/eval/${EXP}_step${STEP}_pipl_eps${EPS_TAG}.out
elif [[ "$SKIP_PIPL" == "1" ]]; then
  OUT=out_file/eval/${EXP}_step${STEP}_ppl_eps${EPS_TAG}.out
else
  OUT=out_file/eval/${EXP}_step${STEP}_ppl_pipl_eps${EPS_TAG}.out
fi

if [[ ! -f "$CKPT" ]]; then
  echo "missing ckpt: $CKPT" >&2
  exit 1
fi

{
  if [[ ! -f "$FACTOR" ]]; then
    echo "=== closed_form_factorization method=${METHOD} rank=${RANK} step=${STEP} ==="
    python closed_form_factorization.py \
      --out ${FACTOR} \
      ${CKPT} \
      "${FACTOR_FLAGS[@]}"
  else
    echo "=== reuse factor ${FACTOR} ==="
  fi

  if [[ "$SKIP_PPL" != "1" ]]; then
    echo "=== PPL method=${METHOD} rank=${RANK} step=${STEP} eps=${EPS} ==="
    python ppl.py \
      --ckpt ${CKPT} \
      --rank ${RANK} \
      ${LORA_FLAG} \
      --out_name ${OUT_NAME} \
      --size 128 \
      --batch 16 \
      --n_sample ${N_SAMPLE} \
      --eps ${EPS} \
      --sampling full
  else
    echo "=== skip PPL (SKIP_PPL=1) ==="
  fi

  if [[ "$SKIP_PIPL" != "1" ]]; then
    echo "=== PIPL method=${METHOD} rank=${RANK} step=${STEP} eps=${EPS} n_sample=${N_SAMPLE} ==="
    python pipl.py \
      --ckpt ${CKPT} \
      --rank ${RANK} \
      ${LORA_FLAG} \
      --out_name ${OUT_NAME} \
      --factor ${FACTOR} \
      --size 128 \
      --batch 16 \
      --n_sample ${N_SAMPLE} \
      --eps ${EPS} \
      --sampling full
  else
    echo "=== skip PIPL (SKIP_PIPL=1) ==="
  fi

  echo "=== done ==="
} > ${OUT} 2>&1
