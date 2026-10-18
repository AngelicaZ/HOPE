#!/bin/bash
# Eval FID + PPL/PIPL for 0927 projector ablations.
# Default: all three arms x steps 0/20000/50000, eps=0.01
#
# Usage:
#   bash SUBMIT_0927_projector_eval.sh
#   STEPS="20000 50000" bash SUBMIT_0927_projector_eval.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFAE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DIFFAE_ROOT}"
EPS=${EPS:-0.01}
STEPS=${STEPS:-"0 20000 50000"}

ARMS=(
  "householder 10"
  "householder 512"
  "lora 10"
)

echo "=== 0927 projector eval ==="
echo "STEPS=${STEPS} EPS=${EPS}"

for arm in "${ARMS[@]}"; do
  set -- $arm
  METHOD=$1
  RANK=$2
  if [[ "$METHOD" == "householder" ]]; then
    EXP=0927_ffhq128_autoenc_rank${RANK}_svd_random
  else
    EXP=0927_ffhq128_autoenc_lora${RANK}_random
  fi
  for STEP in $STEPS; do
    CKPT=checkpoints/${EXP}/checkpoints/step=${STEP}.ckpt
    if [[ ! -f "$CKPT" ]]; then
      echo "skip missing: $CKPT"
      continue
    fi
    J1=$(sbatch --parsable --export=METHOD=${METHOD},RANK=${RANK},STEP=${STEP} \
      scripts/ablation/ffhq128_projector_ablation_fid.sh)
    echo "FID  ${METHOD} r=${RANK} step=${STEP} -> job $J1"
    J2=$(sbatch --parsable --export=METHOD=${METHOD},RANK=${RANK},STEP=${STEP},EPS=${EPS} \
      scripts/ablation/ffhq128_projector_ablation_ppl_pipl.sh)
    echo "PPL/PIPL ${METHOD} r=${RANK} step=${STEP} eps=${EPS} -> job $J2"
  done
done

squeue -u "$USER" | head
