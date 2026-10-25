#!/bin/bash
# Fine-tuning steps ablation (multi projector, Householder rank 10).
# Training saves checkpoints at 0 / 10k / 25k / 50k / 100k under
# checkpoints/0214_ffhq128_autoenc_130M_multi_projector/checkpoints/.
#
# Usage (from models/DiffAE/):
#   bash scripts/submit/steps_ablation.sh train       # train + resume job
#   bash scripts/submit/steps_ablation.sh fid
#   bash scripts/submit/steps_ablation.sh ppl_pipl    # legacy ppl.py / pipl.py
# Env: STEPS (default: every saved step), EPS (default 1e-1), DRY_RUN=1

set -eo pipefail
[[ -f scripts/lib.sh ]] || { echo "error: please run from models/DiffAE/" >&2; exit 1; }
source scripts/lib.sh
cd_diffae_root

EXP=0214_ffhq128_autoenc_130M_multi_projector
CKPT_DIR=checkpoints/$EXP/checkpoints
RANK=10
STEPS=${STEPS:-$(list_steps "$CKPT_DIR")}

case "${1:-}" in
  train)
    mkdir -p "$CKPT_DIR"
    submit_with_resume steps-train --gres gpu:2 -- run_ffhq128_steps_ablation.py >/dev/null
    ;;
  fid)
    for STEP in $STEPS; do
      require_ckpt "$CKPT_DIR/step=$STEP.ckpt" || continue
      submit "steps-fid-$STEP" -- \
        run_ffhq128_steps_ablation.py --mode eval --step "$STEP" --gpus 0 >/dev/null
    done
    ;;
  ppl_pipl)
    EPS=${EPS:-1e-1}
    for STEP in $STEPS; do
      CKPT=$CKPT_DIR/step=$STEP.ckpt
      require_ckpt "$CKPT" || continue
      FACTOR=factors/${EXP}_step${STEP}.pt
      FJ=$(submit_factor "$CKPT" "$FACTOR" --is_ortho --diag_size $RANK)
      OUT_NAME=${EXP}_step${STEP}_eps$(eps_tag "$EPS") \
        submit_ppl_pipl "steps-$STEP" "$CKPT" "$FACTOR" "$FJ" --rank $RANK
    done
    ;;
  *)
    sed -n '2,11p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
