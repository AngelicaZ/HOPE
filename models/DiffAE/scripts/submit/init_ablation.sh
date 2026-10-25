#!/bin/bash
# Projector initialization ablation (multi projector, Householder rank 10).
# The `random` arm is the steps ablation run (0214); this script trains the
# other arms under checkpoints/0222_ffhq128_autoenc_130M_multi_projector_init_<INIT>/.
#
# Usage (from models/DiffAE/):
#   bash scripts/submit/init_ablation.sh train        # train + resume job
#   bash scripts/submit/init_ablation.sh fid
#   bash scripts/submit/init_ablation.sh ppl_pipl     # legacy ppl.py / pipl.py
# Env: INIT (default svd_identity), STEPS (default: every saved step),
#      EPS (default 0.01), DRY_RUN=1

set -eo pipefail
[[ -f scripts/lib.sh ]] || { echo "error: please run from models/DiffAE/" >&2; exit 1; }
source scripts/lib.sh
cd_diffae_root

INIT=${INIT:-svd_identity}
EXP=0222_ffhq128_autoenc_130M_multi_projector_init_${INIT}
CKPT_DIR=checkpoints/$EXP/checkpoints
RANK=10
STEPS=${STEPS:-$(list_steps "$CKPT_DIR")}

case "${1:-}" in
  train)
    submit_with_resume "init-$INIT-train" --gres gpu:2 -- \
      run_ffhq128_init_ablation.py --init "$INIT" >/dev/null
    ;;
  fid)
    for STEP in $STEPS; do
      require_ckpt "$CKPT_DIR/step=$STEP.ckpt" || continue
      submit "init-$INIT-fid-$STEP" -- \
        run_ffhq128_init_ablation.py --mode eval --init "$INIT" --step "$STEP" --gpus 0 >/dev/null
    done
    ;;
  ppl_pipl)
    EPS=${EPS:-0.01}
    for STEP in $STEPS; do
      CKPT=$CKPT_DIR/step=$STEP.ckpt
      require_ckpt "$CKPT" || continue
      FACTOR=factors/${EXP}_step${STEP}.pt
      FJ=$(submit_factor "$CKPT" "$FACTOR" --is_ortho --diag_size $RANK)
      OUT_NAME=${EXP}_step${STEP}_eps$(eps_tag "$EPS") \
        submit_ppl_pipl "init-$INIT-$STEP" "$CKPT" "$FACTOR" "$FJ" --rank $RANK
    done
    ;;
  *)
    sed -n '2,12p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
