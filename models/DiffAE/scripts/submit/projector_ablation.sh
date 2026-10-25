#!/bin/bash
# Projector ablation: cold-start DiffAE + random projector, Householder vs LoRA.
# Saves at 0 / 20k / 50k under
#   checkpoints/0927_ffhq128_autoenc_rank<R>_svd_random/   (householder)
#   checkpoints/0927_ffhq128_autoenc_lora<R>_random/       (lora)
#
# Usage (from models/DiffAE/):
#   bash scripts/submit/projector_ablation.sh train       # train + resume job per arm
#   bash scripts/submit/projector_ablation.sh fid
#   bash scripts/submit/projector_ablation.sh ppl_pipl    # legacy ppl.py / pipl.py
#   bash scripts/submit/projector_ablation.sh sample   [extra args for the sample script]
#   bash scripts/submit/projector_ablation.sh traverse [extra args for the traverse script]
# Env: ARMS (default "householder:10 householder:512 lora:10"),
#      STEPS (default: every saved step), EPS (0.01), N_SAMPLE (5000),
#      SKIP_PPL=1 / SKIP_PIPL=1, DRY_RUN=1

set -eo pipefail
[[ -f scripts/lib.sh ]] || { echo "error: please run from models/DiffAE/" >&2; exit 1; }
source scripts/lib.sh
cd_diffae_root

ARMS=${ARMS:-"householder:10 householder:512 lora:10"}
CMD=${1:-}
shift || true
case "$CMD" in
  train|fid|ppl_pipl|sample|traverse) ;;
  *) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac

for ARM in $ARMS; do
  METHOD=${ARM%%:*}
  RANK=${ARM##*:}
  case "$METHOD" in
    householder)
      EXP=0927_ffhq128_autoenc_rank${RANK}_svd_random
      MODEL_FLAGS=(--rank "$RANK")
      FACTOR_FLAGS=(--is_ortho --diag_size "$RANK")
      ;;
    lora)
      EXP=0927_ffhq128_autoenc_lora${RANK}_random
      MODEL_FLAGS=(--rank "$RANK" --lora)
      FACTOR_FLAGS=(--is_lora --diag_size "$RANK")
      ;;
    *)
      echo "unknown method in ARMS: $ARM" >&2
      exit 1
      ;;
  esac
  CKPT_DIR=checkpoints/$EXP/checkpoints
  ARM_STEPS=${STEPS:-$(list_steps "$CKPT_DIR")}
  TAG=$METHOD$RANK

  if [[ "$CMD" == train ]]; then
    submit_with_resume "proj-$TAG-train" --gres gpu:2 -- \
      run_ffhq128_projector_ablation.py --method "$METHOD" --rank "$RANK" >/dev/null
    continue
  fi

  for STEP in $ARM_STEPS; do
    CKPT=$CKPT_DIR/step=$STEP.ckpt
    require_ckpt "$CKPT" || continue
    FACTOR=factors/${EXP}_step${STEP}.pt
    case "$CMD" in
      fid)
        submit "proj-$TAG-fid-$STEP" -- run_ffhq128_projector_ablation.py \
          --mode eval --method "$METHOD" --rank "$RANK" --step "$STEP" --gpus 0 >/dev/null
        ;;
      ppl_pipl)
        EPS=${EPS:-0.01}
        FJ=$(submit_factor "$CKPT" "$FACTOR" "${FACTOR_FLAGS[@]}")
        OUT_NAME=${EXP}_step${STEP}_eps$(eps_tag "$EPS")_n${N_SAMPLE:-5000} \
          submit_ppl_pipl "proj-$TAG-$STEP" "$CKPT" "$FACTOR" "$FJ" "${MODEL_FLAGS[@]}"
        ;;
      sample)
        submit "proj-$TAG-sample-$STEP" -t 02:00:00 -- run_ffhq128_projector_ablation_sample.py \
          --method "$METHOD" --rank "$RANK" --step "$STEP" "$@" >/dev/null
        ;;
      traverse)
        FJ=$(submit_factor "$CKPT" "$FACTOR" "${FACTOR_FLAGS[@]}")
        submit "proj-$TAG-traverse-$STEP" -t 04:00:00 $(after_ok "$FJ") -- \
          run_ffhq128_projector_ablation_traverse.py \
          --method "$METHOD" --rank "$RANK" --step "$STEP" --factor "$FACTOR" "$@" >/dev/null
        ;;
    esac
  done
done
