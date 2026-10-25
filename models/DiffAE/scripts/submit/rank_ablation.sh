#!/bin/bash
# Householder rank ablation (rank 5 / 10 / 512).
#
# Usage (from models/DiffAE/):
#   bash scripts/submit/rank_ablation.sh train           # run_ffhq128_rank_ablation.py
#   bash scripts/submit/rank_ablation.sh train_svd       # SVD-init variant (+ resume job)
#   bash scripts/submit/rank_ablation.sh train_lowrank   # LoRA-style variant (+ resume job)
#   bash scripts/submit/rank_ablation.sh fid
#   bash scripts/submit/rank_ablation.sh ppl_pipl        # legacy ppl.py / pipl.py, eps 1e-1
#   bash scripts/submit/rank_ablation.sh pipl_fair       # PIPL over the top-N_EIG directions only
# Env: RANKS (default "5 10 512"), EPS (1e-1), N_EIG (5),
#      CKPT (override the checkpoint table below; use with a single rank), DRY_RUN=1

set -eo pipefail
[[ -f scripts/lib.sh ]] || { echo "error: please run from models/DiffAE/" >&2; exit 1; }
source scripts/lib.sh
cd_diffae_root

RANKS=${RANKS:-"5 10 512"}

rank_ckpt() {
  local dir=checkpoints/ffhq128_autoenc_rank$1/checkpoints
  case "$1" in
    5)   echo "$dir/epoch=56-step=124659.ckpt" ;;
    10)  echo "$dir/epoch=128-step=281250.ckpt" ;;
    512) echo "$dir/epoch=191-step=418750.ckpt" ;;
    *)   echo "unsupported rank: $1" >&2; return 1 ;;
  esac
}

for RANK in $RANKS; do
  case "${1:-}" in
    train)
      submit "rank$RANK-train" --gres gpu:2 -- \
        run_ffhq128_rank_ablation.py --rank "$RANK" --mode train >/dev/null
      ;;
    train_svd)
      submit_with_resume "rank$RANK-svd-train" --gres gpu:2 -- \
        run_ffhq128_rank_svd_ablation.py --rank "$RANK" >/dev/null
      ;;
    train_lowrank)
      submit_with_resume "rank$RANK-lowrank-train" --gres gpu:2 -- \
        run_ffhq128_rank_lowrank_ablation.py --rank "$RANK" >/dev/null
      ;;
    fid)
      submit "rank$RANK-fid" -- \
        run_ffhq128_rank_ablation.py --rank "$RANK" --mode eval --gpus 0 >/dev/null
      ;;
    ppl_pipl|pipl_fair)
      CKPT_R=${CKPT:-$(rank_ckpt "$RANK")}
      require_ckpt "$CKPT_R" || continue
      # one factor per checkpoint, so a newer checkpoint never reuses a stale factor
      FACTOR=factors/ffhq128_autoenc_rank${RANK}_$(basename "$CKPT_R" .ckpt).pt
      FJ=$(submit_factor "$CKPT_R" "$FACTOR" --is_ortho --diag_size "$RANK")
      if [[ "$1" == pipl_fair ]]; then
        SKIP_PPL=1 N_EIG=${N_EIG:-5} EPS=${EPS:-1e-1} OUT_NAME=ffhq128_autoenc_rank${RANK}_fair \
          submit_ppl_pipl "rank$RANK-fair" "$CKPT_R" "$FACTOR" "$FJ" --rank "$RANK"
      else
        EPS=${EPS:-1e-1} \
          submit_ppl_pipl "rank$RANK" "$CKPT_R" "$FACTOR" "$FJ" --rank "$RANK"
      fi
      ;;
    *)
      sed -n '2,13p' "${BASH_SOURCE[0]}"
      exit 1
      ;;
  esac
done
