#!/bin/bash
# Evaluate the original DiffAE (checkpoints/ffhq128_autoenc_130M/last.ckpt).
#
# Usage (from models/DiffAE/):
#   bash scripts/submit/baseline_eval.sh fid
#   bash scripts/submit/baseline_eval.sh ppl_pipl     # legacy ppl.py / pipl.py
# Env: EPS (default 1e-1), DRY_RUN=1

set -eo pipefail
[[ -f scripts/lib.sh ]] || { echo "error: please run from models/DiffAE/" >&2; exit 1; }
source scripts/lib.sh
cd_diffae_root

NAME=ffhq128_autoenc_130M
CKPT=checkpoints/$NAME/last.ckpt
require_ckpt "$CKPT" || exit 1

case "${1:-}" in
  fid)
    submit base-fid -- run_ffhq128_base_eval.py --mode fid >/dev/null
    ;;
  ppl_pipl)
    FACTOR=factors/$NAME.pt
    FJ=$(submit_factor "$CKPT" "$FACTOR")
    EPS=${EPS:-1e-1} OUT_NAME=$NAME \
      submit_ppl_pipl base "$CKPT" "$FACTOR" "$FJ" --vanilla
    ;;
  *)
    sed -n '2,7p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
