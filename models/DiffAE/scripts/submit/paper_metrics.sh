#!/bin/bash
# PPL / PIPL with the paper protocol (evaluate_ffhq128_*_readme.py):
# fixed test[0:2000], eps 1e-4, seed 0. See docs/DIFFAE_PPL_PIPL.md.
#
# Usage (from models/DiffAE/):
#   bash scripts/submit/paper_metrics.sh            # PPL and PIPL for every arm
#   bash scripts/submit/paper_metrics.sh ppl
#   bash scripts/submit/paper_metrics.sh pipl
# Env: ARMS (default "householder5 householder10 householder512 lora10 vanilla"),
#      EPS (1e-4), SAMPLES (2000), DRY_RUN=1

set -eo pipefail
[[ -f scripts/lib.sh ]] || { echo "error: please run from models/DiffAE/" >&2; exit 1; }
source scripts/lib.sh
cd_diffae_root

ARMS=${ARMS:-"householder5 householder10 householder512 lora10 vanilla"}
EPS=${EPS:-1e-4}
SAMPLES=${SAMPLES:-2000}

case "${1:-both}" in
  ppl) METRICS="ppl" ;;
  pipl) METRICS="pipl" ;;
  both) METRICS="ppl pipl" ;;
  *) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac

for ARM in $ARMS; do
  for METRIC in $METRICS; do
    submit "paper-$METRIC-$ARM" -t 12:00:00 -- evaluate_ffhq128_${METRIC}_readme.py \
      --arm "$ARM" --epsilon "$EPS" --samples "$SAMPLES" >/dev/null
  done
done
