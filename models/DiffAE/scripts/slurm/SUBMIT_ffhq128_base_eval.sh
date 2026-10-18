#!/bin/bash
# Submit FID + PPL/PIPL for original DiffAE (ffhq128_autoenc_130M).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFAE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DIFFAE_ROOT}"
ls -lh checkpoints/ffhq128_autoenc_130M/last.ckpt

J1=$(sbatch --parsable scripts/eval/ffhq128_base_fid_eval.sh)
echo "submitted FID  -> job $J1"

J2=$(sbatch --parsable scripts/eval/ffhq128_base_ppl_pipl.sh)
echo "submitted PPL/PIPL -> job $J2"

squeue -u "$USER" | head
