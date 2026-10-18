#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFAE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${DIFFAE_ROOT}"
echo "Will resume from latest step=*.ckpt under checkpoints/0214_.../checkpoints/"
ls -lh checkpoints/0214_ffhq128_autoenc_130M_multi_projector/checkpoints/
J=$(sbatch --parsable scripts/ablation/ffhq128_steps_ablation_train.sh)
echo "submitted primary job: $J"
J2=$(sbatch --parsable --dependency=afterany:$J scripts/ablation/ffhq128_steps_ablation_train.sh)
echo "submitted resume job:  $J2"
squeue -u $USER | head
