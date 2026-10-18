#!/bin/bash
#SBATCH -p chaos
#SBATCH -A shared-mhug-staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/ffhq128_rank%j.o
#SBATCH -e e_file/eval/ffhq128_rank%j.e

# Usage:
#   sbatch --export=RANK=5 scripts/ablation/ffhq128_rank_eval.sh
#   sbatch --export=RANK=10 scripts/ablation/ffhq128_rank_eval.sh
#   sbatch --export=RANK=512 scripts/ablation/ffhq128_rank_eval.sh

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}" 2>/dev/null || true

conda activate hope
export TMPDIR="${TMPDIR:-/tmp}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RANK=${RANK:-5}

python run_ffhq128_rank_ablation.py --rank ${RANK} --mode eval --gpus 0 \
    > out_file/eval/ffhq128_rank${RANK}.out 2>&1
