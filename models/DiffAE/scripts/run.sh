#!/bin/bash
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o logs/%x-%j.log
#
# Generic DiffAE job: run one Python entry point inside the job environment.
# Run from models/DiffAE/, e.g.
#   sbatch -J ffhq128 --gres gpu:2 scripts/run.sh run_ffhq128.py
#   bash scripts/run.sh evaluate_ffhq128_ppl_readme.py --arm householder10   # without Slurm
#
# Partition / account are cluster-specific: pass -p / -A to sbatch, or export
# SBATCH_PARTITION / SBATCH_ACCOUNT once in your shell.

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/lib.sh
cd_diffae_root
setup_job_env

echo "[$(date '+%F %T')] $(hostname) job=${SLURM_JOB_ID:-local}: python $*"
python -u "$@"
