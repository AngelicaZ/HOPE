#!/bin/bash
#SBATCH -p chaos
#SBATCH -A shared-mhug-staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:2
#SBATCH -o o_file/train/1020_ffhq128_projector.o
#SBATCH -e e_file/train/1020_ffhq128_projector.e

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}" 2>/dev/null || true

conda activate hope
export TMPDIR="${TMPDIR:-/tmp}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
wandb login '61bbe1cdd46fd39ea897e6088bb2113126178cd8'


python run_ffhq128.py  > out_file/train/1020_ffhq128_projector.out 