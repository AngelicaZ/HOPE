#!/bin/bash
#SBATCH -p lambda
#SBATCH -A staff
#SBATCH -t 23:59:00
#SBATCH --gres gpu:1
#SBATCH -o o_file/eval/0108_horse128_HP.o
#SBATCH -e e_file/eval/0108_horse128_HP.e

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}" 2>/dev/null || true

conda activate hope
export TMPDIR="${TMPDIR:-/tmp}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
wandb login '61bbe1cdd46fd39ea897e6088bb2113126178cd8'


python run_horse128.py  > out_file/eval/0108_horse128_HP.out 
