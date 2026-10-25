#!/bin/bash
#
# Convert an image folder / lmdb to a StyleGAN3 dataset zip.
# Usage (from models/StyleGANHuman/training_scripts/sg3/):
#   SRC=/path/to/ffhq1024 DEST=/path/to/ffhq-1024x1024.zip sbatch -p <partition> scripts/processing_data.sh
# Env: RES (default 1024x1024)

set -eo pipefail
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-stylegan3}"

python dataset_tool.py --source="${SRC:?set SRC}" --dest="${DEST:?set DEST}" \
  --resolution="${RES:-1024x1024}"
