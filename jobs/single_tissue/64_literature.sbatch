#!/bin/bash
#SBATCH --job-name=sm-64-lit
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/single_tissue/64_%j.out
#SBATCH --error=logs/single_tissue/64_%j.err
# Slurm job: run the literature variant screen (script 64), forwarding the breast-epithelium weights and split templates.

set -euo pipefail
umask 027
source "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"/jobs/single_tissue/_common.sh
cd "$ROOT"
STAMP=logs/single_tissue/stamp_64_$SLURM_JOB_ID
stamp_start "$STAMP"
require_cuda
$PY scripts/64_literature_variant_screen.py \
  --split-template 'data/datafiles_breast_epithelium/{split}.csv' \
  --weights-template "$FUSION_W" \
  --existing-candidate-csv data/datafiles_breast_epithelium/testing_data.csv \
  --seeds 42 43 44 \
  --output-dir "$ABL/literature_variant_screen"
check_no_clobber "$STAMP"
