#!/bin/bash
#SBATCH --job-name=sm-63-known
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/single_tissue/63_%j.out
#SBATCH --error=logs/single_tissue/63_%j.err
# Slurm job: score the known-variant application set (script 63).

set -euo pipefail
umask 027
source "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"/jobs/single_tissue/_common.sh
cd "$ROOT"
STAMP=logs/single_tissue/stamp_63_$SLURM_JOB_ID
stamp_start "$STAMP"
require_cuda
$PY scripts/63_known_variant_application.py \
  --split-template 'data/datafiles_breast_epithelium/{split}.csv' \
  --weights-template "$FUSION_W" \
  --seeds 42 43 44 \
  --output-dir "$ABL/known_variant_application"
check_no_clobber "$STAMP"
