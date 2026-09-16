#!/bin/bash
#SBATCH --job-name=r7-64-lit
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/r7_ablation/64_%j.out
#SBATCH --error=logs/r7_ablation/64_%j.err
set -euo pipefail
umask 027
source "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"/jobs/r6_ablation/_common.sh
cd "$ROOT"
STAMP=logs/r7_ablation/stamp_64_$SLURM_JOB_ID
stamp_start "$STAMP"
# 64 shells out to 63. It DOES forward --weights-template and --split-template
# (verified 13 Sep 2026); both must be set here or the screen scores the
# published MCF-10A checkpoints and reports a result that looks correct.
$PY scripts/64_literature_variant_screen.py \
  --split-template 'data/datafiles_breast_epithelium/{split}.csv' \
  --weights-template "$FUSION_W" \
  --existing-candidate-csv data/datafiles_breast_epithelium/testing_data.csv \
  --seeds 42 43 44 \
  --output-dir "$ABL/literature_variant_screen"
check_no_clobber "$STAMP"
