#!/bin/bash
#SBATCH --job-name=r7-63-known
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/r7_ablation/63_%j.out
#SBATCH --error=logs/r7_ablation/63_%j.err
set -euo pipefail
umask 027
source "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"/jobs/r6_ablation/_common.sh
cd "$ROOT"
STAMP=logs/r7_ablation/stamp_63_$SLURM_JOB_ID
stamp_start "$STAMP"
$PY scripts/63_known_variant_application.py \
  --split-template 'data/datafiles_breast_epithelium/{split}.csv' \
  --weights-template "$FUSION_W" \
  --seeds 42 43 44 \
  --output-dir "$ABL/known_variant_application"
check_no_clobber "$STAMP"
