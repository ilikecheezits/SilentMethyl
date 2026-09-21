#!/bin/bash
#SBATCH --job-name=sm-60-cand
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/single_tissue/60_%j.out
#SBATCH --error=logs/single_tissue/60_%j.err
# Slurm job: run the synonymous candidate matched-background screen (script 60).

set -euo pipefail
umask 027
source "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"/jobs/single_tissue/_common.sh
cd "$ROOT"
STAMP=logs/single_tissue/stamp_60_$SLURM_JOB_ID
stamp_start "$STAMP"
require_cuda
$PY scripts/60_candidate_background.py \
  --input-csv "$COHORT" \
  --weights-template "$FUSION_W" \
  --output-dir "$CAND"
check_no_clobber "$STAMP"
