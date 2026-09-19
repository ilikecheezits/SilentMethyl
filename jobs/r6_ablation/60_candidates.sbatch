#!/bin/bash
#SBATCH --job-name=r6-60-cand
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/r6_ablation/60_%j.out
#SBATCH --error=logs/r6_ablation/60_%j.err
# Slurm job: run the synonymous candidate matched-background screen (script 60).

set -euo pipefail
umask 027
source "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"/jobs/r6_ablation/_common.sh
cd "$ROOT"
STAMP=logs/r6_ablation/stamp_60_$SLURM_JOB_ID
stamp_start "$STAMP"
$PY scripts/60_candidate_background.py \
  --input-csv "$COHORT" \
  --weights-template "$FUSION_W" \
  --output-dir "$CAND"
check_no_clobber "$STAMP"
