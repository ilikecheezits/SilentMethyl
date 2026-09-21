#!/bin/bash
#SBATCH --job-name=sm-62-cmp
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/single_tissue/62_%j.out
#SBATCH --error=logs/single_tissue/62_%j.err
# Slurm job: run the candidate cross-model comparison (script 62).

set -euo pipefail
umask 027
source "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"/jobs/single_tissue/_common.sh
cd "$ROOT"
STAMP=logs/single_tissue/stamp_62_$SLURM_JOB_ID
stamp_start "$STAMP"
require_cuda
$PY scripts/62_candidate_comparison.py \
  --input-csv "$COHORT" \
  --sequence-weights-template "$SEQ_W" \
  --fusion-aggregate "$CAND/candidate_matched_background_statistics.csv" \
  --fusion-seed-scores "$CAND/candidate_seed_scores_long.csv" \
  --output-dir "$CAND/model_comparison"
check_no_clobber "$STAMP"
