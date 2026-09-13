#!/bin/bash
#SBATCH --job-name=r6-22-ctx
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/r6_ablation/22_%j.out
#SBATCH --error=logs/r6_ablation/22_%j.err
set -euo pipefail
umask 027
source /ocean/projects/med250012p/szhang37/SilentMethyl/jobs/r6_ablation/_common.sh
cd "$ROOT"
STAMP=logs/r6_ablation/stamp_22_$SLURM_JOB_ID
stamp_start "$STAMP"
echo "[!] CROSS-CONTEXT INPUT: --mqtl-path is the published MCF-10A positive control;"
echo "[!] no breast-epithelium rebuild of it exists. Flagged, not silently mixed."
$PY scripts/22_context_stratification.py \
  --test-path data/datafiles_breast_epithelium/test.csv \
  --prediction-template "$ABL/seed{seed}/{model}/predictions.csv" \
  --candidate-path "$CAND/candidate_matched_background_statistics.csv" \
  --mqtl-path results/journal/egtex_mqtl_positive_control/mqtl_predictions_seed_aggregate.csv \
  --output-dir "$ABL/biological_context"
check_no_clobber "$STAMP"
