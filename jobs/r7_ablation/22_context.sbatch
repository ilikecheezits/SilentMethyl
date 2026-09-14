#!/bin/bash
#SBATCH --job-name=r7-22-ctx
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/r7_ablation/22_%j.out
#SBATCH --error=logs/r7_ablation/22_%j.err
set -euo pipefail
umask 027
source /ocean/projects/med250012p/szhang37/SilentMethyl/jobs/r6_ablation/_common.sh
cd "$ROOT"
STAMP=logs/r7_ablation/stamp_22_$SLURM_JOB_ID
stamp_start "$STAMP"
MQTL="$ABL/egtex_mqtl_positive_control/mqtl_predictions_seed_aggregate.csv"
[[ -f "$MQTL" ]] || { echo "[!!] missing new-context mQTL input: $MQTL" >&2; exit 1; }
echo "[*] all inputs are breast-epithelium; supersedes the R6 cross-context run"
$PY scripts/22_context_stratification.py \
  --test-path data/datafiles_breast_epithelium/test.csv \
  --prediction-template "$ABL/seed{seed}/{model}/predictions.csv" \
  --candidate-path "$CAND/candidate_matched_background_statistics.csv" \
  --mqtl-path "$MQTL" \
  --output-dir "$ABL/biological_context"
check_no_clobber "$STAMP"
