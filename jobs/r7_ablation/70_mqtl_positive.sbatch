#!/bin/bash
#SBATCH --job-name=r7-70-mqtl
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/r7_ablation/70_%j.out
#SBATCH --error=logs/r7_ablation/70_%j.err
set -euo pipefail
umask 027
source "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"/jobs/r6_ablation/_common.sh
cd "$ROOT"
STAMP=logs/r7_ablation/stamp_70_$SLURM_JOB_ID
stamp_start "$STAMP"
# --test-csv carries the CONTEXT: script 70 merges TABULAR_FEATURES off this file,
# so pointing it at the breast-epithelium split is what makes this a new-context run.
# The sequence arm stays on checkpoints_journal by design (see _common.sh).
$PY scripts/70_mqtl_positive_control.py \
  --test-csv data/datafiles_breast_epithelium/test.csv \
  --models fusion sequence \
  --seeds 42 43 44 \
  --fusion-weights-template "$FUSION_W" \
  --sequence-weights-template "$SEQ_W" \
  --output-dir "$ABL/egtex_mqtl_positive_control"
check_no_clobber "$STAMP"
