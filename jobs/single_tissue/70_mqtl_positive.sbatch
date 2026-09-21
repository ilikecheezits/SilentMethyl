#!/bin/bash
#SBATCH --job-name=sm-70-mqtl
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/single_tissue/70_%j.out
#SBATCH --error=logs/single_tissue/70_%j.err
# Slurm job: run the eGTEx lead-mQTL positive control (script 70) against the breast-epithelium split.

set -euo pipefail
umask 027
source "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"/jobs/single_tissue/_common.sh
cd "$ROOT"
STAMP=logs/single_tissue/stamp_70_$SLURM_JOB_ID
stamp_start "$STAMP"
require_cuda
$PY scripts/70_mqtl_positive_control.py \
  --test-csv data/datafiles_breast_epithelium/test.csv \
  --models fusion sequence \
  --seeds 42 43 44 \
  --fusion-weights-template "$FUSION_W" \
  --sequence-weights-template "$SEQ_W" \
  --output-dir "$ABL/egtex_mqtl_positive_control"
check_no_clobber "$STAMP"
