#!/bin/bash
#SBATCH --job-name=r7-71-mneg
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/r7_ablation/71_%j.out
#SBATCH --error=logs/r7_ablation/71_%j.err
# Slurm job: run the matched significant-versus-nonsignificant eGTEx lead control (script 71).

set -euo pipefail
umask 027
source "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"/jobs/r6_ablation/_common.sh
cd "$ROOT"
STAMP=logs/r7_ablation/stamp_71_$SLURM_JOB_ID
stamp_start "$STAMP"
$PY scripts/71_mqtl_matched_negative.py \
  --lead-mqtl-file data/external/egtex_breast/BreastMammaryTissue.regular.perm.fdr.txt \
  --test-csv data/datafiles_breast_epithelium/test.csv \
  --models fusion sequence \
  --seeds 42 43 44 \
  --fusion-weights-template "$FUSION_W" \
  --sequence-weights-template "$SEQ_W" \
  --output-dir "$ABL/egtex_mqtl_matched_negative"
check_no_clobber "$STAMP"
