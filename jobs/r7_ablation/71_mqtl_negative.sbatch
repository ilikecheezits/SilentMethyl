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
set -euo pipefail
umask 027
source /ocean/projects/med250012p/szhang37/SilentMethyl/jobs/r6_ablation/_common.sh
cd "$ROOT"
STAMP=logs/r7_ablation/stamp_71_$SLURM_JOB_ID
stamp_start "$STAMP"
# The script default (data/BreastMammaryTissue.regular.perm.fdr.txt) is the path
# the published run used; the file has since moved under data/external/egtex_breast/.
# Verified byte-identical to the published input, sha256 604c34e4...03743da0.
$PY scripts/71_mqtl_matched_negative.py \
  --lead-mqtl-file data/external/egtex_breast/BreastMammaryTissue.regular.perm.fdr.txt \
  --test-csv data/datafiles_breast_epithelium/test.csv \
  --models fusion sequence \
  --seeds 42 43 44 \
  --fusion-weights-template "$FUSION_W" \
  --sequence-weights-template "$SEQ_W" \
  --output-dir "$ABL/egtex_mqtl_matched_negative"
check_no_clobber "$STAMP"
