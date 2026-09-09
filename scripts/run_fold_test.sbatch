#!/bin/bash
# Score each fold's HELD-OUT TEST chromosomes with that fold's own checkpoints.
#
# Why this matters
# ----------------
# The fold validation numbers reported so far are on each fold's VALIDATION
# chromosome and are not comparable to the published figures, which are test-set
# (chr8+chr9). This job produces the comparable numbers: fusion and sequence on
# each fold's test chromosomes, using only that fold's weights.
#
#   fold 1  test chr12,chr18   26,748 probes
#   fold 2  test chr20,chr4    26,783 probes
#   fold 3  test chr13,chr15,chr21  26,806 probes
#   fold 0  test chr8,chr9     26,570 probes  <- the published split, already scored
#
# Test sizes are matched by construction, so the spread across folds is a spread
# in difficulty, not in sample size.
#
#   sbatch --array=1-3 scripts/run_fold_test.sbatch
#SBATCH --job-name=foldtest
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/folds/test_%A_%a.out
#SBATCH --error=logs/folds/test_%A_%a.err
set -eo pipefail

PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python

# Fail fast if the allocated node cannot actually do CUDA. Node v005 returned
# CUDA error 803 more than once; torch then falls back to CPU *silently* and the
# job runs at 13.3 s/it instead of 3.66 it/s -- a 49x slowdown that only becomes
# visible hours into the log, after the walltime is already spent.
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "STOP: CUDA is not available on $(hostname). Nothing was run." >&2
  echo "Requeue with:  sbatch --exclude=$(hostname) <this script>" >&2
  exit 4
fi

FOLD=${SLURM_ARRAY_TASK_ID}
SEED=42

TEST=data/datafiles/splits/fold${FOLD}/test.csv
CKPT=checkpoints_folds/fold${FOLD}
OUT=results/journal/folds/fold${FOLD}

for f in "${TEST}" \
         "${CKPT}/fusion_seed${SEED}/best_weights.pth" \
         "${CKPT}/sequence_seed${SEED}/best_weights.pth" \
         "${CKPT}/epi_seed${SEED}/best_weights.pth"; do
  [ -s "$f" ] || { echo "STOP: missing or empty ${f}" >&2; exit 1; }
done

mkdir -p "${OUT}"
echo "=== fold ${FOLD} test  $(date -u +%FT%TZ) ==="
echo "test ${TEST}  ($(( $(wc -l < "${TEST}") - 1 )) probes)"
md5sum "${TEST}" | tee "${OUT}/test_checksum.txt"

# All three arms, so the fusion-over-sequence gain can be recomputed per fold
# rather than assumed from the validation curves.
for ARM in fusion sequence epi; do
  echo "--- ${ARM} ---"
  $PY -u scripts/13_test_model.py \
      --model "${ARM}" --seed ${SEED} \
      --test_path "${TEST}" \
      --weights_path "${CKPT}/${ARM}_seed${SEED}/best_weights.pth" \
      --output_dir "${OUT}/${ARM}"
done

echo "=== fold ${FOLD} test done $(date -u +%FT%TZ) ==="
