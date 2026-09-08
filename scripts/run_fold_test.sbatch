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
