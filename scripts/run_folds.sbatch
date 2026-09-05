#!/bin/bash
# Repeated chromosome-blocked splits: one array task per fold.
#
# Fold 0 is the published split. If scripts/17_chromosome_splits.py reports
# "fold 0 reproduces the published split EXACTLY", the existing seed-42
# checkpoints already ARE fold 0 and must not be retrained -- submit 1-3 only.
# If it reports otherwise, submit 0-3 and treat fold 0 as the reproduction check.
#
#   sbatch --array=1-3 scripts/run_folds.sbatch
#
# Each task runs the full pipeline for its fold, in order, because the fusion
# model consumes the sequence and epi towers trained on the SAME split. Reusing
# towers across folds would leak the published split's training data into every
# other fold's test set, which is the exact failure this whole exercise exists
# to rule out.
#SBATCH --job-name=fold
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=47:00:00
#SBATCH --output=logs/folds/%A_%a.out
#SBATCH --error=logs/folds/%A_%a.err
set -eo pipefail

PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python
FOLD=${SLURM_ARRAY_TASK_ID}
SEED=42

SPLIT=data/datafiles/splits/fold${FOLD}
TRAIN=${SPLIT}/train.csv
VAL=${SPLIT}/val.csv
OUT=checkpoints_folds/fold${FOLD}

if [ ! -f "${TRAIN}" ] || [ ! -f "${VAL}" ]; then
  echo "STOP: ${TRAIN} or ${VAL} missing -- run scripts/17_chromosome_splits.py first" >&2
  exit 1
fi

mkdir -p "${OUT}"
echo "=== fold ${FOLD}  seed ${SEED}  $(date -u +%FT%TZ) ==="
echo "train ${TRAIN}"
echo "val   ${VAL}"

# Provenance guard. Regenerating the splits while a job is running once left
# three jobs training on a fold definition that no longer matched
# splits_summary.json -- silently, because nothing recorded which bytes were
# actually used. Snapshot the checksums up front and re-check them at the end;
# a mismatch means the run and the recorded fold definition have diverged.
SNAP=${OUT}/split_checksums.txt
md5sum "${TRAIN}" "${VAL}" > "${SNAP}"
cp data/datafiles/splits/splits_summary.json "${OUT}/splits_summary_at_launch.json"
cat "${SNAP}"

echo "--- 1/3 sequence tower ---"
$PY -u scripts/10_train_sequence.py \
    --train_path "${TRAIN}" --val_path "${VAL}" \
    --save_dir "${OUT}/sequence_seed${SEED}" --seed ${SEED}

echo "--- 2/3 epigenomic tower ---"
$PY -u scripts/11_train_epi.py \
    --train_path "${TRAIN}" --val_path "${VAL}" \
    --save_dir "${OUT}/epi_seed${SEED}" --seed ${SEED}

echo "--- 3/3 gated fusion ---"
$PY -u scripts/12_train_fusion.py \
    --train_path "${TRAIN}" --val_path "${VAL}" \
    --sequence_weights "${OUT}/sequence_seed${SEED}" \
    --epi_weights "${OUT}/epi_seed${SEED}" \
    --save_dir "${OUT}/fusion_seed${SEED}" --seed ${SEED}

echo "--- provenance check ---"
if md5sum -c "${SNAP}" --status; then
  echo "split files unchanged for the whole run"
else
  echo "WARNING: the split CSVs CHANGED while this job was running." >&2
  echo "This fold's checkpoints do not correspond to the recorded fold" >&2
  echo "definition and must not be used. Regenerate and retrain." >&2
  md5sum -c "${SNAP}" >&2 || true
  exit 3
fi

echo "=== fold ${FOLD} done $(date -u +%FT%TZ) ==="
