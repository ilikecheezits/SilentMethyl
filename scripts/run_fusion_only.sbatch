#!/bin/bash
# Fusion stage only, against tower checkpoints that already exist.
#
# Why this exists
# ---------------
# run_folds.sbatch passed the tower SAVE DIRECTORIES to --sequence_weights and
# --epi_weights, but 12_train_fusion.py takes the .pth FILE (it sha256's the
# path). All three folds therefore trained both towers to completion -- sequence
# 10/10 epochs, epi 15/15 -- and then died at the fusion step ~32 h in with
# IsADirectoryError. The towers are intact; only the fusion stage was lost.
#
# This job reruns just that stage. run_folds.sbatch has been corrected too, so a
# fold started from scratch in future will not hit this.
#
#   sbatch --array=1-3 scripts/run_fusion_only.sbatch
#
# Walltime is generous on purpose: the sequence tower ran at ~3.2 h/epoch rather
# than the ~2.5 h assumed, so fusion is nearer 13 h than 10, and there is no
# reason to run a second job close to its limit.
#SBATCH --job-name=fusion
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=logs/folds/fusion_%A_%a.out
#SBATCH --error=logs/folds/fusion_%A_%a.err
set -eo pipefail

PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python
FOLD=${SLURM_ARRAY_TASK_ID}
SEED=42

SPLIT=data/datafiles/splits/fold${FOLD}
TRAIN=${SPLIT}/train.csv
VAL=${SPLIT}/val.csv
OUT=checkpoints_folds/fold${FOLD}
SEQ_W=${OUT}/sequence_seed${SEED}/best_weights.pth
EPI_W=${OUT}/epi_seed${SEED}/best_weights.pth

for f in "${TRAIN}" "${VAL}" "${SEQ_W}" "${EPI_W}"; do
  if [ ! -s "$f" ]; then
    echo "STOP: missing or empty ${f}" >&2
    echo "Both towers must have finished before the fusion stage can run." >&2
    exit 1
  fi
done

echo "=== fold ${FOLD} fusion  seed ${SEED}  $(date -u +%FT%TZ) ==="
echo "train ${TRAIN}"
echo "val   ${VAL}"
echo "seq   ${SEQ_W}"
echo "epi   ${EPI_W}"

# Same provenance guard as run_folds.sbatch: the split CSVs must not change
# under a running job, and the towers must be the ones this fold trained.
SNAP=${OUT}/split_checksums_fusion.txt
md5sum "${TRAIN}" "${VAL}" "${SEQ_W}" "${EPI_W}" > "${SNAP}"
cat "${SNAP}"

$PY -u scripts/12_train_fusion.py \
    --train_path "${TRAIN}" --val_path "${VAL}" \
    --sequence_weights "${SEQ_W}" \
    --epi_weights "${EPI_W}" \
    --save_dir "${OUT}/fusion_seed${SEED}" --seed ${SEED}

echo "--- provenance check ---"
if md5sum -c "${SNAP}" --status; then
  echo "split files and tower weights unchanged for the whole run"
else
  echo "WARNING: an input CHANGED while this job was running." >&2
  echo "This fold's fusion checkpoint does not correspond to the recorded" >&2
  echo "fold definition or towers, and must not be used." >&2
  md5sum -c "${SNAP}" >&2 || true
  exit 3
fi

echo "=== fold ${FOLD} fusion done $(date -u +%FT%TZ) ==="
