#!/bin/bash
#SBATCH --job-name=SM_seq
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=30:00:00
#SBATCH --output=logs/training/sequence_%j.out
#SBATCH --error=logs/training/sequence_%j.err
#
# Sequence-only tower (DNABERT-2), published chr8+chr9 split. ~26 h on one V100-32.
#
#   mkdir -p logs/training
#   sbatch --export=ALL,SEED=42,DATA=data/datafiles_breast_epithelium run_baseline.sh
#
# This tower reads DNA only. The published towers under checkpoints_journal/ were
# trained on data/datafiles/, whose sequence, target and split columns are
# byte-identical to data/datafiles_breast_epithelium/ -- the two builds differ
# only in the seven context columns, which this model never reads. Either DATA
# therefore reproduces the same tower; the breast-epithelium build is the one a
# fresh clone has.
#
# Refuses to overwrite an existing tower: the published ones are reused verbatim
# by every fusion model. Set SAVE_DIR to train somewhere else.

set -euo pipefail

ROOT="${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT"

PY="${SILENTMETHYL_PY:-python}"
if [[ -z "${SILENTMETHYL_PY:-}" ]]; then
  module load anaconda3
  module load cuda/12.4.0
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate silentmethyl
fi

: "${DATA:?set DATA, e.g. --export=ALL,SEED=42,DATA=data/datafiles_breast_epithelium}"
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
SEED=${SEED:-42}
SAVE_DIR="${SAVE_DIR:-checkpoints_journal/seed${SEED}/sequence}"

if [[ -s "$SAVE_DIR/best_weights.pth" ]]; then
  echo "STOP: $SAVE_DIR/best_weights.pth exists. Set SAVE_DIR to train elsewhere." >&2
  exit 2
fi
mkdir -p "$SAVE_DIR"

echo "[*] Host: $(hostname)"
echo "[*] Job ID: ${SLURM_JOB_ID:-NA}"
echo "[*] seed=$SEED data=$DATA save=$SAVE_DIR physical_batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS effective_batch=$((BATCH_SIZE * GRAD_ACCUM_STEPS))"
nvidia-smi || true

"$PY" -u scripts/10_train_sequence.py \
  --train_path "$DATA/train.csv" \
  --val_path "$DATA/val.csv" \
  --save_dir "$SAVE_DIR" \
  --local_model_dir ./dnabert2_local \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  --epochs 10 \
  --window_size 1000 \
  --rc_probability 0.50 \
  --seed "$SEED" \
  --num_workers 4

echo "[✓] Sequence-only training complete: $SAVE_DIR"
