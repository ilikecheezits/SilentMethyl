#!/bin/bash
#SBATCH --job-name=SM_seq_s42
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=30:00:00
#SBATCH --output=logs/training/sequence_seed42_%j.out
#SBATCH --error=logs/training/sequence_seed42_%j.err

set -euo pipefail

ROOT=/ocean/projects/med250012p/szhang37/SilentMethyl
cd "$ROOT"

module load anaconda3
module load cuda/12.4.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate silentmethyl

BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}
SEED=${SEED:-42}
SAVE_DIR="checkpoints_journal/seed${SEED}/sequence"

mkdir -p "$SAVE_DIR"

echo "[*] Host: $(hostname)"
echo "[*] Job ID: ${SLURM_JOB_ID:-NA}"
echo "[*] Starting journal sequence-only training"
echo "[*] seed=$SEED physical_batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS effective_batch=$((BATCH_SIZE * GRAD_ACCUM_STEPS))"
nvidia-smi || true

python -u scripts/01_train_sequence_journal.py \
  --train_path data/datafiles/train.csv \
  --val_path data/datafiles/val.csv \
  --save_dir "$SAVE_DIR" \
  --local_model_dir ./dnabert2_local \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  --epochs 10 \
  --window_size 1000 \
  --rc_probability 0.50 \
  --seed "$SEED" \
  --num_workers 4

echo "[✓] Sequence-only journal training complete: $SAVE_DIR"
