#!/bin/bash
#SBATCH --job-name=SM_epi_s42
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/training/epi_seed42_%j.out
#SBATCH --error=logs/training/epi_seed42_%j.err

set -euo pipefail

ROOT=/ocean/projects/med250012p/szhang37/SilentMethyl
cd "$ROOT"

module load anaconda3
module load cuda/12.4.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate silentmethyl

SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-128}
SAVE_DIR="checkpoints_journal/seed${SEED}/epi"

mkdir -p "$SAVE_DIR"

echo "[*] Host: $(hostname)"
echo "[*] Job ID: ${SLURM_JOB_ID:-NA}"
echo "[*] Starting journal context-only training"
echo "[*] seed=$SEED batch=$BATCH_SIZE"
nvidia-smi || true

python -u scripts/01_train_epi_journal.py \
  --train_path data/datafiles/train.csv \
  --val_path data/datafiles/val.csv \
  --save_dir "$SAVE_DIR" \
  --local_model_dir ./dnabert2_local \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps 1 \
  --epochs 15 \
  --rc_probability 0.50 \
  --seed "$SEED" \
  --num_workers 4

echo "[✓] Context-only journal training complete: $SAVE_DIR"
