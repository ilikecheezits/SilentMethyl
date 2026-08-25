#!/bin/bash
#SBATCH --job-name=SM_gate_s42
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=20:00:00
#SBATCH --output=logs/training/fusion_seed42_%j.out
#SBATCH --error=logs/training/fusion_seed42_%j.err

set -euo pipefail

ROOT=/ocean/projects/med250012p/szhang37/SilentMethyl
cd "$ROOT"

module load anaconda3
module load cuda/12.4.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate silentmethyl

SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
SEED_DIR="checkpoints_journal/seed${SEED}"
SEQUENCE_WEIGHTS="$SEED_DIR/sequence/best_weights.pth"
EPI_WEIGHTS="$SEED_DIR/epi/best_weights.pth"
SAVE_DIR="$SEED_DIR/fusion"

if [[ ! -f "$SEQUENCE_WEIGHTS" ]]; then
  echo "[!] Missing sequence checkpoint: $SEQUENCE_WEIGHTS" >&2
  exit 2
fi
if [[ ! -f "$EPI_WEIGHTS" ]]; then
  echo "[!] Missing context checkpoint: $EPI_WEIGHTS" >&2
  exit 2
fi

mkdir -p "$SAVE_DIR"

echo "[*] Host: $(hostname)"
echo "[*] Job ID: ${SLURM_JOB_ID:-NA}"
echo "[*] Starting journal gated fusion training"
echo "[*] seed=$SEED physical_batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM_STEPS effective_batch=$((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "[*] Sequence ancestor: $SEQUENCE_WEIGHTS"
echo "[*] Context ancestor:  $EPI_WEIGHTS"
echo "[*] Modality towers remain frozen; only gated fusion logic + fresh heads are trained."
nvidia-smi || true

python -u scripts/01_train_fusion_journal.py \
  --train_path data/datafiles/train.csv \
  --val_path data/datafiles/val.csv \
  --sequence_weights "$SEQUENCE_WEIGHTS" \
  --epi_weights "$EPI_WEIGHTS" \
  --save_dir "$SAVE_DIR" \
  --local_model_dir ./dnabert2_local \
  --batch_size "$BATCH_SIZE" \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  --epochs 10 \
  --window_size 1000 \
  --rc_probability 0.50 \
  --seed "$SEED" \
  --num_workers 4

echo "[✓] Gated fusion journal training complete: $SAVE_DIR"
