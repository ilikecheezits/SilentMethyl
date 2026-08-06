#!/bin/bash
#SBATCH --job-name=SilentMethyl_V1
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=30:00:00
#SBATCH --output=training_log_%j.txt
#SBATCH --error=error_log_%j.txt

cd /ocean/projects/med250012p/szhang37/SilentMethyl

# Load environment
module load anaconda3
module load cuda/12.4.0
conda activate silentmethyl
set -e

echo "[*] Starting Phase 1 Baseline Training on V100 GPU..."

python scripts/01_train_model_baseline.py \
  --train_path "data/datafiles/train.csv" \
  --val_path "data/datafiles/val.csv" \
  --save_dir "checkpoints_baseline" \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --epochs 10 \
  --window_size 1000

echo "[✓] Training Job Complete!"