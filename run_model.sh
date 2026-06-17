#!/bin/bash
#SBATCH --job-name=SilentMethyl
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=24:00:00
#SBATCH --output=training_log.txt
#SBATCH --error=sus.txt

cd /ocean/projects/med250012p/szhang37/SilentMethyl
module load anaconda3
module load cuda/12.4.0
conda activate silentmethyl
set -e

echo "[*] Starting Phase 1 Baseline Training on V100 GPU..."

# Execute the training script on the full dataset
python scripts/02_train_model_baseline.py \
  --train_path "data/datafiles/train.csv" \
  --val_path "data/datafiles/val.csv" \
  --save_dir "checkpoints_baseline" \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --epochs 10

echo "[✓] Training Job Complete!"
