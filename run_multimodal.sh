#!/bin/bash
#SBATCH --job-name=SilentMethyl_Multi
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=40:00:00
#SBATCH --output=multimodal_training_log_%j.txt
#SBATCH --error=multimodal_error_log_%j.txt

cd /ocean/projects/med250012p/szhang37/SilentMethyl

# Load environment
module load anaconda3
module load cuda/12.4.0
conda activate silentmethyl
set -e

echo "[*] Starting Phase 2 Multimodal Training..."

# Execute the multimodal training script on the full dataset
python scripts/01_train_model.py \
  --train_path "data/datafiles/train.csv" \
  --val_path "data/datafiles/val.csv" \
  --train_shape_tsv "data/datafiles/train_3d_shapes.tsv" \
  --val_shape_tsv "data/datafiles/val_3d_shapes.tsv" \
  --save_dir "checkpoints_multimodal" \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --epochs 10 \
  --seq_window_size 1000 \
  --shape_window_size 100

echo "[✓] Multimodal Training Job Complete!"
