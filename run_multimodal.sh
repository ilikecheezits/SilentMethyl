#!/bin/bash
#SBATCH --job-name=SilentMethyl_GatedFusion
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=20:00:00
#SBATCH --output=multimodal_gated_log_%j.txt
#SBATCH --error=multimodal_gated_err_%j.txt

cd /ocean/projects/med250012p/szhang37/SilentMethyl

# Load environment
module load anaconda3
module load cuda/12.4.0
conda activate silentmethyl
set -e

echo "[*] Starting Phase 3: Gated Fusion Training..."

# Execute the multimodal training script with Dual Ancestor Weights
python scripts/01_train_model.py \
  --train_path "data/datafiles/train.csv" \
  --val_path "data/datafiles/val.csv" \
  --train_shape_tsv "data/datafiles/train_3d_shapes.tsv" \
  --val_shape_tsv "data/datafiles/val_3d_shapes.tsv" \
  --baseline_weights "checkpoints_baseline/baseline_best_weights.pth" \
  --pure_nn_weights "checkpoints_nn/pure_nn_best.pth" \
  --save_dir "checkpoints_multimodal_gated" \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --epochs 10 \
  --seq_window_size 1000 \
  --shape_window_size 100

echo "[✓] Gated Fusion Training Job Complete!"
