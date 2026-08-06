#!/bin/bash
#SBATCH --job-name=Train_Fusion
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=24:00:00
#SBATCH --output=fusion_log_%j.txt
#SBATCH --error=fusion_err_%j.txt

cd /ocean/projects/med250012p/szhang37/SilentMethyl

# Load environment
module load anaconda3
module load cuda/12.4.0
conda activate silentmethyl
set -e

echo "[*] Starting Sequence + Epigenomics Fusion Training on V100 GPU..."

python scripts/01_train_model.py \
  --train_path "data/datafiles/train.csv" \
  --val_path "data/datafiles/val.csv" \
  --baseline_weights "checkpoints_baseline/best_weights.pth" \
  --pure_epi_weights "checkpoints_epi_only/best_weights.pth" \
  --save_dir "checkpoints_seq_epi_fusion" \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --epochs 10 \
  --seq_window_size 1000

echo "[✓] Final Fusion Training Job Complete!"