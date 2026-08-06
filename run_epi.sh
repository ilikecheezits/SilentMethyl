#!/bin/bash
#SBATCH --job-name=Train_Epi
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=10:00:00
#SBATCH --output=epi_log_%j.txt
#SBATCH --error=epi_err_%j.txt

cd /ocean/projects/med250012p/szhang37/SilentMethyl

# Load environment
module load anaconda3
module load cuda/12.4.0
conda activate silentmethyl
set -e

echo "[*] Starting Pure Epigenetic NN Training on V100 GPU..."

python scripts/01_train_epi.py \
  --train_path "data/datafiles/train.csv" \
  --val_path "data/datafiles/val.csv" \
  --save_dir "checkpoints_epi_only" \
  --batch_size 128 \
  --epochs 15

echo "[✓] Pure Epigenomic Training Job Complete!"