#!/bin/bash
#SBATCH --job-name=SilentMethyl_PureNN
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=20:00:00
#SBATCH --output=nn_log_%j.txt
#SBATCH --error=nn_err_%j.txt

cd /ocean/projects/med250012p/szhang37/SilentMethyl

# Load environment
module load anaconda3
module load cuda/12.4.0
conda activate silentmethyl
set -e

echo "[*] Starting Pure Epigenetic NN Training on V100 GPU..."

# Execute the pure NN training script on the full dataset
python scripts/01_train_nn.py \
  --train_path "data/datafiles/train.csv" \
  --val_path "data/datafiles/val.csv" \
  --train_shape_tsv "data/datafiles/train_3d_shapes.tsv" \
  --val_shape_tsv "data/datafiles/val_3d_shapes.tsv" \
  --save_dir "checkpoints_nn" \
  --batch_size 128 \
  --epochs 10 \
  --shape_window_size 100

echo "[✓] Pure NN Training Job Complete!"


