#!/bin/bash
#SBATCH --job-name=SeqEpi_Model
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=20:00:00
#SBATCH --output=seq_epi_log_%j.txt
#SBATCH --error=seq_epi_err_%j.txt

cd /ocean/projects/med250012p/szhang37/SilentMethyl

module load anaconda3
module load cuda/12.4.0
conda activate silentmethyl
set -e

echo "[*] RUN: Training Sequence + Epigenomics Full Model (No Shape)"
python scripts/01_train_model.py \
  --train_path "data/datafiles/train.csv" \
  --val_path "data/datafiles/val.csv" \
  --baseline_weights "checkpoints_baseline/best_weights.pth" \
  --pure_nn_weights "checkpoints_nn/best_weights.pth" \
  --save_dir "checkpoints_seq_epi_model" \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --epochs 10 

echo "[✓] Sequence-Epi Model Training Complete!"