#!/bin/bash
#SBATCH --job-name=Abl_Concat
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=20:00:00
#SBATCH --output=ablation_concat_log_%j.txt
#SBATCH --error=ablation_concat_err_%j.txt

cd /ocean/projects/med250012p/szhang37/SilentMethyl

module load anaconda3
module load cuda/12.4.0
conda activate silentmethyl
set -e

echo "[*] RUN 3: Training Full Features with standard Concatenation (No Gating)"
python scripts/01_train_abalations.py \
  --train_path "data/datafiles/train.csv" \
  --val_path "data/datafiles/val.csv" \
  --train_shape_tsv "data/datafiles/train_3d_shapes.tsv" \
  --val_shape_tsv "data/datafiles/val_3d_shapes.tsv" \
  --baseline_weights "checkpoints_baseline/best_weights.pth" \
  --pure_nn_weights "checkpoints_nn/best_weights.pth" \
  --save_dir "checkpoints_ablation_concat" \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --epochs 10 \
  --ablation_mode "none" \
  --disable_gating

echo "[✓] Concatenation Ablation Complete!"
