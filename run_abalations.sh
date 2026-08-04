#!/bin/bash
#SBATCH --job-name=SilentMethyl_Ablations
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=40:00:00
#SBATCH --output=ablation_log_%j.txt
#SBATCH --error=ablation_err_%j.txt

cd /ocean/projects/med250012p/szhang37/SilentMethyl

# Load environment
module load anaconda3
module load cuda/12.4.0
conda activate silentmethyl
set -e

echo "[*] Starting Ablation Studies on V100 GPU..."

# ---------------------------------------------------------
# RUN 1: Baseline + Shape (Epigenomics Masked)
# ---------------------------------------------------------
echo "[*] RUN 1: Training with Shape ONLY (No Epigenomics)"
python scripts/train_ablations.py \
  --train_path "data/datafiles/train.csv" \
  --val_path "data/datafiles/val.csv" \
  --train_shape_tsv "data/datafiles/train_3d_shapes.tsv" \
  --val_shape_tsv "data/datafiles/val_3d_shapes.tsv" \
  --save_dir "checkpoints_shape_only" \
  --batch_size 32 \
  --epochs 10 \
  --ablation_mode "no_epi"

# ---------------------------------------------------------
# RUN 2: Baseline + Epigenomics (Shape Masked)
# ---------------------------------------------------------
echo "[*] RUN 2: Training with Epigenomics ONLY (No Shape)"
python scripts/train_ablations.py \
  --train_path "data/datafiles/train.csv" \
  --val_path "data/datafiles/val.csv" \
  --train_shape_tsv "data/datafiles/train_3d_shapes.tsv" \
  --val_shape_tsv "data/datafiles/val_3d_shapes.tsv" \
  --save_dir "checkpoints_epi_only" \
  --batch_size 32 \
  --epochs 10 \
  --ablation_mode "no_shape"

# ---------------------------------------------------------
# RUN 3: Full Concat (Gating Mechanism Disabled)
# ---------------------------------------------------------
echo "[*] RUN 3: Training Full Features with standard Concatenation (No Gating)"
python scripts/train_ablations.py \
  --train_path "data/datafiles/train.csv" \
  --val_path "data/datafiles/val.csv" \
  --train_shape_tsv "data/datafiles/train_3d_shapes.tsv" \
  --val_shape_tsv "data/datafiles/val_3d_shapes.tsv" \
  --save_dir "checkpoints_concat" \
  --batch_size 32 \
  --epochs 10 \
  --ablation_mode "none" \
  --disable_gating

echo "[✓] All Ablation Runs Complete!"