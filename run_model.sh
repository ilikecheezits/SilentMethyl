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

python -u scripts/02_train_model.py \
    --data_path data/processed/train_val_data.csv \
    --epochs 5 \
    --batch_size 16 \
    --grad_accum_steps 2 \
    --window_size 1000

