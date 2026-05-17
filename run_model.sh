#!/bin/bash
#SBATCH --job-name=SilentMethyl
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=04:00:00
#SBATCH --output=training_log.txt

cd /ocean/projects/med250012p/szhang37/SilentMethyl
module load anaconda3
module load cuda/12.4.0
conda activate silentmethyl

python scripts/02_train_model.py --matrix_path data/processed/cleaned_training_data.csv --dict_path data/raw/DNA_Sequence_Dictionary.csv --region_vocab_path data/processed/SilentMethyl_RegionVocab.pkl --island_vocab_path data/processed/SilentMethyl_IslandVocab.pkl --save_dir ./checkpoints --batch_size 16 --steps_per_epoch 2000 --epochs 10 --lr 1e-4 --num_workers 4
