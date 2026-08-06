#!/bin/bash
#SBATCH --job-name=MC_Stability
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=12:00:00
#SBATCH --output=stability_log_%j.txt
#SBATCH --error=stability_err_%j.txt

cd /ocean/projects/med250012p/szhang37/SilentMethyl

module load anaconda3
module load cuda/12.4.0
conda activate silentmethyl
set -e

python scripts/06_stability.py