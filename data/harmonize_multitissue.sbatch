#!/usr/bin/env bash
#SBATCH -J harmonize_egtex
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 04:00:00
#SBATCH -o logs/harmonize_multitissue.log
cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate silentmethyl
bash data/harmonize_egtex_multitissue.sh
