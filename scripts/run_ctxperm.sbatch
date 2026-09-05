#!/bin/bash
#SBATCH --job-name=ctxperm
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ctxperm/%j.out
#SBATCH --error=logs/ctxperm/%j.err
set -euo pipefail
source ~/.bashrc && conda activate silentmethyl
python -u scripts/23_context_permutation.py
