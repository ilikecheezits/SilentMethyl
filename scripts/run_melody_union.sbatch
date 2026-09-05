#!/bin/bash
#SBATCH --job-name=melody_union
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/melody/union_%j.out
#SBATCH --error=logs/melody/union_%j.err
set -euo pipefail
source ~/.bashrc && conda activate melody
python -u scripts/33_melody_scoring.py --format ours \
  --input-csv data/external/egtex_multitissue/scoring/union_scoring_input_heldout.csv \
  --tracks "${MELODY_TRACKS}" --margin "${MELODY_MARGIN}" \
  --output results/journal/melody/union_pair_scores.csv
