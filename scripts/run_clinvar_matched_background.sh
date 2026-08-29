#!/bin/bash
#SBATCH --job-name=SM_clinvar_mb
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/clinvar_mb/clinvar_mb_%A.out
#SBATCH --error=logs/clinvar_mb/clinvar_mb_%A.err
set -euo pipefail
PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python
mkdir -p logs/clinvar_mb
echo "[*] Host: $(hostname)  Started: $(date)"
$PY -u scripts/05_matched_background.py \
  --input-csv results/journal/clinvar_matched_background/clinvar_heldout_nontruncating_cohort.csv \
  --seeds 42 43 44 \
  --output-dir results/journal/clinvar_matched_background/scored
echo "[done] $(date)"
