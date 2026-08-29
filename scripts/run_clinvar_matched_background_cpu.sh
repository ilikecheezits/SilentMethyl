#!/bin/bash
#SBATCH --job-name=SM_clinvar_mb
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=03:00:00
#SBATCH --output=logs/clinvar_mb/clinvar_mb_%A.out
#SBATCH --error=logs/clinvar_mb/clinvar_mb_%A.err
set -euo pipefail
PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python
mkdir -p logs/clinvar_mb
echo "[*] Host: $(hostname)  Started: $(date)"
$PY -u data/prepare_clinvar_matched_background.py --run --device cpu
echo "[done] $(date)"
