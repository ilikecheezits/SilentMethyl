#!/bin/bash
#SBATCH --job-name=SM_gwas_mb
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gwas_mb/clinvar_mb_%A.out
#SBATCH --error=logs/gwas_mb/clinvar_mb_%A.err
#
# Breast-cancer GWAS risk-variant matched-background test. Inference only.
#
# Pre-registered in results/journal/gwas_matched_background/preregistration.json.
# Input is the GWAS cohort concatenated with the same 14,996-variant synthetic
# background pool used before, because scripts/05 draws comparators from the
# frame it is given.
#
# 2 h rather than 3 so it backfills; the comparable ClinVar run took 48 min of
# inference after an 8 min import stall on a cold node.
#
#     mkdir -p logs/gwas_mb && sbatch scripts/run_gwas_matched_background.sh

set -euo pipefail

PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python
IN=results/journal/gwas_matched_background/gwas_plus_background_scoring_input.csv
OUT=results/journal/gwas_matched_background/scored

mkdir -p logs/gwas_mb

echo "[*] Host:    $(hostname)"
echo "[*] Job:     ${SLURM_JOB_ID:-interactive}"
echo "[*] Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

test -f "${IN}" || { echo "missing ${IN}; run --build first"; exit 1; }

"${PY}" -u scripts/05_matched_background.py \
  --input-csv "${IN}" \
  --seeds 42 43 44 \
  --output-dir "${OUT}"

echo "[done] $(date)"
