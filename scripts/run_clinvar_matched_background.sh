#!/bin/bash
#SBATCH --job-name=SM_clinvar_mb
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/clinvar_mb/clinvar_mb_%A.out
#SBATCH --error=logs/clinvar_mb/clinvar_mb_%A.err
#
# ClinVar matched-background test. Inference only; no checkpoint is written.
#
# The input is the 35 held-out ClinVar variants CONCATENATED WITH a synthetic
# background pool, because scripts/05 draws each variant's comparators from the
# frame it is handed. Scoring the 35 alone made every variant's background the
# other 34 pathogenic variants, which invalidated the first attempt.
#
# Pre-flight confirmed 32 of 35 match at T2 (same SBS96, same CpG effect,
# distance within 50 bp) against a 14,996-variant pool, with no variant drawing
# a material share of its background from the ClinVar set.
#
# 5 cpus / 48G is the same allocation the GENOA and eGTEx arrays ran on, so the
# combination is known-good on GPU-shared. 3 h rather than 12 so it backfills;
# the pre-flight estimated ~60 min.
#
#     mkdir -p logs/clinvar_mb && sbatch scripts/run_clinvar_matched_background.sh

set -euo pipefail

PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python
IN=results/journal/clinvar_matched_background/clinvar_plus_background_scoring_input.csv
OUT=results/journal/clinvar_matched_background/scored

mkdir -p logs/clinvar_mb

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
