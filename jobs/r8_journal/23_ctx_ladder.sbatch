#!/bin/bash
#SBATCH --job-name=SM_ctx_ladder
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/r8_journal/ctx_ladder_%A.out
#SBATCH --error=logs/r8_journal/ctx_ladder_%A.err
# Slurm job: run the four-rung context dose-response ladder on the eGTEx held-out set (script 23).

set -euo pipefail
cd "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source jobs/r8_journal/_common.sh

OUT="${ABL}/context_ladder"

STAMP=$(mktemp)
stamp_start "$STAMP"
require_cuda

echo "[*] host $(hostname)  $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY -u scripts/23_context_permutation.py \
    --schemes identity shuffle tissue_Lung xtissue_mean \
    --seed 42 \
    --weights-template "${FUSION_W}" \
    --split-template "${SPLITS}" \
    --device cuda --batch-size 32 \
    --output-dir "${OUT}"

check_no_clobber "$STAMP" "${OUT}"
rm -f "$STAMP"
echo "=== ctx ladder done $(date -u +%FT%TZ) ==="
