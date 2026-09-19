#!/bin/bash
#SBATCH --job-name=SM_gate_instr
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/r8_journal/gate_instr_%a_%A.out
#SBATCH --error=logs/r8_journal/gate_instr_%a_%A.err
# Slurm array job: instrumented forward pass for the gate decomposition, one task per cohort, seed 42 (script 54).

set -euo pipefail
cd "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source jobs/r8_journal/_common.sh

OUT="${ABL}/gate_decomposition"
COHORTS=("genoa" "egtex")
C="${COHORTS[${SLURM_ARRAY_TASK_ID}]}"
REF="${ABL}/${C}_variant_scoring/heldout/fusion/seed42/pair_scores.csv"

STAMP=$(mktemp)
stamp_start "$STAMP"
require_cuda

echo "[*] host $(hostname)  cohort ${C}  $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

$PY -u scripts/54_gate_decomposition.py instrument \
    --cohort "${C}" \
    --seed 42 \
    --weights-template "${FUSION_W}" \
    --split-template "${SPLITS}" \
    --reference-pair-scores "${REF}" \
    --device cuda --batch-size 32 \
    --output-dir "${OUT}"

check_no_clobber "$STAMP" "${OUT}"
rm -f "$STAMP"
echo "=== gate instrument ${C} done $(date -u +%FT%TZ) ==="
