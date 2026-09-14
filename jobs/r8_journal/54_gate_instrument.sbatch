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
#
# Task A -- instrumented forward pass for the gate decomposition.
#
# Inference only, FP32, seed 42 only. One array task per cohort so the two run
# in the time of one; each writes its own directory, so no write collision.
#   sbatch jobs/r8_journal/54_gate_instrument.sbatch
set -euo pipefail
cd /ocean/projects/med250012p/szhang37/SilentMethyl
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

# --split-template MUST point at the breast-epithelium build: it is where the
# seven context features come from, and the default would feed MCF-10A context
# to the new fusion weights.
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
