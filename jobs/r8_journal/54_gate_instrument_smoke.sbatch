#!/bin/bash
#SBATCH --job-name=SM_gate_smoke
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/r8_journal/gate_smoke_%A.out
#SBATCH --error=logs/r8_journal/gate_smoke_%A.err
#
# Task A smoke test. 400 genoa pairs, scratchpad output, sole purpose is to
# prove the hand-rebuilt forward pass still reproduces scripts/20 before the
# full array spends 8 GPU hours finding that out at the very end.
set -euo pipefail
cd /ocean/projects/med250012p/szhang37/SilentMethyl
source jobs/r8_journal/_common.sh

SMOKE_OUT=/tmp/claude-102780/-ocean-projects-med250012p-szhang37-SilentMethyl/334d8211-9c0b-4da6-bb02-eba2c11fc798/scratchpad/gate_smoke
mkdir -p "$SMOKE_OUT"
REF="${ABL}/genoa_variant_scoring/heldout/fusion/seed42/pair_scores.csv"

STAMP=$(mktemp)
stamp_start "$STAMP"
require_cuda
echo "[*] host $(hostname)  $(date -u +%FT%TZ)"

$PY -u scripts/54_gate_decomposition.py instrument \
    --cohort genoa --seed 42 \
    --weights-template "${FUSION_W}" \
    --split-template "${SPLITS}" \
    --reference-pair-scores "${REF}" \
    --device cuda --batch-size 32 --limit 400 \
    --output-dir "$SMOKE_OUT"

# Smoke output goes to scratch, so NOTHING under results/journal may change.
check_no_clobber "$STAMP"
rm -f "$STAMP"
echo "=== smoke ok $(date -u +%FT%TZ) ==="
