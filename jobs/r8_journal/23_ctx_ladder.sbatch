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
#
# Task B -- the full four-rung context dose-response ladder, eGTEx heldout.
#
#   rung 1  identity      native context (control)
#   rung 2  shuffle       another probe's context, same tissue
#   rung 3  tissue_Lung   the SAME locus's context in a different tissue
#   rung 4  xtissue_mean  per-locus mean across all four tissues
#
# Smoke-tested as 46006922 (400 pairs, COMPLETED 0:0): both new rungs replaced
# 400/400 rows with PhyloP max shift 0.000e+00, and all four tissue builds
# resolved 105/105 probes. Measured 0.0575 s/pair/scheme there, so 76,893 pairs
# x 4 schemes ~= 4.9 GPU h inside the 8 h walltime.
#
# Inference only, FP32, seed 42 only.
#   sbatch jobs/r8_journal/23_ctx_ladder.sbatch
set -euo pipefail
cd "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source jobs/r8_journal/_common.sh

# --output-dir MUST be passed: script 23 defaults to results/journal/
# context_permutation, which is outside both declared subtrees.
# --split-template and --weights-template likewise -- the defaults point at
# data/datafiles/ and checkpoints_journal/, i.e. the MCF-10A context and the
# published checkpoints, not the breast-epithelium build this ladder is about.
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
