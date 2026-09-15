#!/bin/bash
#SBATCH --job-name=SM_ctx_smoke
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/r8_journal/ctx_smoke_%A.out
#SBATCH --error=logs/r8_journal/ctx_smoke_%A.err
#
# Task B smoke: all four rungs on 400 pairs, scratch output. Proves the two new
# schemes actually swap context, that PhyloP stays put, and that the level-MAE
# join works -- before the full ladder spends ~5 GPU h finding out.
set -euo pipefail
cd /ocean/projects/med250012p/szhang37/SilentMethyl
source jobs/r8_journal/_common.sh

OUT=/tmp/claude-102780/-ocean-projects-med250012p-szhang37-SilentMethyl/334d8211-9c0b-4da6-bb02-eba2c11fc798/scratchpad/ctx_smoke
mkdir -p "$OUT"
STAMP=$(mktemp); stamp_start "$STAMP"; require_cuda
echo "[*] host $(hostname) $(date -u +%FT%TZ)"

$PY -u scripts/23_context_permutation.py \
    --schemes identity shuffle tissue_Lung xtissue_mean \
    --seed 42 --limit 400 \
    --weights-template "${FUSION_W}" \
    --split-template "${SPLITS}" \
    --device cuda \
    --output-dir "$OUT"

check_no_clobber "$STAMP"
rm -f "$STAMP"
echo "=== ctx smoke ok $(date -u +%FT%TZ) ==="
