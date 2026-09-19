#!/bin/bash
#SBATCH --job-name=SM_asm_score
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/r8_journal/asm_score_%A.out
#SBATCH --error=logs/r8_journal/asm_score_%A.err
# Slurm job: score and evaluate the Rosenski ASM SNV-CpG pairs with the frozen fusion and sequence checkpoints (scripts 53b, 53c).

set -euo pipefail
cd "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source jobs/r8_journal/_common.sh

OUT=results/journal/asm_validation
IN=data/external/asm_atlas/scoring

STAMP=$(mktemp)
stamp_start "$STAMP"
require_cuda

echo "[*] host $(hostname)  $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for f in "${IN}/asm_scoring_pairs.csv" "${IN}/asm_cpg_records.csv"; do
  [[ -s "$f" ]] || { echo "[!!] missing build input: $f -- run 53_asm_build.py first" >&2; exit 1; }
done

$PY -u scripts/53b_asm_score.py \
    --pairs-csv "${IN}/asm_scoring_pairs.csv" \
    --records-csv "${IN}/asm_cpg_records.csv" \
    --models fusion sequence \
    --seed 42 \
    --batch-size 32 \
    --output-dir "${OUT}"

$PY -u scripts/53c_asm_evaluate.py \
    --scores-csv "${OUT}/asm_pair_scores.csv" \
    --output-dir "${OUT}"

check_no_clobber "$STAMP" "${OUT}"
rm -f "$STAMP"
echo "=== asm validation done $(date -u +%FT%TZ) ==="
