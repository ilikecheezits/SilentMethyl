#!/bin/bash
#SBATCH --job-name=SM_tycko_asm
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/r8_journal/tycko_asm_%A.out
#SBATCH --error=logs/r8_journal/tycko_asm_%A.err
# Slurm job: build, score and evaluate the signed Do-Tycko ASM catalogue (scripts 53d, 53e).

set -euo pipefail
cd "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source jobs/r8_journal/_common.sh

OUT=results/journal/asm_validation_tycko
IN=data/external/asm_tycko_gb2020/scoring

STAMP=$(mktemp)
stamp_start "$STAMP"
require_cuda

echo "[*] host $(hostname)  $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for f in "${IN}/tycko_scoring_pairs.csv" "${IN}/tycko_cpg_records.csv"; do
  [[ -s "$f" ]] || { echo "[!!] missing build input: $f -- run 53d_tycko_build.py first" >&2; exit 1; }
done

$PY -u scripts/53b_asm_score.py \
    --pairs-csv "${IN}/tycko_scoring_pairs.csv" \
    --records-csv "${IN}/tycko_cpg_records.csv" \
    --models fusion sequence \
    --seed 42 \
    --batch-size 32 \
    --output-dir "${OUT}"

mv "${OUT}/asm_pair_scores.csv" "${OUT}/tycko_pair_scores.csv"

$PY -u scripts/53e_tycko_evaluate.py \
    --scores-csv "${OUT}/tycko_pair_scores.csv" \
    --output-dir "${OUT}"

check_no_clobber "$STAMP" "${OUT}"
rm -f "$STAMP"
echo "=== tycko ASM validation done $(date -u +%FT%TZ) ==="
