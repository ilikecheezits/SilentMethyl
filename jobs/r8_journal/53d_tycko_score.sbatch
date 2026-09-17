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
#
# Tasks E1 + E2 -- ASM validation against Do & Tycko 2020 (Genome Biology 21:153),
# the only ASM catalogue found that publishes a SIGNED allelic methylation
# difference. That is what makes direction concordance and signed Spearman
# possible; the Nat Commun 2025 atlas gives significance only and CanASM's server
# has been dead throughout.
#
# 722 biallelic ASM index SNPs on chr8+chr9 (the held-out split), 9,822 ASM CpG
# pairs plus distance-matched negatives. All 722 REF alleles agree with hg38, so
# no sign flips were needed.
#
# Inference only, FP32, seed 42.
#   sbatch jobs/r8_journal/53d_tycko_score.sbatch
#
# NOTE TO WHOEVER RUNS THIS: do not write anything under results/journal/ from
# the login node while this is in flight. That is what failed 46096838 (and
# 45997644 before it) on the clobber guard with the science already correct.
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
