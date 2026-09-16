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
#
# Task E1 -- ASM validation. Score the SNV-CpG pairs built by 53_asm_build.py
# with the frozen breast-epithelium fusion checkpoint and the tissue-agnostic
# sequence checkpoint, then evaluate discrimination against distance-matched
# negatives.
#
# Source catalogue: Rosenski et al., Nat Commun 16:2141 (2025), Supplementary
# Table S3 (55,271 ASM SNP loci); underlying WGBS GEO GSE186458. Restricted to
# chr8+chr9, which is the held-out split -- held-out-ness here is CHROMOSOMAL,
# not array membership, which is the whole point of the analysis.
#
# Two contrasts are scored in one pass:
#   positive_vs_bimodal_non_asm   controlled  -- lead with this
#   positive_vs_background        weaker, less specific
#
# Inference only, FP32, seed 42 only. No gradient steps, no new weights.
#   sbatch jobs/r8_journal/53_asm_score.sbatch
set -euo pipefail
cd /ocean/projects/med250012p/szhang37/SilentMethyl
source jobs/r8_journal/_common.sh

# --output-dir MUST be passed. Both scripts default under results/journal/, and
# the 12 Sep incident was exactly a missing --output-dir.
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

# CPU, seconds. Kept in the same job so the result lands with the scores rather
# than waiting on a separate submission.
$PY -u scripts/53c_asm_evaluate.py \
    --scores-csv "${OUT}/asm_pair_scores.csv" \
    --output-dir "${OUT}"

check_no_clobber "$STAMP" "${OUT}"
rm -f "$STAMP"
echo "=== asm validation done $(date -u +%FT%TZ) ==="
