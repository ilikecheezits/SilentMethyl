#!/bin/bash
#SBATCH --job-name=SM_genoa_score
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/genoa_scoring/genoa_score_%a_%A.out
#SBATCH --error=logs/genoa_scoring/genoa_score_%a_%A.err
#
# Stage B.1 -- GENOA variant scoring on the frozen checkpoints.
#
# Inference only. No gradient steps, no checkpoint is written, nothing already on
# disk is modified.
#
# One array task per model-seed combination, six in total. Running them as an
# array rather than a loop matters here: the tokenizer pads every window to 1,000
# tokens and attention is quadratic in that length, so a single pass over the
# 66,495 held-out pairs is roughly 1.5-2.5 hours on one V100. Six of those in
# sequence would exceed any sensible wall clock; six in parallel finish in the
# time of one, and a task that fails can be resubmitted on its own with
#     sbatch --array=3 scripts/run_genoa_scoring.sh
#
# Each task writes to its own directory:
#     results/journal/genoa_variant_scoring/heldout/<model>/seed<seed>/pair_scores.csv
# and its own run_summary_<model>_seed<seed>.json, so there is no write collision.
#
# Submit from the repository root. The mkdir is NOT optional and NOT redundant
# with the one below: Slurm opens the --output/--error files before the script
# body runs, so if logs/genoa_scoring/ does not already exist every task dies
# before its first line with the batch step CANCELLED and no log to explain it.
#
#     mkdir -p logs/genoa_scoring && sbatch scripts/run_genoa_scoring.sh
#
# Smoke-test first -- one minute on CPU, and it exercises every code path:
#     python -u scripts/19_genoa_variant_scoring.py --limit 200 --seeds 42
#
# Watch:
#     squeue -u $USER
#     tail -f logs/genoa_scoring/genoa_score_0_*.out

set -euo pipefail

command -v conda >/dev/null || module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate silentmethyl

mkdir -p logs/genoa_scoring

# Task id -> (model, seed). The sequence model is the tissue-agnostic comparator
# that makes the fusion result on blood-derived GENOA interpretable, so both are
# scored and neither is optional.
COMBINATIONS=(
  "fusion:42" "fusion:43" "fusion:44"
  "sequence:42" "sequence:43" "sequence:44"
)
ENTRY="${COMBINATIONS[${SLURM_ARRAY_TASK_ID}]}"
MODEL="${ENTRY%%:*}"
SEED="${ENTRY##*:}"

echo "[*] Host:    $(hostname)"
echo "[*] Job:     ${SLURM_ARRAY_JOB_ID:-interactive} task ${SLURM_ARRAY_TASK_ID:-0}"
echo "[*] Scoring: model=${MODEL} seed=${SEED}"
echo "[*] Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/19_genoa_variant_scoring.py \
  --input-csv data/external/genoa_meqtl/scoring/genoa_scoring_input_heldout.csv \
  --stratum heldout \
  --models "${MODEL}" \
  --seeds "${SEED}" \
  --device cuda \
  --batch-size 32 \
  --output-dir results/journal/genoa_variant_scoring

echo "[done] ${MODEL} seed ${SEED}: $(date)"
