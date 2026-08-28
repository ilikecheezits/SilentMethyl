#!/bin/bash
#SBATCH --job-name=SM_egtex_score
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/egtex_scoring/egtex_score_%a_%A.out
#SBATCH --error=logs/egtex_scoring/egtex_score_%a_%A.err
#
# eGTEx Breast Mammary Tissue variant scoring on the frozen checkpoints.
#
# This is the TISSUE-MATCHED arm. GENOA is peripheral blood, so a weak GENOA
# correlation is confounded with a tissue change and cannot falsify the model.
# eGTEx Breast is the same tissue the model was trained on, which makes it the
# arm that can. Both are reported; they answer different questions.
#
# Inference only. No gradient steps, no checkpoint is written, nothing already on
# disk is modified.
#
# 76,893 held-out pairs, against GENOA's 66,495 -- so ~1.15x the GENOA runtime,
# roughly 2-3 hours per model-seed on one V100. Six in parallel finish in the
# time of one. A single failed task resubmits on its own:
#     sbatch --array=3 scripts/run_egtex_scoring.sh
#
# The mkdir below is NOT redundant with the one in the submit command: Slurm
# opens --output/--error before the script body runs, so if logs/egtex_scoring/
# does not already exist every task dies with the batch step CANCELLED and no
# log to explain it. That cost us a full array once already.
#
#     mkdir -p logs/egtex_scoring && sbatch scripts/run_egtex_scoring.sh
#
# Smoke-test first -- one minute on CPU, exercises every code path including the
# effect-column resolver, which must report `beta_ref_to_alt`, not the GENOA name:
#     python -u scripts/19_genoa_variant_scoring.py \
#       --input-csv data/external/egtex_breast/scoring/egtex_scoring_input_heldout.csv \
#       --limit 200 --seeds 42 --device cpu \
#       --output-dir results/journal/egtex_variant_scoring

set -euo pipefail

command -v conda >/dev/null || module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate silentmethyl

mkdir -p logs/egtex_scoring

# Both models are scored and neither is optional. The sequence model is what
# makes the fusion number interpretable: the context vector is identical for REF
# and ALT, so any fusion-minus-sequence difference in variant effects can only
# arise through a gate shift. On GENOA that difference was indistinguishable
# from zero. Whether it stays zero in the matched tissue is a real question --
# if context ever mattered for variant effects, it should matter most here.
COMBINATIONS=(
  "fusion:42" "fusion:43" "fusion:44"
  "sequence:42" "sequence:43" "sequence:44"
)
ENTRY="${COMBINATIONS[${SLURM_ARRAY_TASK_ID}]}"
MODEL="${ENTRY%%:*}"
SEED="${ENTRY##*:}"

echo "[*] Host:    $(hostname)"
echo "[*] Job:     ${SLURM_ARRAY_JOB_ID:-interactive} task ${SLURM_ARRAY_TASK_ID:-0}"
echo "[*] Scoring: model=${MODEL} seed=${SEED}  cohort=eGTEx Breast Mammary"
echo "[*] Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/19_genoa_variant_scoring.py \
  --input-csv data/external/egtex_breast/scoring/egtex_scoring_input_heldout.csv \
  --stratum heldout \
  --models "${MODEL}" \
  --seeds "${SEED}" \
  --device cuda \
  --batch-size 32 \
  --output-dir results/journal/egtex_variant_scoring

echo "[done] ${MODEL} seed ${SEED}: $(date)"
