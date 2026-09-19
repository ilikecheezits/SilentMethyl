#!/bin/bash
#SBATCH --job-name=SM_egtex_mt
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/egtex_mt_scoring/score_%a_%A.out
#SBATCH --error=logs/egtex_mt_scoring/score_%a_%A.err
# Score the eGTEx multi-tissue variant set; defaults reproduce the published nine-tissue run.

set -euo pipefail
command -v conda >/dev/null || module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate silentmethyl

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
  echo "FATAL: CUDA unavailable on $(hostname). Resubmit: sbatch --array=${SLURM_ARRAY_TASK_ID} $0"
  nvidia-smi || true; exit 1; }

COMBINATIONS=("fusion:42" "fusion:43" "fusion:44" "sequence:42" "sequence:43" "sequence:44")
ENTRY="${COMBINATIONS[${SLURM_ARRAY_TASK_ID}]}"
MODEL="${ENTRY%%:*}"; SEED="${ENTRY##*:}"

INPUT_CSV="${INPUT_CSV:-data/external/egtex_multitissue/scoring/union_scoring_input_heldout.csv}"
OUT_DIR="${OUT_DIR:-results/journal/egtex_multitissue_scoring}"

EXTRA=()
if [ -n "${WEIGHTS_TEMPLATE:-}" ]; then
  EXTRA+=(--weights-template "${WEIGHTS_TEMPLATE}")
fi
if [ -n "${SPLIT_TEMPLATE:-}" ]; then
  EXTRA+=(--split-template "${SPLIT_TEMPLATE}")
fi

if [ -n "${WEIGHTS_TEMPLATE:-}" ] && [ -z "${SPLIT_TEMPLATE:-}" ]; then
  echo "FATAL: WEIGHTS_TEMPLATE set without SPLIT_TEMPLATE. That scores the new"
  echo "       weights against the default context and yields a hybrid model."
  exit 2
fi
if [ -n "${WEIGHTS_TEMPLATE:-}" ] && [ "${OUT_DIR}" = "results/journal/egtex_multitissue_scoring" ]; then
  echo "FATAL: refusing to write alternative-context scores over the published"
  echo "       path ${OUT_DIR}. Set OUT_DIR."
  exit 2
fi

[ -s "${INPUT_CSV}" ] || { echo "FATAL: missing or empty ${INPUT_CSV}"; exit 1; }

echo "[*] Host: $(hostname)  task ${SLURM_ARRAY_TASK_ID}  ${MODEL} seed ${SEED}"
echo "[*] input   ${INPUT_CSV}"
echo "[*] output  ${OUT_DIR}"
echo "[*] extra   ${EXTRA[*]:-<published defaults>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u scripts/20_variant_scoring.py \
  --input-csv "${INPUT_CSV}" \
  --stratum heldout --models "${MODEL}" --seeds "${SEED}" \
  --device cuda --batch-size 32 \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  --output-dir "${OUT_DIR}"
echo "[done] ${MODEL} seed ${SEED}: $(date)"
