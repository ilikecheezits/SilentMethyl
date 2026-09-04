#!/bin/bash
#SBATCH --job-name=SM_egtex_mt
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/egtex_mt_scoring/score_%a_%A.out
#SBATCH --error=logs/egtex_mt_scoring/score_%a_%A.err
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

echo "[*] Host: $(hostname)  task ${SLURM_ARRAY_TASK_ID}  ${MODEL} seed ${SEED}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u scripts/19_genoa_variant_scoring.py \
  --input-csv data/external/egtex_multitissue/scoring/union_scoring_input_heldout.csv \
  --stratum heldout --models "${MODEL}" --seeds "${SEED}" \
  --device cuda --batch-size 32 \
  --output-dir results/journal/egtex_multitissue_scoring
echo "[done] ${MODEL} seed ${SEED}: $(date)"
