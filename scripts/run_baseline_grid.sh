#!/bin/bash
#SBATCH --job-name=SM_base_grid
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/baselines/grid_%a_%A.out
#SBATCH --error=logs/baselines/grid_%a_%A.err
# Phase 1: hyperparameter selection for the published baselines on the validation split only.

set -euo pipefail
PY="${SILENTMETHYL_PY:-python}"
: "${DATA:?set DATA, e.g. --export=ALL,DATA=data/datafiles_breast_epithelium}"
OUT_DIR="${OUT_DIR:-results/journal/published_baselines}"
mkdir -p logs/baselines
ARCHS=(cpgenie deepcpg)
ARCH="${ARCHS[${SLURM_ARRAY_TASK_ID}]}"

echo "[*] Host: $(hostname)  arch=${ARCH}  started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PY}" -u scripts/15_baselines_published.py \
  --arch "${ARCH}" --grid --epochs 10 --batch-size 128 --num-workers 5 \
  --train "${DATA}/train.csv" --val "${DATA}/val.csv" --test "${DATA}/test.csv" \
  --output-dir "${OUT_DIR}"

echo "[done] ${ARCH} $(date)"
