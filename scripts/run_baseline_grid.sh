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
#
# PHASE 1: hyperparameter selection on the VALIDATION split only.
#
# The test set is never touched here. scripts/28 --grid trains each
# configuration, records validation beta MAE, and writes selected_hyperparameters
# .json. Looking at test during tuning would invalidate every number in Table 1.
#
# Task 0 = CpGenie, 9 configs (the authors' own hyperas grid:
#          dropout {0.3,0.5,0.7} x lr {0.01,0.001,0.0001})
# Task 1 = DeepCpG, 3 configs (dropout {0.0,0.3,0.5} at their default lr)
#
#     mkdir -p logs/baselines && sbatch scripts/run_baseline_grid.sh

set -euo pipefail
PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python
mkdir -p logs/baselines
ARCHS=(cpgenie deepcpg)
ARCH="${ARCHS[${SLURM_ARRAY_TASK_ID}]}"

echo "[*] Host: $(hostname)  arch=${ARCH}  started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PY}" -u scripts/15_baselines_published.py \
  --arch "${ARCH}" --grid --epochs 10 --batch-size 128 --num-workers 5

echo "[done] ${ARCH} $(date)"
