#!/bin/bash
#SBATCH --job-name=SM_base_seeds
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/baselines/seed_%a_%A.out
#SBATCH --error=logs/baselines/seed_%a_%A.err
#
# PHASE 2: three seeds per architecture at the hyperparameters phase 1 selected.
# Only now is the test set evaluated. Run this ONLY after both grid tasks finish.
#
#     sbatch scripts/run_baseline_seeds.sh

set -euo pipefail
PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python
mkdir -p logs/baselines
COMBOS=("cpgenie:42" "cpgenie:43" "cpgenie:44" "deepcpg:42" "deepcpg:43" "deepcpg:44")
ENTRY="${COMBOS[${SLURM_ARRAY_TASK_ID}]}"
ARCH="${ENTRY%%:*}"
SEED="${ENTRY##*:}"
SEL="results/journal/published_baselines/${ARCH}/selected_hyperparameters.json"

test -f "${SEL}" || { echo "missing ${SEL}; run the grid phase first"; exit 1; }
DROPOUT=$("${PY}" -c "import json;print(json.load(open('${SEL}'))['dropout'])")
LR=$("${PY}" -c "import json;print(json.load(open('${SEL}'))['lr'])")

echo "[*] Host: $(hostname)  ${ARCH} seed ${SEED}  dropout ${DROPOUT} lr ${LR}"
echo "[*] Started $(date)"

"${PY}" -u scripts/15_baselines_published.py \
  --arch "${ARCH}" --seed "${SEED}" --dropout "${DROPOUT}" --lr "${LR}" \
  --epochs 10 --batch-size 128 --num-workers 5

echo "[done] ${ARCH} seed ${SEED} $(date)"
