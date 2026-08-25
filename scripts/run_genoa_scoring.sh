#!/bin/bash
#SBATCH --job-name=SM_genoa_score
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/genoa_scoring/genoa_score_%A.out
#SBATCH --error=logs/genoa_scoring/genoa_score_%A.err
#
# Stage B.1 -- GENOA variant scoring on the frozen checkpoints.
#
# Inference only. No gradient steps, no checkpoint is written, nothing already on
# disk is modified. Six passes over the held-out pairs: 2 models x 3 seeds.
#
# Cost. 66,557 held-out pairs x 4 forward passes each (WT/MUT x forward/RC) is
# ~266k sequence evaluations per model-seed. On one V100 that is roughly 25-40
# minutes, so all six land inside a single 8-hour allocation with wide margin.
#
# Submit from the repository root:
#     sbatch scripts/run_genoa_scoring.sh
#
# Before the real run, smoke-test on a login node or an interactive session:
#     python -u scripts/19_genoa_variant_scoring.py --limit 200 --seeds 42
# That exercises every code path in about a minute and will surface a coordinate
# or checkpoint problem before you spend an allocation on it.
#
# Watch:
#     squeue -u $USER
#     tail -f logs/genoa_scoring/genoa_score_*.out

set -euo pipefail

command -v conda >/dev/null || module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate silentmethyl

mkdir -p logs/genoa_scoring

echo "[*] Host:    $(hostname)"
echo "[*] Job:     ${SLURM_JOB_ID:-interactive}"
echo "[*] Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/19_genoa_variant_scoring.py \
  --input-csv data/external/genoa_meqtl/scoring/genoa_scoring_input_heldout.csv \
  --stratum heldout \
  --models fusion sequence \
  --seeds 42 43 44 \
  --batch-size 32 \
  --output-dir results/journal/genoa_variant_scoring

echo "[done] $(date)"
