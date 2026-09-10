#!/bin/bash
#SBATCH --job-name=ctxperm
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ctxperm/%j.out
#SBATCH --error=logs/ctxperm/%j.err
set -eo pipefail
PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python

# Fail fast if the allocated node cannot actually do CUDA. Node v005 returned
# CUDA error 803 more than once; torch then falls back to CPU *silently* and the
# job runs at 13.3 s/it instead of 3.66 it/s -- a 49x slowdown that only becomes
# visible hours into the log, after the walltime is already spent.
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "STOP: CUDA is not available on $(hostname). Nothing was run." >&2
  echo "Requeue with:  sbatch --exclude=$(hostname) <this script>" >&2
  exit 4
fi

$PY -u scripts/23_context_permutation.py
