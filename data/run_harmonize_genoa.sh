#!/bin/bash
#SBATCH --job-name=SM_genoa_harm
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --array=1-22
#SBATCH --output=logs/data_build/harmonize_genoa_chr%a_%A.out
#SBATCH --error=logs/data_build/harmonize_genoa_chr%a_%A.err
# Slurm array job: harmonize GENOA meQTL summary statistics, one chromosome per task.

set -euo pipefail

command -v conda >/dev/null || module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate silentmethyl

mkdir -p logs/data_build

CHROM="${SLURM_ARRAY_TASK_ID}"

echo "[*] Host: $(hostname)"
echo "[*] Job:  ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "[*] Harmonizing GENOA chr${CHROM}"
echo "[*] Started: $(date)"

python -u data/harmonize_genoa_meqtl.py \
  --chrom "${CHROM}" \
  --half-window 500 \
  --output-dir data/external/genoa_meqtl/harmonized

echo "[✓] chr${CHROM} complete: $(date)"
