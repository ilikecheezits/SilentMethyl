#!/bin/bash
#SBATCH --job-name=Build_SilentMethyl_Data
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=33
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --output=logs/data_build/build_data_%j.out
#SBATCH --error=logs/data_build/build_data_%j.err

set -euo pipefail
umask 027

ROOT=/ocean/projects/med250012p/szhang37/SilentMethyl
cd "$ROOT"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate silentmethyl

mkdir -p logs/data_build logs/reproducibility reproducibility

for required in \
  data/build_training_data.py \
  data/build_testing_data.py \
  data/audit_data_purity.py \
  data/TCGA-BRCA.methylation450.tsv.gz \
  data/HM450.hg38.manifest.tsv.gz \
  data/hg38.fa \
  data/hg38.fa.fai \
  data/datafiles/gdc_tcga_brca_synonymous_raw.json.gz; do
  if [[ ! -s "$required" ]]; then
    echo "[!] Missing required file: $required" >&2
    exit 2
  fi
done

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "[*] Host: $(hostname)"
echo "[*] Job ID: ${SLURM_JOB_ID:-NA}"
echo "[*] Git commit: $(git rev-parse HEAD 2>/dev/null || echo unavailable)"

echo "[*] Phase 0a: Building Leakage-Resistant Training & Validation Data..."
python -u data/build_training_data.py \
  --data-dir data \
  --val-chroms chr10 chr11 \
  --test-chroms chr8 chr9

echo "[*] Phase 0b: Building Cleaned Somatic Synonymous Testing Cohort..."
python -u data/build_testing_data.py \
  --data-dir data

echo "[*] Phase 0c: Auditing processed-data integrity..."
python -u data/audit_data_purity.py \
  --data-dir data/datafiles \
  --output data/datafiles/data_purity_audit.json

python -m json.tool data/datafiles/data_purity_audit.json >/dev/null
cp data/datafiles/data_purity_audit.json \
  reproducibility/data_purity_audit.json

sha256sum \
  data/datafiles/train.csv \
  data/datafiles/val.csv \
  data/datafiles/test.csv \
  data/datafiles/testing_data_test_only.csv \
  data/datafiles/split_manifest.json \
  data/datafiles/feature_imputation.json \
  data/datafiles/candidate_cohort_manifest.json \
  data/egtex_breast_mqtl_heldout_qc.csv \
  > reproducibility/processed_data_sha256.txt

echo "[✓] Data build and purity audit complete. Ready for model training."
