#!/bin/bash
#SBATCH --job-name=Build_SilentMethyl_Data
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000M
#SBATCH --time=06:00:00
#SBATCH --output=logs/data_build/build_data_%j.out
#SBATCH --error=logs/data_build/build_data_%j.err
# Build the published processed cohort: TCGA-BRCA normal-breast HM450 targets with the seven primary breast-epithelium context tracks.

set -euo pipefail
umask 027

ROOT="${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$ROOT"

PY="${SILENTMETHYL_PY:-python}"
if [[ -z "${SILENTMETHYL_PY:-}" ]]; then
  command -v conda >/dev/null || module load anaconda3
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate silentmethyl
fi

: "${OUT_DIR:?set OUT_DIR, e.g. --export=ALL,OUT_DIR=data/datafiles_breast_epithelium}"
REFERENCE_DIR="${REFERENCE_DIR:-data/reference/BreastEpithelium}"

mkdir -p logs/data_build "$OUT_DIR"

for required in \
  data/build_training_data.py \
  data/build_testing_data.py \
  data/audit_data_purity.py \
  scripts/17_chromosome_splits.py \
  data/TCGA-BRCA.methylation450.tsv.gz \
  data/HM450.hg38.manifest.tsv.gz \
  data/HM450.hg38.manifest.CpGIsland.tsv.gz \
  data/hg38.fa \
  data/hg38.fa.fai \
  data/reference/hg38.phyloP100way.bw \
  data/reference/gencode.v44.annotation.gtf.gz \
  data/datafiles/gdc_tcga_brca_synonymous_raw.json.gz \
  "$REFERENCE_DIR"/{ATAC_seq,H3K4me3,H3K27ac,H3K27me3,H3K9me3,H3K36me3,H3K4me1}.bw; do
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
echo "[*] Context: $REFERENCE_DIR -> $OUT_DIR"

echo "[*] Phase 0a: training, validation and test splits..."
"$PY" -u data/build_training_data.py \
  --data-dir data \
  --reference-dir "$REFERENCE_DIR" \
  --out-dir "$OUT_DIR" \
  --m-value-precision float64 \
  --val-chroms chr10 chr11 \
  --test-chroms chr8 chr9

echo "[*] Phase 0b: somatic synonymous candidate cohort..."
"$PY" -u data/build_testing_data.py \
  --data-dir data \
  --reference-dir "$REFERENCE_DIR" \
  --out-dir "$OUT_DIR"

echo "[*] Phase 0c: processed-data integrity audit..."
"$PY" -u data/audit_data_purity.py \
  --data-dir "$OUT_DIR" \
  --output "$OUT_DIR/data_purity_audit.json"
"$PY" -m json.tool "$OUT_DIR/data_purity_audit.json" >/dev/null

echo "[*] Phase 0d: chromosome-blocked folds..."
"$PY" -u scripts/17_chromosome_splits.py \
  --datafiles "$OUT_DIR" \
  --out-root "$OUT_DIR/splits" \
  --folds 4

(cd "$OUT_DIR" && find . -type f \( -name '*.csv' -o -name '*.fasta' \) | sort \
   | xargs sha256sum) > "$OUT_DIR/SHA256SUMS.txt"
echo "[*] checksums: $OUT_DIR/SHA256SUMS.txt"

if [[ -n "${VERIFY_AGAINST:-}" ]]; then
  echo "[*] Comparing against $VERIFY_AGAINST ..."
  (cd "$VERIFY_AGAINST" && find . -type f \( -name '*.csv' -o -name '*.fasta' \) | sort \
     | xargs sha256sum) > "$OUT_DIR/SHA256SUMS.reference.txt"
  if diff "$OUT_DIR/SHA256SUMS.reference.txt" "$OUT_DIR/SHA256SUMS.txt"; then
    echo "[✓] byte-identical to $VERIFY_AGAINST"
  else
    echo "[!] differs from $VERIFY_AGAINST (lines above)" >&2
    exit 3
  fi
fi

echo "[✓] Data build complete. Ready for model training."
