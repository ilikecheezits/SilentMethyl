#!/bin/bash
#SBATCH --job-name=melody_st
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/melody_st/%j.out
#SBATCH --error=logs/melody_st/%j.err
# Slurm job: zero-shot application of a single-tissue Melody model across tissues.

set -eo pipefail

PY="${MELODY_PY:-python}"

if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "STOP: CUDA is not available on $(hostname). Nothing was run." >&2
  echo "Requeue with:  sbatch --exclude=$(hostname) <this script>" >&2
  exit 4
fi

DRIVE=data/external/melody/Melody_repo/drive
SCORING=data/external/egtex_multitissue/scoring
OUT=results/journal/melody_st

TISSUES="BreastMammaryTissue ColonTransverse KidneyCortex Lung MuscleSkeletal Ovary Prostate Testis WholeBlood"

mkdir -p "${OUT}"

echo "=== ST-Breast (Luminal) zero-shot across nine tissues ==="
for T in ${TISSUES}; do
  IN=${SCORING}/${T}/egtex_scoring_input_heldout.csv
  if [ ! -f "${IN}" ]; then
    echo "SKIP ${T}: ${IN} not found" >&2
    continue
  fi
  echo "--- ${T} ---"
  $PY -u scripts/33_melody_scoring.py --format ours \
      --checkpoint "${DRIVE}/Melody-ST-Breast-Luminal.pth" \
      --n-track 1 --track-index 0 \
      --tracks GSM5652347_Breast-Luminal-Epithelial-Z000000V2 \
      --input-csv "${IN}" \
      --output "${OUT}/st_breast_${T}.csv"
done

echo "=== ST-Lung (Alveolar) on Lung only -- the tissue-matched arm ==="
$PY -u scripts/33_melody_scoring.py --format ours \
    --checkpoint "${DRIVE}/Melody-ST-Lung-Alveolar.pth" \
    --n-track 1 --track-index 0 \
    --tracks GSM5652354_Lung-Alveolar-Epithelial-Z000000T1 \
    --input-csv "${SCORING}/Lung/egtex_scoring_input_heldout.csv" \
    --output "${OUT}/st_lung_Lung.csv"

echo "=== done $(date -u +%FT%TZ) ==="
