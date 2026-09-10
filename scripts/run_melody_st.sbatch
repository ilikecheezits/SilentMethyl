#!/bin/bash
# Melody-ST zero-shot: a SINGLE-TISSUE Melody applied across tissues.
#
# Why this run exists
# -------------------
# Melody's Fig 3H is cross-TRACK validation on Melody-MT: one model trained on
# all 39 cell types, with 39 output channels, where "cross-track" means reading
# a different channel. That is head selection, not transfer -- the model saw
# every tissue in training. Melody-ST is the genuinely single-tissue variant
# ("trained on one cell type at a time using a single bigWig track as
# supervision", Methods), and the paper never applies one across tissues.
#
# SilentMethyl is single-tissue by construction, so ST-Breast is the correct
# architecture-matched comparison for R3. Without it, "single-tissue training
# still transfers" is our claim and not our evidence.
#
# Two arms:
#   BREAST -> all nine tissues. The zero-shot arm, matched to our own setup.
#   LUNG   -> Lung only. The matched arm. ST-Breast vs ST-Lung on the SAME Lung
#             rows isolates tissue-matching with architecture, training data
#             volume and scoring procedure all held constant.
#
#   sbatch scripts/run_melody_st.sbatch
#SBATCH --job-name=melody_st
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/melody_st/%j.out
#SBATCH --error=logs/melody_st/%j.err
set -eo pipefail

PY=/jet/home/szhang37/.conda/envs/melody/bin/python

# Fail fast if the allocated node cannot actually do CUDA. Node v005 returned
# CUDA error 803 more than once; torch then falls back to CPU *silently* and the
# job runs at 13.3 s/it instead of 3.66 it/s -- a 49x slowdown that only becomes
# visible hours into the log, after the walltime is already spent.
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "STOP: CUDA is not available on $(hostname). Nothing was run." >&2
  echo "Requeue with:  sbatch --exclude=$(hostname) <this script>" >&2
  exit 4
fi

DRIVE=data/external/melody/Melody_repo/drive
SCORING=data/external/egtex_multitissue/scoring
OUT=results/journal/melody_st

# All nine, testis included: we are deliberately using a breast model
# everywhere, so the "no testis track exists" problem that forced testis out of
# the Melody-MT comparison does not arise here.
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
