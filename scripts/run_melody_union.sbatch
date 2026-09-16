#!/bin/bash
#SBATCH --job-name=melody_union
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/melody/union_%j.out
#SBATCH --error=logs/melody/union_%j.err
set -eo pipefail
# Inline, NOT via --export: sbatch splits --export values on commas, which
# silently truncated this list to its first element on the first attempt.
TRACKS="GSM5652347_Breast-Luminal-Epithelial-Z000000V2,GSM5652350_Breast-Basal-Epithelial-Z000000V6,GSM5652370_Colon-Right-Epithelial-Z000000V0,GSM5652198_Colon-Fibroblasts-Z0000042A,GSM5652189_Kidney-Tubular-Endothel-Z0000042R,GSM5652354_Lung-Alveolar-Epithelial-Z000000T1,GSM5652335_Lung-Bronchus-Epithelial-Z000000QD,GSM5652205_Skeletal-Muscle-Z00000427,GSM5652270_Ovary-Epithelial-Z000000QT,GSM5652338_Prostate-Epithelial-Z000000RV,GSM5652317_Blood-B-Z000000UB,GSM5652277_Blood-T-CD3-Z000000TV,GSM5652299_Blood-NK-Z000000TM,GSM5652302_Blood-Monocytes-Z000000TP,GSM5652313_Blood-Granulocytes-Z000000TZ"
PY="${MELODY_PY:-python}"
$PY -u scripts/33_melody_scoring.py --format ours \
  --input-csv data/external/egtex_multitissue/scoring/union_scoring_input_heldout.csv \
  --tracks "$TRACKS" --margin 1 \
  --output results/journal/melody/union_pair_scores_15track.csv
