#!/bin/bash
#SBATCH --job-name=fold
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=47:00:00
#SBATCH --output=logs/folds/%A_%a.out
#SBATCH --error=logs/folds/%A_%a.err
# Slurm array job: repeated chromosome-blocked splits, one task per fold, with fold 0 reproducing the published split.

set -eo pipefail

PY="${SILENTMETHYL_PY:-python}"

if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "STOP: CUDA is not available on $(hostname). Nothing was run." >&2
  echo "Requeue with:  sbatch --exclude=$(hostname) <this script>" >&2
  exit 4
fi

FOLD=${SLURM_ARRAY_TASK_ID}
SEED=42
: "${SPLIT_ROOT:?set SPLIT_ROOT, e.g. --export=ALL,SPLIT_ROOT=data/datafiles_breast_epithelium/splits}"
STAGES=${STAGES:-sequence}

SPLIT=${SPLIT_ROOT}/fold${FOLD}
TRAIN=${SPLIT}/train.csv
VAL=${SPLIT}/val.csv
OUT=checkpoints_folds/fold${FOLD}

if [ ! -f "${TRAIN}" ] || [ ! -f "${VAL}" ]; then
  echo "STOP: ${TRAIN} or ${VAL} missing -- run scripts/17_chromosome_splits.py first" >&2
  exit 1
fi
has_stage() { case " ${STAGES} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
for stage in ${STAGES}; do
  if [ -s "${OUT}/${stage}_seed${SEED}/best_weights.pth" ]; then
    echo "STOP: ${OUT}/${stage}_seed${SEED}/best_weights.pth exists; refusing to overwrite." >&2
    exit 2
  fi
done

mkdir -p "${OUT}"
echo "=== fold ${FOLD}  seed ${SEED}  stages ${STAGES}  $(date -u +%FT%TZ) ==="
echo "train ${TRAIN}"
echo "val   ${VAL}"

SNAP=${OUT}/split_checksums.txt
md5sum "${TRAIN}" "${VAL}" > "${SNAP}"
cp "${SPLIT_ROOT}/splits_summary.json" "${OUT}/splits_summary_at_launch.json"
cat "${SNAP}"

if has_stage sequence; then
echo "--- sequence tower ---"
$PY -u scripts/10_train_sequence.py \
    --train_path "${TRAIN}" --val_path "${VAL}" \
    --save_dir "${OUT}/sequence_seed${SEED}" --seed ${SEED}
fi

if has_stage epi; then
echo "--- epigenomic tower ---"
$PY -u scripts/11_train_epi.py \
    --train_path "${TRAIN}" --val_path "${VAL}" \
    --save_dir "${OUT}/epi_seed${SEED}" --seed ${SEED}
fi

if has_stage fusion; then
echo "--- gated fusion ---"
for w in "${OUT}/sequence_seed${SEED}/best_weights.pth" \
         "${OUT}/epi_seed${SEED}/best_weights.pth"; do
  [ -s "$w" ] || { echo "STOP: missing tower weights ${w}" >&2; exit 1; }
done
$PY -u scripts/12_train_fusion.py \
    --train_path "${TRAIN}" --val_path "${VAL}" \
    --sequence_weights "${OUT}/sequence_seed${SEED}/best_weights.pth" \
    --epi_weights "${OUT}/epi_seed${SEED}/best_weights.pth" \
    --save_dir "${OUT}/fusion_seed${SEED}" --seed ${SEED}
fi

echo "--- provenance check ---"
if md5sum -c "${SNAP}" --status; then
  echo "split files unchanged for the whole run"
else
  echo "WARNING: the split CSVs CHANGED while this job was running." >&2
  echo "This fold's checkpoints do not correspond to the recorded fold" >&2
  echo "definition and must not be used. Regenerate and retrain." >&2
  md5sum -c "${SNAP}" >&2 || true
  exit 3
fi

echo "=== fold ${FOLD} done $(date -u +%FT%TZ) ==="
