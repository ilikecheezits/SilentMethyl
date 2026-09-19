#!/bin/bash
#SBATCH --job-name=ctxabl
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=logs/ablation/ctxabl_%j.out
#SBATCH --error=logs/ablation/ctxabl_%j.err
# Slurm job: retrain the epi and fusion arms on primary breast epithelium in place of MCF-10A; this is the published model since 11 Sep 2026.

set -eo pipefail

PY="${SILENTMETHYL_PY:-python}"
SEED=${SEED:-42}
FOLD=${FOLD:-0}

if [ "${FOLD}" = "0" ]; then
  TAG="seed${SEED}"
  DATA="data/datafiles_breast_epithelium"
  PUBLISHED_SEQ="checkpoints_journal/seed${SEED}/sequence/best_weights.pth"
  BASELINE="results/journal/seed${SEED}/fusion/metrics.json"
  PUBLISHED_SPLIT=""
else
  if [ "${SEED}" != "42" ]; then
    echo "STOP: folds have published sequence towers for seed 42 only." >&2
    echo "Got SEED=${SEED} FOLD=${FOLD}. Drop the SEED override." >&2
    exit 2
  fi
  TAG="fold${FOLD}"
  DATA="data/datafiles_breast_epithelium/splits/fold${FOLD}"
  PUBLISHED_SEQ="checkpoints_folds/fold${FOLD}/sequence_seed42/best_weights.pth"
  BASELINE="results/journal/folds/fold${FOLD}/fusion/metrics.json"
  PUBLISHED_SPLIT=$(dirname "$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['train_path'])" "$(dirname "${PUBLISHED_SEQ}")/run_config.json")")
fi

OUT="checkpoints_ablation/breast_epithelium/${TAG}"
RES="results/journal/ablation_breast_epithelium/${TAG}"

if [ -s "${OUT}/fusion/best_weights.pth" ] || [ -s "${OUT}/epi/best_weights.pth" ]; then
  echo "STOP: ${OUT} already holds trained weights; refusing to overwrite." >&2
  exit 2
fi
mkdir -p logs/ablation "${OUT}"

if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "STOP: CUDA unavailable on $(hostname). Nothing was run." >&2
  echo "Requeue with:  sbatch --exclude=$(hostname) $0" >&2
  exit 4
fi

for f in "${DATA}/train.csv" "${DATA}/val.csv" "${DATA}/test.csv" \
         "${PUBLISHED_SEQ}"; do
  [ -s "$f" ] || { echo "STOP: missing or empty ${f}" >&2; exit 1; }
done

if [ "${FOLD}" = "0" ]; then
  "$PY" - "${DATA}/split_manifest.json" <<'PYEOF'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
if p.is_file():
    m = json.load(p.open())
    test = sorted(m.get("test_chromosomes", []))
    if test != ["chr8", "chr9"]:
        sys.exit(f"STOP: test split is {test}, not chr8+chr9. "
                 "The published numbers are not comparable.")
PYEOF
else
  "$PY" - "${DATA}" "${PUBLISHED_SPLIT}" <<'PYEOF'
import sys, pathlib, pandas as pd
new, old = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
if not old.is_dir():
    sys.exit(f"STOP: published split {old} not found; cannot verify tower reuse.")
for arm in ("train", "val", "test"):
    a = set(pd.read_csv(old / f"{arm}.csv", usecols=["probeID"]).probeID)
    b = set(pd.read_csv(new / f"{arm}.csv", usecols=["probeID"]).probeID)
    if a != b:
        sys.exit(f"STOP: {arm} probe set differs from the published fold "
                 f"({len(a)} published vs {len(b)} here, "
                 f"{len(a ^ b)} not in common). The published sequence tower "
                 "was trained on different data and must not be reused.")
    print(f"  {arm:5s} probe set matches published fold ({len(a)} probes)")
PYEOF
fi

echo "=== context ablation  ${TAG}  $(date -u +%FT%TZ) ==="
echo "host       $(hostname)"
echo "data       ${DATA}   (breast epithelium context)"
echo "seq tower  ${PUBLISHED_SEQ}   (published, reused unchanged)"
echo "baseline   ${BASELINE}"
md5sum "${DATA}/train.csv" "${DATA}/val.csv" "${PUBLISHED_SEQ}" \
    | tee "${OUT}/input_checksums.txt"

echo "--- 1/2 epigenomic tower on the new context ---"
$PY -u scripts/11_train_epi.py \
    --train_path "${DATA}/train.csv" --val_path "${DATA}/val.csv" \
    --save_dir "${OUT}/epi" --seed ${SEED}

echo "--- 2/2 gated fusion ---"
EPI_W="${OUT}/epi/best_weights.pth"
[ -s "${EPI_W}" ] || { echo "STOP: epi tower produced no weights" >&2; exit 1; }

$PY -u scripts/12_train_fusion.py \
    --train_path "${DATA}/train.csv" --val_path "${DATA}/val.csv" \
    --sequence_weights "${PUBLISHED_SEQ}" \
    --epi_weights "${EPI_W}" \
    --save_dir "${OUT}/fusion" --seed ${SEED}

echo "--- test: fusion ---"
$PY -u scripts/13_test_model.py \
    --model fusion --seed ${SEED} \
    --test_path "${DATA}/test.csv" \
    --weights_path "${OUT}/fusion/best_weights.pth" \
    --output_dir "${RES}/fusion"

echo "--- test: context-only ---"
$PY -u scripts/13_test_model.py \
    --model epi --seed ${SEED} \
    --test_path "${DATA}/test.csv" \
    --weights_path "${EPI_W}" \
    --output_dir "${RES}/epi"

echo
echo "=== ${TAG}: breast epithelium vs published MCF-10A ==="
$PY - "${RES}" "${BASELINE}" "${TAG}" <<'PYEOF'
import json, pathlib, sys
res, baseline, tag = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]


def read(p):
    if not p.is_file():
        return None
    m = json.load(p.open())
    return m.get("beta_mae"), m.get("auc")


rows = [("fusion", read(baseline), read(res / "fusion" / "metrics.json"))]
epi_baseline = baseline.parent.parent / "epi" / "metrics.json"
rows.append(("context", read(epi_baseline), read(res / "epi" / "metrics.json")))

print(f"  {'arm':9s} {'published':>21s} {'ablation':>21s} {'difference':>21s}")
for arm, old, new in rows:
    if old is None or new is None:
        print(f"  {arm:9s} (missing: "
              f"{'published' if old is None else 'ablation'} metrics.json)")
        continue
    ob, oa = old
    nb, na = new
    print(f"  {arm:9s} "
          f"{f'MAE {ob:.4f} AUC {oa:.4f}':>21s} "
          f"{f'MAE {nb:.4f} AUC {na:.4f}':>21s} "
          f"{f'MAE {nb-ob:+.4f} AUC {na-oa:+.4f}':>21s}")
print("  (negative MAE and positive AUC favour breast epithelium)")
PYEOF
echo "=== done $(date -u +%FT%TZ) ==="
