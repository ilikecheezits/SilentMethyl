#!/bin/bash
# Context ablation: primary breast epithelium in place of MCF-10A.
#
# The question
# ------------
# The published model's context is MCF-10A, a clonal cell line, while the
# methylation targets are bulk TCGA solid-tissue-normal breast. That domain
# mismatch is a stated limitation. ENCODE also has primary breast epithelium,
# profiled more fully: seven fold-change-over-control tracks from one biosample
# through one pipeline, against MCF-10A's six Mint-ChIP tracks plus a locally
# converted snATAC coverage track whose ATAC has blacklist artifact pileups.
#
# Seed 42 on the published split already answered it:
#
#     context-only   beta MAE 0.1396 -> 0.1022   AUC 0.9188 -> 0.9658
#     fusion         beta MAE 0.0971 -> 0.0885   AUC 0.9699 -> 0.9765
#
# The fusion gain is 4.3 SD below the published three-seed mean (0.0993+-0.0025),
# so it is not seed noise. This script now generalises that run to the other two
# seeds and the other three chromosome folds, which is the full single-tissue
# reprocessing the swap requires.
#
# Why only two stages
# -------------------
# The sequence tower never sees context -- it reads DNA only -- so it does not
# need retraining and the matching published tower is reused verbatim. That is
# ~32 of ~46 h saved per run, and it keeps the sequence-only baseline in the
# paper untouched and still comparable.
#
# Retrained here: the epigenomic tower (which consumes the context) and the gate
# (which weighs the two towers). Roughly 14 h.
#
# How to submit
# -------------
#   mkdir -p logs/ablation
#
#   # remaining seeds, published split
#   sbatch --job-name=ctx-s43 --export=ALL,SEED=43 scripts/run_context_ablation.sbatch
#   sbatch --job-name=ctx-s44 --export=ALL,SEED=44 scripts/run_context_ablation.sbatch
#
#   # chromosome folds, seed 42
#   sbatch --job-name=ctx-f1 --export=ALL,FOLD=1 scripts/run_context_ablation.sbatch
#   sbatch --job-name=ctx-f2 --export=ALL,FOLD=2 scripts/run_context_ablation.sbatch
#   sbatch --job-name=ctx-f3 --export=ALL,FOLD=3 scripts/run_context_ablation.sbatch
#
# The mkdir is NOT optional. Slurm opens --output and --error before the script
# body runs, so a missing logs/ablation/ makes every task die at ~5 s with the
# batch step CANCELLED and no log to say why. The mkdir in the body runs too
# late to help; it is there for reruns only.
#
# --export=ALL,VAR=x is required rather than plain VAR=x: without ALL, Slurm
# drops the rest of the environment and the conda python is not found.
#SBATCH --job-name=ctxabl
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --output=logs/ablation/ctxabl_%j.out
#SBATCH --error=logs/ablation/ctxabl_%j.err
set -eo pipefail

PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python
SEED=${SEED:-42}
FOLD=${FOLD:-0}

# ---------------------------------------------------------------------------
# Layout. FOLD=0 is the published chr8+chr9 split across seeds 42/43/44;
# FOLD=1..3 are the repeated chromosome-blocked splits at seed 42 only, because
# that is the only seed for which published fold sequence towers exist.
# ---------------------------------------------------------------------------
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
  PUBLISHED_SPLIT="data/datafiles/splits/fold${FOLD}"
fi

OUT="checkpoints_ablation/breast_epithelium/${TAG}"
RES="results/journal/ablation_breast_epithelium/${TAG}"

mkdir -p logs/ablation "${OUT}"

# Fail fast if the allocated node cannot do CUDA. Node v005 has returned error
# 803 before, after which torch falls back to CPU silently at ~49x slower.
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "STOP: CUDA unavailable on $(hostname). Nothing was run." >&2
  echo "Requeue with:  sbatch --exclude=$(hostname) $0" >&2
  exit 4
fi

# test.csv is checked here, not at the test step, because the test step is ~14 h
# downstream. A missing split should cost five seconds, not most of a day.
for f in "${DATA}/train.csv" "${DATA}/val.csv" "${DATA}/test.csv" \
         "${PUBLISHED_SEQ}"; do
  [ -s "$f" ] || { echo "STOP: missing or empty ${f}" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
# Split integrity. This is the guard that matters most: the whole saving in this
# script comes from reusing a published sequence tower, and that reuse is only
# valid if this split is the SAME split that tower was trained on. Get it wrong
# and the run completes normally while testing on probes the tower trained on.
# ---------------------------------------------------------------------------
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
# 12_train_fusion.py takes the .pth FILE, not the save directory -- it sha256s
# the path. Passing a directory cost three folds 32 h each once.
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

# The context arm is where the effect is largest (0.1396 -> 0.1022 on seed 42),
# and it costs ten seconds on 26k loci, so it is measured here rather than left
# as a manual follow-up that gets forgotten on four of the five runs.
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
