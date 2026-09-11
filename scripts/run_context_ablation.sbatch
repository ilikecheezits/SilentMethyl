#!/bin/bash
# Context ablation: does primary breast epithelium beat MCF-10A?
#
# The question
# ------------
# The published model's context is MCF-10A, a clonal cell line, while the
# methylation targets are bulk TCGA solid-tissue-normal breast. That domain
# mismatch is already stated as a limitation. ENCODE also has primary breast
# epithelium, profiled more fully: seven fold-change-over-control tracks from
# one biosample through one pipeline, against MCF-10A's six Mint-ChIP tracks
# plus a locally converted snATAC coverage track.
#
# This job answers whether the published choice cost anything, on one seed,
# before committing ~90 GPU-hours to reprocessing three seeds and three folds.
#
# Why only two stages
# -------------------
# The sequence tower never sees context -- it reads DNA only -- so it does not
# need retraining and the published seed-42 tower is reused verbatim. That is
# 32 of the 46 hours saved, and it also means the sequence-only baseline in the
# paper is untouched and still comparable.
#
# What gets retrained: the epigenomic tower (which consumes the context) and
# the gate (which weighs the two towers). Roughly 15-20 h.
#
# Reading the result
# ------------------
# Compare against seed 42's PUBLISHED test numbers on the same chr8+chr9 split:
#
#     beta MAE   0.0971      AUC   0.9699
#
#   clearly better -> the context choice cost something; consider the full swap
#   a wash         -> keep MCF-10A, and cite this as evidence it did not matter,
#                     which closes the limitation with data rather than a caveat
#   worse          -> keep MCF-10A, now with a measured reason rather than an
#                     accident of availability
#
# Before submitting
# -----------------
#   mkdir -p logs/ablation && sbatch scripts/run_context_ablation.sbatch
#
# The mkdir is NOT optional. Slurm opens --output and --error before the script
# body runs, so a missing logs/ablation/ makes every task die at ~5 s with the
# batch step CANCELLED and no log to say why. The `mkdir -p logs/ablation` in
# the body below runs too late to help; it is there for reruns only.
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
SEED=42

DATA=data/datafiles_breast_epithelium
OUT=checkpoints_ablation/breast_epithelium
PUBLISHED_SEQ=checkpoints_journal/seed${SEED}/sequence/best_weights.pth

mkdir -p logs/ablation "${OUT}"

# Fail fast if the allocated node cannot do CUDA. Node v005 has returned error
# 803 before, after which torch falls back to CPU silently at ~49x slower.
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "STOP: CUDA unavailable on $(hostname). Nothing was run." >&2
  echo "Requeue with:  sbatch --exclude=$(hostname) $0" >&2
  exit 4
fi

# test.csv is checked here, not at the test step, because the test step is ~16 h
# downstream. A missing split should cost five seconds, not most of a day.
for f in "${DATA}/train.csv" "${DATA}/val.csv" "${DATA}/test.csv" \
         "${PUBLISHED_SEQ}"; do
  [ -s "$f" ] || { echo "STOP: missing or empty ${f}" >&2; exit 1; }
done

# The comparison against the published 0.0971 / 0.9699 is only meaningful if
# this build used the same held-out chromosomes. build_training_data.py defaults
# to val chr10+chr11 / test chr8+chr9, so this should always pass; it catches a
# build that was run with the chromosome flags overridden.
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

echo "=== context ablation  seed ${SEED}  $(date -u +%FT%TZ) ==="
echo "host       $(hostname)"
echo "data       ${DATA}   (breast epithelium context)"
echo "seq tower  ${PUBLISHED_SEQ}   (published, reused unchanged)"
md5sum "${DATA}/train.csv" "${DATA}/val.csv" "${PUBLISHED_SEQ}" \
    | tee "${OUT}/input_checksums.txt"

echo "--- 1/2 epigenomic tower on the new context ---"
$PY -u scripts/11_train_epi.py \
    --train_path "${DATA}/train.csv" --val_path "${DATA}/val.csv" \
    --save_dir "${OUT}/epi_seed${SEED}" --seed ${SEED}

echo "--- 2/2 gated fusion ---"
# 12_train_fusion.py takes the .pth FILE, not the save directory -- it sha256s
# the path. Passing a directory cost three folds 32 h each once.
EPI_W="${OUT}/epi_seed${SEED}/best_weights.pth"
[ -s "${EPI_W}" ] || { echo "STOP: epi tower produced no weights" >&2; exit 1; }

$PY -u scripts/12_train_fusion.py \
    --train_path "${DATA}/train.csv" --val_path "${DATA}/val.csv" \
    --sequence_weights "${PUBLISHED_SEQ}" \
    --epi_weights "${EPI_W}" \
    --save_dir "${OUT}/fusion_seed${SEED}" --seed ${SEED}

echo "--- test on the published split (chr8 + chr9) ---"
$PY -u scripts/13_test_model.py \
    --model fusion --seed ${SEED} \
    --test_path "${DATA}/test.csv" \
    --weights_path "${OUT}/fusion_seed${SEED}/best_weights.pth" \
    --output_dir "results/journal/ablation_breast_epithelium/fusion"

echo
echo "=== compare against the published seed 42 ==="
echo "  published   beta MAE 0.0971   AUC 0.9699"
$PY - <<'PYEOF'
import json, pathlib
p = pathlib.Path("results/journal/ablation_breast_epithelium/fusion/metrics.json")
if p.is_file():
    m = json.load(p.open())
    b, a = m.get("beta_mae"), m.get("auc")
    print(f"  ablation    beta MAE {b:.4f}   AUC {a:.4f}")
    print(f"  difference  beta MAE {b-0.0971:+.4f}   AUC {a-0.9699:+.4f}")
    print("  (negative beta MAE and positive AUC favour breast epithelium)")
PYEOF
echo "=== done $(date -u +%FT%TZ) ==="
