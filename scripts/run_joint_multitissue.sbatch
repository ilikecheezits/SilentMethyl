#!/bin/bash
# Joint multi-tissue training: four tissues, disjoint probes, shared architecture.
#
# What this is
# ------------
# The published model is breast-only. This trains the same architecture on four
# tissues at once -- breast epithelium, kidney cortex, lung, transverse colon --
# with one context vector per tissue and disjoint training probes, under the
# same ~345k row budget so a joint epoch costs what a single-tissue epoch costs.
#
# Two configurations:
#   all4                       all four tissues; the headline model, 3 seeds
#   holdout_BreastEpithelium   trained on kidney/lung/colon, TESTED on breast,
#                              1 seed. Breast is the held-out tissue because it
#                              is the ONLY tissue with a single-tissue baseline
#                              (3 seeds + 4 folds, both contexts), so it is the
#                              only tissue where a transfer number sits between
#                              a measured floor and a measured ceiling. A lung
#                              held-out number would have nothing to compare to.
#
# Why TWO stages, submitted separately
# ------------------------------------
# Unlike the context ablation, NO published sequence tower matches these splits
# -- the training probes are new -- so the sequence tower must be trained too.
# That is ~32 h on top of ~14 h for epi + fusion, and Bridges-2 caps GPU jobs at
# 48 h. A single job that hits the wall at 47:59 loses everything and burns ~48
# SU, which the remaining allocation cannot absorb. So:
#
#   STAGE=seq    trains the sequence tower          ~32 h
#   STAGE=fuse   trains epi + gate, then tests      ~14 h
#
# The second is chained with --dependency=afterok, so it starts only if the
# first succeeded, and a failure costs one stage rather than both.
#
# How to submit
# -------------
#   mkdir -p logs/joint
#
#   # headline model, seed 42
#   SEQ=$(sbatch --parsable --time=40:00:00 --job-name=jt-all4-s42-seq \
#         --export=ALL,CONFIG=all4,SEED=42,STAGE=seq \
#         scripts/run_joint_multitissue.sbatch)
#   sbatch --dependency=afterok:$SEQ --time=24:00:00 --job-name=jt-all4-s42-fuse \
#         --export=ALL,CONFIG=all4,SEED=42,STAGE=fuse \
#         scripts/run_joint_multitissue.sbatch
#
#   # held-out-breast transfer run, seed 42
#   SEQ=$(sbatch --parsable --time=40:00:00 --job-name=jt-ho-seq \
#         --export=ALL,CONFIG=holdout_BreastEpithelium,SEED=42,STAGE=seq \
#         scripts/run_joint_multitissue.sbatch)
#   sbatch --dependency=afterok:$SEQ --time=24:00:00 --job-name=jt-ho-fuse \
#         --export=ALL,CONFIG=holdout_BreastEpithelium,SEED=42,STAGE=fuse \
#         scripts/run_joint_multitissue.sbatch
#
# --time is overridden on the command line rather than set here, because the two
# stages need very different walls and one file serves both.
#
# --export=ALL,VAR=x, never plain VAR=x: without ALL, Slurm drops the rest of
# the environment and the conda python is not found.
#
# Budget note
# -----------
# At ~1 SU per GPU-hour, all4 x 3 seeds is ~138 SU and the held-out run is ~40.
# Submit all4 seed 42 and the held-out run FIRST, let both land, and only then
# commit to seeds 43 and 44. If the joint pipeline is wrong you find out after
# ~86 SU instead of ~178.
#SBATCH --job-name=joint
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=40:00:00
#SBATCH --output=logs/joint/%x_%j.out
#SBATCH --error=logs/joint/%x_%j.err
set -eo pipefail

PY="${SILENTMETHYL_PY:-python}"
CONFIG=${CONFIG:-all4}
SEED=${SEED:-42}
STAGE=${STAGE:-seq}

case "${STAGE}" in
  seq|fuse) ;;
  *) echo "STOP: STAGE must be 'seq' or 'fuse', got '${STAGE}'" >&2; exit 2 ;;
esac

DATA="data/datafiles_joint/${CONFIG}"
OUT="checkpoints_joint/${CONFIG}/seed${SEED}"
RES="results/journal/joint/${CONFIG}/seed${SEED}"

mkdir -p logs/joint "${OUT}"

# Fail fast if the allocated node cannot do CUDA. Node v005 has returned error
# 803 before, after which torch falls back to CPU silently at ~49x slower --
# which on a 32 h sequence stage would burn the wall and the SUs for nothing.
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "STOP: CUDA unavailable on $(hostname). Nothing was run." >&2
  echo "Requeue with:  sbatch --exclude=$(hostname) $0" >&2
  exit 4
fi

for f in "${DATA}/train.csv" "${DATA}/val.csv" "${DATA}/test.csv" \
         "${DATA}/composition_manifest.json"; do
  [ -s "$f" ] || {
    echo "STOP: missing or empty ${f}" >&2
    echo "Compose it first:" >&2
    echo "  python -u data/compose_multitissue_splits.py --out-dir ${DATA} \\" >&2
    echo "      --seed 42 $([ "${CONFIG}" = all4 ] || echo "--holdout ${CONFIG#holdout_}")" >&2
    exit 1
  }
done

# ---------------------------------------------------------------------------
# Composition guard. The directory name says what this run is supposed to be;
# the manifest says what it actually is. A held-out run whose training set still
# contains the held-out tissue would train and test happily and report a
# transfer number that is not a transfer number at all.
# ---------------------------------------------------------------------------
"$PY" - "${DATA}" "${CONFIG}" <<'PYEOF'
import json, pathlib, sys
import pandas as pd

data, config = pathlib.Path(sys.argv[1]), sys.argv[2]
man = json.loads((data / "composition_manifest.json").read_text())
expected_holdout = config[len("holdout_"):] if config.startswith("holdout_") else None

if man.get("holdout_tissue") != expected_holdout:
    sys.exit(f"STOP: {data} was composed with holdout_tissue="
             f"{man.get('holdout_tissue')!r} but the config name says "
             f"{expected_holdout!r}. One of them is wrong.")

train_tissues = set(pd.read_csv(data / "train.csv", usecols=["Tissue"]).Tissue)
test_tissues = set(pd.read_csv(data / "test.csv", usecols=["Tissue"]).Tissue)

if expected_holdout:
    if expected_holdout in train_tissues:
        sys.exit(f"STOP: {expected_holdout} appears in the training set of a "
                 "run that is supposed to hold it out. The result would not be "
                 "a transfer measurement.")
    if test_tissues != {expected_holdout}:
        sys.exit(f"STOP: test set is {sorted(test_tissues)}, expected only "
                 f"{expected_holdout}.")

print(f"  composition   train={sorted(train_tissues)}")
print(f"                test={sorted(test_tissues)}")
print(f"                assignment seed {man['assignment_seed']}, "
      f"budget {man['row_budget']}, "
      f"{man['rows']['train']} train rows")
PYEOF

echo "=== joint ${CONFIG}  seed ${SEED}  stage ${STAGE}  $(date -u +%FT%TZ) ==="
echo "host    $(hostname)"
echo "data    ${DATA}"
echo "out     ${OUT}"

# ---------------------------------------------------------------------------
if [ "${STAGE}" = "seq" ]; then
  echo "--- sequence tower (no published tower matches these splits) ---"
  md5sum "${DATA}/train.csv" "${DATA}/val.csv" | tee "${OUT}/input_checksums.txt"

  $PY -u scripts/10_train_sequence.py \
      --train_path "${DATA}/train.csv" --val_path "${DATA}/val.csv" \
      --save_dir "${OUT}/sequence" --seed ${SEED}

  [ -s "${OUT}/sequence/best_weights.pth" ] || {
    echo "STOP: sequence stage produced no weights" >&2; exit 1; }
  echo "=== sequence done $(date -u +%FT%TZ) ==="
  echo "Next: sbatch --time=24:00:00 --export=ALL,CONFIG=${CONFIG},SEED=${SEED},STAGE=fuse $0"
  exit 0
fi

# ---------------------------------------------------------------------------
SEQ_W="${OUT}/sequence/best_weights.pth"
[ -s "${SEQ_W}" ] || {
  echo "STOP: ${SEQ_W} is missing. Run STAGE=seq first." >&2
  echo "If that job is still queued, chain this one with --dependency=afterok." >&2
  exit 1; }

# The sequence stage checksummed its inputs. If they changed since, the tower
# was trained on different rows than this stage is about to fuse and test on.
if [ -s "${OUT}/input_checksums.txt" ]; then
  if ! md5sum -c --status "${OUT}/input_checksums.txt"; then
    echo "STOP: train/val no longer match what the sequence tower was trained on." >&2
    echo "Recompose or retrain; do not fuse across a data change." >&2
    exit 1
  fi
  echo "  inputs unchanged since the sequence stage"
fi

echo "--- 1/2 epigenomic tower ---"
$PY -u scripts/11_train_epi.py \
    --train_path "${DATA}/train.csv" --val_path "${DATA}/val.csv" \
    --save_dir "${OUT}/epi" --seed ${SEED}

EPI_W="${OUT}/epi/best_weights.pth"
[ -s "${EPI_W}" ] || { echo "STOP: epi tower produced no weights" >&2; exit 1; }

echo "--- 2/2 gated fusion ---"
# 12_train_fusion.py takes the .pth FILE, not the save directory -- it sha256s
# the path. Passing a directory cost three folds 32 h each once.
$PY -u scripts/12_train_fusion.py \
    --train_path "${DATA}/train.csv" --val_path "${DATA}/val.csv" \
    --sequence_weights "${SEQ_W}" \
    --epi_weights "${EPI_W}" \
    --save_dir "${OUT}/fusion" --seed ${SEED}

echo "--- test: all three arms ---"
# All three are tested, not just fusion. The single-tissue paper reports
# sequence-only, context-only and fusion side by side, and the joint model is
# only comparable to it if the same three arms exist. The two extra tests cost
# seconds on ~100k loci.
for ARM in fusion epi sequence; do
  case "${ARM}" in
    fusion)   W="${OUT}/fusion/best_weights.pth" ;;
    epi)      W="${EPI_W}" ;;
    sequence) W="${SEQ_W}" ;;
  esac
  echo "  -> ${ARM}"
  $PY -u scripts/13_test_model.py \
      --model "${ARM}" --seed ${SEED} \
      --test_path "${DATA}/test.csv" \
      --weights_path "${W}" \
      --output_dir "${RES}/${ARM}"
done

# ---------------------------------------------------------------------------
# Per-tissue metrics. A single pooled number over four tissues hides the case
# that matters most -- one tissue carrying the average while another is near
# chance. probeID in the composed test set is suffixed "__<Tissue>", which is
# what makes this recoverable from predictions.csv alone.
# ---------------------------------------------------------------------------
echo
echo "=== per-tissue test metrics ==="
$PY - "${RES}" <<'PYEOF'
import json, pathlib, sys
import numpy as np, pandas as pd

res = pathlib.Path(sys.argv[1])
for arm in ("fusion", "epi", "sequence"):
    pred = res / arm / "predictions.csv"
    if not pred.is_file():
        continue
    df = pd.read_csv(pred)
    if "probeID" not in df.columns:
        print(f"  {arm}: predictions.csv has no probeID column; skipped")
        continue
    ids = df["probeID"].astype(str)
    df["Tissue"] = np.where(ids.str.contains("__"),
                            ids.str.rsplit("__", n=1).str[-1], "ALL")

    beta_cols = [c for c in df.columns if "beta" in c.lower()]
    pred_col = next((c for c in beta_cols if "pred" in c.lower()), None)
    true_col = next((c for c in beta_cols
                     if "pred" not in c.lower() and "true" in c.lower()
                     or c == "Median_Beta"), None)
    if pred_col is None or true_col is None:
        print(f"  {arm}: could not identify beta columns among {beta_cols}")
        continue

    # Which inference mode this MAE reflects is determined by pred_col above.
    # predictions.csv carries pred_beta_fwd / pred_beta_rc / pred_beta_rc_avg;
    # the first "pred" match is the FORWARD-strand column, so these per-tissue
    # values are forward-only and are NOT the same statistic as the
    # forward/RC-averaged "beta_mae" in metrics.json. Record the mode so the
    # two files cannot be read as contradicting each other.
    mode = {"pred_beta_fwd": "forward_only",
            "pred_beta_rc": "reverse_complement_only",
            "pred_beta_rc_avg": "forward_reverse_complement_average"}.get(
                pred_col, "unknown")

    print(f"\n  {arm}  [inference: {mode}; pred_col={pred_col}]")
    print(f"    {'tissue':<20}{'n':>9}{'beta MAE':>11}")
    rows = {}
    for tissue, g in df.groupby("Tissue"):
        mae = float(np.abs(g[pred_col] - g[true_col]).mean())
        rows[tissue] = {"n": int(len(g)), "beta_mae": mae,
                        "inference": mode,
                        "pred_column": pred_col,
                        "true_column": true_col}
        print(f"    {tissue:<20}{len(g):>9,}{mae:>11.4f}")
    (res / arm / "per_tissue_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n")
PYEOF

echo
echo "=== done $(date -u +%FT%TZ) ==="
echo "Compare against the single-tissue breast numbers in"
echo "  results/journal/ablation_breast_epithelium/seed${SEED}/fusion/metrics.json"
