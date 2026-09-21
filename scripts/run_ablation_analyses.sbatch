#!/bin/bash
#SBATCH --job-name=ablana
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablation_analyses/%x_%A_%a.out
#SBATCH --error=logs/ablation_analyses/%x_%A_%a.err
# Slurm job: run the downstream analyses on the breast-epithelium context models.

set -eo pipefail

PY="${SILENTMETHYL_PY:-python}"
STAGE=${STAGE:-link}

ABL=results/journal/ablation_breast_epithelium
PUB=results/journal
SPLITS=data/datafiles_breast_epithelium/{split}.csv
WEIGHTS='checkpoints_journal/seed{seed}/{model}/best_weights.pth'

GENOA_IN=data/external/genoa_meqtl/scoring/genoa_scoring_input_heldout.csv
EGTEX_IN=data/external/egtex_breast/scoring/egtex_scoring_input_heldout.csv
GENOA_OUT="${ABL}/genoa_variant_scoring"
EGTEX_OUT="${ABL}/egtex_variant_scoring"

mkdir -p logs/ablation_analyses

case "${STAGE}" in
  link|score|ctxperm|analyse) ;;
  *) echo "STOP: STAGE must be link|score|ctxperm|analyse, got '${STAGE}'" >&2
     exit 2 ;;
esac

if [ "${STAGE}" = "score" ] || [ "${STAGE}" = "ctxperm" ]; then
  if ! "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "STOP: CUDA unavailable on $(hostname). Nothing was run." >&2
    echo "Requeue with:  sbatch --exclude=$(hostname) $0" >&2
    exit 4
  fi
fi

echo "=== ablation analyses  stage ${STAGE}  $(date -u +%FT%TZ) ==="
echo "host  $(hostname)"

if [ "${STAGE}" = "link" ]; then
echo "--- verifying the sequence tower is unchanged ---"
"$PY" - <<'PYEOF'
import hashlib, pathlib, sys

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

pub_root = pathlib.Path("checkpoints_journal")
bad = []
for seed in (42, 43, 44):
    pub = pub_root / f"seed{seed}" / "sequence" / "best_weights.pth"
    if not pub.is_file():
        bad.append(f"missing {pub}")
        continue
    cfg = pub_root / f"seed{seed}" / "fusion" / "run_config.json"
    if not cfg.is_file():
        bad.append(f"missing {cfg} -- did the fusion stage finish?")
        continue
    import json
    rec = json.load(cfg.open())
    used = rec.get("sequence_weights_path") or rec.get("config", {}).get("sequence_weights_path")
    used_sha = rec.get("sequence_weights_sha256") or rec.get("config", {}).get("sequence_weights_sha256")
    if used and pathlib.Path(used).resolve() != pub.resolve():
        bad.append(f"seed{seed}: fusion was built on {used}, not {pub}")
    print(f"  seed{seed}  sequence tower md5 {md5(pub)[:12]}  "
          f"sha256-at-fusion {str(used_sha)[:12]}")
if bad:
    sys.exit("STOP: cannot reuse the sequence arm:\n  " + "\n  ".join(bad))
print("  sequence tower identical to the published one -- reuse is sound")
PYEOF

echo "--- linking the unchanged sequence arm into the ablation tree ---"
linked=0
for COHORT in genoa egtex; do
  for SEED in 42 43 44; do
    SRC="${PUB}/${COHORT}_variant_scoring/heldout/sequence/seed${SEED}"
    DST="${ABL}/${COHORT}_variant_scoring/heldout/sequence/seed${SEED}"
    if [ -s "${SRC}/pair_scores.csv" ]; then
      mkdir -p "${DST}"
      ln -sfn "$(readlink -f "${SRC}/pair_scores.csv")" "${DST}/pair_scores.csv"
      linked=$((linked + 1))
    else
      echo "  !! ${SRC}/pair_scores.csv absent -- fusion-vs-sequence comparisons"
      echo "     for ${COHORT} seed${SEED} will be skipped downstream"
    fi
  done
done
for SEED in 42 43 44; do
  SRC="${PUB}/seed${SEED}/sequence"
  DST="${ABL}/seed${SEED}/sequence"
  if [ -s "${SRC}/predictions.csv" ]; then
    mkdir -p "${DST}"
    for f in predictions.csv metrics.json; do
      [ -s "${SRC}/${f}" ] && ln -sfn "$(readlink -f "${SRC}/${f}")" "${DST}/${f}"
    done
    linked=$((linked + 1))
  fi
done
echo "  linked ${linked} sequence-arm artefact set(s)"
echo "=== link done $(date -u +%FT%TZ) ==="
exit 0
fi

if [ "${STAGE}" = "score" ]; then
COMBOS=("GENOA:42" "GENOA:43" "GENOA:44" "eGTEx:42" "eGTEx:43" "eGTEx:44")
IDX=${SLURM_ARRAY_TASK_ID:-0}
ENTRY="${COMBOS[$IDX]}"
COHORT="${ENTRY%%:*}"
SEED="${ENTRY##*:}"

if [ "${COHORT}" = "GENOA" ]; then
  IN="${GENOA_IN}"; OUT="${GENOA_OUT}"
else
  IN="${EGTEX_IN}"; OUT="${EGTEX_OUT}"
fi

MODELS=${MODELS:-fusion}
case "${MODELS}" in
  fusion|sequence) ;;
  *) echo "STOP: MODELS must be 'fusion' or 'sequence', got '${MODELS}'" >&2
     exit 2 ;;
esac

W="checkpoints_journal/seed${SEED}/${MODELS}/best_weights.pth"
for f in "${IN}" "${W}" "data/datafiles_breast_epithelium/test.csv"; do
  [ -s "$f" ] || { echo "STOP: missing or empty ${f}" >&2; exit 1; }
done

echo "cohort ${COHORT}  seed ${SEED}  model ${MODELS}"
echo "weights ${W}"
echo "splits  data/datafiles_breast_epithelium/{split}.csv"

$PY -u scripts/20_variant_scoring.py \
    --input-csv "${IN}" \
    --stratum heldout \
    --models "${MODELS}" \
    --seeds "${SEED}" \
    --weights-template "${WEIGHTS}" \
    --split-template "${SPLITS}" \
    --device cuda --batch-size 32 \
    --output-dir "${OUT}"

echo "=== score done $(date -u +%FT%TZ) ==="
exit 0
fi

if [ "${STAGE}" = "ctxperm" ]; then
$PY -u scripts/23_context_permutation.py \
    --input-csv "${EGTEX_IN}" \
    --stratum heldout \
    --seed 42 \
    --weights-template "${WEIGHTS}" \
    --split-template "${SPLITS}" \
    --device cuda --batch-size 32 \
    --output-dir "${ABL}/context_permutation"

echo "=== ctxperm done $(date -u +%FT%TZ) ==="
exit 0
fi

set +e
FAILED=()
run () {
  local name="$1"; shift
  echo
  echo "--------------------------------------------------------------"
  echo ">>> ${name}   $(date -u +%FT%TZ)"
  echo "--------------------------------------------------------------"
  "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! ${name} exited ${rc}"
    FAILED+=("${name} (exit ${rc})")
  fi
  return 0
}

echo "--- scoring inventory ---"
SCORED_OK=1
for COHORT in genoa egtex; do
  for ARM in fusion sequence; do
    D="${ABL}/${COHORT}_variant_scoring/heldout/${ARM}"
    n=$(ls -1 "${D}"/seed*/pair_scores.csv 2>/dev/null | wc -l)
    printf "  %-6s %-9s %s/3 seed score files\n" "${COHORT}" "${ARM}" "${n}"
    if [ "$n" -ne 3 ]; then
      SCORED_OK=0
      FAILED+=("incomplete scores: ${COHORT}/${ARM} (${n}/3)")
    fi
  done
done
if [ "${SCORED_OK}" -eq 0 ]; then
  echo
  echo "  !! Scoring is incomplete. Steps that read pair_scores.csv (21, 30,"
  echo "     40, 41, 50, 52) will fail and are listed at the end. Steps that"
  echo "     read predictions.csv (16, 51) are unaffected and still run."
  echo "     This stage continues deliberately -- partial results beat none."
fi

run "21 variant evaluation (GENOA)" \
  $PY -u scripts/21_variant_evaluation.py \
      --scores-dir "${GENOA_OUT}" --cohort GENOA \
      --significance 5e-8 --n-boot 2000 \
      --output-dir "${ABL}/genoa_variant_evaluation"

run "21 variant evaluation (eGTEx)" \
  $PY -u scripts/21_variant_evaluation.py \
      --scores-dir "${EGTEX_OUT}" --cohort eGTEx \
      --significance 1.483e-5 --n-boot 2000 \
      --output-dir "${ABL}/egtex_variant_evaluation"

run "30 transfer synthesis" \
  $PY -u scripts/30_transfer_synthesis.py \
      --cohort "GENOA:${GENOA_OUT}:5e-8" \
      --cohort "eGTEx:${EGTEX_OUT}:1.483e-5" \
      --n-boot 500 \
      --output-dir "${ABL}/variant_effect_synthesis"

run "40 tissue specificity (matched)" \
  $PY -u scripts/40_meqtl_tissue_specificity.py \
      --stage matched \
      --cohort "GENOA:${GENOA_OUT}:5e-8" \
      --cohort "eGTEx:${EGTEX_OUT}:1.483e-5" \
      --split-template "${SPLITS}" \
      --model fusion --seeds 42 43 44 --n-boot 1000 \
      --output-dir "${ABL}/tissue_shared_meqtls"

run "40 tissue specificity (chromatin)" \
  $PY -u scripts/40_meqtl_tissue_specificity.py \
      --stage chromatin \
      --cohort "GENOA:${GENOA_OUT}:5e-8" \
      --cohort "eGTEx:${EGTEX_OUT}:1.483e-5" \
      --split-template "${SPLITS}" \
      --model fusion --seeds 42 43 44 --n-boot 1000 \
      --output-dir "${ABL}/meqtl_class_chromatin"

for PUB_DIR in results/journal/tissue_shared_meqtls \
               results/journal/meqtl_class_chromatin; do
  if [ -d "${PUB_DIR}" ] && \
     [ -n "$(find "${PUB_DIR}" -newermt '-30 minutes' -type f 2>/dev/null)" ]; then
    echo "!!! ${PUB_DIR} was modified during this run -- a step wrote to a"
    echo "    PUBLISHED path. Restore it before trusting anything downstream."
    FAILED+=("published path ${PUB_DIR} was overwritten")
  fi
done

run "41 tissue specificity summary" \
  $PY -u scripts/41_tissue_specificity_summary.py \
      --results "${ABL}/tissue_shared_meqtls" \
      --out "${ABL}/tissue_shared_meqtls" \
      --label "SilentMethyl (breast-epithelium context)"

run "16 paired model bootstrap" \
  $PY -u scripts/16_paired_model_bootstrap.py \
      --seeds 42 43 44 --models epi sequence fusion \
      --prediction-template "${ABL}/seed{seed}/{model}/predictions.csv" \
      --output-dir "${ABL}/paired_model_bootstrap" \
      --bootstrap-replicates 5000

run "51 rc uncertainty (base)" \
  $PY -u scripts/51_rc_uncertainty.py --stage base \
      --results-root "${ABL}" --seeds 42 43 44 --models sequence fusion epi \
      --output-dir "${ABL}/rc_uncertainty"

run "51 rc uncertainty (conditional)" \
  $PY -u scripts/51_rc_uncertainty.py --stage conditional \
      --results-root "${ABL}" --seeds 42 43 44 --models sequence fusion epi \
      --strata 10 --output-dir "${ABL}/rc_uncertainty_conditional"

GWAS=data/external/gwas_catalog/gwas-catalog-download-associations-alt-full.tsv
if [ -s "${GWAS}" ]; then
  run "52 gwas enrichment (build)" \
    $PY -u scripts/52_gwas_enrichment.py --build --gwas "${GWAS}" \
        --scores-dir "${GENOA_OUT}" --model fusion --seeds 42 43 44 \
        --output-dir "${ABL}/gwas_regulatory_enrichment"
  run "52 gwas enrichment (analyse)" \
    $PY -u scripts/52_gwas_enrichment.py --analyse \
        --scores-dir "${GENOA_OUT}" --model fusion --seeds 42 43 44 \
        --output-dir "${ABL}/gwas_regulatory_enrichment"
else
  echo "  skipping 52: ${GWAS} not found -- set GWAS= to its real path"
  FAILED+=("52 gwas enrichment (input not found)")
fi

run "50 motif disruption" \
  $PY -u scripts/50_motif_disruption.py \
      --scores-dir "${GENOA_OUT}" --model fusion --seeds 42 43 44 \
      --split-template "${SPLITS}" \
      --n-boot 500 \
      --output-dir "${ABL}/motif_disruption"

echo
echo "=============================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "all analyses completed"
else
  echo "${#FAILED[@]} step(s) failed:"
  for f in "${FAILED[@]}"; do echo "  - $f"; done
  echo
  echo "The rest completed. Rerun a single step by hand rather than"
  echo "resubmitting the whole stage."
fi
echo "outputs under ${ABL}/"
echo "=== analyse done $(date -u +%FT%TZ) ==="
echo "=============================================================="

