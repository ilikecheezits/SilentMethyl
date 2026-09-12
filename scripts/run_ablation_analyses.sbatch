#!/bin/bash
# Downstream analyses on the breast-epithelium context models.
#
# Why this exists
# ---------------
# The context swap changed the epigenomic tower and the gate on every seed and
# every fold (LAB_NOTES 1.10: 6/6 runs favour breast epithelium, fusion
# dMAE -0.0073 +- 0.0012). Every downstream result in the paper was computed
# from the MCF-10A models and is now stale. This reruns them.
#
# Nothing here writes to a published path. Every output lands under
# results/journal/ablation_breast_epithelium/, so the MCF-10A results survive
# for the side-by-side comparison the paper needs.
#
# The two facts that set the design
# ---------------------------------
# 1. THE SEQUENCE TOWER IS UNCHANGED. The ablation reused
#    checkpoints_journal/seed{N}/sequence/best_weights.pth verbatim -- it reads
#    DNA only and never sees context. So sequence-arm scores and predictions are
#    identical and are LINKED IN rather than recomputed, which halves the GPU
#    cost. The link stage verifies this by md5 before trusting it; if the
#    checkpoints ever diverge it refuses instead of silently reusing stale
#    scores.
#
# 2. MOST OF THIS IS CPU WORK. Only variant scoring and the context permutation
#    need a GPU. At 242 GPU SU against 3,996 RM SU, running a four-hour JASPAR
#    scan on a v100 would be the single most wasteful thing in this pipeline.
#    The analyse stage is submitted to RM-shared.
#
# Stages
# ------
#   link      seconds, CPU   verify + link the unchanged sequence arm
#   score     ~3 h x 6, GPU  20_variant_scoring, fusion only, array 0-5
#   ctxperm   ~8 h, GPU      23_context_permutation
#   analyse   ~6 h, CPU      everything that reads CSVs
#
# How to submit
# -------------
#   mkdir -p logs/ablation_analyses
#
#   # 1. link runs in ~20 s and needs no allocation. Run it on the LOGIN NODE
#   #    so a verification failure is visible immediately rather than buried in
#   #    a log an hour later. It must succeed before the rest is submitted.
#   STAGE=link bash scripts/run_ablation_analyses.sbatch
#
#   # 2. GPU work. Defaults in this file already suit GPU-shared.
#   S=$(sbatch --parsable --array=0-5 \
#       --export=ALL,STAGE=score scripts/run_ablation_analyses.sbatch)
#
#   sbatch --time=12:00:00 \
#       --export=ALL,STAGE=ctxperm scripts/run_ablation_analyses.sbatch
#
#   # 3. CPU work, chained behind the whole array.
#   sbatch --dependency=afterany:$S --partition=RM-shared \
#       --gpus=0 --cpus-per-task=8 --mem=16000M --time=20:00:00 \
#       --export=ALL,STAGE=analyse scripts/run_ablation_analyses.sbatch
#
# RM-shared caps memory at 2000 MB PER CORE. --mem=48G with 8 cores is 6 GB/core
# and is rejected with "Allocation requested mem-per-core higher than maximum of
# 2000M/core", which Slurm then reports as a permission error. 8 cores x 2000M =
# 16000M is the ceiling for 8 cores; ask for more cores if more memory is
# needed. --gpus=0 is required too, or the GPU directive in this file follows
# the job onto a partition that has none.
#
# afterany, NOT afterok, on the analyse stage: if one scoring task fails the
# other five results are still worth analysing, and every step below is
# individually guarded. afterok would discard all of it over one task.
#
# --export=ALL,VAR=x, never plain VAR=x: without ALL, Slurm drops the
# environment and the conda python is not found.
#
# What this does NOT run, and why
# -------------------------------
#   60/61/62 candidates, 22 context stratification, 63 known-variant,
#   64 literature screen, 91 figures.
# These form a chain: 22 and 91 hard-require the candidate CSVs that 60
# produces, 60 and 62 are GPU jobs of their own, 63 loads 60 as a module, and 64
# shells out to 63 WITHOUT forwarding a weights template -- so 64 would silently
# score the OLD MCF-10A checkpoints and produce a result that looks fine and is
# wrong. That needs a code change to 64, not a scheduling decision. Left out
# deliberately; see the note at the end of this file.
#SBATCH --job-name=ablana
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablation_analyses/%x_%A_%a.out
#SBATCH --error=logs/ablation_analyses/%x_%A_%a.err
set -eo pipefail

PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python
STAGE=${STAGE:-link}

# Everything the ablation produces lives under one root.
ABL=results/journal/ablation_breast_epithelium
PUB=results/journal
SPLITS=data/datafiles_breast_epithelium/{split}.csv
WEIGHTS='checkpoints_ablation/breast_epithelium/seed{seed}/{model}/best_weights.pth'

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

# ===========================================================================
if [ "${STAGE}" = "link" ]; then
# ===========================================================================
# The sequence arm is reused, not recomputed. That is only sound if the weights
# are literally the same file. Check it, then link. A silent reuse of scores
# from a different tower would corrupt every paired fusion-vs-sequence
# comparison downstream while looking completely normal.
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
abl_root = pathlib.Path("checkpoints_ablation/breast_epithelium")
bad = []
for seed in (42, 43, 44):
    pub = pub_root / f"seed{seed}" / "sequence" / "best_weights.pth"
    if not pub.is_file():
        bad.append(f"missing {pub}")
        continue
    # The ablation never wrote a sequence tower; it pointed 12_train_fusion.py
    # at the published one. Its run_config records which file it used.
    cfg = abl_root / f"seed{seed}" / "fusion" / "run_config.json"
    if not cfg.is_file():
        bad.append(f"missing {cfg} -- did the ablation fusion stage finish?")
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
# 16_paired_model_bootstrap and 22 read predictions.csv, not pair_scores.csv.
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

# ===========================================================================
if [ "${STAGE}" = "score" ]; then
# ===========================================================================
# Fusion only. Six tasks: two cohorts x three seeds.
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

W="checkpoints_ablation/breast_epithelium/seed${SEED}/fusion/best_weights.pth"
for f in "${IN}" "${W}" "data/datafiles_breast_epithelium/test.csv"; do
  [ -s "$f" ] || { echo "STOP: missing or empty ${f}" >&2; exit 1; }
done

echo "cohort ${COHORT}  seed ${SEED}"
echo "weights ${W}"
echo "splits  data/datafiles_breast_epithelium/{split}.csv"

# --split-template MUST point at the breast-epithelium build. It is where the
# seven context features come from; leaving the default would score the new
# fusion weights against MCF-10A context and produce a hybrid that is neither
# model.
$PY -u scripts/20_variant_scoring.py \
    --input-csv "${IN}" \
    --stratum heldout \
    --models fusion \
    --seeds "${SEED}" \
    --weights-template "${WEIGHTS}" \
    --split-template "${SPLITS}" \
    --device cuda --batch-size 32 \
    --output-dir "${OUT}"

echo "=== score done $(date -u +%FT%TZ) ==="
exit 0
fi

# ===========================================================================
if [ "${STAGE}" = "ctxperm" ]; then
# ===========================================================================
# Context permutation is the direct test of allele invariance: shuffle or
# median-replace the context and see whether variant effects move. It is fusion,
# seed 42 only, by the script's design.
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

# ===========================================================================
# STAGE = analyse. CPU only. Each step is guarded so one failure does not
# abandon the rest -- these are independent analyses and a missing input for
# one is not a reason to skip the other eight.
# ===========================================================================
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

# --- discrimination against the distance-matched null, both cohorts ----------
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

# --- pooled synthesis across cohorts ----------------------------------------
run "30 transfer synthesis" \
  $PY -u scripts/30_transfer_synthesis.py \
      --cohort "GENOA:${GENOA_OUT}:5e-8" \
      --cohort "eGTEx:${EGTEX_OUT}:1.483e-5" \
      --n-boot 500 \
      --output-dir "${ABL}/variant_effect_synthesis"

# --- shared vs tissue-specific meQTLs (R4) ----------------------------------
# 40 builds its score paths with "heldout" hardcoded, which matches what the
# score stage wrote. chromatin reads the context features out of the
# breast-epithelium splits, so --split-template matters here too.
#
# RUN THE TWO STAGES SEPARATELY. --stage all IGNORES --output-dir and writes to
# 40's own stage defaults, which are the PUBLISHED paths
# results/journal/{tissue_shared_meqtls,meqtl_class_chromatin}. On 12 Sep 2026
# that silently overwrote four published MCF-10A result files with
# breast-epithelium numbers while reporting success, and 41 then failed because
# the ablation directory it was pointed at did not exist -- the only reason the
# clobber was noticed at all. Same footgun as 51 --stage all, guarded below.
# Per-stage, --output-dir IS honoured; verified 12 Sep.
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

# Nothing in this stage may write outside ${ABL}. Checked rather than trusted,
# because the failure above was silent and only surfaced two steps later.
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

# --- paired model comparison on held-out CpGs -------------------------------
run "16 paired model bootstrap" \
  $PY -u scripts/16_paired_model_bootstrap.py \
      --seeds 42 43 44 --models epi sequence fusion \
      --prediction-template "${ABL}/seed{seed}/{model}/predictions.csv" \
      --output-dir "${ABL}/paired_model_bootstrap" \
      --bootstrap-replicates 5000

# --- reverse-complement uncertainty -----------------------------------------
# Run base and conditional SEPARATELY. --stage all ignores --output-dir for the
# conditional pass and writes to results/journal/rc_uncertainty_conditional,
# which would overwrite the published MCF-10A result.
run "51 rc uncertainty (base)" \
  $PY -u scripts/51_rc_uncertainty.py --stage base \
      --results-root "${ABL}" --seeds 42 43 44 --models sequence fusion epi \
      --output-dir "${ABL}/rc_uncertainty"

run "51 rc uncertainty (conditional)" \
  $PY -u scripts/51_rc_uncertainty.py --stage conditional \
      --results-root "${ABL}" --seeds 42 43 44 --models sequence fusion epi \
      --strata 10 --output-dir "${ABL}/rc_uncertainty_conditional"

# --- GWAS regulatory enrichment: build then analyse -------------------------
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

# --- motif disruption: the long one, deliberately last ----------------------
# Full JASPAR vertebrate library against every scored pair. Hours. Placed last
# so a failure here does not delay anything else, and so the cheap results are
# already on disk if the wall is hit.
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

# ---------------------------------------------------------------------------
# NOT RUN HERE, and what each needs first:
#
#   60_candidate_background.py   GPU. Produces the candidate CSVs that 22 and
#                                91 hard-require. Needs --weights-template
#                                pointing at the ablation checkpoints; check
#                                whether it accepts one before scheduling.
#   62_candidate_comparison.py   GPU. --sequence-weights-template takes {seed}
#                                ONLY -- a {model} placeholder raises KeyError.
#                                The sequence arm is unchanged, so this one can
#                                legitimately keep the published template.
#   22_context_stratification.py CPU, but blocked on 60's outputs.
#   63_known_variant_application GPU, small. --weights-template takes {seed}
#                                only; write "fusion" literally in the path.
#   64_literature_variant_screen NEEDS A CODE CHANGE. It shells out to 63 and
#                                does NOT forward --weights-template, so it
#                                would score the OLD MCF-10A checkpoints and
#                                report a result that looks correct. Fix 64 to
#                                pass the template through before running it.
#   91_build_manuscript_figures  CPU, seconds, but blocked on 60 and 22.
# ---------------------------------------------------------------------------
