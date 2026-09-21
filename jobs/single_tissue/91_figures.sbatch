#!/bin/bash
#SBATCH --job-name=sm-91-fig
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/single_tissue/91_%j.out
#SBATCH --error=logs/single_tissue/91_%j.err
# Slurm job: build the manuscript figure set (script 91) into a dedicated staging directory.

set -euo pipefail
umask 027
source "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"/jobs/single_tissue/_common.sh
cd "$ROOT"
STAMP=logs/single_tissue/stamp_91_$SLURM_JOB_ID
stamp_start "$STAMP"
MQTL="$ABL/egtex_mqtl_positive_control/mqtl_predictions_seed_aggregate.csv"
LIT="$ABL/literature_variant_screen/literature_variant_predictions_ranked.csv"
for f in "$MQTL" "$LIT"; do
  [[ -f "$f" ]] || { echo "[!!] missing new-context input: $f" >&2; exit 1; }
done
echo "[*] all inputs are breast-epithelium; this run is manuscript-ready,"
echo "[*] superseding the earlier staging figure set."
$PY scripts/91_build_manuscript_figures.py \
  --metrics-path "$ABL/paired_model_bootstrap/model_metrics_recomputed.csv" \
  --context-path "$ABL/biological_context/fusion_gain_by_context.csv" \
  --variant-context-path "$ABL/biological_context/variant_response_by_context.csv" \
  --variant-distance-path "$ABL/biological_context/variant_response_by_distance.csv" \
  --mqtl-path "$MQTL" \
  --candidate-path "$CAND/candidate_matched_background_statistics.csv" \
  --candidate-seed-path "$CAND/candidate_seed_scores_long.csv" \
  --comparator-path "$CAND/top_candidate_matched_comparators_long.csv" \
  --case-study-output "$CAND/top_candidate_case_study.csv" \
  --literature-variant-path "$LIT" \
  --output-dir "$ABL/manuscript_figures" \
  --stk11-case-output "$ABL/literature_variant_screen/stk11_case_study_figure_values.csv" \
  --genomic-region-output "$ABL/biological_context/plots/fusion_gain_by_genomic_region.png" \
  --information-gains-output "$ABL/biological_context/plots/information_source_gains.png"
check_no_clobber "$STAMP"
