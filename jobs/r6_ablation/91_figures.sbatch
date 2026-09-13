#!/bin/bash
#SBATCH --job-name=r6-91-fig
#SBATCH --partition=RM-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/r6_ablation/91_%j.out
#SBATCH --error=logs/r6_ablation/91_%j.err
set -euo pipefail
umask 027
source /ocean/projects/med250012p/szhang37/SilentMethyl/jobs/r6_ablation/_common.sh
cd "$ROOT"
STAMP=logs/r6_ablation/stamp_91_$SLURM_JOB_ID
stamp_start "$STAMP"
echo "[!] CROSS-CONTEXT INPUTS: --mqtl-path and --literature-variant-path are"
echo "[!] published MCF-10A products; no breast-epithelium rebuild exists. This"
echo "[!] figure set is a STAGING run, not manuscript-ready."
mkdir -p "$ABL/manuscript_figures" "$ABL/literature_variant_screen" "$ABL/biological_context/plots"
$PY scripts/91_build_manuscript_figures.py \
  --metrics-path "$ABL/paired_model_bootstrap/model_metrics_recomputed.csv" \
  --context-path "$ABL/biological_context/fusion_gain_by_context.csv" \
  --variant-context-path "$ABL/biological_context/variant_response_by_context.csv" \
  --variant-distance-path "$ABL/biological_context/variant_response_by_distance.csv" \
  --mqtl-path results/journal/egtex_mqtl_positive_control/mqtl_predictions_seed_aggregate.csv \
  --candidate-path "$CAND/candidate_matched_background_statistics.csv" \
  --candidate-seed-path "$CAND/candidate_seed_scores_long.csv" \
  --comparator-path "$CAND/top_candidate_matched_comparators_long.csv" \
  --case-study-path "$CAND/top_candidate_case_study.csv" \
  --literature-variant-path results/journal/literature_variant_screen/literature_variant_predictions_ranked.csv \
  --output-dir "$ABL/manuscript_figures" \
  --stk11-case-output "$ABL/literature_variant_screen/stk11_case_study_figure_values.csv" \
  --genomic-region-output "$ABL/biological_context/plots/fusion_gain_by_genomic_region.png" \
  --information-gains-output "$ABL/biological_context/plots/information_source_gains.png"
check_no_clobber "$STAMP"
