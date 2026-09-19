#!/usr/bin/env bash
# Copy the generated figures into the repository root under the names the manuscript expects.

set -euo pipefail
cd "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
DEST="${DEST:-.}"
mkdir -p "$DEST"

MAP=(
  "candidate_response_by_context.png|results/journal/ablation_breast_epithelium/manuscript_figures/candidate_response_by_context.png"
  "fusion_gain_by_epigenomic_context.png|results/journal/ablation_breast_epithelium/manuscript_figures/fusion_gain_by_epigenomic_context.png"
  "model_incremental_performance.png|results/journal/ablation_breast_epithelium/manuscript_figures/model_incremental_performance.png"
  "top_candidate_matched_background.png|results/journal/ablation_breast_epithelium/manuscript_figures/top_candidate_matched_background.png"
  "stk11_nonsynonymous.png|results/journal/ablation_breast_epithelium/manuscript_figures/stk11_variants_vs_nonsynonymous_screen.png"
  "motif_disruption.png|results/journal/ablation_breast_epithelium/motif_disruption/plots/motif_disruption.png"
  "discrimination_vs_distance.png|results/journal/ablation_breast_epithelium/genoa_variant_evaluation/plots/discrimination_vs_distance.png"
  "significance_gradient.png|results/journal/ablation_breast_epithelium/genoa_variant_evaluation/plots/significance_gradient.png"
  "uncertainty_scale_stability.pdf|results/journal/ablation_breast_epithelium/rc_uncertainty_figure/uncertainty_scale_stability.pdf"
)

missing=0
for row in "${MAP[@]}"; do
  dest="${row%%|*}"
  src="${row#*|}"
  if [[ -s "$src" ]]; then
    cp -f "$src" "$DEST/$dest"
    printf '  ok      %-40s <- %s\n' "$dest" "$src"
  else
    printf '  MISSING %-40s <- %s\n' "$dest" "$src"
    missing=$((missing + 1))
  fi
done

if [[ ! -s "$DEST/workflow.pdf" && ! -s workflow.pdf ]]; then
  printf '  MISSING %-40s <- hand-drawn schematic, no generating script\n' workflow.pdf
  missing=$((missing + 1))
fi

if (( missing > 0 )); then
  echo
  echo "[!] ${missing} figure(s) missing; main_revised.tex will draw placeholders for them."
  echo "    Run the analysis that produces each one, then rerun this script."
  exit 1
fi

echo
echo "[ok] all main-text figures in place; build with: pdflatex main_revised.tex"
