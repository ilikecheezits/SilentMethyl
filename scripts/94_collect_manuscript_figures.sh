#!/usr/bin/env bash
#
# Gather the generated figures into the repository root under the exact names
# main_revised.tex asks for, so `pdflatex main_revised.tex` builds the paper.
#
# main_revised.tex wraps every \includegraphics in \IfFileExists, so it compiles
# without this step -- it just draws grey placeholder boxes instead of figures.
# That silent degradation is why this script exists: run it and the missing ones
# are named on stdout instead of quietly becoming a box in the PDF.
#
#   ./scripts/94_collect_manuscript_figures.sh
#
# The copies land at the repository root and are gitignored (*.png, *.pdf), so
# nothing here is tracked. Rerun it after 91_build_manuscript_figures.py,
# 21_variant_evaluation.py, 50_motif_disruption.py or 51_rc_uncertainty.py.
#
# workflow.pdf is a hand-drawn schematic with no generating script. It is not
# produced by any analysis and must be supplied by hand; it is reported below
# as missing rather than silently skipped.

set -euo pipefail
cd "${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# Every source is the breast-epithelium build. Until 16 Sep 2026 this map copied
# the superseded MCF-10A figures from results/journal/<analysis>/.
DEST="${DEST:-.}"
mkdir -p "$DEST"

# tex name <- source path
MAP=(
  "candidate_response_by_context.png|results/journal/ablation_breast_epithelium/manuscript_figures/candidate_response_by_context.png"
  "fusion_gain_by_epigenomic_context.png|results/journal/ablation_breast_epithelium/manuscript_figures/fusion_gain_by_epigenomic_context.png"
  "model_incremental_performance.png|results/journal/ablation_breast_epithelium/manuscript_figures/model_incremental_performance.png"
  "top_candidate_matched_background.png|results/journal/ablation_breast_epithelium/manuscript_figures/top_candidate_matched_background.png"
  "stk11_nonsynonymous.png|results/journal/ablation_breast_epithelium/manuscript_figures/stk11_variants_vs_nonsynonymous_screen.png"
  "motif_disruption.png|results/journal/ablation_breast_epithelium/motif_disruption/plots/motif_disruption.png"
  # Figs. 3 and 4 are the GENOA arm specifically -- not eGTEx, not the
  # pooled baseline evaluation, both of which write identically named files.
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
