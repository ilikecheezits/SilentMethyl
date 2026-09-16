#!/usr/bin/env bash
set -uo pipefail
# Probe split assignments and sequences are identical in every context build;
# a fresh clone has data/datafiles_breast_epithelium.
SPLIT_DIR="${SPLIT_DIR:?set SPLIT_DIR, e.g. SPLIT_DIR=data/datafiles_breast_epithelium}"
SL=data/external/egtex_multitissue/within600
BR=data/external/egtex_breast/egtex_breast_within600.tsv.gz

run () {  # run <tissue> <slice>
  local T="$1" slice="$2"
  local out="data/external/egtex_multitissue/scoring/$T"
  [ -s "$slice" ] || { echo "SKIP $T (no slice)"; return; }
  [ -s "$out/egtex_scoring_summary.json" ] && { echo "SKIP $T (done)"; return; }
  echo "=== $T ==="; mkdir -p "$out"
  python -u data/harmonize_egtex_mqtl.py \
      --prefiltered "$slice" --split-dir "$SPLIT_DIR" --output-dir "$out" 2>&1 | tee "logs/harmonize_${T}.log"
}

run BreastMammaryTissue "$BR"
for T in WholeBlood Lung ColonTransverse Ovary Prostate KidneyCortex Testis MuscleSkeletal; do
  run "$T" "$SL/${T}_within600.tsv.gz"
done

echo "=== per-tissue cohort sizes ==="
for d in data/external/egtex_multitissue/scoring/*/; do
  echo "$(basename $d): $(tail -n +2 $d/egtex_scoring_input_heldout.csv 2>/dev/null | wc -l) heldout pairs"
done
