#!/usr/bin/env bash
# Download the eGTEx conditional and permutation mQTL slices for the multi-tissue cohorts.

set -uo pipefail
BASE=https://storage.googleapis.com/egtex/methylation/epic-arrays/mQTLs
OUT=data/external/egtex_multitissue
mkdir -p "$OUT"

get () {
  echo "[$(date +%H:%M:%S)] $1"
  curl -C - -fsS --retry 5 --retry-delay 10 -o "$OUT/$1" "$BASE/$1" \
    || echo "FAILED: $1"
}

get BreastMammaryTissue.mQTLs.conditional.txt.gz
# data/egtex_significance_threshold.py reads this one; fetch it like the rest.
get BreastMammaryTissue.regular.perm.fdr.txt

for T in MuscleSkeletal Testis KidneyCortex WholeBlood Prostate \
         Ovary ColonTransverse Lung; do
  get "$T.mQTLs.conditional.txt.gz"
  get "$T.regular.perm.fdr.txt"
done
echo "[$(date +%H:%M:%S)] done"
