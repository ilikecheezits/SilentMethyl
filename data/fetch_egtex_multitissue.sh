#!/usr/bin/env bash
set -uo pipefail
BASE=https://storage.googleapis.com/egtex/methylation/epic-arrays/mQTLs
OUT=data/external/egtex_multitissue
mkdir -p "$OUT"

get () {  # get <filename>
  echo "[$(date +%H:%M:%S)] $1"
  curl -C - -fsS --retry 5 --retry-delay 10 -o "$OUT/$1" "$BASE/$1" \
    || echo "FAILED: $1"
}

# breast: conditional only (perm.fdr already in egtex_breast/)
get BreastMammaryTissue.mQTLs.conditional.txt.gz

# remaining eight, small conditional files first
for T in MuscleSkeletal Testis KidneyCortex WholeBlood Prostate \
         Ovary ColonTransverse Lung; do
  get "$T.mQTLs.conditional.txt.gz"
  get "$T.regular.perm.fdr.txt"
done
echo "[$(date +%H:%M:%S)] done"
