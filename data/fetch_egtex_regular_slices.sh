#!/usr/bin/env bash
set -uo pipefail
BASE=https://storage.googleapis.com/egtex/methylation/epic-arrays/mQTLs
OUT=data/external/egtex_multitissue/within600
mkdir -p "$OUT"

AWK=$(command -v mawk || command -v gawk || command -v awk)
if command -v pigz >/dev/null; then UNZ="pigz -dc -p 4"; else UNZ="gzip -dc"; fi
echo "awk=$AWK  unzip=$UNZ"

# byte-identical to AWK_PROGRAM in data/harmonize_egtex_mqtl.py
PROG='
BEGIN { OFS = "\t" }
{
    n++
    if (n % 200000000 == 0) {
        printf("  scanned %.2fB lines, kept %d\n", n / 1000000000, k) > "/dev/stderr"
    }
    if ($3 <= W && $3 >= LO && NF >= 9 && $3 ~ /^-?[0-9]+$/) {
        k++
        print $1, $2, $3, $4, $5, $6, $7, $8, $9
    }
}
END { printf("scanned %d lines, kept %d rows within +/-%d bp\n", n, k, W) > "/dev/stderr" }
'

for T in WholeBlood Lung ColonTransverse Ovary Prostate KidneyCortex Testis MuscleSkeletal; do
  dst="$OUT/${T}_within600.tsv.gz"
  [ -s "$dst" ] && { echo "[$(date +%H:%M:%S)] skip $T"; continue; }
  echo "[$(date +%H:%M:%S)] $T start"
  curl -fsS --retry 5 --retry-delay 10 "$BASE/${T}.mQTLs.regular.txt.gz" \
    | $UNZ \
    | LC_ALL=C $AWK -v W=600 -v LO=-600 "$PROG" \
    | gzip -1 -c > "${dst}.partial"
  st=("${PIPESTATUS[@]}")
  if [ "${st[0]}" -eq 0 ] && [ "${st[1]}" -eq 0 ]; then
    mv "${dst}.partial" "$dst"
    echo "[$(date +%H:%M:%S)] $T ok rows=$(zcat "$dst" | wc -l) size=$(du -h "$dst" | cut -f1)"
  else
    echo "[$(date +%H:%M:%S)] $T FAILED curl=${st[0]} unzip=${st[1]}"
    rm -f "${dst}.partial"
  fi
done
echo "[$(date +%H:%M:%S)] all done"
