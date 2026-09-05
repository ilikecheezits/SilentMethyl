#!/usr/bin/env bash
for T in ColonTransverse Ovary Prostate WholeBlood KidneyCortex \
         BreastMammaryTissue MuscleSkeletal Testis; do
  D=results/journal/egtex_multitissue_scoring/by_tissue/$T
  echo "=== $T $(date +%H:%M:%S)"
  python -u scripts/31_transfer_discrimination.py \
    --reference "$D::fusion::42,43,44" \
    --compare   "$D::sequence::42,43,44" \
    --significance 5e-8 --n-boot 500 \
    --output-dir results/journal/transfer_discrimination/$T \
    > logs/transfer_$T.log 2>&1 || echo "FAILED $T"
done
echo "all done $(date +%H:%M:%S)"
