# Shared guards for the R6 breast-epithelium chain.
ROOT=/ocean/projects/med250012p/szhang37/SilentMethyl
ABL=results/journal/ablation_breast_epithelium
CAND=$ABL/candidates
COHORT=data/datafiles_breast_epithelium/testing_data_test_only.csv
FUSION_W='checkpoints_ablation/breast_epithelium/seed{seed}/fusion/best_weights.pth'
# The sequence arm has no context tower and was never retrained for the swap,
# so it legitimately stays on the published checkpoints. There is no
# checkpoints_ablation/breast_epithelium/*/sequence and there should not be.
SEQ_W='checkpoints_journal/seed{seed}/sequence/best_weights.pth'
PY=/jet/home/szhang37/.conda/envs/silentmethyl/bin/python

stamp_start() { date '+%Y-%m-%d %H:%M:%S' > "$1"; echo "[*] clobber stamp: $(cat "$1")"; }

# Anything this chain writes under results/journal/ but OUTSIDE the ablation
# subtree is a bug: 22 and 91 default to the published manuscript-figure paths
# and every one of them is overridden below.
check_no_clobber() {
  local stamp_file="$1" ; local touched
  touched=$(find results/journal -type f -newermt "$(cat "$stamp_file")" \
              -not -path "results/journal/ablation_breast_epithelium/*" 2>/dev/null || true)
  if [[ -n "$touched" ]]; then
    echo "[!!] PUBLISHED-PATH CLOBBER -- these live outside the ablation subtree:" >&2
    echo "$touched" >&2
    return 1
  fi
  echo "[*] clobber check clean: nothing outside $ABL was written"
}
