# Shared guards for the R6 breast-epithelium job chain.

ROOT="${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ABL=results/journal/ablation_breast_epithelium
CAND=$ABL/candidates
COHORT=data/datafiles_breast_epithelium/testing_data_test_only.csv
FUSION_W='checkpoints_ablation/breast_epithelium/seed{seed}/fusion/best_weights.pth'
SEQ_W='checkpoints_journal/seed{seed}/sequence/best_weights.pth'
PY="${SILENTMETHYL_PY:-python}"

stamp_start() { date '+%Y-%m-%d %H:%M:%S' > "$1"; echo "[*] clobber stamp: $(cat "$1")"; }

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
