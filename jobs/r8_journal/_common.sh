# Shared guards for the R8 journal job chain (gate decomposition, context ladder, gate plasticity, transfer failure, ASM validation).

ROOT="${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ABL=results/journal/ablation_breast_epithelium
JOINT=results/journal/joint
COHORT=data/datafiles_breast_epithelium/testing_data_test_only.csv
SPLITS='data/datafiles_breast_epithelium/{split}.csv'
FUSION_W='checkpoints_ablation/breast_epithelium/seed{seed}/fusion/best_weights.pth'
SEQ_W='checkpoints_journal/seed{seed}/sequence/best_weights.pth'
PY="${SILENTMETHYL_PY:-python}"

stamp_start() { date '+%Y-%m-%d %H:%M:%S' > "$1"; echo "[*] clobber stamp: $(cat "$1")"; }

check_no_clobber() {
  local stamp_file="$1" ; shift
  local args=() ; local p ; local allow=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --concurrent) allow+=("$2") ; shift 2 ;;
      *)            allow+=("$1") ; shift ;;
    esac
  done
  set -- "${allow[@]}"
  for p in "$@"; do args+=(-not -path "${p}/*"); done
  local touched
  touched=$(find results/journal -type f -newermt "$(cat "$stamp_file")" \
              "${args[@]}" 2>/dev/null || true)
  if [[ -n "$touched" ]]; then
    echo "[!!] PUBLISHED-PATH CLOBBER -- these live outside the declared subtrees:" >&2
    echo "$touched" >&2
    return 1
  fi
  echo "[*] clobber check clean: nothing outside $* was written"
}

require_cuda() {
  $PY - <<'PYCHK'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("[!!] CUDA unavailable on this node -- refusing to run on CPU")
print(f"[*] CUDA ok: {torch.cuda.get_device_name(0)}")
PYCHK
}
