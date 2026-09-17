# Shared guards for the R8 journal chain (gate decomposition, context dose-response,
# gate plasticity, transfer failure, ASM validation).
#
# Same contract as jobs/r6_ablation/_common.sh: every job stamps the clock before
# it runs and checks afterwards that nothing outside its declared output subtree
# was touched. Several of the underlying scripts default to the PUBLISHED
# manuscript paths, and on 12 Sep one of them overwrote four MCF-10A results
# because a single --output-dir was left off.
ROOT="${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ABL=results/journal/ablation_breast_epithelium
JOINT=results/journal/joint
COHORT=data/datafiles_breast_epithelium/testing_data_test_only.csv
SPLITS='data/datafiles_breast_epithelium/{split}.csv'
FUSION_W='checkpoints_ablation/breast_epithelium/seed{seed}/fusion/best_weights.pth'
# The sequence arm has no context tower and was never retrained for the swap, so
# it legitimately stays on the published checkpoints.
SEQ_W='checkpoints_journal/seed{seed}/sequence/best_weights.pth'
PY="${SILENTMETHYL_PY:-python}"

stamp_start() { date '+%Y-%m-%d %H:%M:%S' > "$1"; echo "[*] clobber stamp: $(cat "$1")"; }

# $2.. are the subtrees this job is ALLOWED to write. Anything else under
# results/journal/ is a bug.
#
# `--concurrent <subtree>` additionally exempts a subtree that a DIFFERENT
# analysis is known to be writing at the same time. The check is wall-clock
# (`-newermt`), so it cannot tell a sibling job's legitimate output from this
# job clobbering a published path: on 14 Sep it failed 45997644 at the final
# line, after a full hour of GPU work had already been written correctly,
# because Task D wrote transfer_failure/ from the login node mid-run. Exemptions
# are opt-in and stated in the sbatch so they stay visible; everything else
# newer than the stamp still fails the job.
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

# Node v005 silently ran GPU jobs on CPU at a 49x slowdown after a CUDA 803
# driver mismatch. Fail in seconds instead of hours.
require_cuda() {
  $PY - <<'PYCHK'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("[!!] CUDA unavailable on this node -- refusing to run on CPU")
print(f"[*] CUDA ok: {torch.cuda.get_device_name(0)}")
PYCHK
}
