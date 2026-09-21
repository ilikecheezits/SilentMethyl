# Shared guards for the single-tissue breast-epithelium job chain.

ROOT="${SILENTMETHYL_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ABL=results/journal/ablation_breast_epithelium
CAND=$ABL/candidates
COHORT=data/datafiles_breast_epithelium/testing_data_test_only.csv
FUSION_W='checkpoints_journal/seed{seed}/fusion/best_weights.pth'
SEQ_W='checkpoints_journal/seed{seed}/sequence/best_weights.pth'
PY="${SILENTMETHYL_PY:-python}"

stamp_start() { date '+%Y-%m-%d %H:%M:%S' > "$1"; echo "[*] clobber stamp: $(cat "$1")"; }

# Other Slurm jobs of this user that ran in the same tree since the stamp. The clobber
# check uses file timestamps, so their writes would otherwise count as this job's.
concurrent_jobs() {
  local self="${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
  command -v sacct >/dev/null || return 0
  sacct -u "$USER" -S "$(cat "$1")" -X -n -P -o JobID,State,WorkDir 2>/dev/null \
    | awk -F'|' -v d="$PWD" -v a="${SLURM_JOB_ID:-none}" -v b="${self:-none}" \
        '$3==d && $2!="PENDING" && $1!=a && $1!=b {print $1}' | tr '\n' ' '
}

check_no_clobber() {
  local stamp_file="$1" ; local touched
  touched=$(find results/journal -type f -newermt "$(cat "$stamp_file")" \
              -not -path "results/journal/ablation_breast_epithelium/*" 2>/dev/null || true)
  if [[ -n "$touched" ]]; then
    echo "[!!] PUBLISHED-PATH CLOBBER -- these live outside the ablation subtree:" >&2
    echo "$touched" >&2
    local others ; others=$(concurrent_jobs "$stamp_file")
    if [[ -n "$others" ]]; then
      echo "[!] other jobs ran in this tree meanwhile (${others% }); not failing. If none of the paths above are this job's, the outputs are valid." >&2
      return 0
    fi
    return 1
  fi
  echo "[*] clobber check clean: nothing outside $ABL was written"
}

require_cuda() {
  $PY - <<'PYCHK'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("[!!] CUDA unavailable on this node -- refusing to run on CPU")
print(f"[*] CUDA ok: {torch.cuda.get_device_name(0)}")
PYCHK
}
