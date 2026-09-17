#!/usr/bin/env bash
#
# Make the SilentMethyl tree readable again.
#
#   ./cleanup_workspace.sh           # dry run -- prints every action, changes nothing
#   ./cleanup_workspace.sh --apply   # actually do it
#
# The rule this script follows
# ----------------------------
# Anything produced by a GPU is kept. Anything reproducible in minutes on a CPU,
# or re-downloadable with one command, is archived or removed. Nothing is deleted
# that cannot be regenerated from what remains plus a documented command.
#
# NEVER TOUCHED, by design:
#   checkpoints_journal/*/*/best_weights.pth   9 trained models, ~40 GPU-hours
#   data/datafiles/{train,val,test}.csv        the built training data
#   data/hg38.fa, data/reference/*.bw          large, slow downloads
#   data/external/genoa_meqtl/                 30 MB distilled from 5.3 GB raw
#   results/journal/*/pair_scores.csv          ~6 GPU-hours of variant scoring
#   results/journal/seed*/*/predictions.csv    test-set predictions, GPU
#   results/journal/{motif_disruption,genoa_variant_evaluation}/
#   data/external/egtex_breast/*.txt.gz        45 GB, still needed -- see note
#
# ARCHIVED to _archive/ (tarred, then the tree is removed) rather than deleted,
# because it is cited somewhere or cheap to keep compressed.
#
# REMOVED outright only where restoration is a single documented command.

set -euo pipefail

APPLY=0
PRUNE_EMPTY=0
for arg in "$@"; do
    case "$arg" in
        --apply)       APPLY=1 ;;
        --prune-empty) PRUNE_EMPTY=1 ;;
        *) echo "usage: $0 [--apply] [--prune-empty]" >&2; exit 2 ;;
    esac
done

# --- refuse to run anywhere but the project root ---------------------------
# Markers must exist in BOTH checkouts. data/datafiles/ is gitignored and lives
# only on the cluster, and REQUIREMENTS.md was folded into LAB_NOTES.md on
# 9 Sep 2026 -- requiring either made this script refuse to run on the laptop.
for marker in scripts/12_train_fusion.py LAB_NOTES.md; do
    if [[ ! -e "$marker" ]]; then
        echo "ERROR: $marker not found. Run this from the SilentMethyl root." >&2
        exit 1
    fi
done

ARCHIVE="_archive"
FREED=0

human () { numfmt --to=iec --suffix=B "${1:-0}" 2>/dev/null || echo "${1:-0} bytes"; }

# BSD du (macOS) has no -b. Fall back to -k so this runs on the laptop as well
# as on Bridges-2; always echo a number, because `set -u` plus an empty string
# turns the FREED arithmetic below into a fatal error.
size_of () {
    [[ -e "$1" ]] || { echo 0; return; }
    local b
    b=$(du -sb "$1" 2>/dev/null | cut -f1) \
        || b=$(( $(du -sk "$1" 2>/dev/null | cut -f1 || echo 0) * 1024 ))
    [[ -n "${b:-}" ]] || b=$(( $(du -sk "$1" 2>/dev/null | cut -f1 || echo 0) * 1024 ))
    echo "${b:-0}"
}

# Same idea for a glob: sum the matches without relying on `du -c -b`.
size_of_glob () {
    local total=0 f
    for f in $1; do
        [[ -e "$f" ]] || continue
        total=$(( total + $(size_of "$f") ))
    done
    echo "$total"
}

banner () { printf '\n\033[1m%s\033[0m\n' "$1"; }

# remove a path outright
drop () {
    local path="$1" why="$2"
    [[ -e "$path" ]] || return 0
    local bytes; bytes=$(size_of "$path")
    FREED=$((FREED + bytes))
    printf '  remove   %-58s %8s  %s\n' "$path" "$(human "$bytes")" "$why"
    if [[ $APPLY -eq 1 ]]; then rm -rf -- "$path"; fi
}

# tar into _archive/<name>.tar.gz, then remove the original
stow () {
    local path="$1" name="$2" why="$3"
    [[ -e "$path" ]] || return 0
    local bytes; bytes=$(size_of "$path")
    printf '  archive  %-58s %8s  %s\n' "$path" "$(human "$bytes")" "$why"
    if [[ $APPLY -eq 1 ]]; then
        mkdir -p "$ARCHIVE"
        tar -czf "$ARCHIVE/${name}.tar.gz" -- "$path"
        rm -rf -- "$path"
    fi
}

# Archive files matching a glob but KEEP the directory. Slurm opens its
# --output/--error paths before the job body runs, so removing a log directory
# breaks any queued job that writes into it. Use this for log trees, and `stow`
# only for directories nothing will write to again.
stow_glob () {
    local pattern="$1" name="$2" why="$3"
    compgen -G "$pattern" > /dev/null 2>&1 || return 0
    local n bytes
    n=$(compgen -G "$pattern" | wc -l | tr -d ' ')
    bytes=$(size_of_glob "$pattern")
    FREED=$((FREED + ${bytes:-0}))
    printf '  archive  %-58s %8s  %s\n' "$pattern" "$(human "${bytes:-0}")" "$why"
    printf '           %s file(s) -> %s/%s.tar.gz, directory kept\n' "$n" "$ARCHIVE" "$name"
    if [[ $APPLY -eq 1 ]]; then
        mkdir -p "$ARCHIVE"
        tar -czf "$ARCHIVE/${name}.tar.gz" $pattern
        rm -f -- $pattern
    fi
}

move () {
    local src="$1" dst="$2" why="$3"
    [[ -e "$src" ]] || return 0
    printf '  move     %-58s %8s  %s\n' "$src" "-> $(basename "$dst")" "$why"
    if [[ $APPLY -eq 1 ]]; then
        mkdir -p "$(dirname "$dst")"
        mv -n -- "$src" "$dst"
    fi
}

BEFORE_FILES=$(find . -type f -not -path './.git/*' | wc -l)
BEFORE_SIZE=$(size_of .)

if [[ $APPLY -eq 0 ]]; then
    banner "DRY RUN -- nothing will change. Re-run with --apply to execute."
else
    banner "APPLYING"
fi

# ---------------------------------------------------------------------------
banner "1. Dead weight"
# ---------------------------------------------------------------------------

# The harmonizer removes its own partial on a clean failure. A surviving
# .partial means the job was SIGKILLed (time limit or OOM), so this is debris
# from a run that must be repeated anyway -- wget -c does not resume it, and
# the harmonizer will rewrite it from scratch.
drop "data/external/egtex_breast/egtex_breast_within600.tsv.gz.partial" \
     "debris from the killed harmonize job"

while IFS= read -r d; do drop "$d" "bytecode cache"; done < <(
    find . -type d -name __pycache__ -not -path './.git/*' 2>/dev/null)

drop ".DS_Store" "macOS turd"
while IFS= read -r f; do drop "$f" "macOS turd"; done < <(
    find . -name .DS_Store -not -path './.git/*' 2>/dev/null)

# Shell accidents. A mistyped redirect or an unquoted flag creates a file or
# directory named after the command or the flag; they then sit in the tree
# looking like they mean something. Four of them (done/echo/grep/tail) reached
# a commit on 10 Sep 2026 before GitHub rejected the push for unrelated
# reasons, which is how they were noticed at all.
for junk in done echo grep tail head cat nohup true false null; do
    [[ -f "$junk" ]] && drop "$junk" "shell accident, not a real file"
done
for flag in data/external/melody/Melody_repo/--input-csv \
            data/external/melody/Melody_repo/--limit \
            data/external/melody/Melody_repo/--output \
            data/external/melody/Melody_repo/--tracks; do
    drop "$flag" "directory named after a CLI flag; unquoted argument"
done

# Misspelling of data/external. Contains only an empty Melody_repo/scripts
# skeleton -- a mkdir -p that went to the wrong path.
drop "data/exteral" "typo for data/external; empty skeleton"

# Tree dumps generated to inspect the repo. Regenerate with `tree` on demand.
drop "a.md" "scratch tree dump"

# ---------------------------------------------------------------------------
banner "2. Re-clonable third-party source trees"
# ---------------------------------------------------------------------------

# ~200 files of 2017-era Keras/Theano source, which is most of what makes the
# tree unreadable. scripts/14_baselines_simple.py is self-contained and does
# not use either of them. If the phase-2 architecture comparison happens, both
# are one git clone away -- recorded in the RESTORE file written below.
if [[ -d data/external/baselines ]]; then
    drop "data/external/baselines" "CpGenie + DeepCpG git clones (see RESTORE.md)"
    if [[ $APPLY -eq 1 ]]; then
        mkdir -p data/external/baselines
        cat > data/external/baselines/RESTORE.md <<'RESTORE'
# Baseline source trees, removed for readability

Neither is used by `scripts/14_baselines_simple.py`, which is self-contained.
They are only needed for the phase-2 architecture comparison (a CpGenie-style
CNN retrained on our splits) or for scoring the published cross-tissue weights.

    git clone https://github.com/gifford-lab/CpGenie.git
    git clone https://github.com/cangermueller/deepcpg.git

Provenance for the original clones is in `data/external/external_manifest.json`.

Caveat worth knowing before spending time on it: both are 2017-era Keras/Theano
and TF1. Reimplementing the CpGenie architecture in current PyTorch and training
it on our splits is both easier than resurrecting the environment AND a fairer
comparison, because it removes the cross-tissue handicap of the published
weights (CpGenie was trained on GM12878 lymphoblastoid, not breast).
RESTORE
    fi
fi

# ---------------------------------------------------------------------------
banner "3. Finished job logs"
# ---------------------------------------------------------------------------

if compgen -G "logs/data_build/harmonize_genoa_chr*.out" > /dev/null; then
    printf '  archive  %-58s %8s  %s\n' "logs/data_build/harmonize_genoa_chr*.out" \
        "22 files" "GENOA harmonization finished; summary is in harmonization_summary.json"
    if [[ $APPLY -eq 1 ]]; then
        mkdir -p "$ARCHIVE"
        tar -czf "$ARCHIVE/logs_harmonize_genoa.tar.gz" logs/data_build/harmonize_genoa_chr*.out
        rm -f logs/data_build/harmonize_genoa_chr*.out
    fi
fi

stow "logs/genoa_scoring" "logs_genoa_scoring" \
     "scoring finished; pair_scores.csv is the product"
stow "logs/training" "logs_training" \
     "training finished; run_config.json is beside each checkpoint"
stow "logs/testing" "logs_testing" \
     "testing finished; metrics.json is the product"

# TensorBoard event files are training telemetry, not weights. Keeping them
# compressed keeps checkpoints_journal/ down to what it is for: 9 .pth files.
if compgen -G "checkpoints_journal/*/*/tensorboard" > /dev/null; then
    printf '  archive  %-58s %8s  %s\n' "checkpoints_journal/*/*/tensorboard" \
        "9 dirs" "training curves; weights and run_config.json stay put"
    if [[ $APPLY -eq 1 ]]; then
        mkdir -p "$ARCHIVE"
        tar -czf "$ARCHIVE/tensorboard_events.tar.gz" checkpoints_journal/*/*/tensorboard
        rm -rf checkpoints_journal/*/*/tensorboard
    fi
fi

# ---------------------------------------------------------------------------
banner "4. Stale or superseded outputs"
# ---------------------------------------------------------------------------

# Built before scripts 19-22 existed, so it describes a state of the project
# that no longer exists. scripts/90_build_supplement_package.py regenerates it
# in minutes, and it must be rebuilt before submission regardless.
stow "supplementary_package" "supplementary_package_stale" \
     "predates scripts 19-22; rebuild with scripts/11 before submission"

# Strata-count sensitivity variants of the main uncertainty analysis. The s50
# run is cited in scripts/19's comments, so archive rather than delete.
stow "results/journal/rc_uncertainty_conditional_s20" "rc_uncertainty_s20" \
     "sensitivity variant; main run kept"
stow "results/journal/rc_uncertainty_conditional_s50" "rc_uncertainty_s50" \
     "sensitivity variant; main run kept (cited in scripts/19)"

# ---------------------------------------------------------------------------
banner "4b. Melody tuning sweeps and superseded runs  (added 10 Sep 2026)"
# ---------------------------------------------------------------------------

# The whole-blood margin sweep. Its pair_scores were archived already
# (_archive/melody_margin_sweep.tar.gz) but the 14 loose .summary.json files
# were left behind, which is most of what makes results/journal/melody/ look
# busy. The sweep settled the scoring parameters -- margin >= 1, sigmoid_first,
# midpoint window -- and those are now fixed in scripts/33_melody_scoring.py,
# so the summaries are a record of how we got there, not an input to anything.
stow_glob "results/journal/melody/wb_*.summary.json" "melody_margin_sweep_summaries" \
          "margin/threshold sweep; settled parameters live in scripts/33"

stow_glob "results/journal/melody/smoke_wholeblood.summary.json" "melody_smoke" \
          "smoke test for the scoring harness"

# Two intermediate union runs from before the full 39-track scoring existed.
# The 15-track run and the breast-luminal-only run are both superseded by
# by_tissue/melody_by_tissue_summary.json, which is what the manuscript cites.
stow_glob "results/journal/melody/union_*.summary.json" \
          "melody_partial_track_runs" \
          "15-track and breast-luminal runs, superseded by the 39-track by_tissue/"

# Smoke run of scripts/40. The real nine-tissue result is in
# results/journal/tissue_shared_meqtls/, and the Melody arm in
# tissue_shared_meqtls_melody/. Both are cited; this is neither.
stow "results/journal/tissue_specificity_smoke" "tissue_specificity_smoke" \
     "smoke run; real results in tissue_shared_meqtls{,_melody}/"

# The 15-track scores themselves. Their summary was archived above; this is the
# bulk. Superseded by the full 39-track run, and regenerable from the released
# checkpoint with one documented scripts/33 invocation.
drop "results/journal/melody/union_pair_scores_15track.csv" \
     "15-track run superseded by the 39-track by_tissue/ scores"

# ---------------------------------------------------------------------------
banner "4d. Fold checkpoints: same treatment checkpoints_journal already got"
# ---------------------------------------------------------------------------

# checkpoints_journal/ is already down to best_weights.pth + run_config.json
# per model. checkpoints_folds/ was never given the same pass and still carries
# both resume state and training telemetry.
#
# THIS IS THE ONE EXCEPTION to "anything produced by a GPU is kept", so it is
# in its own section and easy to skip. latest_checkpoint.pt is optimiser state
# for resuming an interrupted run. All four folds finished, were tested, and
# their metrics are committed; scripts/13_test_model.py loads best_weights.pth,
# never this. The only capability lost is resuming to more epochs, which would
# invalidate the reported numbers anyway. Nine files, several GB each.
for f in checkpoints_folds/*/*/latest_checkpoint.pt; do
    drop "$f" "resume state; folds finished, best_weights.pth is the artifact"
done

if compgen -G "checkpoints_folds/*/*/tensorboard" > /dev/null; then
    printf '  archive  %-58s %8s  %s\n' "checkpoints_folds/*/*/tensorboard" \
        "12 dirs" "training curves; weights and run_config.json stay put"
    if [[ $APPLY -eq 1 ]]; then
        mkdir -p "$ARCHIVE"
        tar -czf "$ARCHIVE/tensorboard_events_folds.tar.gz" checkpoints_folds/*/*/tensorboard
        rm -rf checkpoints_folds/*/*/tensorboard
    fi
fi

# ---------------------------------------------------------------------------
banner "4c. Logs from finished or dead jobs  (added 10 Sep 2026)"
# ---------------------------------------------------------------------------

# Job 45290715 is the fold array that trained both towers and then died at the
# fusion stage with IsADirectoryError (--sequence_weights got the save DIRECTORY
# rather than the .pth file). The towers it produced were reused by the
# fusion-only rerun, so the run mattered -- but its logs are ~240 MB of tqdm
# progress bars for a job whose ending is already documented in LAB_NOTES.md
# and guarded against in scripts/run_folds.sbatch.
stow_glob "logs/folds/45290715_*" "logs_folds_dead_fusion_run" \
          "died at fusion; towers survived and were reused by run_fusion_only"

# The successful fusion rerun and the fold test-set scoring. Same story: the
# .err files are tqdm output, ~120 MB each, and the products (fusion
# best_weights.pth, results/journal/folds/*/metrics.json) are all tracked. The
# provenance lines that matter -- "split files and tower weights unchanged for
# the whole run" -- are quoted in LAB_NOTES.md 7.D.
stow_glob "logs/folds/fusion_45427618_*" "logs_folds_fusion_rerun" \
          "fusion rerun finished and passed its provenance check"
stow_glob "logs/folds/test_*" "logs_folds_test" \
          "fold test scoring finished; metrics.json per fold is the product"

# Interactive run logs at the logs/ root: transfer_*, h2h_*, st_h2h_*,
# harmonize_*, r4_*, fetch_*. All small, all from finished work whose outputs
# are under results/journal/ or data/external/.
stow_glob "logs/*.log" "logs_interactive_runs" \
          "finished interactive runs; outputs are tracked"

stow_glob "logs/egtex_mt_scoring/*" "logs_egtex_mt_scoring" \
          "scoring finished; pair_scores.csv per tissue is the product"
stow_glob "logs/ctxperm/*" "logs_ctxperm" \
          "finished; results/journal/context_permutation/ is the product"
stow_glob "logs/melody/*" "logs_melody_union" \
          "finished; results/journal/melody/by_tissue/ is the product"
stow_glob "logs/melody_st/*" "logs_melody_st" \
          "finished; results/journal/melody_st/ is the product"

# NOT touched: logs/r41_*.txt if present. Those are the two-model R4 summaries
# read directly in LAB_NOTES.md 7.A2, small, and the only human-readable copy
# of that comparison outside decision.json.

# ---------------------------------------------------------------------------
banner "5. Misfiled"
# ---------------------------------------------------------------------------

move "data/BreastMammaryTissue.regular.perm.fdr.txt" \
     "data/external/egtex_breast/BreastMammaryTissue.regular.perm.fdr.txt" \
     "eGTEx permutation FDR file belongs with the cohort"

# ---------------------------------------------------------------------------
banner "6. Empty directories left behind (opt-in: --prune-empty)"
# ---------------------------------------------------------------------------

# OFF BY DEFAULT. Removing empty directories is purely cosmetic, and doing it
# while jobs are queued or running destroys them: a job creates its output
# directory at startup and writes to it minutes later, and Slurm opens its log
# files before the script body runs. This prune killed a six-task GPU array and
# a three-minute analysis in one evening. Enable with --prune-empty only when
# `squeue -u $USER` is empty.
#
# Restricted to the generated trees, and logs/ is DELIBERATELY EXCLUDED.
# Slurm opens --output/--error before the job script body runs, so an empty
# logs/<jobname>/ directory is load-bearing for any queued job: deleting it makes
# every array task die at ~5 s with the batch step CANCELLED and no log to
# explain why. Pruning empty log directories cost us a six-task array once.
EMPTY=""
if [[ $PRUNE_EMPTY -eq 1 ]]; then
    EMPTY=$(find data/external results checkpoints_journal -type d -empty \
            -not -path "*/$ARCHIVE/*" 2>/dev/null | sort || true)
fi
if [[ $PRUNE_EMPTY -eq 0 ]]; then
    printf '  skipped (pass --prune-empty, and only when squeue is empty)\n'
elif [[ -n "$EMPTY" ]]; then
    while IFS= read -r d; do printf '  rmdir    %s\n' "$d"; done <<< "$EMPTY"
    if [[ $APPLY -eq 1 ]]; then
        # repeat until stable: removing a leaf can empty its parent
        for _ in 1 2 3 4 5; do
            find data/external results checkpoints_journal -type d -empty \
                 -not -path "*/$ARCHIVE/*" -delete 2>/dev/null || true
        done
    fi
else
    printf '  none\n'
fi

# ---------------------------------------------------------------------------
banner "Summary"
# ---------------------------------------------------------------------------

if [[ $APPLY -eq 1 ]]; then
    AFTER_FILES=$(find . -type f -not -path './.git/*' | wc -l)
    AFTER_SIZE=$(size_of .)
    printf '  files : %s -> %s\n' "$BEFORE_FILES" "$AFTER_FILES"
    printf '  size  : %s -> %s\n' "$(human "$BEFORE_SIZE")" "$(human "$AFTER_SIZE")"
    printf '  archives written to %s/\n' "$ARCHIVE"
else
    printf '  would remove roughly %s outright\n' "$(human "$FREED")"
    printf '  current tree: %s files, %s\n' "$BEFORE_FILES" "$(human "$BEFORE_SIZE")"
    printf '\n  re-run with --apply to execute\n'
fi

cat <<'NOTE'

  NOT touched, and deliberately so:
    results/journal/genoa_variant_scoring/  and  results/journal/seed*/
      Every pair_scores.csv and predictions.csv is GPU output. Retraining will
      supersede them, but until the new checkpoints exist they are the only
      copy of ~6 GPU-hours of scoring.

    checkpoints_folds/fold{1,2,3}/
      Four folds x three stages of retraining. The fold table in
      main_revised.tex is the only published summary of them.

    scripts/70_mqtl_positive_control.py, scripts/71_mqtl_matched_negative.py
      Superseded by scripts/20-21, but still cited by Supplementary S2/S3.
      Removing them means editing the supplement in the same commit; do the two
      together or not at all.

    logs/r41_melody.txt, logs/r41_silentmethyl.txt
      The two-model R4 comparison, read directly in LAB_NOTES.md 7.A2.
      (.txt, so the logs/*.log archive step above does not match them.)

    checkpoints_folds/*/*/best_weights.pth, run_config.json,
    split_checksums*.txt, splits_summary_at_launch.json
      The fold models and the provenance that ties each to its split.

    data/external/melody/Melody_repo/ and dnabert2_local/
      Both are needed to score: Melody's released checkpoints live under
      Melody_repo/drive/, and dnabert2_local/ is the frozen encoder.

  Already gone, recorded rather than deleted silently:
    data/external/egtex_breast/BreastMammaryTissue.mQTLs.regular.txt.gz  (45 GB)
      Removed 9 Sep 2026 after the harmonized within-600bp file was produced.
      URL, byte count and sha256 are in data/external/external_manifest.json,
      so it is one documented download away.
NOTE

# ===========================================================================
# RECORD OF THE 16 Sep 2026 PASS (applied by hand, not by this script)
# ===========================================================================
#
# The sections above had already been applied and now free only ~2 MB. This
# pass removed a further ~9.3 GB and ~500 files. Recorded here so the tree's
# current state is auditable and every removal has a restore path.
#
# REMOVED -- resume state, 6.32 GB, 18 files
#   checkpoints_ablation/*/*/latest_checkpoint.pt
#   checkpoints_joint/*/*/latest_checkpoint.pt
#     Same argument section 4d already made for checkpoints_folds, verified
#     again: the three training scripts (10/11/12) WRITE this file and nothing
#     READS it. Scoring loads best_weights.pth. All 36 best_weights.pth are
#     intact. The only lost capability is resuming to more epochs, which would
#     invalidate the reported numbers anyway.
#     RESTORE: not restorable, and deliberately so -- retraining is the only
#     path, and it would supersede rather than reproduce these runs.
#
# ARCHIVED then removed -- training telemetry, 33 MB -> 11 MB, 36 dirs
#   checkpoints_{ablation,joint}/*/*/tensorboard
#     -> _archive/tensorboard_events_ablation_joint.tar.gz  (36 entries)
#     Matches the treatment checkpoints_folds got in section 4d.
#
# ARCHIVED then removed -- API response cache, 25 MB -> 1.7 MB, 499 files
#   data/cache/literature_variant_screen/*.{xml,json}
#     -> _archive/literature_screen_api_cache.tar.gz  (500 entries, gzip -t OK)
#     Content-addressed PubMed/Entrez responses for scripts/64. Nineteen percent
#     of the repository's file count for 0.02% of its bytes. Archived rather
#     than deleted because it is the record of what the API returned at the
#     time; the screen's resolved outputs are already final under
#     results/journal/literature_variant_screen/.
#     RESTORE: tar -xzf _archive/literature_screen_api_cache.tar.gz
#
# COMPRESSED in place -- 234 MB, lossless
#   logs/joint/jt-all4-s42-fuse_45812872.err  ->  .gz
#   logs/joint/jt-ho-fuse_45812874.err        ->  .gz
#     The only two uncompressed .err files left; every sibling was already .gz.
#
# REMOVED -- python bytecode caches
#   data/__pycache__, scripts/__pycache__, 12 *.pyc
#
# ---------------------------------------------------------------------------
# NOT DONE, and left for an explicit decision. Both are large and both were
# deliberately not taken unilaterally.
#
#   data/{datafiles,datafiles_breast_epithelium}/splits/fold*/{train,val,test}.csv
#     16.76 GB, 24 files. Deterministically regenerable:
#         python -u scripts/17_chromosome_splits.py --folds 4
#         python -u scripts/17_chromosome_splits.py --folds 4 \
#             --datafiles data/datafiles_breast_epithelium \
#             --out-root  data/datafiles_breast_epithelium/splits
#     and md5-verifiable against checkpoints_folds/*/split_checksums.txt and
#     checkpoints_ablation/breast_epithelium/*/input_checksums.txt. This meets
#     the script's own "restorable by a single documented command" bar.
#     splits_summary.json must be KEPT in both trees -- it is the provenance.
#
#   The 27 pre-11-Sep result directories duplicated under
#   results/journal/ablation_breast_epithelium/
#     2.91 GB, 408 files. Every one predates the 11 Sep context swap and was
#     re-run into the ablation tree; LAB_NOTES section 0 says single-tissue
#     numbers must be read from there and that figures built before 12 Sep must
#     not be circulated. Having both copies is the live hazard -- the stale one
#     sits at the shorter, more guessable path.
#     BLOCKER: scripts, main_revised.tex and RESULTS_REVISED.md still reference
#     the stale top-level paths (genoa_variant_scoring 7, candidates 6,
#     egtex_multitissue_scoring 4, ...). Deleting without repointing those
#     references first would break them. Do the two together or not at all.
