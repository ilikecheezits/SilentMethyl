#!/usr/bin/env bash
# Archive or remove CPU-reproducible artefacts and leave every GPU-derived product in place; dry run by default, --apply to act.

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

for marker in scripts/12_train_fusion.py LAB_NOTES.md; do
    if [[ ! -e "$marker" ]]; then
        echo "ERROR: $marker not found. Run this from the SilentMethyl root." >&2
        exit 1
    fi
done

ARCHIVE="_archive"
FREED=0

human () { numfmt --to=iec --suffix=B "${1:-0}" 2>/dev/null || echo "${1:-0} bytes"; }

size_of () {
    [[ -e "$1" ]] || { echo 0; return; }
    local b
    b=$(du -sb "$1" 2>/dev/null | cut -f1) \
        || b=$(( $(du -sk "$1" 2>/dev/null | cut -f1 || echo 0) * 1024 ))
    [[ -n "${b:-}" ]] || b=$(( $(du -sk "$1" 2>/dev/null | cut -f1 || echo 0) * 1024 ))
    echo "${b:-0}"
}

size_of_glob () {
    local total=0 f
    for f in $1; do
        [[ -e "$f" ]] || continue
        total=$(( total + $(size_of "$f") ))
    done
    echo "$total"
}

banner () { printf '\n\033[1m%s\033[0m\n' "$1"; }

drop () {
    local path="$1" why="$2"
    [[ -e "$path" ]] || return 0
    local bytes; bytes=$(size_of "$path")
    FREED=$((FREED + bytes))
    printf '  remove   %-58s %8s  %s\n' "$path" "$(human "$bytes")" "$why"
    if [[ $APPLY -eq 1 ]]; then rm -rf -- "$path"; fi
}

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

banner "1. Dead weight"

drop "data/external/egtex_breast/egtex_breast_within600.tsv.gz.partial" \
     "debris from the killed harmonize job"

while IFS= read -r d; do drop "$d" "bytecode cache"; done < <(
    find . -type d -name __pycache__ -not -path './.git/*' 2>/dev/null)

drop ".DS_Store" "macOS turd"
while IFS= read -r f; do drop "$f" "macOS turd"; done < <(
    find . -name .DS_Store -not -path './.git/*' 2>/dev/null)

for junk in done echo grep tail head cat nohup true false null; do
    [[ -f "$junk" ]] && drop "$junk" "shell accident, not a real file"
done
for flag in data/external/melody/Melody_repo/--input-csv \
            data/external/melody/Melody_repo/--limit \
            data/external/melody/Melody_repo/--output \
            data/external/melody/Melody_repo/--tracks; do
    drop "$flag" "directory named after a CLI flag; unquoted argument"
done

drop "data/exteral" "typo for data/external; empty skeleton"

drop "a.md" "scratch tree dump"

banner "2. Re-clonable third-party source trees"

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

banner "3. Finished job logs"

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

if compgen -G "checkpoints_journal/*/*/tensorboard" > /dev/null; then
    printf '  archive  %-58s %8s  %s\n' "checkpoints_journal/*/*/tensorboard" \
        "9 dirs" "training curves; weights and run_config.json stay put"
    if [[ $APPLY -eq 1 ]]; then
        mkdir -p "$ARCHIVE"
        tar -czf "$ARCHIVE/tensorboard_events.tar.gz" checkpoints_journal/*/*/tensorboard
        rm -rf checkpoints_journal/*/*/tensorboard
    fi
fi

banner "4. Stale or superseded outputs"

stow "supplementary_package" "supplementary_package_stale" \
     "predates scripts 19-22; rebuild with scripts/11 before submission"

stow "results/journal/rc_uncertainty_conditional_s20" "rc_uncertainty_s20" \
     "sensitivity variant; main run kept"
stow "results/journal/rc_uncertainty_conditional_s50" "rc_uncertainty_s50" \
     "sensitivity variant; main run kept (cited in scripts/19)"

banner "4b. Melody tuning sweeps and superseded runs  (added 10 Sep 2026)"

stow_glob "results/journal/melody/wb_*.summary.json" "melody_margin_sweep_summaries" \
          "margin/threshold sweep; settled parameters live in scripts/33"

stow_glob "results/journal/melody/smoke_wholeblood.summary.json" "melody_smoke" \
          "smoke test for the scoring harness"

stow_glob "results/journal/melody/union_*.summary.json" \
          "melody_partial_track_runs" \
          "15-track and breast-luminal runs, superseded by the 39-track by_tissue/"

stow "results/journal/tissue_specificity_smoke" "tissue_specificity_smoke" \
     "smoke run; real results in tissue_shared_meqtls{,_melody}/"

drop "results/journal/melody/union_pair_scores_15track.csv" \
     "15-track run superseded by the 39-track by_tissue/ scores"

banner "4d. Fold checkpoints: same treatment checkpoints_journal already got"

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

banner "4c. Logs from finished or dead jobs  (added 10 Sep 2026)"

stow_glob "logs/folds/45290715_*" "logs_folds_dead_fusion_run" \
          "died at fusion; towers survived and were reused by run_fusion_only"

stow_glob "logs/folds/fusion_45427618_*" "logs_folds_fusion_rerun" \
          "fusion rerun finished and passed its provenance check"
stow_glob "logs/folds/test_*" "logs_folds_test" \
          "fold test scoring finished; metrics.json per fold is the product"

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


banner "5. Misfiled"

move "data/BreastMammaryTissue.regular.perm.fdr.txt" \
     "data/external/egtex_breast/BreastMammaryTissue.regular.perm.fdr.txt" \
     "eGTEx permutation FDR file belongs with the cohort"

banner "6. Empty directories left behind (opt-in: --prune-empty)"

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
        for _ in 1 2 3 4 5; do
            find data/external results checkpoints_journal -type d -empty \
                 -not -path "*/$ARCHIVE/*" -delete 2>/dev/null || true
        done
    fi
else
    printf '  none\n'
fi

banner "Summary"

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

