# The `verdict` block in this directory's run_summary.json is WRONG

Read this before quoting anything from `run_summary.json`'s `verdict` field or
from the banner in `logs/r8_journal/ctx_ladder_46007255.out`.

## What is wrong

Job `46007255` was run with a version of `scripts/23_context_permutation.py`
whose verdict banner tested a **Pearson** inequality:

    deltas_pearson > levels_pearson

On that test two of the three rungs "fail", and the banner printed:

    At least one scheme moved the deltas as much as the levels.
    The allele-invariance argument does NOT hold empirically here.

**That conclusion does not follow, and the comparison is invalid.** Section 1 R2
of `LAB_NOTES.md` had already documented why, before this job ran: methylation
levels are bimodal with SD ~3.18 M-units, so a high Pearson on levels is cheap
and is not comparable against the same statistic on deltas, whose SD is ~0.10.
Under the realistic rungs both quantities also barely move at all — levels
Spearman 0.9865 against deltas 0.9856 under `xtissue_mean` — so every
correlation-based statistic saturates and flips sign on noise.

## What the data actually show

The **CSV outputs in this directory are correct and unchanged.** Only the derived
verdict was wrong. On normalised MAE (each quantity divided by its own SD, which
puts them on one scale), computed from `agreement_with_identity.csv`:

    rung           normMAE levels   normMAE deltas   ratio
    shuffle            0.5260           0.2266       2.32x
    tissue_Lung        0.1577           0.1154       1.37x
    xtissue_mean       0.0774           0.0650       1.19x

Deltas are preserved better than levels at **every** rung. Stratifying on the
OBSERVED effect size (`beta_ref_to_alt`, model-independent) the separation holds
in every quintile of every rung — see
`../context_ladder_stratified/` and `scripts/58_ladder_effect_size_stratification.py`.

Binning instead on the model's own predicted delta manufactures a false reversal
through regression to the mean. That artefact is reproduced deliberately under
`--stratifier predicted` and must not be reported as a result.

## What was changed

`scripts/23_context_permutation.py` was fixed on 15 Sep 2026 to decide the
verdict on normalised MAE, to print Pearson alongside it, and to state
explicitly when the two disagree and why. Re-running the script now would emit
the corrected banner and a `verdict` block with these keys:

    levels_normalised_mae, deltas_normalised_mae, levels_over_deltas_ratio,
    deltas_preserved_more_normalised_mae,
    levels_pearson, deltas_pearson, deltas_preserved_more_pearson

The bare `deltas_preserved_more` key no longer exists, deliberately: anything
that assumed the old Pearson semantics should fail loudly rather than silently
read the new verdict under the old name.

**`run_summary.json` in this directory was NOT edited.** It is the record of what
job `46007255` actually produced, and rewriting it would destroy that. The job
was not re-run because it costs ~4.9 GPU h and none of its data would change —
the correction is entirely in the derived verdict.

## Reporting

Report the dissociation as *consistent on scale-fair normalised error at all
three rungs, dose-dependent (2.32x, 1.37x, 1.19x), and strongest for the largest
observed effects*, and **report the Pearson/Spearman disagreement explicitly**
rather than selecting the metric that agrees. See `LAB_NOTES.md` section 6,
subsection B, for the wording to use and the wording to avoid.
