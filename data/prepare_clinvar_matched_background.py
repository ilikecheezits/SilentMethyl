#!/usr/bin/env python3
"""
Do ClinVar-pathogenic variants in breast-cancer genes show larger predicted
methylation effects than their matched backgrounds?

Why this framing and not an enrichment test
-------------------------------------------
The literature cohort was assembled by selecting ClinVar-pathogenic variants in
cancer-relevant genes: 310 of 322 entered through `Discovery_Source =
ClinVar-pathogenic`, and there are ZERO benign variants. So "do pathogenic
variants rank higher than benign ones" is undefined -- there is no comparison
group, and a high rank inside an all-pathogenic list is close to arithmetically
guaranteed.

The matched background replaces the missing benign class. Each variant is
compared against comparators matched on distance to the target CpG, sequence
context and substitution class, and the question becomes: does this variant's
predicted effect sit in the upper tail of variants that look like it? That needs
no benign class, and it is the framing scripts/05 already implements for the
somatic candidate cohort.

Eligibility, and why each filter is not optional
------------------------------------------------
* **Held-out probes only** (`Model_Split == test`). The two STK11 case-study
  variants sit on TRAINING probes; the model saw those loci. An effect-size
  claim cannot rest on them.
* **Non-truncating only.** A variant annotated `Ter` or `fs` is pathogenic
  because it truncates the protein. Its ClinVar classification says nothing
  about methylation, so including it tests the wrong thing.
* Nearby-CpG-altering variants are KEPT and stratified, not excluded. Only
  target-CpG-altering variants are unscoreable, and scripts/05 handles that.
  (rs148928808 destroys a CpG 205 bp from its target; the target is intact.)

Declared before the numbers exist
---------------------------------
* **Primary cohort:** all held-out non-truncating variants (n = 35).
* **Secondary cohort:** those with unambiguous calls -- Pathogenic,
  Pathogenic/Likely pathogenic, Likely pathogenic (n = 16). The 18 "Conflicting
  classifications" are variants whose submitters disagree; including them
  dilutes, excluding them halves the sample, so BOTH are reported.
* **Primary statistic:** the count of variants whose matched-background tail
  probability is below 0.05, against a binomial null of 0.05. At n = 35 that
  expects 1.75 by chance; 6 or more gives p < 0.01.
* **Secondary statistic:** mean matched-background percentile against a null of
  50, by block bootstrap.
* Two-sided throughout.

**Power, stated in advance.** At n = 35 a mean percentile of 65 is about 3 SE
from the null and detectable; 58 is not. A null result here will NOT distinguish
"no enrichment" from "underpowered", and must be reported as inconclusive rather
than negative.

Usage (run from the repository root)
------------------------------------
    python -u data/prepare_clinvar_matched_background.py --prepare
    # then scripts/05 on the emitted CSV, then:
    python -u data/prepare_clinvar_matched_background.py --analyse \
        --scored results/journal/clinvar_matched_background/<file>.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger("silentmethyl.clinvar")

DEFAULT_RANKED = Path("results/journal/literature_variant_screen/"
                      "literature_variant_predictions_ranked.csv")
DEFAULT_OUT = Path("results/journal/clinvar_matched_background")
CLEAN_CALLS = {"Pathogenic", "Pathogenic/Likely pathogenic", "Likely pathogenic"}
TRUNCATING = r"Ter|fs"
ALPHA = 0.05

PERCENTILE_ALIASES = ("Matched_Background_Absolute_Effect_Percentile",)
TAIL_ALIASES = ("Matched_Empirical_Tail_Probability",)


def resolve(frame, aliases, what):
    for name in aliases:
        if name in frame.columns:
            return name
    raise SystemExit(
        f"no {what} column found (looked for {list(aliases)}). Run scripts/05 "
        f"on the prepared CSV first; its output carries these columns.")


def cmd_prepare(args) -> int:
    if not args.ranked.is_file():
        raise SystemExit(f"not found: {args.ranked}")
    d = pd.read_csv(args.ranked, low_memory=False)
    LOGGER.info("literature cohort: %d rows", len(d))

    ann = d["Transcript_Annotation"].astype(str)
    truncating = ann.str.contains(TRUNCATING, na=False, regex=True)
    held = d["Model_Split"].astype(str).eq("test")
    primary = d[held & ~truncating].copy()
    clean = primary["Clinical_Significance"].isin(CLEAN_CALLS)

    counts = {
        "cohort_total": int(len(d)),
        "held_out": int(held.sum()),
        "held_out_truncating_excluded": int((held & truncating).sum()),
        "primary_n": int(len(primary)),
        "secondary_n": int(clean.sum()),
        # str() the keys: value_counts(dropna=False) yields a NaN key alongside
        # string keys, and json.dump(sort_keys=True) cannot order float against
        # str. The real cohort has one unclassified variant; a synthetic test
        # without NaN will not catch this.
        "primary_clinical_significance":
            {str(k): int(v) for k, v in
             primary["Clinical_Significance"].value_counts(dropna=False).items()},
        "primary_cpg_effect":
            {str(k): int(v) for k, v in
             primary["CpG_Effect"].value_counts(dropna=False).items()},
    }
    LOGGER.info("primary n=%d, secondary n=%d", counts["primary_n"],
                counts["secondary_n"])
    if counts["primary_n"] < 20:
        LOGGER.warning("primary cohort under 20; the test will not be "
                       "informative either way")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary["ClinVar_Clean_Call"] = clean.to_numpy()

    # scripts/05 generates its own columns and merges them back onto the cohort
    # on Variant_UID. Any name already present on our side collides into _x/_y
    # and the plain name vanishes -- which is what produced
    # KeyError: 'HM450_MASK_general' inside apply_probe_qc, and then
    # KeyError: 'Predicted_Delta_Beta' inside aggregate_seeds.
    #
    # The literature cohort carries model scores from scripts/15; the somatic
    # candidate cohort scripts/05 was built against does not, which is why this
    # only bites here. Fixing the names one crash at a time is whack-a-mole, so
    # drop the whole set scripts/05 regenerates and let it be the single source
    # of truth for the scores it produces. A stale score column silently
    # shadowing a fresh one would be a far worse failure than the crash: the
    # matched-background percentile would be computed against scores from a
    # different run.
    SCRIPT05_GENERATED = (
        # probe QC, re-derived from the HM450 manifest
        "HM450_MASK_general",
        # per-variant aggregate emitted by aggregate_seeds()
        "Seed_Count", "Seeds",
        "Predicted_Delta_Beta", "Predicted_Delta_Beta_Mean",
        "Predicted_Delta_Beta_SD", "Predicted_Delta_Beta_Median",
        "Predicted_Delta_Beta_Min", "Predicted_Delta_Beta_Max",
        "Delta_Beta_Sign_Consistency", "Mean_Absolute_Delta_Beta",
        "Mean_Within_Seed_Rank", "SD_Within_Seed_Rank",
        "Best_Within_Seed_Rank", "Worst_Within_Seed_Rank",
        "Top10_Seed_Frequency", "Top20_Seed_Frequency",
        "Mean_Delta_RC_Absolute_Difference", "Delta_RC_Sign_Agreement_Fraction",
        "WT_Gate_DNA_Mean", "WT_Gate_EPI_Mean", "WT_Gate_DNA_Share_Mean",
        "MUT_Gate_DNA_Mean", "MUT_Gate_EPI_Mean", "MUT_Gate_DNA_Share_Mean",
        # derived after that merge; assignment would overwrite silently
        "Absolute_Delta_Beta", "Absolute_Delta_Beta_Rank",
    )
    collides = [c for c in SCRIPT05_GENERATED if c in primary.columns]
    if collides:
        LOGGER.info("dropping %d column(s) that scripts/05 regenerates, so the "
                    "merge cannot collide into _x/_y: %s", len(collides),
                    collides)
        primary = primary.drop(columns=collides)

    target = args.output_dir / "clinvar_heldout_nontruncating_cohort.csv"
    primary.to_csv(target, index=False)

    payload = {
        "analysis": "ClinVar-pathogenic variants vs matched background",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "PRE-REGISTERED -- written before any matched-background "
                  "score for this cohort existed",
        "source": str(args.ranked),
        "eligibility": {
            "held_out_only": "Model_Split == test; the STK11 case-study variants "
                             "are on training probes and are excluded",
            "non_truncating_only": "Ter/fs variants are pathogenic by protein "
                                   "truncation; their ClinVar call is not "
                                   "evidence about methylation",
            "nearby_cpg_altering": "KEPT and stratified, not excluded",
        },
        "counts": counts,
        "primary_statistic": (
            "count of variants with matched-background tail probability < 0.05, "
            "against a binomial null of 0.05; two-sided"),
        "secondary_statistic": (
            "mean matched-background percentile against a null of 50"),
        "power_statement": (
            "at n=35, a mean percentile of 65 is ~3 SE from the null and "
            "detectable; 58 is not. A null result does NOT distinguish no "
            "enrichment from underpowered and must be reported as inconclusive."),
        "prepared_cohort": str(target),
        "columns_dropped_for_scripts05": collides,
    }
    with (args.output_dir / "preregistration.json").open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    print()
    print("=" * 74)
    print("PRE-REGISTERED before any score exists")
    print(f"  cohort total                    {counts['cohort_total']:>6}")
    print(f"  held out                        {counts['held_out']:>6}")
    print(f"  held out, truncating (excluded) {counts['held_out_truncating_excluded']:>6}")
    print(f"  PRIMARY   (all non-truncating)  {counts['primary_n']:>6}")
    print(f"  SECONDARY (clean ClinVar calls) {counts['secondary_n']:>6}")
    print("\n  clinical significance in the primary cohort:")
    for k, v in counts["primary_clinical_significance"].items():
        print(f"    {str(k):<46} {v:>4}")
    print("\n  CpG effect (kept and stratified, not excluded):")
    for k, v in counts["primary_cpg_effect"].items():
        print(f"    {str(k):<46} {v:>4}")
    print(f"\nwrote {target}")
    print("\nNext: score it with scripts/05, then re-run with --analyse")
    print("=" * 74)
    return 0


def cmd_analyse(args) -> int:
    if not args.scored or not Path(args.scored).is_file():
        raise SystemExit("--analyse needs --scored pointing at scripts/05 output")
    d = pd.read_csv(args.scored, low_memory=False)
    pct_col = resolve(d, PERCENTILE_ALIASES, "matched-background percentile")
    tail_col = resolve(d, TAIL_ALIASES, "matched-background tail probability")

    pre = args.output_dir / "preregistration.json"
    if not pre.is_file():
        raise SystemExit(f"{pre} missing -- run --prepare first so the strata "
                         f"and statistic are fixed before the numbers are seen")

    results = {}
    for label, frame in (("primary", d),
                         ("secondary_clean_calls",
                          d[d.get("ClinVar_Clean_Call", pd.Series(dtype=bool))
                            .fillna(False).astype(bool)]
                          if "ClinVar_Clean_Call" in d.columns else d.iloc[0:0])):
        if frame.empty:
            continue
        tail = pd.to_numeric(frame[tail_col], errors="coerce").dropna()
        pct = pd.to_numeric(frame[pct_col], errors="coerce").dropna()
        n = int(len(tail))
        hits = int((tail < ALPHA).sum())
        binom = stats.binomtest(hits, n, ALPHA, alternative="two-sided")
        se = float(np.std(pct, ddof=1) / np.sqrt(len(pct))) if len(pct) > 1 else np.nan
        results[label] = {
            "n": n,
            "expected_hits_under_null": round(n * ALPHA, 2),
            "observed_hits": hits,
            "binomial_p": float(binom.pvalue),
            "mean_percentile": float(pct.mean()),
            "median_percentile": float(pct.median()),
            "percentile_se": se,
            "percentile_z_vs_50": float((pct.mean() - 50) / se) if se and np.isfinite(se) else None,
        }

    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "scored_file": str(args.scored), "alpha": ALPHA, "results": results}
    with (args.output_dir / "enrichment_result.json").open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    print()
    print("=" * 74)
    for label, r in results.items():
        print(f"\n{label.upper()}  (n = {r['n']})")
        print(f"  tail probability < {ALPHA}: {r['observed_hits']} observed, "
              f"{r['expected_hits_under_null']} expected   p = {r['binomial_p']:.4f}")
        print(f"  mean matched-background percentile: {r['mean_percentile']:.1f} "
              f"(null 50)", end="")
        if r["percentile_z_vs_50"] is not None:
            print(f"   z = {r['percentile_z_vs_50']:+.2f}")
        else:
            print()
    print()
    print("Reminder from the pre-registration: at this sample size a null result")
    print("does NOT distinguish 'no enrichment' from 'underpowered'. If nothing")
    print("reaches significance, report it as inconclusive, not as negative.")
    print("=" * 74)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranked", type=Path, default=DEFAULT_RANKED)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--scored", type=Path, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if args.prepare:
        return cmd_prepare(args)
    if args.analyse:
        return cmd_analyse(args)
    raise SystemExit("pass --prepare or --analyse")


if __name__ == "__main__":
    sys.exit(main())
