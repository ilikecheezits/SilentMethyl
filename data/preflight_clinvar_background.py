#!/usr/bin/env python
"""Pre-flight for the ClinVar matched-background test. Reads only; scores nothing.

Each held-out ClinVar-pathogenic variant is compared against comparators matched
on substitution class, CpG effect and distance to the target CpG. Those
comparators are drawn from whatever frame is handed to scripts/05 -- which is
why passing it the 35 ClinVar variants alone was wrong: every variant's
"background" was the other 34 pathogenic variants.

The fix is to score the 35 inside a pool of ordinary held-out variants. Before
spending a GPU on that, this answers the questions that decide whether the
design works:

  1. Do both files carry what scripts/05 requires?
  2. How many rows of each survive scripts/05's model-visibility validation?
     (that count, not the row count, is what a GPU run actually costs)
  3. Do the two sets collide on Variant_UID?
  4. Does every ClinVar variant find enough comparators in the pool, at a tier
     that still constrains substitution class, without drawing its background
     from other ClinVar variants?

Question 4 is the one that matters, and it has three ways to fail, not one.

IMPORTANT -- the matching keys are NOT columns you supply. scripts/05 derives
Variant_UID, Canonical_SBS6/96, CpG_Effect and the distance itself, inside
build_model_visible_cohort, by diffing the WT and mutant sequences through
annotate_variant. This script imports and calls those same functions rather than
reimplementing them, so it cannot drift from what scripts/05 will actually do.

No model is loaded and no GPU is used, though torch is imported transitively by
training_common.

    python -u data/preflight_clinvar_background.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

DEFAULT_CLINVAR = ("results/journal/clinvar_matched_background/"
                   "clinvar_heldout_nontruncating_cohort.csv")
DEFAULT_POOL = "data/datafiles/testing_data_test_only.csv"

WINDOW_SIZE = 1000          # scripts/05 --window-size default
MIN_COMPARATORS = 20        # scripts/05 --min-comparators default
MAX_COMPARATORS = 1000      # scripts/05 --max-comparators default

# Tiers where the substitution-class constraint has been dropped. A variant
# matched only here is "matched" in name only.
WEAK_TIERS = ("T7_CpG_distance250", "T8_all_observed_synonymous")

# With 35 ClinVar variants inside a pool of thousands, a few will match each
# other by chance; one comparator in forty does not move a percentile. What
# matters is a background substantially made of other pathogenic variants --
# the original flaw at smaller scale. Threshold fixed here, before any result.
CONTAMINATION_LIMIT = 0.10

TABULAR_FEATURES = [
    "Ref_ATAC_Signal", "Ref_H3K4me3_Signal", "Ref_H3K27ac_Signal",
    "Ref_H3K27me3_Signal", "Ref_H3K9me3_Signal", "Ref_H3K36me3_Signal",
    "Ref_H3K4me1_Signal", "Target_Base_PhyloP_100way_1",
    "Target_Base_PhyloP_100way_2",
]
MISSING_FEATURES = [f"{n}_Missing" for n in TABULAR_FEATURES]
SCRIPT05_REQUIRED = ["probeID", "Healthy_5000bp_DNA", "Mutated_5000bp_DNA",
                     "Model_Split", *TABULAR_FEATURES, *MISSING_FEATURES]


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def derive(df: pd.DataFrame, label: str, centered_crop, annotate_variant,
           protected: set) -> pd.DataFrame:
    """Replicate scripts/05 build_model_visible_cohort, minus the tensors.

    Same functions, same order, same skip conditions -- so the surviving row
    count here is the surviving row count there.
    """
    records: list[dict] = []
    skipped = {"invalid_sequence": 0, "mutation_outside_model_crop": 0,
               "invalid_or_noncentral_snv": 0, "duplicate_uid": 0}
    seen: set[str] = set()

    for _, row in df.iterrows():
        try:
            wt = centered_crop(str(row["Healthy_5000bp_DNA"]), WINDOW_SIZE)
            mut = centered_crop(str(row["Mutated_5000bp_DNA"]), WINDOW_SIZE)
        except ValueError:
            skipped["invalid_sequence"] += 1
            continue
        diffs = [i for i, (a, b) in enumerate(zip(wt, mut)) if a != b]
        if len(diffs) == 0:
            skipped["mutation_outside_model_crop"] += 1
            continue
        if len(diffs) != 1 or diffs[0] in protected:
            skipped["invalid_or_noncentral_snv"] += 1
            continue
        try:
            meta = annotate_variant(row, wt, mut, window_size=WINDOW_SIZE)
        except ValueError:
            skipped["invalid_or_noncentral_snv"] += 1
            continue
        uid = str(meta["Variant_UID"])
        if uid in seen:
            skipped["duplicate_uid"] += 1
            continue
        seen.add(uid)
        records.append(meta)

    out = pd.DataFrame(records)
    print(f"  {label}: {len(df)} rows in -> {len(out)} model-visible")
    if any(skipped.values()):
        print(f"    skipped: {skipped}")
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Pre-flight for the ClinVar test")
    ap.add_argument("--clinvar", default=DEFAULT_CLINVAR)
    ap.add_argument("--pool", default=DEFAULT_POOL,
                    help="Background pool CSV. Must satisfy the same scripts/05 "
                         "input contract as the ClinVar cohort.")
    args = ap.parse_args()
    CLINVAR, POOL = Path(args.clinvar), Path(args.pool)

    for p in (CLINVAR, POOL):
        if not p.exists():
            print(f"MISSING: {p}")
            return 1

    rule("1. SCHEMA  (scripts/05 derives the matching keys; they are not inputs)")
    cv_cols = list(pd.read_csv(CLINVAR, nrows=0).columns)
    pool_cols = list(pd.read_csv(POOL, nrows=0).columns)
    print(f"clinvar cohort : {len(cv_cols)} columns")
    print(f"background pool: {len(pool_cols)} columns "
          f"({POOL.stat().st_size / 1e6:.0f} MB)")
    ok = True
    for name, cols in (("clinvar", cv_cols), ("pool", pool_cols)):
        missing = [c for c in SCRIPT05_REQUIRED if c not in cols]
        if missing:
            ok = False
            print(f"  {name}: MISSING {missing}")
        else:
            print(f"  {name}: has everything scripts/05 requires")
    if not ok:
        print("\nSTOP: scripts/05 would refuse this input.")
        return 1

    rule("2. MODEL-VISIBLE COHORTS  (what a GPU run actually costs)")
    try:
        from training_common import centered_crop
        from matched_background_utils import (
            PROTECTED_CPG_INDICES, annotate_variant, choose_matched_comparators)
    except Exception as exc:
        print(f"import failed: {type(exc).__name__}: {exc}")
        print("run from the repository root with the silentmethyl env active")
        return 1

    cv_raw = pd.read_csv(CLINVAR, low_memory=False)
    pool_raw = pd.read_csv(POOL, low_memory=False)
    for name, df in (("clinvar", cv_raw), ("pool", pool_raw)):
        bad = ~df["Model_Split"].astype(str).eq("test")
        if bad.any():
            print(f"  STOP: {name} has {int(bad.sum())} non-test rows; "
                  f"scripts/05 requires held-out only")
            return 1

    cv = derive(cv_raw, "clinvar", centered_crop, annotate_variant,
                set(PROTECTED_CPG_INDICES))
    pool = derive(pool_raw, "pool", centered_crop, annotate_variant,
                  set(PROTECTED_CPG_INDICES))
    if cv.empty or pool.empty:
        print("\nSTOP: one cohort has no model-visible variants.")
        return 1

    rule("3. IDENTITY COLLISIONS")
    overlap = set(cv["Variant_UID"].astype(str)) & set(pool["Variant_UID"].astype(str))
    print(f"Variant_UID shared between the two sets: {len(overlap)}")
    if overlap:
        print(f"  e.g. {sorted(overlap)[:5]}")
        print("  dropped from the pool below so no variant is its own comparator")
    pool = pool[~pool["Variant_UID"].astype(str).isin(overlap)].copy()

    rule("4. COMPARATOR AVAILABILITY  <- the question that decides the design")
    union = pd.concat([cv, pool], ignore_index=True)
    union["Predicted_Delta_Beta"] = 0.0        # required column; never read here
    n_cv = len(cv)

    tiers: dict[str, int] = {}
    counts, short, weak, shares = [], [], [], []
    for i in range(n_cv):
        idx, tier, _ = choose_matched_comparators(
            union, i, min_comparators=MIN_COMPARATORS,
            max_comparators=MAX_COMPARATORS)
        idx = idx[idx != i]                        # never its own comparator
        n_cv_used = int((idx < n_cv).sum())        # ClinVar rows lead the union
        uid = str(union.iloc[i]["Variant_UID"])
        tiers[tier] = tiers.get(tier, 0) + 1
        counts.append(len(idx))
        if len(idx) < MIN_COMPARATORS:
            short.append((uid, tier, len(idx)))
        if tier in WEAK_TIERS:
            weak.append(uid)
        if n_cv_used:
            shares.append((uid, n_cv_used, len(idx),
                           n_cv_used / len(idx) if len(idx) else 1.0))

    arr = np.array(counts)
    print(f"variants checked       : {n_cv}")
    print(f"comparators per variant: min {arr.min()}  median "
          f"{int(np.median(arr))}  max {arr.max()}")
    print(f"variants with >= {MIN_COMPARATORS}   : "
          f"{int((arr >= MIN_COMPARATORS).sum())} / {n_cv}")
    print("\ntier reached (T1 strictest, T8 matches everything):")
    for tier in sorted(tiers):
        flag = "   <- constraint effectively dropped" if tier in WEAK_TIERS else ""
        print(f"  {tier:<36} {tiers[tier]}{flag}")

    if short:
        print(f"\n{len(short)} variant(s) below {MIN_COMPARATORS} comparators:")
        for uid, tier, n in short[:10]:
            print(f"  {uid:<34} {tier:<32} {n}")
    material = [s for s in shares if s[3] > CONTAMINATION_LIMIT]
    if shares:
        print(f"\n{len(shares)}/{n_cv} variant(s) draw at least one comparator from "
              f"the ClinVar set; worst share {100 * max(s[3] for s in shares):.1f}%")
        for uid, used, tot, share in sorted(shares, key=lambda s: -s[3])[:8]:
            mark = "  <- above threshold" if share > CONTAMINATION_LIMIT else ""
            print(f"  {uid:<34} {used}/{tot} = {100 * share:4.1f}%{mark}")
        if not material:
            print(f"  all below {100 * CONTAMINATION_LIMIT:.0f}% — incidental, "
                  f"not a design problem")

    rule("VERDICT")
    problems = []
    if short:
        problems.append(f"{len(short)} variant(s) below {MIN_COMPARATORS} comparators")
    if weak:
        problems.append(f"{len(weak)} variant(s) matched only at {WEAK_TIERS[0]} "
                        f"or looser")
    if material:
        problems.append(f"{len(material)} variant(s) draw over "
                        f"{100 * CONTAMINATION_LIMIT:.0f}% of their background "
                        f"from other ClinVar variants")
    if problems:
        print("CAUTION -- resolve before spending a GPU:")
        for p in problems:
            print(f"  - {p}")
        print("\nDecide what to do about these BEFORE scoring, not after seeing")
        print("which way the result falls.")
    else:
        print(f"GO. All {n_cv} variants find at least {MIN_COMPARATORS} comparators,")
        print("each at a tier that still holds substitution class fixed, and none")
        print("draws a material share of its background from other ClinVar")
        print("variants. Proceed to scoring.")

    total = len(pool) + n_cv
    print(f"\nGPU cost: {total} model-visible variants x 4 passes x 3 seeds")
    print(f"  ~{total * 12 / 50 / 60:.0f} min on one V100 at ~50 seq/s")
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
