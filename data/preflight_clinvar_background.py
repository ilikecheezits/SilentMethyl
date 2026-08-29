#!/usr/bin/env python
"""Pre-flight for the ClinVar matched-background test. Reads only; runs nothing.

The test compares each held-out ClinVar-pathogenic variant against comparators
matched on substitution class, CpG effect and distance to the target CpG. Those
comparators are drawn from whatever frame is handed to scripts/05 -- which is
why feeding it the 35 ClinVar variants alone was wrong: each variant's
"background" was the other 34 pathogenic variants.

The fix is to score the 35 inside a pool of ordinary held-out variants. Before
spending a GPU on that, this answers the three questions that decide whether the
design works at all:

  1. Do the two files share a compatible schema, so they can be concatenated
     and pass scripts/05's validate_candidate_table?
  2. Do the two sets overlap on Variant_UID? (scripts/05 drops duplicates, and a
     ClinVar variant already in the pool would be its own comparator.)
  3. Does every ClinVar variant actually find >= min_comparators matches in the
     pool, and at which tier?

Question 3 is the one that matters. If a variant falls through to the loosest
tier with few matches, its tail probability is computed against almost nothing
and the pre-registered statistic is meaningless for that variant.

No torch, no model, no GPU. The large DNA columns are never read.

    python -u data/preflight_clinvar_background.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

CLINVAR = Path("results/journal/clinvar_matched_background/"
               "clinvar_heldout_nontruncating_cohort.csv")
POOL = Path("data/datafiles/testing_data_test_only.csv")

# Keys the matching tiers use, plus identity. Deliberately NOT the 5,000 bp
# sequence columns -- reading those would take minutes and we do not need them.
MATCH_KEYS = [
    "Variant_UID", "probeID", "Gene", "Model_Split",
    "Canonical_SBS6", "Canonical_SBS96", "Transition_Transversion",
    "CpG_Effect", "Absolute_Distance_From_Target_CpG",
]

TABULAR_FEATURES = [
    "Ref_ATAC_Signal", "Ref_H3K4me3_Signal", "Ref_H3K27ac_Signal",
    "Ref_H3K27me3_Signal", "Ref_H3K9me3_Signal", "Ref_H3K36me3_Signal",
    "Ref_H3K4me1_Signal", "Target_Base_PhyloP_100way_1",
    "Target_Base_PhyloP_100way_2",
]
MISSING_FEATURES = [f"{n}_Missing" for n in TABULAR_FEATURES]
SCRIPT05_REQUIRED = ["probeID", "Healthy_5000bp_DNA", "Mutated_5000bp_DNA",
                     "Model_Split", *TABULAR_FEATURES, *MISSING_FEATURES]

MIN_COMPARATORS = 20


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def header_of(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def main() -> int:
    for p in (CLINVAR, POOL):
        if not p.exists():
            print(f"MISSING: {p}")
            return 1

    rule("1. SCHEMA")
    cv_cols, pool_cols = header_of(CLINVAR), header_of(POOL)
    print(f"clinvar cohort : {len(cv_cols)} columns   {CLINVAR}")
    print(f"background pool: {len(pool_cols)} columns   {POOL}"
          f"   ({POOL.stat().st_size / 1e9:.2f} GB on disk)")

    ok = True
    for name, cols in (("clinvar", cv_cols), ("pool", pool_cols)):
        missing_05 = [c for c in SCRIPT05_REQUIRED if c not in cols]
        missing_mb = [c for c in MATCH_KEYS if c not in cols]
        if missing_05 or missing_mb:
            ok = False
            print(f"  {name}: MISSING for scripts/05 {missing_05}")
            print(f"  {name}: MISSING for matching   {missing_mb}")
        else:
            print(f"  {name}: has every column scripts/05 and the matcher need")

    shared = [c for c in cv_cols if c in pool_cols]
    print(f"  shared columns: {len(shared)}   "
          f"clinvar-only: {len(set(cv_cols) - set(pool_cols))}   "
          f"pool-only: {len(set(pool_cols) - set(cv_cols))}")
    if not ok:
        print("\nSTOP: schema incompatible. Do not proceed to scoring.")
        return 1

    rule("2. COHORTS")
    cv = pd.read_csv(CLINVAR, usecols=lambda c: c in MATCH_KEYS)
    pool = pd.read_csv(POOL, usecols=lambda c: c in MATCH_KEYS)
    print(f"clinvar rows : {len(cv)}")
    print(f"pool rows    : {len(pool)}")
    for name, df in (("clinvar", cv), ("pool", pool)):
        vc = df["Model_Split"].astype(str).value_counts().to_dict()
        print(f"  {name} Model_Split: {vc}")
        if set(vc) != {"test"}:
            print(f"  WARNING: {name} is not held-out only; scripts/05 will refuse it")

    overlap = set(cv["Variant_UID"].astype(str)) & set(pool["Variant_UID"].astype(str))
    print(f"  Variant_UID overlap: {len(overlap)}")
    if overlap:
        print(f"    e.g. {sorted(overlap)[:5]}")
        print("    these must be dropped from the pool so a ClinVar variant is "
              "never its own comparator")

    rule("3. COMPARATOR AVAILABILITY  <- the question that decides the design")
    try:
        from matched_background_utils import choose_matched_comparators
    except Exception as exc:
        print(f"could not import matched_background_utils: {type(exc).__name__}: {exc}")
        print("run this from the repository root")
        return 1

    pool_clean = pool[~pool["Variant_UID"].astype(str).isin(overlap)].copy()
    union = pd.concat([cv, pool_clean], ignore_index=True)
    union["Predicted_Delta_Beta"] = 0.0          # matcher never reads it
    n_cv = len(cv)

    # Two failure modes, not one. A variant can have plenty of comparators and
    # still be badly matched:
    #   - it fell through to a tier that no longer constrains substitution class
    #     (T7 drops SBS entirely, T8 matches everything), so "matched" is a
    #     misnomer and the tail probability tests almost nothing;
    #   - its comparators are other ClinVar variants rather than ordinary ones,
    #     which is the flaw that sank the first attempt, in miniature.
    # Counting comparators alone reports GO in both cases. It should not.
    WEAK_TIERS = ("T7_CpG_distance250", "T8_all_observed_synonymous")

    tiers: dict[str, int] = {}
    counts: list[int] = []
    short: list[tuple[str, str, int]] = []
    weak: list[tuple[str, str, int]] = []
    contaminated: list[tuple[str, int, int]] = []
    for i in range(n_cv):
        idx, tier, _ = choose_matched_comparators(
            union, i, min_comparators=MIN_COMPARATORS, max_comparators=1000)
        idx = idx[idx != i]                       # never its own comparator
        n_from_clinvar = int((idx < n_cv).sum())  # union puts ClinVar rows first
        uid = str(union.iloc[i]["Variant_UID"])
        tiers[tier] = tiers.get(tier, 0) + 1
        counts.append(len(idx))
        if len(idx) < MIN_COMPARATORS:
            short.append((uid, tier, len(idx)))
        if tier in WEAK_TIERS:
            weak.append((uid, tier, len(idx)))
        if n_from_clinvar:
            contaminated.append((uid, n_from_clinvar, len(idx)))

    counts_arr = np.array(counts)
    print(f"variants checked       : {n_cv}")
    print(f"comparators per variant: min {counts_arr.min()}  "
          f"median {int(np.median(counts_arr))}  max {counts_arr.max()}")
    print(f"variants with >= {MIN_COMPARATORS}   : "
          f"{int((counts_arr >= MIN_COMPARATORS).sum())} / {n_cv}")
    print("\ntier reached (T1 strictest, T8 matches everything):")
    for tier in sorted(tiers):
        flag = "   <- constraint effectively dropped" if tier in WEAK_TIERS else ""
        print(f"  {tier:<36} {tiers[tier]}{flag}")

    if short:
        print(f"\n{len(short)} variant(s) below the {MIN_COMPARATORS} minimum:")
        for uid, tier, n in short[:10]:
            print(f"  {uid:<30} {tier:<34} {n}")
    # With 35 ClinVar variants inside a pool of thousands, a few will match each
    # other by chance. That is not the failure mode we care about -- one ClinVar
    # comparator among thirty barely moves a percentile. What matters is a
    # variant whose background is SUBSTANTIALLY made of other pathogenic
    # variants, which is the original flaw at smaller scale. Threshold declared
    # here, before any result exists.
    CONTAMINATION_LIMIT = 0.10

    shares = [(uid, n_used, n_tot, n_used / n_tot if n_tot else 1.0)
              for uid, n_used, n_tot in contaminated]
    material = [s for s in shares if s[3] > CONTAMINATION_LIMIT]
    if contaminated:
        worst = max(s[3] for s in shares)
        print(f"\n{len(contaminated)}/{n_cv} variant(s) draw at least one comparator "
              f"from the ClinVar set itself; worst share {100 * worst:.1f}%")
        for uid, n_used, n_tot, share in sorted(shares, key=lambda s: -s[3])[:10]:
            mark = "  <- above threshold" if share > CONTAMINATION_LIMIT else ""
            print(f"  {uid:<30} {n_used}/{n_tot} = {100 * share:4.1f}%{mark}")
        if not material:
            print(f"  all below the {100 * CONTAMINATION_LIMIT:.0f}% threshold — "
                  f"incidental, not a design problem")

    rule("VERDICT")
    problems = []
    if short:
        problems.append(f"{len(short)} variant(s) below {MIN_COMPARATORS} comparators")
    if weak:
        problems.append(f"{len(weak)} variant(s) matched only at {WEAK_TIERS[0]} or "
                        f"looser, where substitution class is no longer held fixed")
    if material:
        problems.append(f"{len(material)} variant(s) draw more than "
                        f"{100 * CONTAMINATION_LIMIT:.0f}% of their background from "
                        f"other ClinVar variants")

    if not problems:
        print(f"GO. All {n_cv} variants find at least {MIN_COMPARATORS} comparators,")
        print("every one at a tier that still constrains substitution class, and")
        print("none draws a material share of its background from other ClinVar")
        print("variants. The design is sound. Proceed to scoring.")
    else:
        print("CAUTION -- resolve before spending a GPU:")
        for pr in problems:
            print(f"  - {pr}")
        print("\nDecide what to do about these BEFORE scoring, not after seeing")
        print("which way the result falls.")
    print(f"\nGPU estimate: pool {len(pool_clean)} + clinvar {n_cv} = "
          f"{len(pool_clean) + n_cv} variants x 4 passes x 3 seeds")
    print(f"  at ~50 seq/s on a V100: "
          f"~{(len(pool_clean) + n_cv) * 12 / 50 / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
