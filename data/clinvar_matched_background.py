#!/usr/bin/env python
"""ClinVar matched-background test: build the scoring input, then analyse it.

Do held-out ClinVar-pathogenic variants in breast-cancer genes produce larger
predicted methylation effects than substitution- and distance-matched SNVs at
the same class of locus?

    python -u data/clinvar_matched_background.py --build
    sbatch scripts/run_clinvar_matched_background.sh
    python -u data/clinvar_matched_background.py --analyse

PRE-REGISTERED, declared before any score existed (preregistration.json):

  * primary cohort   : all 35 held-out non-truncating ClinVar variants
  * secondary cohort : the 16 with unambiguous ClinVar calls
  * primary statistic: count with matched-background tail probability < 0.05,
    against a binomial null of 0.05. At n = 35 that expects 1.75; 6 or more
    gives p < 0.01
  * secondary        : mean matched-background percentile against a null of 50
  * two-sided throughout
  * POWER, STATED IN ADVANCE: at n = 35 a mean percentile of 65 is about 3 SE
    from the null and detectable; 58 is not. A null result CANNOT distinguish
    "no enrichment" from "underpowered" and must be reported as inconclusive
    rather than negative.

--build emits one CSV containing the 35 ClinVar variants followed by the
synthetic background pool, because scripts/05 draws each variant's comparators
from the frame it is given. Feeding it the 35 alone made every variant's
background the other 34 pathogenic variants, which is what invalidated the
first attempt.

--analyse restricts to the 35 by Variant_UID before computing anything. The
pool is scored so it can serve as background; it is not part of the cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path("results/journal/clinvar_matched_background")
CLINVAR = OUT_DIR / "clinvar_heldout_nontruncating_cohort.csv"
POOL = OUT_DIR / "synthetic_background_pool.csv"
UNION = OUT_DIR / "clinvar_plus_background_scoring_input.csv"
COHORT_IDS = OUT_DIR / "clinvar_cohort_ids.json"

NULL_TAIL_RATE = 0.05
TAIL_ALPHA = 0.05
NULL_PERCENTILE = 50.0
BOOTSTRAP = 10000

TAIL_COL = "Matched_Empirical_Tail_Probability"
PCT_COL = "Matched_Background_Absolute_Effect_Percentile"

CLEAN_CALLS = {"Pathogenic", "Pathogenic/Likely pathogenic", "Likely pathogenic"}

# scripts/05 regenerates these and merges them back on Variant_UID. Any that we
# also carry collide into _x/_y and the plain name vanishes -- that is what
# produced KeyError: 'HM450_MASK_general' and then KeyError:
# 'Predicted_Delta_Beta' on the first attempt. Drop the whole set rather than
# fixing them one crash at a time, and let scripts/05 own its own outputs.
SCRIPT05_GENERATED = (
    "HM450_MASK_general", "Seed_Count", "Seeds",
    "Predicted_Delta_Beta", "Predicted_Delta_Beta_Mean", "Predicted_Delta_Beta_SD",
    "Predicted_Delta_Beta_Median", "Predicted_Delta_Beta_Min",
    "Predicted_Delta_Beta_Max", "Delta_Beta_Sign_Consistency",
    "Mean_Absolute_Delta_Beta", "Mean_Within_Seed_Rank", "SD_Within_Seed_Rank",
    "Best_Within_Seed_Rank", "Worst_Within_Seed_Rank", "Top10_Seed_Frequency",
    "Top20_Seed_Frequency", "Mean_Delta_RC_Absolute_Difference",
    "Delta_RC_Sign_Agreement_Fraction", "WT_Gate_DNA_Mean", "WT_Gate_EPI_Mean",
    "WT_Gate_DNA_Share_Mean", "MUT_Gate_DNA_Mean", "MUT_Gate_EPI_Mean",
    "MUT_Gate_DNA_Share_Mean", "Absolute_Delta_Beta", "Absolute_Delta_Beta_Rank",
    "Matched_Background_Tier", "Matched_Comparator_Count",
    "Matched_Comparator_Count_Below_Minimum", "Matched_Background_Mean",
    "Matched_Background_Median", "Matched_Background_STD", PCT_COL, TAIL_COL,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true")
    p.add_argument("--analyse", action="store_true")
    p.add_argument("--scored", default=None,
                   help="candidate_matched_background_statistics.csv from scripts/05")
    return p.parse_args()


def resolve(frame: pd.DataFrame, name: str) -> str:
    if name in frame.columns:
        return name
    near = [c for c in frame.columns if c.startswith(name)]
    raise SystemExit(f"scored file lacks {name}. Similar columns: {near[:5]}")


def cmd_build() -> int:
    for p in (CLINVAR, POOL):
        if not p.exists():
            print(f"MISSING: {p}")
            if p is POOL:
                print("run data/build_synthetic_background.py first")
            return 1

    cv = pd.read_csv(CLINVAR, low_memory=False)
    pool = pd.read_csv(POOL, low_memory=False)
    print(f"clinvar {len(cv)} rows, {len(cv.columns)} cols")
    print(f"pool    {len(pool)} rows, {len(pool.columns)} cols")

    dropped = [c for c in SCRIPT05_GENERATED if c in cv.columns or c in pool.columns]
    if dropped:
        print(f"dropping {len(dropped)} column(s) scripts/05 regenerates: {dropped}")
        cv = cv.drop(columns=[c for c in dropped if c in cv.columns])
        pool = pool.drop(columns=[c for c in dropped if c in pool.columns])

    # ClinVar rows first: --analyse identifies them by UID, not position, but a
    # stable order makes the emitted file easy to inspect.
    union = pd.concat([cv, pool], ignore_index=True, sort=False)
    union["Model_Split"] = "test"

    # Which rows are the cohort. The UIDs scripts/05 will derive are not present
    # yet, so record the identifying fields it uses and let --analyse match on
    # the derived UID via the ClinVar rows' own provenance columns.
    id_cols = [c for c in ("Candidate_ID", "GDC_Genomic_DNA_Change", "probeID")
               if c in cv.columns]
    if not id_cols:
        print("STOP: clinvar cohort has none of Candidate_ID / "
              "GDC_Genomic_DNA_Change / probeID; cannot identify it after scoring")
        return 1

    clean_col = next((c for c in cv.columns
                      if c in ("ClinVar_Clean_Call", "Clinical_Significance",
                               "ClinVar_Clinical_Significance")), None)
    if clean_col is None:
        print("WARNING: no ClinVar call column found; secondary cohort will be "
              "empty and only the primary statistic is reported")
        secondary_mask = pd.Series(False, index=cv.index)
    else:
        vals = cv[clean_col].astype(str).str.strip()
        secondary_mask = vals.isin(CLEAN_CALLS)
        print(f"secondary cohort from {clean_col}: {int(secondary_mask.sum())} of "
              f"{len(cv)}")
        print(f"  call counts: {vals.value_counts().to_dict()}")

    UNION.parent.mkdir(parents=True, exist_ok=True)
    union.to_csv(UNION, index=False)
    COHORT_IDS.write_text(json.dumps({
        "id_columns": id_cols,
        "primary": cv[id_cols].astype(str).to_dict(orient="records"),
        "secondary": cv.loc[secondary_mask, id_cols].astype(str)
                       .to_dict(orient="records"),
        "clinvar_rows": int(len(cv)),
        "pool_rows": int(len(pool)),
        "clean_call_column": clean_col,
    }, indent=2))

    print(f"\nwrote {UNION}  ({len(union)} rows, "
          f"{UNION.stat().st_size / 1e6:.0f} MB)")
    print(f"wrote {COHORT_IDS}")
    print("\nnext: sbatch scripts/run_clinvar_matched_background.sh")
    return 0


def identify(scored: pd.DataFrame, records: list[dict], id_cols: list[str]
             ) -> pd.Series:
    """Flag scored rows belonging to a recorded cohort.

    scripts/05 derives Variant_UID itself, so we match on the provenance columns
    it carried through rather than on a UID we guessed.
    """
    usable = [c for c in id_cols if c in scored.columns]
    if not usable:
        raise SystemExit(f"scored file lacks all of {id_cols}; cannot identify "
                         f"the cohort")
    keys = {tuple(str(r[c]) for c in usable) for r in records}
    have = scored[usable].astype(str).apply(tuple, axis=1)
    return have.isin(keys)


def summarise(frame: pd.DataFrame, label: str, tail_col: str, pct_col: str,
              rng: np.random.Generator) -> dict:
    n = len(frame)
    if n == 0:
        print(f"\n{label}: empty")
        return {"label": label, "n": 0}
    tails = pd.to_numeric(frame[tail_col], errors="coerce").to_numpy(float)
    pcts = pd.to_numeric(frame[pct_col], errors="coerce").to_numpy(float)
    hits = int(np.sum(tails < TAIL_ALPHA))
    binom = stats.binomtest(hits, n, NULL_TAIL_RATE, alternative="two-sided")
    boot = np.array([rng.choice(pcts, n, replace=True).mean()
                     for _ in range(BOOTSTRAP)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    se = float(boot.std(ddof=1))

    print(f"\n{label}  (n = {n})")
    print(f"  tail probability < {TAIL_ALPHA}: {hits}"
          f"   expected under null {NULL_TAIL_RATE * n:.2f}"
          f"   binomial p = {binom.pvalue:.4f}")
    print(f"  mean matched-background percentile: {pcts.mean():.1f} "
          f"[{lo:.1f}, {hi:.1f}]   (null = {NULL_PERCENTILE:.0f})")
    if se > 0:
        print(f"  z vs null: {(pcts.mean() - NULL_PERCENTILE) / se:+.2f}")
    print(f"  median percentile: {np.median(pcts):.1f}")
    return {
        "label": label, "n": n, "hits": hits,
        "expected_hits": NULL_TAIL_RATE * n,
        "binomial_p": float(binom.pvalue),
        "mean_percentile": float(pcts.mean()),
        "percentile_ci": [float(lo), float(hi)],
        "percentile_se": se,
        "median_percentile": float(np.median(pcts)),
    }


def cmd_analyse(scored_path: str | None) -> int:
    if not COHORT_IDS.exists():
        print(f"MISSING: {COHORT_IDS}. Run --build first.")
        return 1
    ids = json.loads(COHORT_IDS.read_text())

    path = Path(scored_path) if scored_path else (
        OUT_DIR / "scored" / "candidate_matched_background_statistics.csv")
    if not path.exists():
        print(f"MISSING: {path}")
        print("pass --scored with the statistics CSV scripts/05 wrote")
        return 1

    scored = pd.read_csv(path, low_memory=False)
    tail_col = resolve(scored, TAIL_COL)
    pct_col = resolve(scored, PCT_COL)
    print(f"scored rows: {len(scored)}   from {path}")

    primary_mask = identify(scored, ids["primary"], ids["id_columns"])
    secondary_mask = identify(scored, ids["secondary"], ids["id_columns"])
    primary = scored[primary_mask]
    secondary = scored[secondary_mask]

    print(f"identified {len(primary)} of {ids['clinvar_rows']} primary-cohort "
          f"variants in the scored output")
    if len(primary) != ids["clinvar_rows"]:
        print("  NOTE: some cohort variants were dropped by scripts/05's "
              "model-visibility validation; they are absent, not zero")

    if "Matched_Background_Tier" in scored.columns:
        print("\nmatch quality for the primary cohort:")
        print(f"  {primary['Matched_Background_Tier'].value_counts().to_dict()}")
    if "Matched_Comparator_Count" in scored.columns:
        c = pd.to_numeric(primary["Matched_Comparator_Count"], errors="coerce")
        print(f"  comparators: min {c.min():.0f}  median {c.median():.0f}  "
              f"max {c.max():.0f}")

    rng = np.random.default_rng(42)
    print("\n" + "=" * 72)
    print("PRE-REGISTERED RESULT")
    print("=" * 72)
    results = [summarise(primary, "PRIMARY  (all held-out non-truncating)",
                         tail_col, pct_col, rng),
               summarise(secondary, "SECONDARY (unambiguous ClinVar calls)",
                         tail_col, pct_col, rng)]

    print("\n" + "=" * 72)
    print("INTERPRETATION (rule fixed in advance, not chosen now)")
    print("=" * 72)
    p = results[0]
    if p["n"] == 0:
        verdict = "no primary-cohort variants scored"
    elif p["hits"] >= 6 and p["binomial_p"] < 0.01:
        verdict = ("POSITIVE. ClinVar-pathogenic variants are enriched in the "
                   "upper tail of their matched backgrounds.")
    elif p["binomial_p"] < 0.05:
        verdict = ("SUGGESTIVE. Below the pre-declared bar of 6 hits at "
                   "p < 0.01; report as such, not as a positive result.")
    else:
        verdict = ("INCONCLUSIVE, not negative. The declared power statement "
                   "says n = 35 cannot distinguish absence of enrichment from "
                   "insufficient power. Report it that way.")
    print(verdict)

    (OUT_DIR / "clinvar_matched_background_result.json").write_text(json.dumps({
        "scored_file": str(path),
        "results": results,
        "verdict": verdict,
        "preregistered": {
            "null_tail_rate": NULL_TAIL_RATE,
            "tail_alpha": TAIL_ALPHA,
            "primary_bar": "6 or more hits, binomial p < 0.01",
            "power_statement": (
                "at n = 35 a mean percentile of 65 is ~3 SE from the null and "
                "detectable; 58 is not. A null result cannot distinguish no "
                "enrichment from underpowered."),
        },
    }, indent=2))
    print(f"\nwrote {OUT_DIR / 'clinvar_matched_background_result.json'}")
    return 0


def main() -> int:
    args = parse_args()
    if args.build == args.analyse:
        print("choose exactly one of --build or --analyse")
        return 1
    return cmd_build() if args.build else cmd_analyse(args.scored)


if __name__ == "__main__":
    raise SystemExit(main())
