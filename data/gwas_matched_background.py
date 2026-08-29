#!/usr/bin/env python
"""GWAS risk-variant matched-background test: build the scoring input, then analyse.

Implements results/journal/gwas_matched_background/preregistration.json. Read
that first; nothing here may deviate from it.

    python -u data/gwas_matched_background.py --build
    sbatch scripts/run_gwas_matched_background.sh
    python -u data/gwas_matched_background.py --analyse

Two things differ from the ClinVar version, both because that analysis got them
wrong:

  1. The secondary statistic CLUSTERS BY PROBE. Variants at the same CpG share
     the window and the context features; treating them as independent is what
     turned a null ClinVar result into an apparent z = -3.30.
  2. The verdict has a branch for "significantly below 50". The ClinVar code
     only tested for enrichment and mislabelled depletion as inconclusive.

The pre-specified context check runs before the verdict and is reported whatever
it shows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = Path("results/journal/gwas_matched_background")
COHORT = OUT_DIR / "gwas_cohort.csv"
POOL = Path("results/journal/clinvar_matched_background/synthetic_background_pool.csv")
UNION = OUT_DIR / "gwas_plus_background_scoring_input.csv"
COHORT_IDS = OUT_DIR / "gwas_cohort_ids.json"
SCORED = OUT_DIR / "scored" / "candidate_matched_background_statistics.csv"

NULL_TAIL_RATE = 0.05
TAIL_ALPHA = 0.05
NULL_PERCENTILE = 50.0
PRIMARY_HIT_BAR = 7          # pre-registered at n = 46
BOOTSTRAP = 10000

TAIL_COL = "Matched_Empirical_Tail_Probability"
PCT_COL = "Matched_Background_Absolute_Effect_Percentile"

FEATURES = [
    "Ref_ATAC_Signal", "Ref_H3K4me3_Signal", "Ref_H3K27ac_Signal",
    "Ref_H3K27me3_Signal", "Ref_H3K9me3_Signal", "Ref_H3K36me3_Signal",
    "Ref_H3K4me1_Signal", "Target_Base_PhyloP_100way_1",
    "Target_Base_PhyloP_100way_2",
]

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
    p.add_argument("--scored", default=None)
    return p.parse_args()


def cmd_build() -> int:
    for p in (COHORT, POOL):
        if not p.exists():
            print(f"MISSING: {p}")
            return 1
    cv = pd.read_csv(COHORT, low_memory=False)
    pool = pd.read_csv(POOL, low_memory=False)
    print(f"cohort {len(cv)} rows across {cv['probeID'].nunique()} probes")
    print(f"pool   {len(pool)} rows")

    dropped = [c for c in SCRIPT05_GENERATED if c in cv.columns or c in pool.columns]
    if dropped:
        print(f"dropping {len(dropped)} column(s) scripts/05 regenerates: {dropped}")
        cv = cv.drop(columns=[c for c in dropped if c in cv.columns])
        pool = pool.drop(columns=[c for c in dropped if c in pool.columns])

    overlap = set(cv["Candidate_ID"]) & set(pool.get("Candidate_ID", pd.Series(dtype=str)))
    if overlap:
        print(f"removing {len(overlap)} pool rows sharing a Candidate_ID with the cohort")
        pool = pool[~pool["Candidate_ID"].isin(overlap)]

    union = pd.concat([cv, pool], ignore_index=True, sort=False)
    union["Model_Split"] = "test"
    UNION.parent.mkdir(parents=True, exist_ok=True)
    union.to_csv(UNION, index=False)
    COHORT_IDS.write_text(json.dumps({
        "candidate_ids": cv["Candidate_ID"].astype(str).tolist(),
        "cohort_rows": int(len(cv)),
        "distinct_probes": int(cv["probeID"].nunique()),
        "pool_rows": int(len(pool)),
    }, indent=2) + "\n")
    print(f"\nwrote {UNION} ({len(union)} rows, {UNION.stat().st_size/1e6:.0f} MB)")
    print(f"wrote {COHORT_IDS}")
    print("\nnext: sbatch scripts/run_gwas_matched_background.sh")
    return 0


def clustered_mean(values: np.ndarray, groups: np.ndarray, rng) -> dict:
    """Probe-level mean with a cluster bootstrap. Resamples PROBES, not variants."""
    df = pd.DataFrame({"v": values, "g": groups})
    per = df.groupby("g")["v"].mean()
    n = len(per)
    boot = np.array([rng.choice(per.to_numpy(), n, replace=True).mean()
                     for _ in range(BOOTSTRAP)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    se = float(per.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t = (per.mean() - NULL_PERCENTILE) / se if se and np.isfinite(se) else np.nan
    p = float(2 * stats.t.sf(abs(t), n - 1)) if np.isfinite(t) else float("nan")
    return {"n_clusters": int(n), "mean": float(per.mean()),
            "ci": [float(lo), float(hi)], "se": se,
            "t": float(t) if np.isfinite(t) else None,
            "p": p if np.isfinite(p) else None}


def cmd_analyse(scored_path: str | None) -> int:
    if not COHORT_IDS.exists():
        print(f"MISSING: {COHORT_IDS}. Run --build first.")
        return 1
    ids = json.loads(COHORT_IDS.read_text())
    path = Path(scored_path) if scored_path else SCORED
    if not path.exists():
        print(f"MISSING: {path}")
        return 1

    s = pd.read_csv(path, low_memory=False)
    for col in (TAIL_COL, PCT_COL, "Candidate_ID", "probeID"):
        if col not in s.columns:
            print(f"STOP: scored file lacks {col}")
            return 1
    mask = s["Candidate_ID"].astype(str).isin(set(ids["candidate_ids"]))
    cohort, background = s[mask], s[~mask]
    print(f"scored rows {len(s)}; cohort identified {len(cohort)} of "
          f"{ids['cohort_rows']}, across {cohort['probeID'].nunique()} probes")
    if cohort.empty:
        print("STOP: no cohort rows found in the scored output.")
        return 1

    if "Matched_Background_Tier" in s.columns:
        print(f"match tiers: {cohort['Matched_Background_Tier'].value_counts().to_dict()}")
    if "Matched_Comparator_Count" in s.columns:
        c = pd.to_numeric(cohort["Matched_Comparator_Count"], errors="coerce")
        print(f"comparators: min {c.min():.0f} median {c.median():.0f} max {c.max():.0f}")

    # ---- pre-specified context check, BEFORE the verdict ---------------------
    print("\n" + "=" * 70)
    print("PRE-SPECIFIED CONTEXT CHECK (run before interpreting anything)")
    print("=" * 70)
    context = {}
    material = []
    for f in FEATURES:
        if f not in s.columns:
            continue
        a = pd.to_numeric(cohort[f], errors="coerce").dropna()
        b = pd.to_numeric(background[f], errors="coerce").dropna()
        if a.empty or b.empty:
            continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d = float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0
        context[f] = {"cohort_mean": float(a.mean()), "background_mean": float(b.mean()),
                      "cohens_d": d}
        flag = "  <- material" if abs(d) >= 0.5 else ""
        print(f"  {f:<30} {a.mean():+8.3f} vs {b.mean():+8.3f}   d={d:+.2f}{flag}")
        if abs(d) >= 0.5:
            material.append(f)
    print(f"\nfeatures differing by |d| >= 0.5: {len(material)}"
          + (f" -> {material}" if material else ""))

    # ---- statistics ---------------------------------------------------------
    rng = np.random.default_rng(42)
    tails = pd.to_numeric(cohort[TAIL_COL], errors="coerce").to_numpy(float)
    pcts = pd.to_numeric(cohort[PCT_COL], errors="coerce").to_numpy(float)
    n = len(cohort)
    hits = int(np.sum(tails < TAIL_ALPHA))
    binom = stats.binomtest(hits, n, NULL_TAIL_RATE, alternative="two-sided")
    variant_level = float(np.mean(pcts))
    clus = clustered_mean(pcts, cohort["probeID"].to_numpy(), rng)

    print("\n" + "=" * 70)
    print("PRE-REGISTERED RESULT")
    print("=" * 70)
    print(f"PRIMARY   tail probability < {TAIL_ALPHA}: {hits} of {n}"
          f"   expected {NULL_TAIL_RATE * n:.1f}   binomial p = {binom.pvalue:.4f}"
          f"   (bar: {PRIMARY_HIT_BAR}+ at p < 0.01)")
    print(f"SECONDARY mean percentile, clustered by probe: {clus['mean']:.1f} "
          f"[{clus['ci'][0]:.1f}, {clus['ci'][1]:.1f}]   "
          f"n_probes = {clus['n_clusters']}")
    if clus["t"] is not None:
        print(f"          t = {clus['t']:+.2f}, p = {clus['p']:.3f}  (null = 50)")
    print(f"          variant-level mean, NOT the inferential number: {variant_level:.1f}")

    # ---- verdict, rules fixed in the pre-registration ------------------------
    sig_below = clus["p"] is not None and clus["p"] < 0.05 and clus["mean"] < NULL_PERCENTILE
    sig_above = clus["p"] is not None and clus["p"] < 0.05 and clus["mean"] > NULL_PERCENTILE
    if hits >= PRIMARY_HIT_BAR and binom.pvalue < 0.01 and clus["mean"] > NULL_PERCENTILE:
        verdict = ("POSITIVE. Risk variants are enriched in the upper tail of their "
                   "matched backgrounds.")
    elif binom.pvalue < 0.05:
        verdict = ("SUGGESTIVE. Below the pre-declared bar of "
                   f"{PRIMARY_HIT_BAR} hits at p < 0.01. Report as suggestive, "
                   "explicitly not as positive.")
    elif sig_below:
        verdict = ("DEPLETION. The clustered mean percentile is significantly BELOW "
                   "50. This is a result, not an inconclusive one.")
    elif sig_above:
        verdict = ("ELEVATED on the secondary statistic only; the primary count "
                   "statistic did not clear its bar. Report both.")
    else:
        verdict = ("NULL. Consistent with no enrichment. Per the declared power "
                   "statement, this constrains the effect to roughly 8 percentile "
                   "points or less; it does not establish absence.")
    if material:
        verdict += (f" CONTEXT CAVEAT: {len(material)} feature(s) differ materially "
                    f"between cohort and background, so the raw percentile is not "
                    f"interpretable on its own.")
    print("\n" + "=" * 70)
    print("VERDICT (rules fixed in the pre-registration)")
    print("=" * 70)
    print(verdict)

    (OUT_DIR / "gwas_matched_background_result.json").write_text(json.dumps({
        "scored_file": str(path), "n_variants": n,
        "n_probes": clus["n_clusters"], "hits": hits,
        "binomial_p": float(binom.pvalue),
        "clustered_percentile": clus, "variant_level_mean": variant_level,
        "context_check": context, "material_context_differences": material,
        "verdict": verdict,
    }, indent=2) + "\n")
    print(f"\nwrote {OUT_DIR / 'gwas_matched_background_result.json'}")
    return 0


def main() -> int:
    args = parse_args()
    if args.build == args.analyse:
        print("choose exactly one of --build or --analyse")
        return 1
    return cmd_build() if args.build else cmd_analyse(args.scored)


if __name__ == "__main__":
    raise SystemExit(main())
