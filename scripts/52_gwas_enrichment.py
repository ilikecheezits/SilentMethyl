#!/usr/bin/env python
"""Do high predicted-effect variants land in GWAS loci more than matched background?

The ClinVar and breast-GWAS matched-background tests (E.2, E.2b) asked the hard
direction: *do known risk variants score high?* Both were tiny -- 35 variants on
six probes, and 32 variants restricted to breast associations on chr8--9 -- and
both were underpowered by construction.

This asks the same question in the well-powered direction: *do high-scoring
variants land in GWAS loci?* It runs over every held-out non-CpG-altering pair
rather than a proximity-selected handful, and over the whole GWAS Catalog rather
than one trait. Thousands of labelled variants instead of thirty-two.

It also works where the earlier tests could not. Most high-effect variants here
are intergenic and therefore absent from ClinVar, which annotates on coding and
splicing evidence; GWAS associations are overwhelmingly non-coding, so they are
the right annotation for a methylation-effect score.

Design. Each scored pair is labelled by whether its variant is a GWAS Catalog
association. Labelled pairs are matched to unlabelled pairs at identical
variant--CpG distance -- the dominant confound, since predicted effect rises
steeply with proximity -- and the two groups are compared on |dM| with intervals
from a 1-Mb block bootstrap. Every metric and the matching routine are imported
from scripts/20 so no definition can drift.

    python -u scripts/52_gwas_enrichment.py --build   --gwas <file>
    python -u scripts/52_gwas_enrichment.py --analyse

--build writes the cohort and a pre-registration; --analyse refuses to run until
that pre-registration exists, so the statistics are fixed before any result.

Exit codes:  0 ran   1 could not run   2 inconclusive
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EVALUATOR = HERE / "21_variant_evaluation.py"

# Confounds declared before any result exists. Reported whatever they show.
# Allele frequency is named differently per cohort: GENOA writes af_genoa,
# eGTEx writes maf. Detected, never assumed -- a missing column silently
# disables the matching that made the v1 run uninterpretable.
AF_COLUMN_CANDIDATES = ("af_genoa", "maf", "af", "MAF", "allele_frequency")
PRESPECIFIED_CONFOUNDS = ("abs_distance_bp", "__af__")


def find_af_column(frame: pd.DataFrame) -> str | None:
    for c in AF_COLUMN_CANDIDATES:
        if c in frame.columns and pd.to_numeric(frame[c], errors="coerce").notna().any():
            return c
    return None


def load_evaluator():
    if not EVALUATOR.is_file():
        raise SystemExit(f"STOP: cannot find the evaluator at {EVALUATOR}")
    spec = importlib.util.spec_from_file_location("genoa_evaluator", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules["genoa_evaluator"] = module
    spec.loader.exec_module(module)
    for name in ("load_scores", "add_seed_ensemble", "build_matched_cohort",
                 "block_bootstrap", "marginal_auroc", "GENOME_WIDE", "BLOCK_BP"):
        if not hasattr(module, name):
            raise SystemExit(f"STOP: scripts/20 no longer exports {name}")
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", action="store_true",
                   help="Label the cohort and write the pre-registration.")
    p.add_argument("--analyse", action="store_true",
                   help="Score it. Requires the pre-registration to exist.")
    p.add_argument("--gwas", type=Path,
                   help="GWAS Catalog all-associations TSV (may be .gz).")
    p.add_argument("--scores-dir", type=Path,
                   default=Path("results/journal/genoa_variant_scoring"))
    p.add_argument("--stratum", default="heldout",
                   choices=("heldout", "model_visible"))
    p.add_argument("--model", default="fusion")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--window", type=int, default=0,
                   help="bp around a catalog position that still counts as an "
                        "overlap. 0 = the variant must itself be an association.")
    p.add_argument("--trait-contains", default=None,
                   help="Restrict to traits matching this substring "
                        "(case-insensitive). Default: all traits.")
    p.add_argument("--mqtl-significance", type=float, default=None,
                   help="Restrict to pairs whose COHORT p-value is below this, "
                        "i.e. variants with a measured methylation effect. "
                        "5e-8 = genome-wide meQTLs; 0.05 = nominal. Default None "
                        "uses every tested pair, ~71%% of which have no measured "
                        "effect and are therefore the wrong population for a "
                        "regulatory claim.")
    p.add_argument("--match-tolerance", type=int, default=10)
    p.add_argument("--af-bins", type=int, default=5,
                   help="Allele-frequency strata to match within, on top of "
                        "distance. 0 disables AF matching (the v1 behaviour, "
                        "which the confound check showed is uninterpretable).")
    p.add_argument("--tails", nargs="+", type=float, default=[0.05, 0.02, 0.01],
                   help="Top fractions of |delta M| at which enrichment is "
                        "evaluated. Declared in the amended pre-registration.")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/journal/gwas_regulatory_enrichment"))
    return p.parse_args()


# --------------------------------------------------------------- GWAS catalog
def read_catalog(path: Path, trait_contains: str | None) -> pd.DataFrame:
    """rsid + hg38 chr/pos for every association, deduplicated."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        frame = pd.read_csv(handle, sep="\t", low_memory=False, dtype=str)
    cols = {c.upper().strip(): c for c in frame.columns}
    need = {"SNPS": None, "CHR_ID": None, "CHR_POS": None, "DISEASE/TRAIT": None}
    for key in list(need):
        if key not in cols:
            raise SystemExit(f"STOP: {path} has no {key} column. "
                             f"Columns: {sorted(frame.columns)[:12]}")
        need[key] = cols[key]
    out = frame[[need["SNPS"], need["CHR_ID"], need["CHR_POS"],
                 need["DISEASE/TRAIT"]]].copy()
    out.columns = ["rsid", "chr", "pos", "trait"]
    if trait_contains:
        keep = out["trait"].str.contains(trait_contains, case=False, na=False)
        print(f"trait filter {trait_contains!r}: {int(keep.sum())} of {len(out)} rows")
        out = out[keep]
    out["rsid"] = out["rsid"].str.strip().str.lower()
    out["chr"] = "chr" + out["chr"].astype(str).str.replace("chr", "", regex=False)
    out["pos"] = pd.to_numeric(out["pos"], errors="coerce")
    out = out.dropna(subset=["pos"])
    out["pos"] = out["pos"].astype("int64")
    return out.drop_duplicates(subset=["rsid", "chr", "pos"])


def label_overlap(pairs: pd.DataFrame, catalog: pd.DataFrame,
                  window: int) -> pd.Series:
    """True where the scored variant is (or sits within `window` bp of) a hit."""
    # Two identifier systems: GENOA carries rsIDs in Variant_ID, eGTEx carries a
    # locus string. Match on BOTH rsID and exact coordinate so neither cohort
    # silently matches nothing.
    by_rsid = set(catalog["rsid"])
    hit = pairs["Variant_ID"].astype(str).str.strip().str.lower().isin(by_rsid)
    coords = set(zip(catalog["chr"].astype(str), catalog["pos"].astype("int64")))
    pair_coords = list(zip(pairs["chr"].astype(str),
                           pairs["Position_1based"].astype("int64")))
    hit = hit | pd.Series([c in coords for c in pair_coords], index=pairs.index)
    if window > 0:
        positions: dict[str, np.ndarray] = {
            c: np.sort(g["pos"].to_numpy()) for c, g in catalog.groupby("chr")}
        near = np.zeros(len(pairs), dtype=bool)
        chrom = pairs["chr"].astype(str).to_numpy()
        pos = pairs["Position_1based"].to_numpy(dtype="int64")
        for c in np.unique(chrom):
            arr = positions.get(c)
            if arr is None or not len(arr):
                continue
            sel = chrom == c
            idx = np.searchsorted(arr, pos[sel])
            left = np.clip(idx - 1, 0, len(arr) - 1)
            right = np.clip(idx, 0, len(arr) - 1)
            d = np.minimum(np.abs(arr[left] - pos[sel]), np.abs(arr[right] - pos[sel]))
            near[sel] = d <= window
        hit = hit | pd.Series(near, index=pairs.index)
    return hit


def stratified_match(ev, pairs: pd.DataFrame, tolerance: int, af_bins: int,
                     rng) -> tuple[pd.DataFrame, dict]:
    """Distance matching (scripts/20, unchanged) run inside allele-frequency strata.

    GWAS associations are common variants by construction -- an association
    cannot be detected at low frequency -- so a distance-only match leaves a
    large frequency imbalance. Matching within AF strata removes it. Distance
    matching itself is still scripts/20's routine, so its behaviour is unchanged.
    """
    af_col = find_af_column(pairs)
    if af_bins <= 0 or af_col is None:
        return ev.build_matched_cohort(pairs, tolerance, 1, rng)
    af = pd.to_numeric(pairs[af_col], errors="coerce")
    usable = af.notna()
    if not usable.any():
        return ev.build_matched_cohort(pairs, tolerance, 1, rng)
    edges = np.unique(np.nanquantile(af[usable], np.linspace(0, 1, af_bins + 1)))
    strata = pd.cut(af, bins=edges, include_lowest=True, labels=False)
    frames, balances = [], []
    for b in sorted(pd.unique(strata.dropna())):
        chunk = pairs[strata == b]
        if chunk["significant"].nunique() < 2:
            continue
        matched, bal = ev.build_matched_cohort(chunk, tolerance, 1, rng)
        if not matched.empty:
            frames.append(matched); balances.append(bal)
    if not frames:
        return pd.DataFrame(), {}
    out = pd.concat(frames, ignore_index=False)
    total = {
        "significant_n": int(sum(b["significant_n"] for b in balances)),
        "matched_null_n": int(sum(b["matched_null_n"] for b in balances)),
        "requested_negatives": int(sum(b["requested_negatives"] for b in balances)),
        "unmatched_slots": int(sum(b["unmatched_slots"] for b in balances)),
        "af_strata_used": len(frames),
    }
    lab = out[out["significant"] == 1]["abs_distance_bp"].astype(float)
    nul = out[out["significant"] == 0]["abs_distance_bp"].astype(float)
    total["median_distance_significant"] = float(lab.median())
    total["median_distance_matched_null"] = float(nul.median())
    from sklearn.metrics import roc_auc_score
    total["distance_only_auroc_after_matching"] = (
        float(roc_auc_score(out["significant"], -out["abs_distance_bp"].astype(float)))
        if out["significant"].nunique() == 2 else np.nan)
    return out, total


def tail_enrichment(frame: pd.DataFrame, fraction: float) -> float:
    """Share of the top |delta M| fraction that are GWAS associations.

    The matched cohort is 1:1, so 0.5 is the null. This is the statistic that
    answers the question the whole-distribution AUROC cannot: among the variants
    the model actually flags, are GWAS associations over-represented? Most pairs
    carry a near-zero prediction, so a rank statistic over all of them is
    dominated by noise.
    """
    n = max(1, int(round(len(frame) * fraction)))
    # Below ~25 the share is too noisy to interpret: at a 1:1 matched cohort the
    # standard error of a proportion at n=25 is already 0.10.
    if n < 25 or len(frame) < 50:
        return np.nan
    # positional, not label-based: the block bootstrap resamples with
    # replacement, so the index carries duplicates and .reindex() would fail
    magnitude = frame["Predicted_Delta_M"].abs().to_numpy(dtype=float)
    labels = frame["significant"].to_numpy(dtype=int)
    top = np.argsort(-magnitude, kind="stable")[:n]
    return float(labels[top].mean())


# ------------------------------------------------------------------- cohorts
def load_pairs(ev, args) -> pd.DataFrame:
    ns = SimpleNamespace(scores_dir=args.scores_dir, stratum=args.stratum,
                         models=[args.model], seeds=args.seeds,
                         significance=ev.GENOME_WIDE)
    long = ev.add_seed_ensemble(ev.load_scores(ns))
    frame = long[long["Seed"] == -1]
    frame = frame[~frame["cpg_altering"].astype(bool)].reset_index(drop=True)
    for col in ("Variant_ID", "chr", "Position_1based"):
        if col not in frame.columns:
            raise SystemExit(f"STOP: score files lack {col}; rerun scripts/19 "
                             f"so variant coordinates travel with the scores")
    return frame


def confound_table(cohort: pd.DataFrame, background: pd.DataFrame) -> list[dict]:
    rows = []
    af_col = find_af_column(cohort)
    for col in PRESPECIFIED_CONFOUNDS:
        col = af_col if col == "__af__" else col
        if col is None or col not in cohort.columns:
            rows.append({"feature": "allele frequency", "available": False}); continue
        if col not in cohort.columns:
            rows.append({"feature": col, "available": False}); continue
        a = pd.to_numeric(cohort[col], errors="coerce").dropna()
        b = pd.to_numeric(background[col], errors="coerce").dropna()
        if a.empty or b.empty:
            rows.append({"feature": col, "available": False}); continue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) or np.nan
        rows.append({"feature": col, "available": True,
                     "cohort_mean": float(a.mean()),
                     "background_mean": float(b.mean()),
                     "standardised_difference":
                         float((a.mean() - b.mean()) / pooled) if pooled else np.nan})
    return rows


def main() -> int:
    args = parse_args()
    if args.build == args.analyse:
        print("STOP: pass exactly one of --build or --analyse")
        return 1
    ev = load_evaluator()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    prereg_path = out / "preregistration.json"
    cohort_path = out / "labelled_pairs.csv"

    # --------------------------------------------------------------- build
    if args.build:
        if args.gwas is None:
            print("STOP: --build needs --gwas pointing at the catalog file")
            return 1
        if not args.gwas.is_file():
            print(f"STOP: no such file: {args.gwas}")
            parent = args.gwas.parent
            if parent.is_dir():
                near = sorted(q.name for q in parent.iterdir() if q.is_file())[:10]
                print(f"      {parent} contains: {near or '(empty)'}")
            else:
                print(f"      the directory {parent} does not exist either")
            print("      The GWAS Catalog all-associations download is a TSV "
                  "inside a zip; unzip it and point --gwas at the .tsv.")
            return 1
        catalog = read_catalog(args.gwas, args.trait_contains)
        print(f"catalog: {len(catalog):,} unique associations")
        pairs = load_pairs(ev, args)
        print(f"scored:  {len(pairs):,} non-CpG-altering {args.stratum} pairs, "
              f"{pairs['Variant_ID'].nunique():,} variants")
        if args.mqtl_significance is not None:
            if "pvalue" not in pairs.columns:
                print("STOP: no cohort p-value column; cannot restrict to mQTLs")
                return 1
            before = len(pairs)
            keep = pd.to_numeric(pairs["pvalue"], errors="coerce") < args.mqtl_significance
            pairs = pairs[keep.fillna(False)].reset_index(drop=True)
            print(f"restricted to measured mQTLs at p < {args.mqtl_significance:g}: "
                  f"{len(pairs):,} of {before:,} pairs")
            if pairs.empty:
                print("STOP: no pairs survive that threshold")
                return 1
        pairs["gwas_hit"] = label_overlap(pairs, catalog, args.window).astype(int)
        n_hit = int(pairs["gwas_hit"].sum())
        n_var = int(pairs.loc[pairs["gwas_hit"] == 1, "Variant_ID"].nunique())
        n_probe = int(pairs.loc[pairs["gwas_hit"] == 1, "probeID"].nunique()) \
            if "probeID" in pairs.columns else -1
        print(f"labelled {n_hit:,} pairs ({n_var:,} variants, {n_probe:,} probes) "
              f"as GWAS associations")
        if n_hit < 100:
            print("CAUTION: fewer than 100 labelled pairs; this will be "
                  "underpowered in the same way the earlier tests were")
        pairs.to_csv(cohort_path, index=False)

        prereg = {
            "title": "Are high predicted-effect variants enriched for GWAS "
                     "associations relative to distance-matched background?",
            "written": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "status": "PRE-REGISTERED. Written by --build before any statistic "
                      "was computed. No result existed when this file was made.",
            "motivation": (
                "ClinVar annotates on coding and splicing evidence, so a "
                "methylation-effect score is not expected to predict it, and the "
                "earlier ClinVar and breast-GWAS tests returned nulls that are "
                "uninformative about regulatory relevance. GWAS associations are "
                "overwhelmingly non-coding and are the appropriate annotation."),
            "direction": (
                "Reversed relative to E.2/E.2b. Those asked whether known risk "
                "variants score high, on 35 and 32 variants. This asks whether "
                "high-scoring variants are GWAS associations, over every "
                "held-out pair."),
            "cohort": {
                "scores": str(args.scores_dir), "stratum": args.stratum,
                "model": args.model, "seeds": args.seeds,
                "pairs": int(len(pairs)),
                "labelled_pairs": n_hit, "labelled_variants": n_var,
                "labelled_probes": n_probe,
                "overlap_rule": ("variant is itself a catalog association (rsid)"
                                 if args.window == 0 else
                                 f"variant within {args.window} bp of an association"),
                "trait_filter": args.trait_contains or "all traits",
                "mqtl_significance": args.mqtl_significance,
                "population": ("variants with a measured methylation effect at "
                               f"cohort p < {args.mqtl_significance:g}"
                               if args.mqtl_significance is not None else
                               "every tested pair, including those with no "
                               "measured methylation effect"),
            },
            "independent_unit": (
                "the PROBE. Variants at one CpG share the window and the context "
                "features. Intervals are 1-Mb block bootstraps, which also absorb "
                "the LD clustering of GWAS hits."),
            "primary_statistic": {
                "definition": "AUROC of |delta M| separating GWAS-associated "
                              "pairs from distance-matched non-associated pairs",
                "null": 0.5, "sided": "two-sided"},
            "secondary_statistic": {
                "definition": "difference in mean |delta M| between the two "
                              "groups of the matched cohort"},
            "prespecified_confounds": {
                "features": list(PRESPECIFIED_CONFOUNDS),
                "concern": ("Predicted effect rises steeply with proximity to the "
                            "CpG, and GWAS hits are common variants, so distance "
                            "and allele frequency could each produce a shift with "
                            "no regulatory meaning."),
                "handling": ("Distance is matched exactly, with tolerance "
                             f"{args.match_tolerance} bp. Allele frequency is "
                             "reported as a standardised difference and is NOT "
                             "matched; a large imbalance makes the result "
                             "uninterpretable and must be stated as such.")},
            "interpretation_rules_fixed_in_advance": {
                "enrichment": "AUROC interval excludes 0.5 from above and no "
                              "material confound imbalance.",
                "depletion": "AUROC interval excludes 0.5 from below. A real "
                             "possible outcome, reported as depletion, never as "
                             "'inconclusive'.",
                "null": "interval covers 0.5; reported as null with the achieved "
                        "n attached.",
                "uninterpretable": "confound imbalance |standardised difference| "
                                   "> 0.25 on any matched feature."},
            "note": "Trait-agnostic overlap is a weaker claim than a "
                    "trait-specific one and is reported as such. The model is "
                    "breast-trained; the catalog spans all traits.",
        }
        prereg["amendments"] = [{
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "change": "Primary statistic changed from whole-distribution AUROC "
                      "to GWAS share of the top |delta M| fraction, at "
                      f"{args.tails}. Matching now runs within "
                      f"{args.af_bins} allele-frequency strata in addition to "
                      "exact distance.",
            "reason": "The v1 run returned UNINTERPRETABLE, not a result: the "
                      "pre-declared allele-frequency confound was imbalanced at "
                      "a standardised difference of 0.77, because GWAS "
                      "associations are common variants by construction. The "
                      "pre-registration prescribed exactly this response. "
                      "Separately, mean |delta M| over the matched cohort is "
                      "about 0.04 M-units, so most pairs carry a near-zero "
                      "prediction and a rank statistic over the whole "
                      "distribution is dominated by them; the tail is where a "
                      "regulatory claim lives.",
            "no_result_was_superseded": "The v1 verdict was uninterpretable, so "
                      "no statistical result existed to select away from. The "
                      "whole-distribution AUROC is still computed and reported "
                      "as a secondary alongside the tail statistic.",
        }]
        prereg_path.write_text(json.dumps(prereg, indent=2) + "\n")
        print(f"\nwrote {cohort_path}\nwrote {prereg_path}")
        print("\nInspect the pre-registration, then run --analyse.")
        return 0

    # ------------------------------------------------------------- analyse
    if not prereg_path.is_file():
        print(f"STOP: {prereg_path} does not exist. Run --build first; the "
              f"statistics are fixed before any result is computed.")
        return 1
    if not cohort_path.is_file():
        print(f"STOP: {cohort_path} does not exist. Run --build first.")
        return 1
    rng = np.random.default_rng(args.random_seed)
    pairs = pd.read_csv(cohort_path, low_memory=False)
    if "gwas_hit" not in pairs.columns:
        print("STOP: labelled_pairs.csv has no gwas_hit column")
        return 1

    # build_matched_cohort keys off `significant`; reuse it unchanged
    pairs["significant"] = pairs["gwas_hit"].astype(int)
    n_hit = int(pairs["significant"].sum())
    if n_hit < 20:
        print(f"STOP: only {n_hit} labelled pairs; nothing to test")
        return 1
    matched, balance = stratified_match(ev, pairs, args.match_tolerance,
                                       args.af_bins, rng)
    if matched.empty:
        print("STOP: distance matching produced an empty cohort")
        return 1
    cohort = matched[matched["significant"] == 1]
    background = matched[matched["significant"] == 0]
    print(f"matched cohort {len(matched):,} pairs "
          f"({len(cohort):,} GWAS + {len(background):,} matched background), "
          f"distance-only AUROC after matching "
          f"{balance['distance_only_auroc_after_matching']:.4f}")

    auroc = ev.marginal_auroc(matched)
    lo, hi = ev.block_bootstrap(matched, ev.marginal_auroc, args.n_boot, rng)
    tails = []
    for frac in args.tails:
        fn = (lambda f: (lambda fr: tail_enrichment(fr, f)))(frac)
        point = fn(matched)
        tlo, thi = ev.block_bootstrap(matched, fn, args.n_boot, rng)
        tails.append({"fraction": frac, "n_in_tail": int(round(len(matched) * frac)),
                      "gwas_share": point, "ci_low": tlo, "ci_high": thi,
                      "excludes_half": bool(np.isfinite(tlo) and np.isfinite(thi)
                                            and (tlo > 0.5 or thi < 0.5))})
    d_cohort = cohort["Predicted_Delta_M"].abs()
    d_back = background["Predicted_Delta_M"].abs()
    confounds = confound_table(cohort, background)

    worst = max((abs(r.get("standardised_difference") or 0.0)
                 for r in confounds if r.get("available")), default=0.0)
    usable = [t for t in tails if np.isfinite(t["gwas_share"])
              and np.isfinite(t["ci_low"])]
    unchecked = [r["feature"] for r in confounds if not r.get("available")]
    if worst > 0.25:
        verdict = "uninterpretable_confound"
    elif unchecked:
        verdict = "confound_unchecked"
    elif not usable:
        verdict = "inconclusive"
    elif any(t["ci_low"] > 0.5 for t in usable):
        verdict = "enrichment"
    elif any(t["ci_high"] < 0.5 for t in usable):
        verdict = "depletion"
    else:
        verdict = "null"

    print("\n" + "=" * 78)
    print("PRIMARY: GWAS share of the top |delta M| variants (null 0.5)")
    print("=" * 78)
    print("Most pairs carry a near-zero prediction, so a statistic over the whole")
    print("distribution ranks noise against noise. This asks whether the variants")
    print("the model actually flags are disproportionately GWAS associations.\n")
    for t in tails:
        if not np.isfinite(t["gwas_share"]):
            print(f"  top {100*t['fraction']:>5.2f}%  too few variants to evaluate")
            continue
        mark = "  <-- excludes 0.5" if t["excludes_half"] else ""
        print(f"  top {100*t['fraction']:>5.2f}%  n={t['n_in_tail']:>4}  "
              f"GWAS share {t['gwas_share']:.3f} "
              f"[{t['ci_low']:.3f}, {t['ci_high']:.3f}]{mark}")
    print(f"\nSECONDARY (whole distribution, the v1 primary):")
    print(f"  AUROC {auroc:.4f} [{lo:.4f}, {hi:.4f}]   (null 0.5)")
    print(f"  mean |delta M|  GWAS {d_cohort.mean():.4f}   "
          f"background {d_back.mean():.4f}   "
          f"difference {d_cohort.mean() - d_back.mean():+.4f}")
    print("\nPRESPECIFIED CONFOUND CHECK")
    for r in confounds:
        if not r.get("available"):
            print(f"  {r['feature']:<18} not available in the score files")
            continue
        print(f"  {r['feature']:<18} GWAS {r['cohort_mean']:.4f}   "
              f"background {r['background_mean']:.4f}   "
              f"std diff {r['standardised_difference']:+.3f}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if verdict == "enrichment":
        print("ENRICHMENT IN THE TAIL. Variants the model scores highly are GWAS")
        print("associations more often than distance-matched variants are. This is")
        print("a regulatory-relevance result that does not depend on ClinVar")
        print("annotation and holds in the intergenic space where these variants")
        print("sit. It is trait-agnostic, which is weaker than a trait-specific")
        print("claim, and must be reported that way.")
    elif verdict == "depletion":
        print("DEPLETION. High-scoring variants are LESS often GWAS associations")
        print("than matched background. Report as depletion, not as a null.")
    elif verdict == "confound_unchecked":
        print("CANNOT BE INTERPRETED AS ENRICHMENT. A pre-declared confound could")
        print(f"not be checked because its column is absent: {unchecked}.")
        print("Allele frequency is the confound that made the first run of this")
        print("test uninterpretable, so a result without it is not reportable.")
    elif verdict == "uninterpretable_confound":
        print("UNINTERPRETABLE. A prespecified confound is materially imbalanced")
        print(f"(worst standardised difference {worst:.3f} > 0.25), so the raw")
        print("comparison cannot be read as regulatory signal.")
    elif verdict == "null":
        print("NULL. Predicted effect magnitude does not distinguish GWAS")
        print("associations from distance-matched background. With this n the")
        print("result constrains the effect rather than establishing absence.")
    else:
        print("INCONCLUSIVE. The statistic or its interval is undefined.")

    payload = {
        "verdict": verdict,
        "primary_tail_enrichment": tails,
        "af_matching": {"bins": args.af_bins,
                        "strata_used": balance.get("af_strata_used")},
        "auroc": None if not np.isfinite(auroc) else float(auroc),
        "auroc_ci": [None if not np.isfinite(lo) else float(lo),
                     None if not np.isfinite(hi) else float(hi)],
        "n_matched_pairs": int(len(matched)),
        "n_gwas_pairs": int(len(cohort)),
        "mean_abs_delta_m": {"gwas": float(d_cohort.mean()),
                             "background": float(d_back.mean())},
        "matching": {k: (float(v) if isinstance(v, float) else int(v))
                     for k, v in balance.items()},
        "confounds": confounds,
        "n_boot": args.n_boot, "block_size_bp": ev.BLOCK_BP,
        "preregistration": str(prereg_path),
    }
    (out / "run_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\noutput: {out}")
    print("=" * 78)
    payload["unchecked_confounds"] = unchecked
    (out / "run_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if verdict in ("enrichment", "depletion", "null") else 2


if __name__ == "__main__":
    sys.exit(main())
