#!/usr/bin/env python3
"""
Are tissue-SHARED meQTLs different from tissue-SPECIFIC ones -- to the model, and
in the genome?

This file merges what were scripts 25 and 26. They were one argument split in
two: script 25 asked whether the model predicts shared meQTLs better, found a
small effect, and honestly flagged it as fragile; script 26 then asked whether
that split is a real biological partition (chromatin, no model involved) and
whether the model difference survives without matching. The second only makes
sense as a response to the first, so they belong together.

    --stage matched     model accuracy, shared vs specific, matched on |Z|
    --stage chromatin   chromatin characterisation + regression-adjusted accuracy
    --stage all         both

The hypothesis
--------------
SilentMethyl's variant-effect pathway is sequence-only. That is established, not
assumed: fusion and sequence-only models are statistically equivalent on every
variant-effect metric in both external cohorts, because the context vector is
identical for reference and alternate alleles and can enter a paired contrast
only through a gate shift.

A sequence-only predictor should therefore succeed on meQTLs whose mechanism is
sequence-intrinsic and fail on those mediated by tissue-specific chromatin. Those
classes are separable empirically: an effect present in two tissues is more
likely sequence-intrinsic; one present in only one is more likely
chromatin-mediated. If the model is more accurate on shared meQTLs at MATCHED
discovery effect size, it is not merely predicting meQTLs -- it is separating
them by mechanism.

The confound that would fake this, and how it is removed
--------------------------------------------------------
Cohorts differ enormously in power (GENOA ~1,000 donors, eGTEx Breast ~50-100).
A pair significant in one and not the other is by default far more likely to be
*underpowered* in the second than *absent* there. Three safeguards:

1. **Matched discovery effect size.** Pairs are compared only against pairs with
   the same |Z| in the discovery cohort (Z = effect / SE, scale-free and so
   comparable across cohorts with different phenotype normalisations). Without
   this, "shared" would simply mean "larger effect", and larger effects are
   easier to predict.
2. **An explicit power screen.** A pair is called tissue-specific only when the
   replication cohort had the precision to detect an effect of the observed
   magnitude. Pairs failing the screen are counted and dropped, never pooled.
3. **Both directions.** A finding that appears in only one direction is a cohort
   artifact, not a tissue effect.

The chromatin stage, which never touches the model
---------------------------------------------------
Every probe carries seven MCF-10A breast chromatin tracks and two phyloP scores.
A meQTL discovered in blood that does NOT replicate in breast should, if the
tissue interpretation is right, sit in sequence that is chromatin-INACTIVE in
breast. If blood-specific meQTLs are depleted for breast ATAC and H3K27ac
relative to shared ones at matched effect size, the split is a real biological
partition rather than a relabelling of model error -- and it explains *why* the
model does worse on them: it learned a breast sequence-to-methylation function
and those loci are not breast-active.

The same stage replaces matching with regression. Matching on |Z| discarded pairs
that found no partner and left the arms imbalanced anyway. Logistic regression
controls |Z| continuously, uses every pair, and is what the matched comparison was
approximating:

    P(model gets the direction right) ~ shared + |Z| + |distance|

If matching and regression disagree, the matched result was an artifact of which
pairs happened to find partners.

Cohorts
-------
`--cohort NAME:PATH:THRESHOLD`, repeatable. The default is the historical
GENOA/eGTEx pair. Supplying more cohorts runs every ordered pair, which is how
this extends to the nine-tissue eGTEx panel without changing any logic.

Usage (run from the repository root)
------------------------------------
    python -u scripts/40_meqtl_tissue_specificity.py --stage all

    # nine-tissue panel. NOTE the `/by_tissue/<Tissue>` suffix: the parent
    # directory holds the UNION scoring run, whose label columns (pvalue, beta,
    # se) are BreastMammaryTissue placeholders. Pointing at it gives nine
    # identical breast cohorts and a plausible, wrong table. Always use
    # by_tissue, which data/split_predictions_by_tissue.py writes.
    # Threshold 5e-8 matches scripts/31, so R3 and R4 partition the same
    # significant set.
    python -u scripts/40_meqtl_tissue_specificity.py --stage matched \\
        --cohort Lung:results/journal/egtex_multitissue_scoring/by_tissue/Lung:5e-8 \\
        --cohort ColonTransverse:results/journal/egtex_multitissue_scoring/by_tissue/ColonTransverse:5e-8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

LOGGER = logging.getLogger("silentmethyl.tissue")

BLOCK_BP = 1_000_000
EFFECT_ALIASES = ("beta_ref_to_alt", "beta_genoa_ref_to_alt")
PVALUE_ALIASES = ("pvalue", "p_wald", "pval_nominal")
SE_ALIASES = ("se", "se_genoa", "slope_se")
KEY = ["chr", "Position_1based", "Ref", "Alt", "probeID"]

DEFAULT_COHORTS = [
    "GENOA:results/journal/genoa_variant_scoring:5e-8",
    "eGTEx:results/journal/egtex_variant_scoring:1.483e-5",
]

DEFAULT_OUT = {
    "matched": Path("results/journal/tissue_shared_meqtls"),
    "chromatin": Path("results/journal/meqtl_class_chromatin"),
}

# Declared locally rather than imported from training_common, which pulls in
# torch, transformers and huggingface_hub for one list of strings and makes a
# CPU-only analysis unrunnable on a plain node. Duplicated AND verified: when
# training_common imports, the two are asserted identical at startup.
TABULAR_FEATURES = [
    "Ref_ATAC_Signal",
    "Ref_H3K4me3_Signal",
    "Ref_H3K27ac_Signal",
    "Ref_H3K27me3_Signal",
    "Ref_H3K9me3_Signal",
    "Ref_H3K36me3_Signal",
    "Ref_H3K4me1_Signal",
    "Target_Base_PhyloP_100way_1",
    "Target_Base_PhyloP_100way_2",
]


def verify_feature_list() -> dict:
    try:
        import training_common as tc
    except Exception as exc:  # noqa: BLE001
        return {"checked": False, "reason": type(exc).__name__}
    if list(tc.TABULAR_FEATURES) != TABULAR_FEATURES:
        raise RuntimeError(
            "TABULAR_FEATURES here disagrees with training_common:\n"
            f"  here: {TABULAR_FEATURES}\n  there: {list(tc.TABULAR_FEATURES)}\n"
            "Chromatin columns would be mislabelled -- fix before trusting output.")
    return {"checked": True}


# =========================================================================
# shared loading and statistics
# =========================================================================

def resolve(frame, aliases, what):
    for name in aliases:
        if name in frame.columns:
            return name
    raise SystemExit(f"no {what} column; looked for {list(aliases)}")


def parse_cohorts(items) -> list:
    out = []
    for it in items:
        parts = it.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--cohort must be NAME:PATH:THRESHOLD, got {it!r}")
        out.append({"name": parts[0], "path": Path(parts[1]),
                    "threshold": float(parts[2])})
    return out


def load(scores_dir: Path, model: str, seeds, threshold: float,
         label: str) -> pd.DataFrame:
    """Seed-ensembled pair scores with effect/p/SE resolved to canonical names.

    Union of the two pre-merge loaders. `cpg_altering_nearby` is retained as a
    column (script 25 kept it) and used for the same filter both applied.
    """
    frames = []
    for seed in seeds:
        path = scores_dir / "heldout" / model / f"seed{seed}" / "pair_scores.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"no score files under {scores_dir}")
    long = pd.concat(frames, ignore_index=True)
    eff = resolve(long, EFFECT_ALIASES, "effect")
    pv = resolve(long, PVALUE_ALIASES, "p-value")
    se = resolve(long, SE_ALIASES, "standard error")
    long = long.rename(columns={eff: "effect", pv: "pvalue", se: "effect_se"})

    for col in KEY:
        if col not in long.columns:
            raise SystemExit(f"{scores_dir}: pair_scores.csv lacks {col}; "
                             f"rescoring with the current scripts/19 is required")

    # Seed ensemble: one prediction per pair.
    pred = long.groupby(KEY, sort=False)["Predicted_Delta_M"].mean()
    meta = long.drop_duplicates(subset=KEY).set_index(KEY)
    out = meta.drop(columns=["Predicted_Delta_M"]).join(pred).reset_index()

    out["Z"] = out["effect"] / out["effect_se"]
    out["significant"] = out["pvalue"] < threshold
    out["_block"] = (out["cpg_chr"].astype(str) + ":"
                     + (out["cpg_pos_hg38"] // BLOCK_BP).astype(int).astype(str))
    out["cpg_altering_nearby"] = (out["creates_cpg"].astype(bool)
                                  | out["destroys_cpg"].astype(bool))
    out = out[~out["cpg_altering_nearby"]].reset_index(drop=True)
    LOGGER.info("%s: %d pairs, %d significant at p<%.3g",
                label, len(out), int(out["significant"].sum()), threshold)
    return out


def load_all(cohorts, model, seeds) -> dict:
    return {c["name"]: load(c["path"], model, seeds, c["threshold"], c["name"])
            for c in cohorts}


def block_bootstrap(frame, metric, n_boot, rng):
    """Percentile interval over 1-Mb genomic blocks, so LD is respected."""
    blocks = frame["_block"].to_numpy()
    uniq = np.unique(blocks)
    if len(uniq) < 5:
        return (np.nan, np.nan)
    idx = {b: np.flatnonzero(blocks == b) for b in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        v = metric(frame.iloc[np.concatenate([idx[b] for b in pick])])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < max(20, n_boot // 10):
        return (np.nan, np.nan)
    return tuple(float(x) for x in np.percentile(vals, [2.5, 97.5]))


def direction_agreement(f):
    if len(f) < 10:
        return np.nan
    a = np.sign(f["Predicted_Delta_M"].to_numpy(float))
    b = np.sign(f["effect_discovery"].to_numpy(float))
    keep = (a != 0) & (b != 0)
    return float((a[keep] == b[keep]).mean()) if keep.sum() >= 10 else np.nan


def signed_rho(f):
    if len(f) < 10:
        return np.nan
    r = stats.spearmanr(f["Predicted_Delta_M"], f["effect_discovery"]).statistic
    return float(r) if np.isfinite(r) else np.nan


# =========================================================================
# stage: matched
# =========================================================================

def match_on_discovery_z(shared, specific, tolerance, rng):
    """One specific pair per shared pair, matched on |Z| in the discovery cohort.

    Without replacement. This is the control that makes the comparison mean
    anything: both arms end up with equally strong effects where they were
    discovered, so a difference in model accuracy cannot be 'shared meQTLs are
    just bigger'.

    Returns BOTH arms. Returning only the specific arm, and letting the caller
    recover the shared arm as `shared.iloc[:len(matched)]`, is wrong twice over:
    the shared rows that actually found partners are scattered through the loop
    rather than sitting in a prefix, so the arms are not paired; and a positional
    prefix is genomically clustered, which starves the 1-Mb block bootstrap and
    returns nan intervals. Both were observed before this was fixed.
    """
    pool = specific.copy()
    pool["_used"] = False
    picked_specific, picked_shared = [], []
    for shared_idx, z in shared["absZ_discovery"].items():
        cand = pool.index[(~pool["_used"])
                          & (np.abs(pool["absZ_discovery"] - z) <= tolerance)]
        if len(cand) == 0:
            continue
        choice = rng.choice(cand)
        pool.loc[choice, "_used"] = True
        picked_specific.append(choice)
        picked_shared.append(shared_idx)
    return (pool.loc[picked_specific].drop(columns="_used"),
            shared.loc[picked_shared])


def analyse(discovery, replication, disc_name, rep_name, args, rng, rows):
    """One direction of the comparison. Body unchanged from the pre-merge
    script 25 so that bootstrap draws, and therefore intervals, are identical."""
    merged = discovery.merge(
        replication[KEY + ["effect", "effect_se", "pvalue", "Z", "significant"]],
        on=KEY, how="inner", suffixes=("_discovery", "_replication"))
    LOGGER.info("%s -> %s: %d pairs tested in both cohorts",
                disc_name, rep_name, len(merged))
    if merged.empty:
        return None

    sig = merged[merged["significant_discovery"]].copy()
    if len(sig) < 40:
        LOGGER.warning("only %d significant pairs tested in both; skipping",
                       len(sig))
        return None

    sig["absZ_discovery"] = sig["Z_discovery"].abs()
    # Replication: same sign AND nominally supported. A sign flip is not
    # replication even at a small p-value.
    same_sign = np.sign(sig["effect_discovery"]) == np.sign(sig["effect_replication"])
    sig["replicates"] = same_sign & (sig["pvalue_replication"] < args.replication_p)

    # Power screen: could the replication cohort have detected an effect as
    # large as the one observed in discovery? Compared in Z units, which are
    # scale-free. A pair the replication cohort could never have seen is not
    # evidence of tissue specificity.
    z_needed = stats.norm.isf(args.replication_p / 2)
    sig["replication_powered"] = (
        sig["absZ_discovery"] * (sig["effect_se_discovery"]
                                 / sig["effect_se_replication"]).abs()) >= z_needed
    dropped = int((~sig["replicates"] & ~sig["replication_powered"]).sum())

    shared = sig[sig["replicates"]]
    specific = sig[(~sig["replicates"]) & sig["replication_powered"]]
    LOGGER.info("  shared=%d  tissue-specific=%d  underpowered-dropped=%d",
                len(shared), len(specific), dropped)
    if len(shared) < 20 or len(specific) < 20:
        LOGGER.warning("  too few in one class; reporting counts only")
        return {"discovery": disc_name, "replication": rep_name,
                "tested_in_both": int(len(merged)),
                "significant_in_discovery": int(len(sig)),
                "shared": int(len(shared)), "specific": int(len(specific)),
                "underpowered_dropped": dropped, "matched": 0,
                "note": "insufficient pairs for a matched comparison"}

    matched, shared_use = match_on_discovery_z(shared, specific, args.z_tolerance, rng)
    LOGGER.info("  matched %d of %d shared pairs to a specific counterpart",
                len(matched), len(shared))
    if len(matched) < 20:
        return {"discovery": disc_name, "replication": rep_name,
                "shared": int(len(shared)), "specific": int(len(specific)),
                "matched": int(len(matched)),
                "note": "matching failed; widen --z-tolerance"}

    # |Z| balance is the whole point of matching, so check it rather than assume
    # it. A gap this large means the pool could not cover the shared |Z| range
    # and the comparison is not controlled -- report it instead of hiding it.
    z_gap = float(abs(shared_use["absZ_discovery"].median()
                      - matched["absZ_discovery"].median()))
    if z_gap > args.z_tolerance:
        LOGGER.warning("  |Z| medians differ by %.2f (> tolerance %.2f): the "
                       "matched arms are NOT balanced; treat this direction as "
                       "uncontrolled", z_gap, args.z_tolerance)

    out = {"discovery": disc_name, "replication": rep_name,
           "median_absZ_gap": z_gap,
           "z_balance_ok": bool(z_gap <= args.z_tolerance),
           "tested_in_both": int(len(merged)),
           "significant_in_discovery": int(len(sig)),
           "shared": int(len(shared)), "specific": int(len(specific)),
           "underpowered_dropped": dropped, "matched": int(len(matched)),
           "median_absZ_shared": float(shared_use["absZ_discovery"].median()),
           "median_absZ_specific": float(matched["absZ_discovery"].median())}

    for name, fn in (("direction_agreement", direction_agreement),
                     ("signed_rho", signed_rho)):
        for label, frame in (("shared", shared_use), ("tissue_specific", matched)):
            point = fn(frame)
            lo, hi = block_bootstrap(frame, fn, args.n_boot, rng)
            out[f"{name}_{label}"] = point
            out[f"{name}_{label}_ci"] = [lo, hi]
            rows.append({"direction": f"{disc_name}->{rep_name}", "class": label,
                         "metric": name, "n": int(len(frame)),
                         "value": point, "ci_low": lo, "ci_high": hi})
        # paired difference, bootstrapped over the same blocks
        diffs = []
        for _ in range(args.n_boot):
            blocks = np.unique(np.concatenate([shared_use["_block"],
                                               matched["_block"]]))
            pick = set(rng.choice(blocks, size=len(blocks), replace=True))
            a = fn(shared_use[shared_use["_block"].isin(pick)])
            b = fn(matched[matched["_block"].isin(pick)])
            if np.isfinite(a) and np.isfinite(b):
                diffs.append(a - b)
        if len(diffs) >= max(20, args.n_boot // 10):
            out[f"{name}_difference"] = float(np.mean(diffs))
            out[f"{name}_difference_ci"] = [float(np.percentile(diffs, 2.5)),
                                            float(np.percentile(diffs, 97.5))]
    return out


def run_matched(args, loaded: dict) -> int:
    out_dir = args.output_dir or DEFAULT_OUT["matched"]
    rng = np.random.default_rng(args.random_seed)

    rows, results = [], []
    for dn, rn in permutations(loaded.keys(), 2):
        r = analyse(loaded[dn], loaded[rn], dn, rn, args, rng, rows)
        if r:
            results.append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "by_class.csv", index=False)
    with (out_dir / "run_summary.json").open("w") as fh:
        json.dump({
            "analysis": "model accuracy on tissue-shared vs tissue-specific meQTLs",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "hypothesis": ("a sequence-only variant pathway should predict "
                           "sequence-intrinsic (tissue-shared) meQTLs better than "
                           "chromatin-mediated (tissue-specific) ones"),
            "confound_controls": {
                "matched_on": "|Z| in the discovery cohort, without replacement",
                "why": ("shared meQTLs have larger effects and larger effects are "
                        "easier to predict; without matching the result would be "
                        "a restatement of effect size"),
                "power_screen": ("a pair counts as tissue-specific only if the "
                                 "replication cohort could have detected an effect "
                                 "of the observed magnitude, compared in Z units"),
            },
            "both_directions": ("every ordered cohort pair is run; a result "
                                "appearing in only one direction is a cohort "
                                "artifact, not a tissue effect"),
            "parameters": vars(args),
            "results": results,
        }, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    print()
    print("=" * 78)
    for r in results:
        print(f"\n{r['discovery']} discovered -> replicated in {r['replication']}")
        print(f"  tested in both cohorts        {r.get('tested_in_both', 0):>8,}")
        print(f"  significant in discovery      {r.get('significant_in_discovery', 0):>8,}")
        print(f"  tissue-shared                 {r.get('shared', 0):>8,}")
        print(f"  tissue-specific (powered)     {r.get('specific', 0):>8,}")
        print(f"  dropped, underpowered         {r.get('underpowered_dropped', 0):>8,}")
        if r.get("note"):
            print(f"  -> {r['note']}")
            continue
        print(f"  matched pairs per arm         {r.get('matched', 0):>8,}")
        print(f"  median |Z| shared / specific  "
              f"{r.get('median_absZ_shared', float('nan')):.2f} / "
              f"{r.get('median_absZ_specific', float('nan')):.2f}   "
              f"(matched, so these should be close)")
        for metric in ("direction_agreement", "signed_rho"):
            for cls in ("shared", "tissue_specific"):
                v, ci = r.get(f"{metric}_{cls}"), r.get(f"{metric}_{cls}_ci", [np.nan]*2)
                if v is not None and np.isfinite(v):
                    print(f"  {metric:<20} {cls:<16} {v:+.4f} "
                          f"[{ci[0]:+.4f}, {ci[1]:+.4f}]")
            d, dci = r.get(f"{metric}_difference"), r.get(f"{metric}_difference_ci")
            if d is not None:
                verdict = "SHARED HIGHER" if dci and dci[0] > 0 else \
                          ("SPECIFIC HIGHER" if dci and dci[1] < 0 else "no difference")
                print(f"  {metric:<20} {'difference':<16} {d:+.4f} "
                      f"[{dci[0]:+.4f}, {dci[1]:+.4f}]  {verdict}")
    print()
    print("A difference in the SAME direction in BOTH analyses is the finding.")
    print("One direction only means a cohort artifact. No difference means the")
    print("sequence-only account does not separate meQTLs by mechanism -- which")
    print("is a real answer and must be reported as one.")
    print("=" * 78)
    print(f"\nwrote {out_dir}")
    return 0


# =========================================================================
# stage: chromatin
# =========================================================================

def build_classes(discovery, replication, replication_p):
    m = discovery.merge(
        replication[KEY + ["effect", "effect_se", "pvalue"]],
        on=KEY, how="inner", suffixes=("_disc", "_rep"))
    sig = m[m["significant"]].copy()
    sig["absZ"] = sig["Z"].abs()
    same_sign = np.sign(sig["effect_disc"]) == np.sign(sig["effect_rep"])
    sig["shared"] = same_sign & (sig["pvalue_rep"] < replication_p)
    z_needed = stats.norm.isf(replication_p / 2)
    powered = (sig["absZ"] * (sig["effect_se_disc"] / sig["effect_se_rep"]).abs()) >= z_needed
    sig = sig[sig["shared"] | powered].reset_index(drop=True)
    LOGGER.info("classes: %d shared, %d tissue-specific (powered)",
                int(sig["shared"].sum()), int((~sig["shared"]).sum()))
    return sig


def probe_features(split_template: str, probes: set) -> pd.DataFrame:
    need = ["probeID", *TABULAR_FEATURES]
    frames = []
    for split in ("train", "val", "test"):
        path = Path(split_template.format(split=split))
        if not path.is_file():
            continue
        for chunk in pd.read_csv(path, usecols=need, chunksize=20_000):
            chunk = chunk[chunk["probeID"].astype(str).isin(probes)]
            if not chunk.empty:
                frames.append(chunk)
    if not frames:
        raise SystemExit("no probe features found")
    return pd.concat(frames, ignore_index=True).drop_duplicates("probeID")


def run_chromatin(args, loaded: dict) -> int:
    from sklearn.linear_model import LogisticRegression

    out_dir = args.output_dir or DEFAULT_OUT["chromatin"]
    rng = np.random.default_rng(args.random_seed)
    LOGGER.info("feature-list check: %s", verify_feature_list())
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(loaded.keys())
    disc_name, rep_name = names[0], names[1]
    pairs = build_classes(loaded[disc_name], loaded[rep_name], args.replication_p)

    feats = probe_features(args.split_template, set(pairs["probeID"].astype(str)))
    pairs = pairs.merge(feats, on="probeID", how="left")

    # ---- A. chromatin characterisation, model-independent -----------------
    # One row per PROBE, not per pair: chromatin is a property of the locus, and
    # counting a probe once per variant would inflate n by its variant count.
    probes = pairs.groupby("probeID").agg(
        shared=("shared", "max"), absZ=("absZ", "max"),
        _block=("_block", "first"),
        **{f: (f, "first") for f in TABULAR_FEATURES}).reset_index()
    LOGGER.info("chromatin comparison over %d unique probes (%d shared)",
                len(probes), int(probes["shared"].sum()))

    rows = []
    for feature in TABULAR_FEATURES:
        a = probes.loc[probes["shared"], feature].to_numpy(float)
        b = probes.loc[~probes["shared"], feature].to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 20 or len(b) < 20:
            continue
        diff = lambda f: float(  # noqa: E731
            np.median(f.loc[f["shared"], feature]) -
            np.median(f.loc[~f["shared"], feature]))
        lo, hi = block_bootstrap(probes, diff, args.n_boot, rng)
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        # rank-biserial: scale-free effect size, interpretable as a probability
        rb = 2 * u.statistic / (len(a) * len(b)) - 1
        rows.append({
            "feature": feature, "n_shared": len(a), "n_specific": len(b),
            "median_shared": float(np.median(a)),
            "median_specific": float(np.median(b)),
            "median_difference": float(np.median(a) - np.median(b)),
            "ci_low": lo, "ci_high": hi,
            "rank_biserial": float(rb), "mannwhitney_p": float(u.pvalue),
        })
    chrom = pd.DataFrame(rows).sort_values("rank_biserial", key=abs, ascending=False)
    chrom.to_csv(out_dir / "chromatin_by_class.csv", index=False)

    # ---- B. regression-adjusted accuracy, no matching ----------------------
    pairs["correct"] = (np.sign(pairs["Predicted_Delta_M"])
                        == np.sign(pairs["effect_disc"])).astype(int)
    design = ["shared", "absZ", "abs_distance_bp"]
    frame = pairs.dropna(subset=design + ["correct"]).copy()
    frame["shared"] = frame["shared"].astype(float)

    def shared_coefficient(f):
        if f["correct"].nunique() < 2 or f["shared"].nunique() < 2:
            return np.nan
        X = f[design].to_numpy(float)
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
        try:
            fit = LogisticRegression(max_iter=2000, C=1e6).fit(X, f["correct"])
        except Exception:  # noqa: BLE001
            return np.nan
        return float(fit.coef_[0][0])

    point = shared_coefficient(frame)
    lo, hi = block_bootstrap(frame, shared_coefficient, args.n_boot, rng)
    regression = {"n_pairs": int(len(frame)),
                  "n_shared": int(frame["shared"].sum()),
                  "coefficient_shared": point, "ci_low": lo, "ci_high": hi,
                  "covariates": design,
                  "note": ("logistic coefficient on standardised predictors; "
                           "positive means the model gets tissue-shared meQTLs "
                           "right more often at equal effect size and distance")}

    with (out_dir / "run_summary.json").open("w") as fh:
        json.dump({
            "analysis": "chromatin characterisation of meQTL classes",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "discovery": disc_name, "replication": rep_name,
            "chromatin": rows, "regression": regression,
            "feature_list_verified": verify_feature_list(),
            "interpretation": (
                "The chromatin comparison is independent of the model. If "
                "blood-discovered meQTLs that do NOT replicate in breast sit in "
                "breast-inactive chromatin, the shared/specific split is a "
                "biological partition rather than a relabelling of model error."),
        }, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    print()
    print("=" * 78)
    print("A. CHROMATIN BY CLASS  (MCF-10A breast tracks, one row per probe)")
    print("   Positive rank-biserial = higher in tissue-SHARED meQTLs.")
    print("   This analysis never uses the model's predictions.")
    print(f"{'feature':<34}{'shared':>10}{'specific':>10}{'rank-bis':>10}{'p':>12}")
    for r in chrom.to_dict("records"):
        print(f"{r['feature']:<34}{r['median_shared']:>10.3f}"
              f"{r['median_specific']:>10.3f}{r['rank_biserial']:>+10.3f}"
              f"{r['mannwhitney_p']:>12.2e}")
    print()
    print("=" * 78)
    print("B. REGRESSION-ADJUSTED ACCURACY  (all pairs, |Z| controlled continuously)")
    print(f"   n = {regression['n_pairs']:,} pairs, {regression['n_shared']:,} shared")
    print(f"   coefficient on `shared`: {point:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    if np.isfinite(lo) and lo > 0:
        print("   -> survives without matching; the matched result was not an")
        print("      artifact of which pairs found partners")
    elif np.isfinite(hi) and hi < 0:
        print("   -> REVERSED without matching; the matched result was an artifact")
    else:
        print("   -> does NOT survive without matching. The matched comparison")
        print("      depended on pair selection and should not be reported as a")
        print("      finding. Say so.")
    print("=" * 78)
    print(f"\nwrote {out_dir}")
    return 0


# =========================================================================
# main
# =========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("matched", "chromatin", "all"), default="all")
    ap.add_argument("--cohort", action="append", metavar="NAME:PATH:THRESHOLD",
                    help="repeatable; defaults to the GENOA/eGTEx pair")
    ap.add_argument("--replication-p", type=float, default=0.05,
                    help="Nominal p in the replication cohort, with matching "
                         "sign, that counts as the effect being present there.")
    ap.add_argument("--z-tolerance", type=float, default=0.5,
                    help="|Z| window for matching on discovery effect size.")
    ap.add_argument("--split-template", default="data/datafiles/{split}.csv")
    ap.add_argument("--model", default="fusion")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="stage-specific default if omitted")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    cohorts = parse_cohorts(args.cohort or DEFAULT_COHORTS)
    if len(cohorts) < 2:
        raise SystemExit("at least two cohorts are required")
    loaded = load_all(cohorts, args.model, args.seeds)

    # Each stage gets a FRESH generator seeded identically. The pre-merge scripts
    # each created their own; sharing one across stages here would silently shift
    # every bootstrap interval in whichever stage ran second.
    if args.stage in ("matched", "all"):
        rc = run_matched(argparse.Namespace(
            **{**vars(args), "output_dir": args.output_dir
               if args.stage == "matched" else None}), loaded)
        if rc:
            return rc

    if args.stage in ("chromatin", "all"):
        rc = run_chromatin(argparse.Namespace(
            **{**vars(args), "output_dir": args.output_dir
               if args.stage == "chromatin" else None}), loaded)
        if rc:
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
