#!/usr/bin/env python3
"""
Does SilentMethyl predict tissue-SHARED meQTLs better than tissue-SPECIFIC ones?

The hypothesis, and why it follows from what is already established
-------------------------------------------------------------------
SilentMethyl's variant-effect pathway is sequence-only. That is not an
assumption here: fusion and sequence-only models are statistically equivalent on
every variant-effect metric in BOTH external cohorts, because the context vector
is identical for reference and alternate alleles and can enter a paired contrast
only through a gate shift.

A sequence-only predictor should therefore succeed on meQTLs whose mechanism is
sequence-intrinsic and fail on those mediated by tissue-specific chromatin. Those
two classes are separable empirically: an effect present in both blood and breast
is more likely sequence-intrinsic; one present in only one tissue is more likely
chromatin-mediated.

If the model is more accurate on tissue-shared meQTLs at MATCHED discovery
effect size, then it is not merely predicting meQTLs -- it is separating them by
mechanism. That is a biological finding derived computationally rather than a
performance number.

The confound that would fake this result, and how it is removed
---------------------------------------------------------------
GENOA has ~1,000 donors; eGTEx Breast has ~50-100. A pair significant in GENOA
and not in eGTEx is, by default, far more likely to be *underpowered in eGTEx*
than *absent in breast*. Two safeguards:

1. **Matched discovery effect size.** Pairs are compared only against pairs with
   the same |Z| in the discovery cohort (Z = effect / SE, which is scale-free and
   so comparable across cohorts with different phenotype normalisations). Both
   groups therefore have equally strong effects where they were discovered, and
   the only difference is whether the effect is also present in the other tissue.
   Without this, "shared" would simply mean "larger effect", and larger effects
   are easier to predict.

2. **An explicit power screen.** A pair is called tissue-specific only when the
   replication cohort had the precision to detect an effect of the observed
   magnitude. Pairs failing that screen are counted and dropped, not silently
   pooled into the specific class.

Both cohorts keep their own pre-declared significance thresholds. The analysis is
run in both directions -- GENOA-discovered and eGTEx-discovered -- because a
finding that only appears in one direction is a cohort artifact.

Usage (run from the repository root)
------------------------------------
    python -u scripts/25_tissue_shared_meqtls.py \
        --genoa results/journal/genoa_variant_scoring \
        --egtex results/journal/egtex_variant_scoring
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger("silentmethyl.tissue")

BLOCK_BP = 1_000_000
EFFECT_ALIASES = ("beta_ref_to_alt", "beta_genoa_ref_to_alt")
PVALUE_ALIASES = ("pvalue", "p_wald", "pval_nominal")
SE_ALIASES = ("se", "se_genoa", "slope_se")


def resolve(frame, aliases, what):
    for name in aliases:
        if name in frame.columns:
            return name
    raise SystemExit(f"no {what} column; looked for {list(aliases)}")


def load(scores_dir: Path, model: str, seeds: list[int], threshold: float,
         label: str) -> pd.DataFrame:
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

    # Seed ensemble: one prediction per pair.
    key = ["chr", "Position_1based", "Ref", "Alt", "probeID"]
    for col in key:
        if col not in long.columns:
            raise SystemExit(f"{scores_dir}: pair_scores.csv lacks {col}; "
                             f"rescoring with the current scripts/19 is required")
    pred = long.groupby(key, sort=False)["Predicted_Delta_M"].mean()
    meta = long.drop_duplicates(subset=key).set_index(key)
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


def block_bootstrap(frame, metric, n_boot, rng):
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


def match_on_discovery_z(shared, specific, tolerance, rng):
    """One specific pair per shared pair, matched on |Z| in the discovery cohort.

    Without replacement. This is the control that makes the comparison mean
    anything: both arms end up with equally strong effects where they were
    discovered, so a difference in model accuracy cannot be 'shared meQTLs are
    just bigger'.
    """
    pool = specific.copy()
    pool["_used"] = False
    picked = []
    for z in shared["absZ_discovery"].to_numpy():
        cand = pool.index[(~pool["_used"])
                          & (np.abs(pool["absZ_discovery"] - z) <= tolerance)]
        if len(cand) == 0:
            continue
        choice = rng.choice(cand)
        pool.loc[choice, "_used"] = True
        picked.append(choice)
    return pool.loc[picked].drop(columns="_used")


def analyse(discovery, replication, disc_name, rep_name, args, rng, rows):
    """One direction of the comparison."""
    key = ["chr", "Position_1based", "Ref", "Alt", "probeID"]
    merged = discovery.merge(
        replication[key + ["effect", "effect_se", "pvalue", "Z", "significant"]],
        on=key, how="inner", suffixes=("_discovery", "_replication"))
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

    matched = match_on_discovery_z(shared, specific, args.z_tolerance, rng)
    LOGGER.info("  matched %d of %d shared pairs to a specific counterpart",
                len(matched), len(shared))
    if len(matched) < 20:
        return {"discovery": disc_name, "replication": rep_name,
                "shared": int(len(shared)), "specific": int(len(specific)),
                "matched": int(len(matched)),
                "note": "matching failed; widen --z-tolerance"}

    shared_use = shared.iloc[:len(matched)] if len(shared) > len(matched) else shared
    out = {"discovery": disc_name, "replication": rep_name,
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genoa", type=Path,
                    default=Path("results/journal/genoa_variant_scoring"))
    ap.add_argument("--egtex", type=Path,
                    default=Path("results/journal/egtex_variant_scoring"))
    ap.add_argument("--genoa-threshold", type=float, default=5e-8)
    ap.add_argument("--egtex-threshold", type=float, default=1.483e-5)
    ap.add_argument("--replication-p", type=float, default=0.05,
                    help="Nominal p in the replication cohort, with matching "
                         "sign, that counts as the effect being present there.")
    ap.add_argument("--z-tolerance", type=float, default=0.5,
                    help="|Z| window for matching on discovery effect size.")
    ap.add_argument("--model", default="fusion")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/journal/tissue_shared_meqtls"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    rng = np.random.default_rng(args.random_seed)

    genoa = load(args.genoa, args.model, args.seeds, args.genoa_threshold, "GENOA")
    egtex = load(args.egtex, args.model, args.seeds, args.egtex_threshold, "eGTEx")

    rows, results = [], []
    for disc, rep, dn, rn in ((genoa, egtex, "GENOA", "eGTEx"),
                              (egtex, genoa, "eGTEx", "GENOA")):
        r = analyse(disc, rep, dn, rn, args, rng, rows)
        if r:
            results.append(r)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(args.output_dir / "by_class.csv", index=False)
    with (args.output_dir / "run_summary.json").open("w") as fh:
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
            "both_directions": ("run GENOA->eGTEx and eGTEx->GENOA; a result "
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
    print(f"\nwrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
