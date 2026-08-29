#!/usr/bin/env python3
"""
What distinguishes tissue-shared from tissue-specific meQTLs, and does the model
difference survive without matching?

Two analyses, and the first one is the point
--------------------------------------------
scripts/25 showed that SilentMethyl predicts tissue-shared meQTLs slightly better
than tissue-specific ones. That is a statement about the model. On its own it is
also fragile: the effect is small and one of the two metrics loses significance
under tighter matching.

This script asks a different question, and the answer does not depend on the
model at all.

**A. Chromatin characterisation (model-independent).**
Every probe carries seven MCF-10A breast chromatin tracks and two phyloP scores.
A meQTL discovered in blood that does NOT replicate in breast should, if the
tissue interpretation is right, sit in sequence that is chromatin-INACTIVE in
breast. A shared meQTL should sit in active, conserved, CpG-dense sequence.

That prediction is testable with data already on disk and it never touches the
model's predictions. If blood-specific meQTLs are depleted for breast ATAC and
H3K27ac relative to shared ones -- at matched effect size -- then the
shared/specific split is a real biological partition, not a relabelling of model
error. It also explains *why* the model does worse on them: it learned a breast
sequence-to-methylation function, and those loci are not breast-active.

**B. Regression-adjusted accuracy (replaces matching).**
scripts/25 matched pairs on discovery |Z| and discarded those that could not be
matched, which cost 130 of 868 shared pairs and left the arms imbalanced anyway
(median |Z| 11.73 vs 10.95). Logistic regression controls |Z| continuously
instead, uses every pair, and is what the matched comparison was approximating:

    P(model gets the direction right) ~ shared + |Z| + |distance| + GC

The coefficient on `shared` is the quantity of interest, with a 1-Mb block
bootstrap for its interval. If matching and regression disagree, the matched
result was an artifact of which pairs happened to find partners.

Usage (run from the repository root)
------------------------------------
    python -u scripts/26_meqtl_class_chromatin.py
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
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

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

LOGGER = logging.getLogger("silentmethyl.classes")
BLOCK_BP = 1_000_000
EFFECT_ALIASES = ("beta_ref_to_alt", "beta_genoa_ref_to_alt")
PVALUE_ALIASES = ("pvalue", "p_wald", "pval_nominal")
SE_ALIASES = ("se", "se_genoa", "slope_se")


def resolve(frame, aliases, what):
    for n in aliases:
        if n in frame.columns:
            return n
    raise SystemExit(f"no {what} column; looked for {list(aliases)}")


def load(scores_dir: Path, model, seeds, threshold, label):
    frames = [pd.read_csv(p) for p in
              (scores_dir / "heldout" / model / f"seed{s}" / "pair_scores.csv"
               for s in seeds) if p.is_file()]
    if not frames:
        raise SystemExit(f"no score files under {scores_dir}")
    long = pd.concat(frames, ignore_index=True)
    long = long.rename(columns={resolve(long, EFFECT_ALIASES, "effect"): "effect",
                                resolve(long, PVALUE_ALIASES, "p"): "pvalue",
                                resolve(long, SE_ALIASES, "se"): "effect_se"})
    key = ["chr", "Position_1based", "Ref", "Alt", "probeID"]
    pred = long.groupby(key, sort=False)["Predicted_Delta_M"].mean()
    out = (long.drop_duplicates(subset=key).set_index(key)
           .drop(columns=["Predicted_Delta_M"]).join(pred).reset_index())
    out["Z"] = out["effect"] / out["effect_se"]
    out["significant"] = out["pvalue"] < threshold
    out["_block"] = (out["cpg_chr"].astype(str) + ":"
                     + (out["cpg_pos_hg38"] // BLOCK_BP).astype(int).astype(str))
    out = out[~(out["creates_cpg"].astype(bool)
                | out["destroys_cpg"].astype(bool))].reset_index(drop=True)
    LOGGER.info("%s: %d pairs, %d significant", label, len(out),
                int(out["significant"].sum()))
    return out


def build_classes(discovery, replication, replication_p):
    key = ["chr", "Position_1based", "Ref", "Alt", "probeID"]
    m = discovery.merge(
        replication[key + ["effect", "effect_se", "pvalue"]],
        on=key, how="inner", suffixes=("_disc", "_rep"))
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


def block_ci(frame, fn, n_boot, rng):
    blocks = frame["_block"].to_numpy()
    uniq = np.unique(blocks)
    if len(uniq) < 5:
        return (np.nan, np.nan)
    idx = {b: np.flatnonzero(blocks == b) for b in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        v = fn(frame.iloc[np.concatenate([idx[b] for b in pick])])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < max(20, n_boot // 10):
        return (np.nan, np.nan)
    return tuple(float(x) for x in np.percentile(vals, [2.5, 97.5]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genoa", type=Path, default=Path("results/journal/genoa_variant_scoring"))
    ap.add_argument("--egtex", type=Path, default=Path("results/journal/egtex_variant_scoring"))
    ap.add_argument("--genoa-threshold", type=float, default=5e-8)
    ap.add_argument("--egtex-threshold", type=float, default=1.483e-5)
    ap.add_argument("--replication-p", type=float, default=0.05)
    ap.add_argument("--split-template", default="data/datafiles/{split}.csv")
    ap.add_argument("--model", default="fusion")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/journal/meqtl_class_chromatin"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    rng = np.random.default_rng(args.random_seed)
    LOGGER.info("feature-list check: %s", verify_feature_list())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    genoa = load(args.genoa, args.model, args.seeds, args.genoa_threshold, "GENOA")
    egtex = load(args.egtex, args.model, args.seeds, args.egtex_threshold, "eGTEx")
    pairs = build_classes(genoa, egtex, args.replication_p)

    feats = probe_features(args.split_template,
                           set(pairs["probeID"].astype(str)))
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
        lo, hi = block_ci(probes, diff, args.n_boot, rng)
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
    chrom.to_csv(args.output_dir / "chromatin_by_class.csv", index=False)

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
    lo, hi = block_ci(frame, shared_coefficient, args.n_boot, rng)
    regression = {"n_pairs": int(len(frame)),
                  "n_shared": int(frame["shared"].sum()),
                  "coefficient_shared": point, "ci_low": lo, "ci_high": hi,
                  "covariates": design,
                  "note": ("logistic coefficient on standardised predictors; "
                           "positive means the model gets tissue-shared meQTLs "
                           "right more often at equal effect size and distance")}

    with (args.output_dir / "run_summary.json").open("w") as fh:
        json.dump({
            "analysis": "chromatin characterisation of meQTL classes",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
    print(f"\nwrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
