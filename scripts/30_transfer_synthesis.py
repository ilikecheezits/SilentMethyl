#!/usr/bin/env python3
"""
Cross-cohort synthesis of SilentMethyl's variant-effect performance.

Why this exists
---------------
The variant-effect results have been reported as a list of separate, modest
numbers. That undersells them. Three things the per-cohort analyses cannot show,
all of which are legitimate strengths and none of which required new data:

1. **Replication across cohorts is itself the result.** GENOA (peripheral blood,
   African American, ~1,000 donors) and eGTEx Breast Mammary (breast, largely
   European, ~50-100 donors) differ in tissue, ancestry, platform generation and
   genotyping pipeline. An effect that holds across that gap is far stronger
   evidence than either cohort alone. Reported as an inverse-variance
   meta-analysis with Cochran's Q, so consistency is measured rather than
   asserted.

2. **Rank agreement understates a model that also gets the scale right.**
   Everything so far is Spearman and sign agreement. The regression of measured
   effect on predicted effect asks a harder question -- is the predicted
   magnitude proportional to the truth? -- and a significantly positive slope is
   a quantitative claim that rank statistics cannot make.

3. **Direction agreement is bounded by measurement error in the TRUTH, not only
   by the model.** A meQTL measured at |slope|/se = 2 has a poorly determined
   sign; one at |slope|/se = 20 does not. If agreement rises with the precision
   of the measurement, part of the apparent weakness is noise in the reference,
   and the model's ceiling is higher than the pooled number suggests. If it stays
   flat, the model is genuinely weak. Either answer is worth having, and nobody
   has asked the question.

The reporting discipline this enforces
--------------------------------------
Every stratum is written out, not only the flattering ones. The precision
analysis emits all deciles; the distance analysis emits all bins; the
CpG-context split emits both arms. A stratified result is a legitimate finding
ONLY when the whole stratification is shown -- otherwise it is the same number
selected after the fact. This script makes showing everything the path of least
resistance, because it writes the full tables whether you ask for them or not.

Usage (run from the repository root)
------------------------------------
    python -u scripts/30_transfer_synthesis.py \
        --cohort GENOA:results/journal/genoa_variant_scoring:5e-8 \
        --cohort eGTEx:results/journal/egtex_variant_scoring:1.483e-5
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

LOGGER = logging.getLogger("silentmethyl.synthesis")

BLOCK_BP = 1_000_000
EFFECT_ALIASES = ("beta_ref_to_alt", "beta_genoa_ref_to_alt")
PVALUE_ALIASES = ("pvalue", "p_wald", "pval_nominal")
SE_ALIASES = ("se", "se_genoa", "slope_se")
DISTANCE_BINS = [0, 50, 100, 200, 300, 400, 501]


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def resolve(frame: pd.DataFrame, aliases: tuple[str, ...], what: str) -> str:
    for name in aliases:
        if name in frame.columns:
            return name
    raise SystemExit(f"no {what} column found; looked for {list(aliases)}")


def load_cohort(name: str, scores_dir: Path, stratum: str, models: list[str],
                seeds: list[int], threshold: float) -> pd.DataFrame:
    frames = []
    for model in models:
        for seed in seeds:
            path = scores_dir / stratum / model / f"seed{seed}" / "pair_scores.csv"
            if not path.is_file():
                LOGGER.warning("missing %s -- skipping", path)
                continue
            f = pd.read_csv(path)
            f["Model"] = model
            f["Seed"] = int(seed)
            frames.append(f)
    if not frames:
        raise SystemExit(f"no score files found for cohort {name} under {scores_dir}")
    long = pd.concat(frames, ignore_index=True)

    effect = resolve(long, EFFECT_ALIASES, "effect")
    pvalue = resolve(long, PVALUE_ALIASES, "p-value")
    long = long.rename(columns={effect: "effect", pvalue: "pvalue"})
    try:
        se = resolve(long, SE_ALIASES, "standard error")
        long = long.rename(columns={se: "effect_se"})
    except SystemExit:
        LOGGER.warning("%s: no standard-error column; precision analysis skipped", name)
        long["effect_se"] = np.nan

    long["Cohort"] = name
    long["significant"] = (long["pvalue"] < threshold).astype(int)
    long["_block"] = (long["cpg_chr"].astype(str) + ":"
                      + (long["cpg_pos_hg38"] // BLOCK_BP).astype(int).astype(str))
    long["cpg_altering_nearby"] = (long["creates_cpg"].astype(bool)
                                   | long["destroys_cpg"].astype(bool))
    # Report both counts. The pre-filter number is what this function loaded;
    # the primary stratum drops nearby-CpG-altering pairs to match scripts/20,
    # so the analysis table below will show the smaller one. Printing only the
    # larger figure here is how "4,037 vs 6,604" got into the project in the
    # first place.
    n_seeds = max(len(frames), 1)
    sig_all = int(long["significant"].sum())
    sig_clean = int((long["significant"] & ~long["cpg_altering_nearby"]).sum())
    LOGGER.info("%s: %d rows | significant at p < %.3g: %d per model-seed "
                "(%d excluding nearby CpG-altering -- this is the primary stratum)",
                name, len(long), threshold, sig_all // n_seeds, sig_clean // n_seeds)
    return long


def seed_ensemble(long: pd.DataFrame) -> pd.DataFrame:
    """Mean predicted delta across seeds, labelled Seed = -1."""
    if long["Seed"].nunique() <= 1:
        return long.assign(Seed=-1)
    keys = ["Cohort", "Model", "Pair_UID"]
    mean = (long.groupby(keys, sort=False)["Predicted_Delta_M"]
            .mean().reset_index())
    meta = (long.drop(columns=["Predicted_Delta_M", "Seed"])
            .drop_duplicates(subset=keys))
    out = meta.merge(mean, on=keys, how="inner", validate="one_to_one")
    out["Seed"] = -1
    return out


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def block_bootstrap(frame: pd.DataFrame, metric, n_boot: int,
                    rng: np.random.Generator) -> tuple[float, float]:
    """1 Mb block resampling. Pairs are in LD; naive intervals are far too narrow."""
    blocks = frame["_block"].to_numpy()
    unique = np.unique(blocks)
    if len(unique) < 5:
        return (np.nan, np.nan)
    index = {b: np.flatnonzero(blocks == b) for b in unique}
    values = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[b] for b in picked])
        value = metric(frame.iloc[rows])
        if np.isfinite(value):
            values.append(value)
    if len(values) < max(20, n_boot // 10):
        return (np.nan, np.nan)
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def signed_rho(frame: pd.DataFrame) -> float:
    if len(frame) < 10:
        return np.nan
    r = stats.spearmanr(frame["Predicted_Delta_M"], frame["effect"]).statistic
    return float(r) if np.isfinite(r) else np.nan


def direction_agreement(frame: pd.DataFrame) -> float:
    if len(frame) < 10:
        return np.nan
    a = np.sign(frame["Predicted_Delta_M"].to_numpy(dtype=float))
    b = np.sign(frame["effect"].to_numpy(dtype=float))
    keep = (a != 0) & (b != 0)
    return float((a[keep] == b[keep]).mean()) if keep.sum() >= 10 else np.nan


def calibration_slope(frame: pd.DataFrame) -> float:
    """OLS slope of measured effect on predicted effect.

    Rank statistics say the ordering is right. This asks whether the magnitude is
    proportional -- a strictly harder question, and the only one of the two that
    supports a quantitative statement.
    """
    if len(frame) < 20:
        return np.nan
    x = frame["Predicted_Delta_M"].to_numpy(dtype=float)
    y = frame["effect"].to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 20 or np.std(x[keep]) == 0:
        return np.nan
    return float(np.polyfit(x[keep], y[keep], 1)[0])


METRICS = {
    "signed_rho": signed_rho,
    "direction_agreement": direction_agreement,
    "calibration_slope": calibration_slope,
}
SLIM = ["Predicted_Delta_M", "effect", "_block"]


def measure(frame: pd.DataFrame, name: str, n_boot: int,
            rng: np.random.Generator, **context) -> dict:
    slim = frame[[c for c in SLIM if c in frame.columns]]
    fn = METRICS[name]
    point = fn(slim)
    low, high = block_bootstrap(slim, fn, n_boot, rng)
    return {**context, "metric": name, "n": int(len(slim)),
            "n_blocks": int(slim["_block"].nunique()),
            "value": point, "ci_low": low, "ci_high": high}


# --------------------------------------------------------------------------
# analyses
# --------------------------------------------------------------------------

def precision_strata(sig: pd.DataFrame, n_bins: int) -> pd.Series:
    """Bin by |effect| / se -- how precisely the TRUTH is measured.

    This is the ceiling question. A meQTL at |slope|/se = 2 has a barely
    determined sign, so no model can agree with it much above chance. If
    agreement climbs with precision, the pooled number is limited by the
    reference measurement and not only by the model.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(sig["effect"].to_numpy(dtype=float)) / sig["effect_se"].to_numpy(dtype=float)
    z = pd.Series(z, index=sig.index)
    if not np.isfinite(z).any():
        return pd.Series(index=sig.index, dtype=object)
    try:
        return pd.qcut(z, n_bins, labels=[f"q{i+1}" for i in range(n_bins)],
                       duplicates="drop")
    except ValueError:
        return pd.Series(index=sig.index, dtype=object)


def meta_analyse(rows: pd.DataFrame) -> list[dict]:
    """Inverse-variance meta-analysis of signed rho across cohorts, on Fisher z.

    Cochran's Q tests whether the cohorts disagree by more than sampling error.
    Low heterogeneity across a blood/breast, AFR/EUR split is the substantive
    claim: the effect is not a property of one cohort's idiosyncrasies.
    """
    out = []
    subset = rows[(rows["metric"] == "signed_rho") & (rows["stratum"] == "significant")]
    for model, group in subset.groupby("model"):
        group = group[np.isfinite(group["value"]) & (group["n"] > 6)]
        if len(group) < 2:
            continue
        r = group["value"].to_numpy(dtype=float)
        # Effective n from independent LD blocks, not pairs -- using the pair
        # count here would understate the variance by an order of magnitude.
        n_eff = group["n_blocks"].to_numpy(dtype=float)
        z = np.arctanh(np.clip(r, -0.999999, 0.999999))
        var = 1.0 / np.maximum(n_eff - 3.0, 1.0)
        w = 1.0 / var
        z_meta = float((w * z).sum() / w.sum())
        se_meta = float(np.sqrt(1.0 / w.sum()))
        q = float((w * (z - z_meta) ** 2).sum())
        df = len(group) - 1
        i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
        out.append({
            "model": model,
            "cohorts": ", ".join(sorted(group["cohort"])),
            "k": int(len(group)),
            "rho_meta": float(np.tanh(z_meta)),
            "ci_low": float(np.tanh(z_meta - 1.96 * se_meta)),
            "ci_high": float(np.tanh(z_meta + 1.96 * se_meta)),
            "z": z_meta / se_meta,
            "p": float(2 * stats.norm.sf(abs(z_meta / se_meta))),
            "cochran_q": q, "df": df,
            "q_p": float(stats.chi2.sf(q, df)) if df > 0 else np.nan,
            "i_squared": i2,
        })
    return out


def atomic(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(tmp, index=False)
    else:
        with tmp.open("w") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", action="append", required=True,
                    help="NAME:SCORES_DIR:THRESHOLD, repeatable. THRESHOLD is the "
                         "PRE-DECLARED significance cutoff for that cohort.")
    ap.add_argument("--stratum", default="heldout")
    ap.add_argument("--models", nargs="+", default=["fusion", "sequence"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--precision-bins", type=int, default=5)
    ap.add_argument("--primary-includes-cpg-altering", action="store_true",
                    help="Pool CpG-altering-nearby pairs into the primary "
                         "stratum. OFF by default so the headline matches "
                         "scripts/20. Both arms are reported either way.")
    ap.add_argument("--allow-missing-cohorts", action="store_true", default=True,
                    help="Analyse whichever cohorts have score files instead of "
                         "failing. The summary records requested vs analysed.")
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/journal/variant_effect_synthesis"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    rng = np.random.default_rng(args.random_seed)

    cohorts, thresholds = {}, {}
    requested, missing = [], []
    for spec in args.cohort:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--cohort must be NAME:DIR:THRESHOLD, got {spec!r}")
        name, path, thresh = parts[0], Path(parts[1]), float(parts[2])
        requested.append(name)
        try:
            loaded = load_cohort(name, path, args.stratum, args.models,
                                 args.seeds, thresh)
        except SystemExit as exc:
            if not args.allow_missing_cohorts:
                raise
            LOGGER.warning("=" * 70)
            LOGGER.warning("COHORT %s SKIPPED -- %s", name, exc)
            LOGGER.warning("Results below cover only: %s",
                           ", ".join(sorted(cohorts)) or "(none yet)")
            LOGGER.warning("=" * 70)
            missing.append(name)
            continue
        thresholds[name] = thresh
        cohorts[name] = seed_ensemble(loaded)
    if not cohorts:
        raise SystemExit("no cohort had score files; nothing to analyse")

    rows = []
    for name, data in cohorts.items():
        for model, group in data.groupby("Model"):
            # The primary stratum EXCLUDES pairs with a nearby CpG-altering
            # variant, matching scripts/20. Without this the two scripts report
            # different counts for "significant GENOA pairs" -- 4,037 there and
            # 6,604 here -- and a reviewer who notices has found an inconsistency
            # in the paper rather than a stratum definition.
            eligible = group if args.primary_includes_cpg_altering \
                else group[~group["cpg_altering_nearby"]]
            sig = eligible[eligible["significant"] == 1]
            null = eligible[eligible["pvalue"] > 0.5]
            common = {"cohort": name, "model": model,
                      "threshold": thresholds[name]}

            for metric in METRICS:
                rows.append(measure(sig, metric, args.n_boot, rng,
                                    stratum="significant", **common))
            # The null stratum is the internal negative control and is reported
            # every time, unasked. A signed rho at zero here is what separates a
            # real effect from a fitted one.
            for metric in ("signed_rho", "direction_agreement"):
                rows.append(measure(null, metric, args.n_boot, rng,
                                    stratum="null_p_gt_0.5", **common))

            # measurement-precision deciles of the TRUTH
            bins = precision_strata(sig, args.precision_bins)
            for label in [b for b in bins.dropna().unique()]:
                part = sig[bins == label]
                for metric in ("signed_rho", "direction_agreement"):
                    rows.append(measure(part, metric, args.n_boot, rng,
                                        stratum=f"precision_{label}", **common))

            # distance bins
            for lo, hi in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
                part = sig[(sig["abs_distance_bp"] >= lo)
                           & (sig["abs_distance_bp"] < hi)]
                for metric in ("signed_rho", "direction_agreement"):
                    rows.append(measure(part, metric, args.n_boot, rng,
                                        stratum=f"distance_{lo}_{hi}", **common))

            # nearby CpG-altering context, both arms
            for label, part in (("cpg_altering_nearby", sig[sig["cpg_altering_nearby"]]),
                                ("no_cpg_altering_nearby", sig[~sig["cpg_altering_nearby"]])):
                for metric in ("signed_rho", "direction_agreement"):
                    rows.append(measure(part, metric, args.n_boot, rng,
                                        stratum=label, **common))

    table = pd.DataFrame(rows)
    atomic(table, args.output_dir / "all_strata.csv")

    meta = meta_analyse(table) if len(cohorts) >= 2 else []
    if len(cohorts) < 2:
        LOGGER.warning("meta-analysis needs >= 2 cohorts; have %d (%s). "
                       "Per-cohort results below are complete and valid.",
                       len(cohorts), ", ".join(sorted(cohorts)))
    atomic(pd.DataFrame(meta), args.output_dir / "cross_cohort_meta_analysis.csv")

    atomic({
        "analysis": "cross-cohort variant-effect synthesis",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cohorts_requested": requested,
        "cohorts_analysed": sorted(cohorts),
        "cohorts_missing_scores": missing,
        "primary_stratum": ("significant AND no nearby CpG-altering variant"
                            if not args.primary_includes_cpg_altering
                            else "significant, CpG-altering pooled in"),
        "cohorts": {k: {"threshold": v, "rows": int(len(cohorts[k]))}
                    for k, v in thresholds.items()},
        "models": args.models, "seeds": args.seeds, "n_boot": args.n_boot,
        "block_size_bp": BLOCK_BP,
        "meta_analysis": meta,
        "reporting_rule": (
            "all_strata.csv contains every stratum computed, including the null "
            "control and every precision and distance bin. A stratified result "
            "is a finding only when the full stratification is shown; quoting one "
            "bin without the table is the same number selected after the fact."),
        "meta_note": (
            "Meta-analysis weights use independent 1 Mb LD blocks, not pair "
            "counts. Using pairs would understate the variance by roughly an "
            "order of magnitude and manufacture significance."),
    }, args.output_dir / "run_summary.json")

    # ---- report ----------------------------------------------------------
    def show(cohort, model, stratum, metric):
        r = table[(table["cohort"] == cohort) & (table["model"] == model)
                  & (table["stratum"] == stratum) & (table["metric"] == metric)]
        if r.empty or not np.isfinite(r.iloc[0]["value"]):
            return "        n/a"
        r = r.iloc[0]
        return f"{r['value']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  n={r['n']:,}"

    print()
    print("=" * 78)
    print("PER COHORT -- significant stratum, seed ensemble")
    for cohort in cohorts:
        for model in sorted(table[table["cohort"] == cohort]["model"].unique()):
            print(f"\n{cohort} / {model}   (p < {thresholds[cohort]:.3g})")
            for metric in ("signed_rho", "direction_agreement", "calibration_slope"):
                print(f"  {metric:<22} {show(cohort, model, 'significant', metric)}")
            print(f"  {'null control rho':<22} "
                  f"{show(cohort, model, 'null_p_gt_0.5', 'signed_rho')}")

    print()
    print("=" * 78)
    print("MEASUREMENT PRECISION OF THE TRUTH  (|effect| / se quintiles)")
    print("  Rising agreement means the reference measurement, not the model,")
    print("  bounds the pooled number. Flat means the model is the limit.")
    for cohort in cohorts:
        for model in sorted(table[table["cohort"] == cohort]["model"].unique()):
            print(f"\n{cohort} / {model}")
            for i in range(args.precision_bins):
                label = f"precision_q{i+1}"
                print(f"  {label:<22} "
                      f"dir {show(cohort, model, label, 'direction_agreement')}")

    if meta:
        print()
        print("=" * 78)
        print("CROSS-COHORT META-ANALYSIS  (signed rho, Fisher z, LD-block weights)")
        for m in meta:
            print(f"  {m['model']:<10} rho = {m['rho_meta']:+.4f} "
                  f"[{m['ci_low']:+.4f}, {m['ci_high']:+.4f}]  p = {m['p']:.2e}")
            print(f"             cohorts: {m['cohorts']}  |  "
                  f"Cochran Q = {m['cochran_q']:.2f} (p = {m['q_p']:.3f}), "
                  f"I2 = {100*m['i_squared']:.0f}%")
        print("\n  Low heterogeneity across a blood/breast, AFR/EUR split is the")
        print("  substantive claim: the effect is not one cohort's idiosyncrasy.")
    print("=" * 78)
    print(f"\nfull table: {args.output_dir / 'all_strata.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
