#!/usr/bin/env python3
"""
Evaluate the GENOA variant scores: the analysis half of Stage B.1.

Consumes the six pair_scores.csv files written by
scripts/20_variant_scoring.py (2 models x 3 seeds, held-out stratum) and
produces the numbers and figures for mentor requirement 6, independent variant
evaluation.

CPU only. Minutes, not hours.

What the exploratory pass established, and what this script is built to survive
------------------------------------------------------------------------------
1. **The pooled correlation is diluted.** GENOA's summary statistics contain every
   cis pair tested, not the meQTLs discovered: 71% of non-CpG-altering held-out
   pairs have p > 0.05. Signed agreement rises monotonically as p falls and lands
   exactly on chance in the null stratum -- a dose-response, which is the strongest
   evidence available that the signal is real. That gradient is a primary figure.

2. **Distance is a confound and it is a big one.** Significant meQTLs sit closer to
   their CpG (median 191 bp vs 258 bp), the model's |delta| is larger for nearer
   variants (rho = -0.29), and distance ALONE classifies significant vs null at
   AUROC 0.595 -- as well as the model does marginally. So the marginal AUROC is
   not reportable on its own. This script's primary discrimination metric is
   computed against distance-matched negatives, and the distance-only baseline is
   reported beside it every time.

3. **Fusion and sequence perform identically on variant effects, by construction.**
   The context vector is allele-invariant: it depends on the probe, not on REF vs
   ALT. In the fusion difference

       delta_fused = [dna_mut*g_mut - dna_wt*g_wt] + epi*[g_epi,mut - g_epi,wt]

   the epigenomic term survives only through a gate shift that is itself driven by
   the sequence change. Context can modulate a variant effect; it cannot create
   one. The right claim is therefore EQUIVALENCE, tested with a paired interval,
   not "sequence beat fusion" from a 0.008 AUROC gap on one seed. Section 5 then
   asks the one question where context could legitimately help: does the fusion
   advantage vary with the gate's DNA share?

Inference is on the M scale throughout. results/journal/rc_uncertainty_conditional_s50/
showed that bounded beta compresses exactly the signal of interest -- an estimator
that looks fine on beta can be worthless on M, and the reverse.

Every interval is a block bootstrap over 1 Mb genomic blocks, matching scripts
16-17. These pairs are emphatically not independent: 39,657 variants in LD across
19,081 probes. A naive interval would be far too narrow and would overstate every
result in the file.

Usage (run from the repository root)
------------------------------------
    python -u scripts/21_variant_evaluation.py
    python -u scripts/21_variant_evaluation.py --n-boot 2000    # publication
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/silentmethyl_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger("silentmethyl.genoa_eval")

# Categorical palette carried over from scripts/18_uncertainty_figure.py, where it
# passed the project's contrast checks. Colour is never the only channel: marker
# and dash carry the same distinction so the figures survive greyscale printing.
COLOURS = {"fusion": "#B03A2E", "sequence": "#1F6FB2", "baseline": "#B4761A"}
MARKERS = {"fusion": "o", "sequence": "s", "baseline": "^"}
DASHES = {"fusion": "-", "sequence": "--", "baseline": ":"}

# Baseline models (scripts/23) come through this evaluator under their own names,
# so styling must not assume the two original models. Unknown names cycle through
# a spare palette instead of raising a KeyError halfway through figure drawing.
_SPARE_COLOURS = ["#2E7D4F", "#6A4C93", "#8D6E63", "#00838F", "#AD1457"]
_SPARE_MARKERS = ["D", "v", "P", "X", "*"]
_SPARE_DASHES = ["-.", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (4, 1, 1, 1, 1, 1))]


def style_for(model: str, index: int) -> dict:
    return {
        "color": COLOURS.get(model, _SPARE_COLOURS[index % len(_SPARE_COLOURS)]),
        "marker": MARKERS.get(model, _SPARE_MARKERS[index % len(_SPARE_MARKERS)]),
        "linestyle": DASHES.get(model, _SPARE_DASHES[index % len(_SPARE_DASHES)]),
    }

GENOME_WIDE = 5e-8
SIGNIFICANCE_STRATA = [
    (0.0, 5e-8, "p < 5e-8"),
    (5e-8, 1e-5, "5e-8 - 1e-5"),
    (1e-5, 1e-3, "1e-5 - 1e-3"),
    (1e-3, 1e-2, "1e-3 - 1e-2"),
    (1e-2, 5e-2, "1e-2 - 0.05"),
    (5e-2, 5e-1, "0.05 - 0.5"),
    (5e-1, 1.01, "p > 0.5"),
]
DISTANCE_BINS = [0, 50, 100, 200, 300, 400, 501]

# Cohort-specific wording, so the same evaluator can run on the tissue-matched
# eGTEx arm and the cross-tissue GENOA arm without either one inheriting the
# other's caveats. The tissue caveat in particular is the whole reason the two
# arms answer different questions.
COHORTS = {
    "GENOA": {
        "label": "GENOA",
        "tissue_caveat": ("GENOA is peripheral blood; SilentMethyl is trained on "
                          "breast and its context features are MCF-10A breast. "
                          "This is cross-tissue, cross-ancestry transfer, not "
                          "tissue-matched validation."),
        "effect_scale": ("GENOA betas are on a normalized-phenotype scale. Rank "
                         "and sign comparisons only; no magnitude calibration "
                         "claim is available."),
    },
    "eGTEx": {
        "label": "eGTEx Breast Mammary Tissue",
        "tissue_caveat": ("eGTEx Breast Mammary Tissue matches the training "
                          "tissue, so a weak result here is attributable to the "
                          "model rather than to a tissue change. Donors are "
                          "predominantly European-ancestry."),
        "effect_scale": ("tensorQTL slopes are on the inverse-normalised "
                         "methylation scale. Rank and sign comparisons only."),
    },
}
BLOCK_BP = 1_000_000


# --------------------------------------------------------------------------- io

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores-dir", type=Path,
                   default=Path("results/journal/genoa_variant_scoring"))
    p.add_argument("--stratum", default="heldout", choices=("heldout", "model_visible"))
    p.add_argument("--models", nargs="+", default=["fusion", "sequence"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--significance", type=float, default=GENOME_WIDE,
                   help="Primary significance threshold on the cohort p-value.")
    p.add_argument("--n-boot", type=int, default=500,
                   help="Block-bootstrap resamples. Use 2000 for the final figures.")
    p.add_argument("--match-tolerance", type=int, default=10,
                   help="bp window for distance matching of negatives.")
    p.add_argument("--match-ratio", type=int, default=1,
                   help="Negatives matched per significant pair.")
    p.add_argument("--cohort", default="GENOA", choices=sorted(COHORTS),
                   help="Selects cohort wording and caveats. It does not change any computation.")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/journal/genoa_variant_evaluation"))
    return p.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def load_scores(args: argparse.Namespace) -> pd.DataFrame:
    """Long frame: one row per (pair, model, seed), plus a per-model seed ensemble."""
    frames = []
    for model in args.models:
        for seed in args.seeds:
            path = (args.scores_dir / args.stratum / model / f"seed{seed}"
                    / "pair_scores.csv")
            if not path.is_file():
                raise SystemExit(f"missing scores: {path}")
            frame = pd.read_csv(path)
            frame["Model"] = model
            frame["Seed"] = int(seed)
            frames.append(frame)
            LOGGER.info("%s seed %d: %d pairs", model, seed, len(frame))
    long = pd.concat(frames, ignore_index=True)

    # Score files written before the eGTEx arm existed name the cohort effect
    # `beta_genoa_ref_to_alt` and the p-value `p_wald`. Alias them so this script
    # reads GENOA and eGTEx outputs identically without regenerating the former.
    for canonical, legacy in (("beta_ref_to_alt", "beta_genoa_ref_to_alt"),
                              ("pvalue", "p_wald")):
        if canonical not in long.columns and legacy in long.columns:
            long[canonical] = long[legacy]

    required = {"Pair_UID", "Predicted_Delta_M", "beta_ref_to_alt", "pvalue",
                "abs_distance_bp", "cpg_chr", "cpg_pos_hg38", "creates_cpg",
                "destroys_cpg"}
    absent = sorted(required - set(long.columns))
    if absent:
        raise SystemExit(f"score files are missing {absent}; rerun script 19")

    # Guard against a silent duplication: one row per pair per model per seed.
    counts = long.groupby(["Model", "Seed"])["Pair_UID"].agg(["size", "nunique"])
    duplicated = counts[counts["size"] != counts["nunique"]]
    if not duplicated.empty:
        raise SystemExit(f"duplicate Pair_UIDs within a model-seed:\n{duplicated}")

    long["cpg_altering"] = (long["creates_cpg"].astype(bool)
                            | long["destroys_cpg"].astype(bool))
    long["_block"] = (long["cpg_chr"].astype(str) + ":"
                      + (long["cpg_pos_hg38"] // BLOCK_BP).astype(int).astype(str))
    long["significant"] = (long["pvalue"] < args.significance).astype(int)
    return long


def add_seed_ensemble(long: pd.DataFrame) -> pd.DataFrame:
    """Mean delta across seeds, as its own pseudo-'seed' labelled -1.

    Reported alongside the per-seed values rather than instead of them: the spread
    across seeds is the honest measure of how stable any of this is.
    """
    # A single-seed input is ALREADY the ensemble. Without this guard, running
    # with --seeds=-1 (deterministic baselines from scripts/23, which have no
    # seeds to average) computes the mean of one value, labels it -1, and
    # concatenates it onto rows that are already labelled -1 -- duplicating every
    # Pair_UID. Point estimates survive that, but n doubles and the block
    # bootstrap resamples duplicated rows, so the intervals come out too narrow.
    # Relabel and return instead of appending.
    if long["Seed"].nunique() <= 1:
        return long.assign(Seed=-1)

    keys = ["Model", "Pair_UID"]
    means = (long.groupby(keys, sort=False)["Predicted_Delta_M"]
             .mean().rename("Predicted_Delta_M").reset_index())
    metadata = (long[long["Seed"] == long["Seed"].min()]
                .drop(columns=["Predicted_Delta_M", "Seed"]))
    ensemble = metadata.merge(means, on=keys, how="inner", validate="one_to_one")
    ensemble["Seed"] = -1
    out = pd.concat([long, ensemble], ignore_index=True)
    counts = out.groupby(["Model", "Seed"])["Pair_UID"].agg(["size", "nunique"])
    bad = counts[counts["size"] != counts["nunique"]]
    if not bad.empty:
        raise SystemExit(f"seed ensemble duplicated Pair_UIDs:\n{bad}")
    return out


# ---------------------------------------------------------------------- metrics

def signed_rho(frame: pd.DataFrame) -> float:
    if len(frame) < 20:
        return np.nan
    result = spearmanr(frame["Predicted_Delta_M"], frame["beta_ref_to_alt"])
    return float(result.statistic)


def direction_agreement(frame: pd.DataFrame) -> float:
    predicted = np.sign(frame["Predicted_Delta_M"].to_numpy(dtype=float))
    observed = np.sign(frame["beta_ref_to_alt"].to_numpy(dtype=float))
    usable = (predicted != 0) & (observed != 0)
    if usable.sum() < 20:
        return np.nan
    return float((predicted[usable] == observed[usable]).mean())


def marginal_auroc(frame: pd.DataFrame) -> float:
    labels = frame["significant"].to_numpy(dtype=int)
    if len(np.unique(labels)) < 2 or labels.sum() < 20:
        return np.nan
    return float(roc_auc_score(labels, frame["Predicted_Delta_M"].abs()))


def distance_only_auroc(frame: pd.DataFrame) -> float:
    """How well distance alone separates significant from null.

    The number the model must beat. Reported beside every AUROC in this file --
    volunteering it is stronger than having a reviewer extract it.
    """
    labels = frame["significant"].to_numpy(dtype=int)
    if len(np.unique(labels)) < 2 or labels.sum() < 20:
        return np.nan
    return float(roc_auc_score(labels, -frame["abs_distance_bp"].to_numpy(dtype=float)))


def within_bin_auroc(frame: pd.DataFrame) -> float:
    """AUROC pooled across distance bins, weighted by bin size.

    Cheap companion to the matched-negative analysis: the confound cannot operate
    inside a bin, so agreement between this and the matched AUROC is a useful check
    that the matching did what it claims.
    """
    total_weight = total = 0.0
    for low, high in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
        subset = frame[(frame["abs_distance_bp"] >= low)
                       & (frame["abs_distance_bp"] < high)]
        labels = subset["significant"].to_numpy(dtype=int)
        if len(np.unique(labels)) < 2 or labels.sum() < 20:
            continue
        auc = roc_auc_score(labels, subset["Predicted_Delta_M"].abs())
        total += auc * len(subset)
        total_weight += len(subset)
    return float(total / total_weight) if total_weight else np.nan


def build_matched_cohort(frame: pd.DataFrame, tolerance: int, ratio: int,
                         rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Pair every significant variant with distance-matched null variants.

    Sampling is without replacement, so a single convenient null cannot be reused
    across many significant pairs and inflate the result. Matching runs on the
    exact bp distance with a small tolerance, widening only if a bucket is empty.
    """
    significant = frame[frame["significant"] == 1]
    nulls = frame[frame["significant"] == 0]
    if significant.empty or nulls.empty:
        return pd.DataFrame(), {}

    pool: dict[int, list] = {}
    for index, distance in zip(nulls.index, nulls["abs_distance_bp"].astype(int)):
        pool.setdefault(distance, []).append(index)
    for value in pool.values():
        rng.shuffle(value)

    chosen: list = []
    unmatched = 0
    for distance in significant["abs_distance_bp"].astype(int):
        taken = 0
        for offset in range(0, tolerance + 1):
            for candidate in ({distance} if offset == 0
                              else {distance - offset, distance + offset}):
                bucket = pool.get(candidate)
                while bucket and taken < ratio:
                    chosen.append(bucket.pop())
                    taken += 1
                if taken >= ratio:
                    break
            if taken >= ratio:
                break
        if taken < ratio:
            unmatched += ratio - taken

    matched = pd.concat([significant, frame.loc[chosen]])
    balance = {
        "significant_n": int(len(significant)),
        "matched_null_n": int(len(chosen)),
        "requested_negatives": int(len(significant) * ratio),
        "unmatched_slots": int(unmatched),
        "median_distance_significant": float(significant["abs_distance_bp"].median()),
        "median_distance_matched_null": float(
            frame.loc[chosen, "abs_distance_bp"].median()) if chosen else np.nan,
        "distance_only_auroc_after_matching": (
            float(roc_auc_score(matched["significant"],
                                -matched["abs_distance_bp"].astype(float)))
            if matched["significant"].nunique() == 2 else np.nan),
    }
    return matched, balance


# -------------------------------------------------------------- block bootstrap

def block_bootstrap(frame: pd.DataFrame, metric, n_boot: int,
                    rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI resampling whole 1 Mb blocks, not rows.

    Rows are variant-CpG pairs in LD; resampling them independently would treat
    correlated observations as independent and produce an interval several times
    too narrow.
    """
    if frame.empty:
        return (np.nan, np.nan)
    blocks = frame["_block"].to_numpy()
    unique, inverse = np.unique(blocks, return_inverse=True)
    if len(unique) < 5:
        return (np.nan, np.nan)
    positions = [np.flatnonzero(inverse == i) for i in range(len(unique))]

    values = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(unique), len(unique))
        index = np.concatenate([positions[p] for p in picks])
        value = metric(frame.iloc[index])
        if np.isfinite(value):
            values.append(value)
    if len(values) < max(20, n_boot // 10):
        return (np.nan, np.nan)
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


# Only these columns are touched by any metric. Slicing a wide frame 500 times per
# measurement dominates the runtime otherwise -- a bootstrap resample copies every
# column it carries, and the score files have ~40 of them.
METRIC_COLUMNS = [
    "Predicted_Delta_M", "Predicted_Delta_M_sequence", "beta_ref_to_alt",
    "significant", "abs_distance_bp", "_block",
]


def measure(frame: pd.DataFrame, metric, name: str, n_boot: int,
            rng: np.random.Generator, **context) -> dict:
    slim = frame[[c for c in METRIC_COLUMNS if c in frame.columns]]
    point = metric(slim)
    low, high = block_bootstrap(slim, metric, n_boot, rng)
    return {**context, "metric": name, "n": int(len(slim)),
            "n_blocks": int(slim["_block"].nunique()) if len(slim) else 0,
            "value": point, "ci_low": low, "ci_high": high}


# ------------------------------------------------------------------- the report

def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    rng = np.random.default_rng(args.random_seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    long = add_seed_ensemble(load_scores(args))
    clean = long[~long["cpg_altering"]]
    LOGGER.info("non-CpG-altering pairs per model-seed: %d",
                int(len(clean) / (len(args.models) * (len(args.seeds) + 1))))

    seed_labels = [*args.seeds, -1]

    # ---- 1. headline metrics, per model, per seed, per CpG-altering stratum
    rows = []
    for model in args.models:
        for seed in seed_labels:
            base = long[(long["Model"] == model) & (long["Seed"] == seed)]
            for altering, label in ((False, "non_cpg_altering"), (True, "cpg_altering")):
                subset = base[base["cpg_altering"] == altering]
                significant = subset[subset["significant"] == 1]
                context = {"model": model, "seed": seed, "variant_class": label}
                rows.append(measure(significant, signed_rho, "signed_rho_significant",
                                    args.n_boot, rng, **context))
                rows.append(measure(significant, direction_agreement,
                                    "direction_agreement_significant",
                                    args.n_boot, rng, **context))
                rows.append(measure(subset, marginal_auroc, "auroc_marginal",
                                    args.n_boot, rng, **context))
                rows.append(measure(subset, distance_only_auroc,
                                    "auroc_distance_only_baseline",
                                    args.n_boot, rng, **context))
                rows.append(measure(subset, within_bin_auroc, "auroc_within_distance_bin",
                                    args.n_boot, rng, **context))
    primary = pd.DataFrame(rows)
    atomic_csv(primary, out / "primary_metrics.csv")

    # ---- 2. distance-matched negatives: the reportable discrimination number
    matched_rows, balance_rows = [], []
    for model in args.models:
        for seed in seed_labels:
            subset = clean[(clean["Model"] == model) & (clean["Seed"] == seed)]
            matched, balance = build_matched_cohort(
                subset, args.match_tolerance, args.match_ratio, rng)
            if matched.empty:
                continue
            matched_rows.append(measure(matched, marginal_auroc,
                                        "auroc_distance_matched", args.n_boot, rng,
                                        model=model, seed=seed))
            balance_rows.append({"model": model, "seed": seed, **balance})
    atomic_csv(pd.DataFrame(matched_rows), out / "matched_negative_auroc.csv")
    atomic_csv(pd.DataFrame(balance_rows), out / "matching_balance.csv")

    # ---- 3. the dilution gradient: signal vs association strength
    gradient = []
    for model in args.models:
        subset = clean[(clean["Model"] == model) & (clean["Seed"] == -1)]
        for low, high, label in SIGNIFICANCE_STRATA:
            stratum = subset[(subset["pvalue"] >= low) & (subset["pvalue"] < high)]
            context = {"model": model, "stratum": label,
                       "p_low": low, "p_high": high}
            gradient.append(measure(stratum, signed_rho, "signed_rho",
                                    args.n_boot, rng, **context))
            gradient.append(measure(stratum, direction_agreement,
                                    "direction_agreement", args.n_boot, rng, **context))
    gradient = pd.DataFrame(gradient)
    atomic_csv(gradient, out / "significance_gradient.csv")

    # ---- 4. fusion vs sequence: an EQUIVALENCE interval, not a winner
    #
    # Paired on Pair_UID and bootstrapped over the same blocks, so the interval is
    # on the difference rather than on two independent estimates. A CI that spans
    # zero here supports "the context tower adds nothing to variant-effect
    # prediction", which is what the architecture predicts: the context vector is
    # identical for REF and ALT.
    paired_rows = []
    if {"fusion", "sequence"} <= set(args.models):
        fusion = clean[(clean["Model"] == "fusion") & (clean["Seed"] == -1)]
        sequence = clean[(clean["Model"] == "sequence") & (clean["Seed"] == -1)]
        merged = fusion.merge(
            sequence[["Pair_UID", "Predicted_Delta_M"]], on="Pair_UID",
            how="inner", suffixes=("", "_sequence"), validate="one_to_one")
        LOGGER.info("paired on %d pairs", len(merged))

        def difference(metric):
            def inner(frame: pd.DataFrame) -> float:
                fusion_value = metric(frame)
                alternative = frame.copy()
                alternative["Predicted_Delta_M"] = frame["Predicted_Delta_M_sequence"]
                return fusion_value - metric(alternative)
            return inner

        significant = merged[merged["significant"] == 1]
        for metric, name, frame in (
            (signed_rho, "signed_rho", significant),
            (direction_agreement, "direction_agreement", significant),
            (marginal_auroc, "auroc_marginal", merged),
            (within_bin_auroc, "auroc_within_distance_bin", merged),
        ):
            paired_rows.append(measure(frame, difference(metric),
                                       f"fusion_minus_sequence_{name}",
                                       args.n_boot, rng))
    atomic_csv(pd.DataFrame(paired_rows), out / "fusion_vs_sequence_paired.csv")

    # ---- 5. can the gate rescue the context tower anywhere?
    #
    # The one legitimate route by which allele-invariant context could shape a
    # variant effect is the gate: if the DNA share shifts with chromatin state, the
    # same sequence change could produce a larger delta in open chromatin. If the
    # fusion advantage is flat across gate quartiles, that route is closed too, and
    # the equivalence in section 4 is architectural rather than incidental.
    gate_rows = []
    gate_column = "WT_Gate_Avg_DNA_Share"
    if paired_rows and gate_column in merged.columns:
        merged = merged.copy()
        merged["gate_quartile"] = pd.qcut(merged[gate_column], 4,
                                          labels=["Q1", "Q2", "Q3", "Q4"],
                                          duplicates="drop")
        for quartile, group in merged.groupby("gate_quartile", observed=True):
            significant = group[group["significant"] == 1]
            context = {"gate_quartile": str(quartile),
                       "gate_share_median": float(group[gate_column].median())}
            gate_rows.append(measure(significant, difference(signed_rho),
                                     "fusion_minus_sequence_signed_rho",
                                     args.n_boot, rng, **context))
            gate_rows.append(measure(group, difference(within_bin_auroc),
                                     "fusion_minus_sequence_auroc_within_bin",
                                     args.n_boot, rng, **context))
    atomic_csv(pd.DataFrame(gate_rows), out / "gate_modulation.csv")

    # ---- 6. distance bins, for the supplement
    distance_rows = []
    for model in args.models:
        subset = clean[(clean["Model"] == model) & (clean["Seed"] == -1)]
        for low, high in zip(DISTANCE_BINS[:-1], DISTANCE_BINS[1:]):
            binned = subset[(subset["abs_distance_bp"] >= low)
                            & (subset["abs_distance_bp"] < high)]
            context = {"model": model, "distance_bin": f"{low}-{high}"}
            distance_rows.append(measure(binned, marginal_auroc, "auroc",
                                         args.n_boot, rng, **context))
            distance_rows.append(measure(binned[binned["significant"] == 1],
                                         direction_agreement, "direction_agreement",
                                         args.n_boot, rng, **context))
    atomic_csv(pd.DataFrame(distance_rows), out / "distance_bins.csv")

    make_figures(gradient, primary, pd.DataFrame(matched_rows), out, args)

    atomic_json(
        {
            "analysis": (f"{COHORTS[args.cohort]['label']} variant evaluation "
                         f"on frozen SilentMethyl checkpoints"),
            "cohort": args.cohort,
            "purpose": "mentor requirement 6: independent variant evaluation",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stratum": args.stratum,
            "models": args.models,
            "seeds": args.seeds,
            "seed_minus_one_means": "mean delta across seeds (ensemble)",
            "significance_threshold": args.significance,
            "n_boot": args.n_boot,
            "block_size_bp": BLOCK_BP,
            "matching": {"tolerance_bp": args.match_tolerance,
                         "negatives_per_positive": args.match_ratio,
                         "replacement": False},
            "primary_claims": {
                "discrimination": ("AUROC of |delta M| against distance-matched "
                                   "negatives, reported beside the distance-only "
                                   "baseline. The marginal AUROC is NOT reportable "
                                   "alone: distance by itself reaches ~0.595."),
                "direction": ("signed Spearman rho and direction agreement against "
                              "the cohort effect at p < 5e-8, non-CpG-altering. "
                              "Direction has no mechanical relationship to distance, "
                              "so it is the confound-resistant claim."),
                "dilution": ("signal rises monotonically with association strength "
                             "and reaches chance in the null stratum -- the evidence "
                             "that the effect is real rather than fitted noise."),
                "equivalence": ("fusion minus sequence, paired and bootstrapped. An "
                                "interval spanning zero is the expected result: the "
                                "context vector is identical for REF and ALT, so it "
                                "cannot create a variant effect, only modulate one "
                                "through the gate."),
            },
            "caveats": {
                "effect_scale": COHORTS[args.cohort]["effect_scale"],
                "tissue": COHORTS[args.cohort]["tissue_caveat"],
                "dependence": ("Pairs are in LD. All intervals are 1 Mb block "
                               "bootstraps; naive intervals would be far too narrow."),
            },
        },
        out / "run_summary.json",
    )

    print_report(primary, pd.DataFrame(matched_rows), gradient,
                 pd.DataFrame(paired_rows), args, out)
    return 0


def make_figures(gradient: pd.DataFrame, primary: pd.DataFrame,
                 matched: pd.DataFrame, out: Path, args: argparse.Namespace) -> None:
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # Figure 1 -- the dilution gradient. The argument that the signal is real.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    labels = [label for _, _, label in SIGNIFICANCE_STRATA]
    for axis, metric, null, title in (
        (axes[0], "signed_rho", 0.0, "Signed Spearman correlation"),
        (axes[1], "direction_agreement", 0.5, "Direction agreement"),
    ):
        for model in args.models:
            subset = gradient[(gradient["model"] == model)
                              & (gradient["metric"] == metric)]
            subset = subset.set_index("stratum").reindex(labels).reset_index()
            x = np.arange(len(labels))
            axis.errorbar(
                x, subset["value"],
                yerr=[subset["value"] - subset["ci_low"],
                      subset["ci_high"] - subset["value"]],
                **style_for(model, args.models.index(model)),
                capsize=3, markersize=5, linewidth=1.6, label=model)
        axis.axhline(null, color="0.45", linewidth=0.9, linestyle=(0, (1, 2)))
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(f"{COHORTS[args.cohort]['label']} association strength",
                        fontsize=9)
        axis.grid(axis="y", alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("value (95% block-bootstrap CI)", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Agreement with {COHORTS[args.cohort]['label']} rises with "
                 f"association strength and reaches chance where associations "
                 f"are null", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(plots / "significance_gradient.png", dpi=400, bbox_inches="tight",
                facecolor="white")
    fig.savefig(plots / "significance_gradient.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 2 -- discrimination, with the distance baseline in the same frame.
    fig, axis = plt.subplots(figsize=(7.4, 4.0))
    families = [
        ("auroc_marginal", "Marginal", "baseline"),
        ("auroc_within_distance_bin", "Within distance bin", "fusion"),
    ]
    # 'style' here names a palette entry, not a model, and both entries are
    # always present in COLOURS -- it is independent of --models.
    width = 0.32
    positions = np.arange(len(args.models))
    for index, (metric, label, style) in enumerate(families):
        values, errors = [], [[], []]
        for model in args.models:
            row = primary[(primary["model"] == model) & (primary["seed"] == -1)
                          & (primary["variant_class"] == "non_cpg_altering")
                          & (primary["metric"] == metric)]
            value = float(row["value"].iloc[0]) if len(row) else np.nan
            values.append(value)
            errors[0].append(value - float(row["ci_low"].iloc[0]) if len(row) else 0)
            errors[1].append(float(row["ci_high"].iloc[0]) - value if len(row) else 0)
        axis.bar(positions + (index - 0.5) * (width + 0.03), values, width,
                 yerr=errors, capsize=3,
                 color=COLOURS[style], alpha=0.85, label=label,
                 edgecolor="white", linewidth=0.8)
    if not matched.empty:
        for model in args.models:
            row = matched[(matched["model"] == model) & (matched["seed"] == -1)]
            if len(row):
                index = args.models.index(model)
                axis.plot([index - width, index + width],
                          [row["value"].iloc[0]] * 2, color="black",
                          linewidth=1.8, linestyle="-",
                          label="Distance-matched" if index == 0 else None)
    baseline = primary[(primary["seed"] == -1)
                       & (primary["variant_class"] == "non_cpg_altering")
                       & (primary["metric"] == "auroc_distance_only_baseline")]
    if len(baseline):
        axis.axhline(float(baseline["value"].mean()), color=COLOURS["baseline"],
                     linestyle=":", linewidth=1.6,
                     label="Distance alone (baseline)")
    axis.axhline(0.5, color="0.45", linewidth=0.9, linestyle=(0, (1, 2)))
    axis.set_xticks(positions, args.models)
    axis.set_ylim(0.45, 0.70)
    axis.set_ylabel("AUROC, significant vs null (95% CI)", fontsize=9)
    axis.set_title("Discrimination survives distance matching, but the marginal\n"
                   "AUROC does not exceed distance alone", fontsize=10.5)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.15)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(plots / "discrimination_vs_distance.png", dpi=400,
                bbox_inches="tight", facecolor="white")
    fig.savefig(plots / "discrimination_vs_distance.pdf", bbox_inches="tight")
    plt.close(fig)


def show(frame: pd.DataFrame, metric: str, **filters) -> str:
    subset = frame[frame["metric"] == metric]
    for key, value in filters.items():
        subset = subset[subset[key] == value]
    if subset.empty or not np.isfinite(subset["value"].iloc[0]):
        return "        n/a"
    row = subset.iloc[0]
    return f"{row['value']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"


def print_report(primary, matched, gradient, paired, args, out) -> None:
    print()
    print("=" * 78)
    print(f"{COHORTS[args.cohort]['label']} variant evaluation -- "
          f"{args.stratum} stratum, non-CpG-altering, seed ensemble")
    print(f"significance p < {args.significance:.0e} | "
          f"{args.n_boot} block-bootstrap resamples over {BLOCK_BP//1000} kb blocks")
    print("=" * 78)

    for model in args.models:
        common = {"model": model, "seed": -1, "variant_class": "non_cpg_altering"}
        print(f"\n{model.upper()}")
        print(f"  signed rho (significant)      "
              f"{show(primary, 'signed_rho_significant', **common)}")
        print(f"  direction agreement           "
              f"{show(primary, 'direction_agreement_significant', **common)}")
        print(f"  AUROC, marginal               "
              f"{show(primary, 'auroc_marginal', **common)}")
        print(f"  AUROC, within distance bin    "
              f"{show(primary, 'auroc_within_distance_bin', **common)}")
        if not matched.empty:
            print(f"  AUROC, distance-matched       "
                  f"{show(matched, 'auroc_distance_matched', model=model, seed=-1)}")
        print(f"  AUROC, distance alone         "
              f"{show(primary, 'auroc_distance_only_baseline', **common)}"
              "   <- the number to beat")

    if not paired.empty:
        print("\nFUSION MINUS SEQUENCE  (interval spanning zero = equivalent)")
        for _, row in paired.iterrows():
            name = row["metric"].replace("fusion_minus_sequence_", "")
            spans = (np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"])
                     and row["ci_low"] <= 0 <= row["ci_high"])
            verdict = "equivalent" if spans else "DIFFERENT"
            print(f"  {name:<28} {row['value']:+.4f} "
                  f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  {verdict}")

    print(f"\nDILUTION GRADIENT  ({args.models[0]}, seed ensemble)")
    print(f"  {'stratum':<14} {'n':>7} {'signed rho':>22} {'direction':>22}")
    for _, _, label in SIGNIFICANCE_STRATA:
        rho = gradient[(gradient["model"] == args.models[0])
                       & (gradient["metric"] == "signed_rho")
                       & (gradient["stratum"] == label)]
        agree = gradient[(gradient["model"] == args.models[0])
                         & (gradient["metric"] == "direction_agreement")
                         & (gradient["stratum"] == label)]
        if rho.empty:
            continue
        n = int(rho["n"].iloc[0])
        if not np.isfinite(rho["value"].iloc[0]):
            print(f"  {label:<14} {n:>7,} {'too few pairs':>22}")
            continue
        print(f"  {label:<14} {n:>7,} "
              f"{rho['value'].iloc[0]:>+9.4f} [{rho['ci_low'].iloc[0]:+.3f},"
              f"{rho['ci_high'].iloc[0]:+.3f}] "
              f"{agree['value'].iloc[0]:>9.4f} [{agree['ci_low'].iloc[0]:.3f},"
              f"{agree['ci_high'].iloc[0]:.3f}]")

    print(f"\noutput: {out}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
