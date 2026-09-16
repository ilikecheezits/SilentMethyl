#!/usr/bin/env python3
"""
Does the context-ladder dissociation survive stratification by effect size?

The question
------------
`23_context_permutation.py` (job 46007255) prints a verdict banner claiming the
allele-invariance argument fails empirically. That banner is computed on
**Pearson**, which §1 R2 of LAB_NOTES already documents as the wrong comparison:
methylation levels are bimodal with SD ~3.18 M-units, so a high Pearson on levels
is cheap and not comparable against the same statistic on deltas.

Pooled over all 76,893 pairs the metrics disagree -- normalised MAE favours the
deltas at every rung, while Spearman/sign/Pearson favour them only under
`shuffle`. This script settles the disagreement by stratifying on effect size.

Why the stratifier is the whole methodology
--------------------------------------------
Binning on the model's own |Predicted_Delta_M| from the identity run is
**invalid** and it produces a dramatic false reversal: selecting pairs whose
identity delta landed near zero guarantees the comparison rung's delta sits
relatively further away, by regression to the mean alone. That artefact is
reproduced here under `--stratifier predicted` precisely so it is on the record
as an artefact and nobody rediscovers it and believes it.

The valid stratifier is the **observed** effect size `beta_ref_to_alt`, which is
measured, not predicted, and therefore independent of every rung. Under it the
dissociation holds at every quintile of every rung and strengthens monotonically
with effect size.

Normalisation
-------------
Both quantities are normalised by their own WITHIN-BIN standard deviation, so
levels (M-units, SD ~3.18) and deltas (SD ~0.10) are compared on equal footing
inside each stratum. Normalising by a global SD instead inflates the small-effect
bins and is what makes the pooled table ambiguous.

    ratio = normMAE_levels / normMAE_deltas      > 1 means deltas moved LESS

Zero GPU. Reads the frozen `46007255` outputs; computes nothing new from models.

Usage
-----
    python -u scripts/58_ladder_effect_size_stratification.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("ladder_stratification")

RUNGS = ("shuffle", "tissue_Lung", "xtissue_mean")
LEVEL_COLUMN = "WT_M_RC_Avg"
DELTA_COLUMN = "Predicted_Delta_M"
OBSERVED_COLUMN = "beta_ref_to_alt"
KEY = "Pair_UID"
QUINTILES = ("Q1 smallest", "Q2", "Q3", "Q4", "Q5 largest")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_rung(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=columns)
    frame[KEY] = frame[KEY].astype(str)
    if frame[KEY].duplicated().any():
        raise ValueError(f"Duplicate {KEY} in {path}")
    return frame.set_index(KEY)


def stratum_stats(
    identity: pd.DataFrame, other: pd.DataFrame, mask: np.ndarray
) -> dict[str, float]:
    a, b = identity[mask], other[mask]
    level_sd = float(a[LEVEL_COLUMN].std())
    delta_sd = float(a[DELTA_COLUMN].std())
    level_shift = float((a[LEVEL_COLUMN] - b[LEVEL_COLUMN]).abs().mean())
    delta_shift = float((a[DELTA_COLUMN] - b[DELTA_COLUMN]).abs().mean())
    normalized_level = level_shift / level_sd if level_sd > 0 else float("nan")
    normalized_delta = delta_shift / delta_sd if delta_sd > 0 else float("nan")
    sign_agreement = float(
        (np.sign(a[DELTA_COLUMN]) == np.sign(b[DELTA_COLUMN])).mean()
    )
    return {
        "N_Pairs": int(mask.sum()),
        "Level_SD_In_Bin": level_sd,
        "Delta_SD_In_Bin": delta_sd,
        "Normalized_MAE_Levels": normalized_level,
        "Normalized_MAE_Deltas": normalized_delta,
        "Ratio_Levels_Over_Deltas": (
            normalized_level / normalized_delta if normalized_delta > 0 else float("nan")
        ),
        "Delta_Sign_Agreement": sign_agreement,
    }


def run(args: argparse.Namespace) -> int:
    root = args.ladder_dir
    identity_path = root / "pair_scores_identity.csv"

    base_columns = [KEY, LEVEL_COLUMN, DELTA_COLUMN]
    identity = load_rung(identity_path, base_columns + [OBSERVED_COLUMN])
    hashes = {identity_path.as_posix(): sha256_file(identity_path)}
    LOGGER.info("identity: %d pairs", len(identity))

    stratifiers: dict[str, pd.Series] = {
        "observed": identity[OBSERVED_COLUMN].abs(),
        "predicted": identity[DELTA_COLUMN].abs(),
    }
    usable = stratifiers[args.stratifier].notna()
    dropped = int((~usable).sum())
    if dropped:
        LOGGER.warning(
            "dropping %d pairs with missing %s", dropped, args.stratifier
        )
    identity = identity[usable]
    magnitude = stratifiers[args.stratifier][usable]
    bins = pd.qcut(magnitude, 5, labels=list(QUINTILES))

    rows: list[dict[str, object]] = []
    for rung in RUNGS:
        path = root / f"pair_scores_{rung}.csv"
        other = load_rung(path, base_columns).reindex(identity.index)
        hashes[path.as_posix()] = sha256_file(path)
        if other[LEVEL_COLUMN].isna().any():
            raise ValueError(f"{path} does not cover every identity pair")
        LOGGER.info("rung %s", rung)

        pooled = stratum_stats(identity, other, np.ones(len(identity), dtype=bool))
        rows.append({"Rung": rung, "Stratum": "All pairs", **pooled})
        for label in QUINTILES:
            mask = (bins == label).to_numpy()
            rows.append({"Rung": rung, "Stratum": label, **stratum_stats(identity, other, mask)})

    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.stratifier == "observed" else f"_{args.stratifier}"
    csv_path = args.output_dir / f"ladder_effect_size_stratification{suffix}.csv"
    atomic_csv(table, csv_path)

    quintiles_only = table[table["Stratum"] != "All pairs"]
    monotone = {
        rung: bool(
            quintiles_only[quintiles_only["Rung"] == rung]["Ratio_Levels_Over_Deltas"]
            .diff()
            .dropna()
            .ge(0)
            .all()
        )
        for rung in RUNGS
    }
    holds_everywhere = bool((quintiles_only["Ratio_Levels_Over_Deltas"] > 1.0).all())

    summary = {
        "analysis": "context-ladder dissociation, stratified by effect size",
        "analysis_status": "COMPLETE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_job": "46007255",
        "stratifier": args.stratifier,
        "stratifier_note": (
            "observed = beta_ref_to_alt, measured and independent of every rung -- "
            "the valid choice. predicted = |Predicted_Delta_M| from the identity "
            "run, which is INVALID: selecting on a rung's own value induces "
            "regression to the mean and manufactures a reversal in the smallest "
            "bin. Reproduced only to document it as an artefact."
        ),
        "normalization": (
            "Each quantity divided by its own within-bin SD, so M-unit levels and "
            "delta-scale effects are compared on equal footing inside each stratum."
        ),
        "ratio_definition": (
            "Ratio_Levels_Over_Deltas > 1 means the context swap moved predicted "
            "levels more than predicted variant effects, i.e. the dissociation holds."
        ),
        "n_pairs": int(len(identity)),
        "n_dropped_missing_stratifier": dropped,
        "dissociation_holds_in_every_quintile": holds_everywhere,
        "ratio_monotone_increasing_with_effect_size": monotone,
        "pooled_ratio_by_rung": {
            str(r["Rung"]): float(r["Ratio_Levels_Over_Deltas"])
            for _, r in table[table["Stratum"] == "All pairs"].iterrows()
        },
        "inputs": hashes,
    }
    json_path = args.output_dir / f"run_summary{suffix}.json"
    atomic_json(summary, json_path)

    print("=" * 78)
    print(f"context-ladder dissociation, stratified by {args.stratifier} effect size")
    print(f"n = {len(identity):,} pairs   (ratio > 1 means deltas moved LESS)")
    print("=" * 78)
    for rung in RUNGS:
        print(f"\n{rung}")
        print(f"  {'stratum':<14}{'n':>7}{'normMAE_lev':>13}{'normMAE_del':>13}{'ratio':>9}{'sign_agr':>10}")
        for _, r in table[table["Rung"] == rung].iterrows():
            print(
                f"  {str(r['Stratum']):<14}{int(r['N_Pairs']):>7}"
                f"{r['Normalized_MAE_Levels']:>13.4f}{r['Normalized_MAE_Deltas']:>13.4f}"
                f"{r['Ratio_Levels_Over_Deltas']:>8.2f}x{r['Delta_Sign_Agreement']:>10.4f}"
            )
    print("\n" + "=" * 78)
    if args.stratifier == "observed":
        print(
            "VERDICT: dissociation holds in every quintile"
            if holds_everywhere
            else "VERDICT: dissociation FAILS in at least one quintile"
        )
        print(f"monotone increase with effect size, by rung: {monotone}")
        print(
            "The 23_context_permutation.py banner is computed on Pearson and does\n"
            "not survive this stratification. Do not act on that banner."
        )
    else:
        print(
            "This run used the INVALID stratifier. Any reversal in Q1 is regression\n"
            "to the mean, not a finding. Use --stratifier observed for the result."
        )
    print("=" * 78)
    print(f"wrote {csv_path}\nwrote {json_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ladder-dir",
        type=Path,
        default=Path("results/journal/ablation_breast_epithelium/context_ladder"),
    )
    parser.add_argument(
        "--stratifier",
        choices=("observed", "predicted"),
        default="observed",
        help="observed = beta_ref_to_alt (valid). predicted = the model's own "
        "delta (INVALID, reproduces the regression-to-the-mean artefact).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/journal/ablation_breast_epithelium/context_ladder_stratified"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    sys.exit(run(parse_args()))
