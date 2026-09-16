#!/usr/bin/env python3
"""
Where does the fusion gain concentrate?

The question
------------
Results 3.2 needs the converse of Task D. Task D (`56_transfer_failure.py`)
localises where zero-shot cross-tissue transfer FAILS. This localises where
adding epigenomic context HELPS, on the same held-out probes, so the two can be
read against each other.

`22_context_stratification.py` already produces a paired fusion-minus-sequence
beta MAE for four strata (CpG-island class, genomic region, ATAC quartile,
H3K27ac quartile). This script extends that in the two directions 3.2 needs:

  1. all SEVEN reference tracks, not just ATAC and H3K27ac;
  2. a paired AUROC difference alongside the paired beta MAE, block-bootstrapped
     the same way.

Absolute vs relative gain -- read this before quoting a number
--------------------------------------------------------------
Absolute beta-MAE gain is bounded by how much error there is to remove, so a
stratum with a high sequence-only error can show the largest absolute gain
purely because it starts worst. Both are reported:

    Fusion_Minus_Sequence_Beta_MAE      negative = fusion better (absolute)
    Relative_Beta_MAE_Reduction         fraction of sequence error removed
    Fusion_Minus_Sequence_ROC_AUC       positive = fusion better

They do not rank the strata the same way, and the difference is the finding.
Quote the relative reduction when comparing strata with different baselines.

Zero GPU. Post hoc stratification of frozen held-out predictions -- not a
training experiment and not a causal claim.

Split discipline
----------------
Held-out TEST probes of the breast-epithelium context ablation, chr8 + chr9,
26,570 probes. Same probe set as Task D.

Usage
-----
    python -u scripts/57_fusion_gain_stratified.py \
        --output-dir results/journal/ablation_breast_epithelium/fusion_gain_stratified
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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/silentmethyl_matplotlib")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger("fusion_gain")

TRACKS = [
    "Ref_ATAC_Signal",
    "Ref_H3K4me3_Signal",
    "Ref_H3K27ac_Signal",
    "Ref_H3K27me3_Signal",
    "Ref_H3K9me3_Signal",
    "Ref_H3K36me3_Signal",
    "Ref_H3K4me1_Signal",
]
QUARTILE_ORDER = ("Q1 low", "Q2", "Q3", "Q4 high", "Missing")
ISLAND_ORDER = ("Island", "Shore", "Shelf", "Open sea", "Unclassified")


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


def normalize_chr(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    return text.where(text.str.startswith("chr"), "chr" + text)


def quartile_stratum(frame: pd.DataFrame, signal: str) -> pd.Series:
    """Quartiles of the observed values; unobserved probes form a Missing bin.

    Identical binning to 22_context_stratification.py, so the ATAC and H3K27ac
    columns here reproduce that script's strata exactly.
    """
    missing_column = f"{signal}_Missing"
    if missing_column in frame:
        missing = frame[missing_column].astype(float).astype(bool)
    else:
        missing = frame[signal].isna()
    observed = pd.to_numeric(frame.loc[~missing, signal], errors="coerce")
    if observed.isna().any():
        raise ValueError(f"Observed {signal} values contain nonnumeric entries")

    stratum = pd.Series("Missing", index=frame.index, dtype=object)
    if len(observed):
        percentile = observed.rank(method="average", pct=True)
        labels = pd.cut(
            percentile,
            bins=[0.0, 0.25, 0.50, 0.75, 1.0],
            labels=list(QUARTILE_ORDER[:4]),
            include_lowest=True,
        )
        stratum.loc[observed.index] = labels.astype(str)
    return stratum


def load_test_metadata(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    required = ["probeID", "chr", "pos"] + TRACKS
    missing = [c for c in required if c not in header.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    usecols = required + [
        f"{t}_Missing" for t in TRACKS if f"{t}_Missing" in header.columns
    ]
    frame = pd.read_csv(path, usecols=usecols)
    frame["probeID"] = frame["probeID"].astype(str)
    frame["chr"] = normalize_chr(frame["chr"])
    frame["pos"] = pd.to_numeric(frame["pos"], errors="raise").astype(np.int64)
    if frame["probeID"].duplicated().any():
        raise ValueError(f"Duplicate probeID values in {path}")
    for track in TRACKS:
        frame[f"{track}_Stratum"] = quartile_stratum(frame, track)
    return frame


def load_island_annotation(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, sep="\t", nrows=0)
    probe_column = next(
        (n for n in ("probeID", "Probe_ID", "IlmnID", "Name") if n in header.columns),
        None,
    )
    relation_column = next(
        (
            n
            for n in ("CGIposition", "Relation_to_UCSC_CpG_Island", "Relation_to_Island")
            if n in header.columns
        ),
        None,
    )
    if probe_column is None or relation_column is None:
        raise ValueError(f"Could not identify probe/relation columns in {path}")
    frame = pd.read_csv(path, sep="\t", usecols=[probe_column, relation_column])
    frame = frame.rename(
        columns={probe_column: "probeID", relation_column: "CGIposition"}
    )
    frame["probeID"] = frame["probeID"].astype(str)

    def collapse(value: str) -> str:
        lower = str(value).lower().replace("-", "_").replace(" ", "_")
        if lower in {"na", "nan", "none", "", "opensea", "open_sea"}:
            return "Open sea"
        if "shore" in lower:
            return "Shore"
        if "shelf" in lower:
            return "Shelf"
        if "island" in lower or lower == "cgi":
            return "Island"
        return "Unclassified"

    frame["CpG_Island_Context"] = (
        frame["CGIposition"].fillna("NA").astype(str).str.strip().map(collapse)
    )
    return frame[["probeID", "CpG_Island_Context"]]


def load_paired_predictions(
    template: str, seeds: list[int], models: tuple[str, str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Seed-ensembled per-probe predictions for the two arms being compared."""
    required = [
        "probeID",
        "true_beta",
        "binary_true",
        "pred_beta_rc_avg",
        "class_prob_rc_avg",
    ]
    rows: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for seed in seeds:
        for model in models:
            path = Path(template.format(seed=seed, model=model))
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            gap = [c for c in required if c not in frame.columns]
            if gap:
                raise ValueError(f"{path} is missing required columns: {gap}")
            frame = frame[required].copy()
            frame["probeID"] = frame["probeID"].astype(str)
            if frame["probeID"].duplicated().any():
                raise ValueError(f"Duplicate probeID values in {path}")
            frame["Seed"] = int(seed)
            frame["Model"] = str(model)
            rows.append(frame)
            hashes[path.as_posix()] = sha256_file(path)

    long = pd.concat(rows, ignore_index=True)

    # Truth must be identical across every seed/model file, or the pairing is
    # comparing different targets.
    reference = (
        long[(long["Seed"] == seeds[0]) & (long["Model"] == models[0])]
        .sort_values("probeID")
        .reset_index(drop=True)
    )
    for (seed, model), current in long.groupby(["Seed", "Model"], sort=False):
        current = current.sort_values("probeID").reset_index(drop=True)
        if current["probeID"].tolist() != reference["probeID"].tolist():
            raise ValueError(f"Probe set differs for seed={seed}, model={model}")
        for column in ("true_beta", "binary_true"):
            if not np.allclose(current[column], reference[column], rtol=0, atol=1e-10):
                raise ValueError(
                    f"Truth differs for seed={seed}, model={model}, column={column}"
                )

    truth = reference[["probeID", "true_beta", "binary_true"]]
    means = long.groupby(["Model", "probeID"], as_index=False)[
        ["pred_beta_rc_avg", "class_prob_rc_avg"]
    ].mean()
    wide = means.pivot(
        index="probeID", columns="Model", values=["pred_beta_rc_avg", "class_prob_rc_avg"]
    )
    wide.columns = [f"{model}_{value}" for value, model in wide.columns]
    wide = wide.reset_index()
    return truth.merge(wide, on="probeID", validate="one_to_one"), hashes


def block_codes(group: pd.DataFrame, block_size_bp: int) -> tuple[np.ndarray, int]:
    labels = (
        group["chr"].astype(str)
        + ":"
        + (group["pos"].astype(np.int64) // block_size_bp).astype(str)
    )
    codes, unique = pd.factorize(labels, sort=True)
    return codes, len(unique)


def weighted_auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    positive = weight[y == 1].sum()
    negative = weight[y == 0].sum()
    if positive <= 0 or negative <= 0:
        return float("nan")
    return float(roc_auc_score(y, score, sample_weight=weight))


def stratum_row(
    group: pd.DataFrame,
    grouping: str,
    stratum: str,
    block_size_bp: int,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    sequence_error = group["sequence_absolute_error"].to_numpy(dtype=float)
    fusion_error = group["fusion_absolute_error"].to_numpy(dtype=float)
    delta = fusion_error - sequence_error
    labels = group["binary_true"].to_numpy(dtype=int)
    sequence_score = group["sequence_class_prob_rc_avg"].to_numpy(dtype=float)
    fusion_score = group["fusion_class_prob_rc_avg"].to_numpy(dtype=float)

    sequence_mae = float(sequence_error.mean())
    fusion_mae = float(fusion_error.mean())
    observed_delta = float(delta.mean())
    ones = np.ones(len(group), dtype=float)
    sequence_auc = weighted_auc(labels, sequence_score, ones)
    fusion_auc = weighted_auc(labels, fusion_score, ones)

    row: dict[str, object] = {
        "Grouping": grouping,
        "Stratum": stratum,
        "N_CpGs": int(len(group)),
        "N_Positive": int(labels.sum()),
        "Sequence_Beta_MAE": sequence_mae,
        "Fusion_Beta_MAE": fusion_mae,
        "Fusion_Minus_Sequence_Beta_MAE": observed_delta,
        "Relative_Beta_MAE_Reduction": (
            float(-observed_delta / sequence_mae) if sequence_mae > 0 else float("nan")
        ),
        "Sequence_ROC_AUC": sequence_auc,
        "Fusion_ROC_AUC": fusion_auc,
        "Fusion_Minus_Sequence_ROC_AUC": fusion_auc - sequence_auc,
    }

    codes, n_blocks = block_codes(group, block_size_bp)
    row["N_Genomic_Blocks"] = int(n_blocks)
    nan_fields = {
        "Beta_MAE_Difference_CI_Low": float("nan"),
        "Beta_MAE_Difference_CI_High": float("nan"),
        "Beta_MAE_Difference_Probability_Greater_Equal_Zero": float("nan"),
        "Relative_Reduction_CI_Low": float("nan"),
        "Relative_Reduction_CI_High": float("nan"),
        "ROC_AUC_Difference_CI_Low": float("nan"),
        "ROC_AUC_Difference_CI_High": float("nan"),
        "ROC_AUC_Difference_Probability_Less_Equal_Zero": float("nan"),
    }
    if n_blocks < 2 or replicates <= 0:
        row.update(nan_fields)
        return row

    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(
        n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=replicates
    ).astype(float)

    counts = np.bincount(codes, minlength=n_blocks).astype(float)
    delta_sums = np.bincount(codes, weights=delta, minlength=n_blocks).astype(float)
    sequence_sums = np.bincount(
        codes, weights=sequence_error, minlength=n_blocks
    ).astype(float)

    denominator = multiplicity @ counts
    delta_draws = (multiplicity @ delta_sums) / denominator
    sequence_draws = (multiplicity @ sequence_sums) / denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_draws = np.where(
            sequence_draws > 0, -delta_draws / sequence_draws, np.nan
        )

    # AUROC is not a per-CpG mean, so each replicate is recomputed with the
    # block multiplicities carried as sample weights.
    auc_draws = np.full(replicates, np.nan, dtype=float)
    for index in range(replicates):
        weight = multiplicity[index][codes]
        keep = weight > 0
        if not keep.any():
            continue
        sub_weight = weight[keep]
        sub_labels = labels[keep]
        if np.unique(sub_labels).size < 2:
            continue
        auc_draws[index] = weighted_auc(
            sub_labels, fusion_score[keep], sub_weight
        ) - weighted_auc(sub_labels, sequence_score[keep], sub_weight)

    finite_auc = auc_draws[np.isfinite(auc_draws)]
    finite_relative = relative_draws[np.isfinite(relative_draws)]
    row.update({
        "Beta_MAE_Difference_CI_Low": float(np.quantile(delta_draws, 0.025)),
        "Beta_MAE_Difference_CI_High": float(np.quantile(delta_draws, 0.975)),
        "Beta_MAE_Difference_Probability_Greater_Equal_Zero": float(
            np.mean(delta_draws >= 0)
        ),
        "Relative_Reduction_CI_Low": (
            float(np.quantile(finite_relative, 0.025)) if finite_relative.size else float("nan")
        ),
        "Relative_Reduction_CI_High": (
            float(np.quantile(finite_relative, 0.975)) if finite_relative.size else float("nan")
        ),
        "ROC_AUC_Difference_CI_Low": (
            float(np.quantile(finite_auc, 0.025)) if finite_auc.size else float("nan")
        ),
        "ROC_AUC_Difference_CI_High": (
            float(np.quantile(finite_auc, 0.975)) if finite_auc.size else float("nan")
        ),
        "ROC_AUC_Difference_Probability_Less_Equal_Zero": (
            float(np.mean(finite_auc <= 0)) if finite_auc.size else float("nan")
        ),
    })
    return row


def ordered_strata(grouping: str, present: list[str]) -> list[str]:
    order = ISLAND_ORDER if grouping == "CpG_Island_Context" else QUARTILE_ORDER
    ranked = [s for s in order if s in present]
    return ranked + sorted(s for s in present if s not in ranked)


def run(args: argparse.Namespace) -> int:
    LOGGER.info("loading held-out metadata: %s", args.test_path)
    test = load_test_metadata(args.test_path)
    islands = load_island_annotation(args.cpg_island_annotation)
    LOGGER.info("loading predictions: seeds=%s", args.seeds)
    predictions, hashes = load_paired_predictions(
        args.prediction_template, list(args.seeds), ("sequence", "fusion")
    )

    frame = test.merge(predictions, on="probeID", validate="one_to_one")
    if len(frame) != len(predictions):
        raise ValueError(
            f"Probe join lost rows: {len(predictions)} predictions -> {len(frame)} joined"
        )
    frame = frame.merge(islands, on="probeID", how="left", validate="one_to_one")
    frame["CpG_Island_Context"] = frame["CpG_Island_Context"].fillna("Open sea")

    frame["sequence_absolute_error"] = (
        frame["sequence_pred_beta_rc_avg"] - frame["true_beta"]
    ).abs()
    frame["fusion_absolute_error"] = (
        frame["fusion_pred_beta_rc_avg"] - frame["true_beta"]
    ).abs()

    groupings = ["CpG_Island_Context"] + [f"{t}_Stratum" for t in TRACKS]
    rows: list[dict[str, object]] = [
        stratum_row(
            frame,
            grouping="All",
            stratum="All held-out CpGs",
            block_size_bp=args.block_size_bp,
            replicates=args.bootstrap_replicates,
            seed=args.random_seed,
        )
    ]
    for index, grouping in enumerate(groupings):
        present = sorted(frame[grouping].astype(str).unique())
        for offset, stratum in enumerate(ordered_strata(grouping, present)):
            group = frame[frame[grouping].astype(str) == stratum]
            LOGGER.info("  %s = %s (n=%d)", grouping, stratum, len(group))
            rows.append(
                stratum_row(
                    group,
                    grouping=grouping,
                    stratum=stratum,
                    block_size_bp=args.block_size_bp,
                    replicates=args.bootstrap_replicates,
                    seed=args.random_seed + 1000 * (index + 1) + offset,
                )
            )

    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "fusion_gain_stratified.csv"
    atomic_csv(table, csv_path)

    stratified = table[table["Grouping"] != "All"]
    islands_only = stratified[stratified["Grouping"] == "CpG_Island_Context"]
    tracks_only = stratified[stratified["Grouping"] != "CpG_Island_Context"]
    tracks_top = tracks_only[tracks_only["Stratum"] == "Q4 high"]

    def _best(subset: pd.DataFrame, column: str, largest: bool) -> dict[str, object]:
        if subset.empty:
            return {}
        row = subset.loc[subset[column].idxmax() if largest else subset[column].idxmin()]
        return {
            "grouping": str(row["Grouping"]),
            "stratum": str(row["Stratum"]),
            "n": int(row["N_CpGs"]),
            "value": float(row[column]),
        }

    summary = {
        "analysis": "where the fusion gain concentrates, by genomic region",
        "analysis_status": "COMPLETE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "arms": ["sequence", "fusion"],
        "seeds": list(args.seeds),
        "seed_note": (
            "Seed-ensembled predictions, matching 22_context_stratification.py so the "
            "ATAC and H3K27ac rows reproduce that script's frozen numbers."
        ),
        "heldout_cpg_count": int(len(frame)),
        "block_size_bp": int(args.block_size_bp),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "random_seed": int(args.random_seed),
        "tracks": TRACKS,
        "interpretation": (
            "Post hoc descriptive stratification of frozen held-out predictions; not "
            "an additional training experiment and not a causal claim. Negative "
            "Fusion_Minus_Sequence_Beta_MAE and positive Fusion_Minus_Sequence_ROC_AUC "
            "both mean fusion is better."
        ),
        "absolute_vs_relative_note": (
            "Absolute beta-MAE gain is bounded by the sequence-only error in the "
            "stratum, so strata that start worse can show the largest absolute gain "
            "with no greater fractional benefit. Compare strata on "
            "Relative_Beta_MAE_Reduction."
        ),
        "largest_absolute_beta_gain": _best(
            stratified, "Fusion_Minus_Sequence_Beta_MAE", largest=False
        ),
        "largest_relative_beta_gain": _best(
            stratified, "Relative_Beta_MAE_Reduction", largest=True
        ),
        "largest_auroc_gain": _best(
            stratified, "Fusion_Minus_Sequence_ROC_AUC", largest=True
        ),
        "island_context_relative_gain": {
            str(r["Stratum"]): float(r["Relative_Beta_MAE_Reduction"])
            for _, r in islands_only.iterrows()
        },
        "top_quartile_relative_gain_by_track": {
            str(r["Grouping"]).replace("_Stratum", ""): float(
                r["Relative_Beta_MAE_Reduction"]
            )
            for _, r in tracks_top.iterrows()
        },
        "inputs": hashes
        | {
            args.test_path.as_posix(): sha256_file(args.test_path),
            args.cpg_island_annotation.as_posix(): sha256_file(
                args.cpg_island_annotation
            ),
        },
    }
    json_path = args.output_dir / "run_summary.json"
    atomic_json(summary, json_path)

    print("=" * 78)
    print(f"fusion gain stratified -- {len(frame):,} held-out CpGs, seeds {list(args.seeds)}")
    print("=" * 78)
    header = (
        f"{'stratum':<26}{'n':>7}{'seq MAE':>10}{'fus MAE':>10}"
        f"{'abs gain':>11}{'rel':>8}{'dAUROC':>10}"
    )
    for grouping in ["All"] + groupings:
        block = table[table["Grouping"] == grouping]
        if block.empty:
            continue
        print(f"\n{grouping}")
        print(header)
        for _, row in block.iterrows():
            print(
                f"  {str(row['Stratum']):<24}{int(row['N_CpGs']):>7}"
                f"{row['Sequence_Beta_MAE']:>10.4f}{row['Fusion_Beta_MAE']:>10.4f}"
                f"{row['Fusion_Minus_Sequence_Beta_MAE']:>+11.4f}"
                f"{row['Relative_Beta_MAE_Reduction']*100:>7.1f}%"
                f"{row['Fusion_Minus_Sequence_ROC_AUC']:>+10.4f}"
            )
    print("=" * 78)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("data/datafiles_breast_epithelium/test.csv"),
    )
    parser.add_argument(
        "--prediction-template",
        default="results/journal/ablation_breast_epithelium/seed{seed}/{model}/predictions.csv",
    )
    parser.add_argument(
        "--cpg-island-annotation",
        type=Path,
        default=Path("data/HM450.hg38.manifest.CpGIsland.tsv.gz"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--block-size-bp", type=int, default=1_000_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--random-seed", type=int, default=20260915)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/journal/ablation_breast_epithelium/fusion_gain_stratified"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    sys.exit(run(parse_args()))
