#!/usr/bin/env python3
"""Post hoc biological-context analysis using existing SilentMethyl outputs.

This script does not load model checkpoints or perform model inference.  It
combines the held-out prediction tables from seeds 42--44 with the processed
test metadata and the HM450 CpG-island annotation.  It then asks two focused
questions:

1. How do context-only, sequence-only, and fusion performance vary across
   CpG-island position and MCF-10A ATAC-signal strata?
2. How do predicted candidate responses and external mQTL agreement vary with
   variant--CpG distance?

All stratified results are descriptive, post hoc analyses.  The primary model
comparison remains the prespecified chromosome-held-out evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


LOGGER = logging.getLogger("biological_context")
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_MODELS = ("epi", "sequence", "fusion")
ISLAND_ORDER = ("Island", "Shore", "Shelf", "Open sea")
ATAC_ORDER = ("Q1 low", "Q2", "Q3", "Q4 high", "Missing")
DISTANCE_ORDER = ("0--50", "51--100", "101--250", "251--500")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", type=Path, default=Path("data/datafiles/test.csv"))
    parser.add_argument(
        "--prediction-template",
        default="results/journal/seed{seed}/{model}/predictions.csv",
    )
    parser.add_argument(
        "--cpg-island-annotation",
        type=Path,
        default=Path("data/HM450.hg38.manifest.CpGIsland.tsv.gz"),
    )
    parser.add_argument(
        "--candidate-path",
        type=Path,
        default=Path("results/journal/candidates/candidate_matched_background_statistics.csv"),
    )
    parser.add_argument(
        "--mqtl-path",
        type=Path,
        default=Path("results/journal/egtex_mqtl_positive_control/mqtl_predictions_seed_aggregate.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/journal/biological_context"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--block-size-bp", type=int, default=1_000_000)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--random-seed", type=int, default=20260814)
    return parser.parse_args()


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


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path | str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def normalize_chr(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    return text.where(text.str.startswith("chr"), "chr" + text)


def load_test_metadata(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0)
    required = ["probeID", "chr", "pos", "Ref_ATAC_Signal"]
    require_columns(header, required, path)
    usecols = required + [
        column
        for column in ("Ref_ATAC_Signal_Missing",)
        if column in header.columns
    ]
    frame = pd.read_csv(path, usecols=usecols)
    frame["probeID"] = frame["probeID"].astype(str)
    frame["chr"] = normalize_chr(frame["chr"])
    frame["pos"] = pd.to_numeric(frame["pos"], errors="raise").astype(np.int64)
    if frame["probeID"].duplicated().any():
        raise ValueError(f"Duplicate probeID values in {path}")

    if "Ref_ATAC_Signal_Missing" not in frame:
        frame["Ref_ATAC_Signal_Missing"] = frame["Ref_ATAC_Signal"].isna().astype(int)
    missing = frame["Ref_ATAC_Signal_Missing"].astype(bool)
    observed = pd.to_numeric(frame.loc[~missing, "Ref_ATAC_Signal"], errors="coerce")
    if observed.isna().any():
        raise ValueError("Observed ATAC values contain nonnumeric or missing entries")

    frame["ATAC_Stratum"] = "Missing"
    if len(observed):
        percentile = observed.rank(method="average", pct=True)
        labels = pd.cut(
            percentile,
            bins=[0.0, 0.25, 0.50, 0.75, 1.0],
            labels=list(ATAC_ORDER[:4]),
            include_lowest=True,
        )
        frame.loc[observed.index, "ATAC_Stratum"] = labels.astype(str)
    return frame


def load_island_annotation(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing CpG-island annotation: {path}. Download the file listed in instructions.md."
        )
    header = pd.read_csv(path, sep="\t", nrows=0)
    probe_column = next(
        (name for name in ("probeID", "Probe_ID", "IlmnID", "Name") if name in header.columns),
        None,
    )
    relation_column = next(
        (
            name
            for name in (
                "CGIposition",
                "Relation_to_UCSC_CpG_Island",
                "Relation_to_Island",
            )
            if name in header.columns
        ),
        None,
    )
    if probe_column is None or relation_column is None:
        raise ValueError(
            f"Could not identify probe/relation columns in {path}; columns={list(header.columns)}"
        )
    frame = pd.read_csv(path, sep="\t", usecols=[probe_column, relation_column])
    frame = frame.rename(columns={probe_column: "probeID", relation_column: "CGIposition"})
    frame["probeID"] = frame["probeID"].astype(str)
    if frame["probeID"].duplicated().any():
        raise ValueError(f"Duplicate probeID values in {path}")

    raw = frame["CGIposition"].fillna("NA").astype(str).str.strip()

    def collapse(value: str) -> str:
        lower = value.lower().replace("-", "_").replace(" ", "_")
        if lower in {"na", "nan", "none", "", "opensea", "open_sea"}:
            return "Open sea"
        if "shore" in lower:
            return "Shore"
        if "shelf" in lower:
            return "Shelf"
        if "island" in lower or lower == "cgi":
            return "Island"
        return "Unclassified"

    frame["CpG_Island_Context"] = raw.map(collapse)
    return frame[["probeID", "CpG_Island_Context"]]


def load_predictions(
    template: str,
    seeds: list[int],
    models: list[str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    required = [
        "probeID",
        "chr",
        "pos",
        "true_beta",
        "true_m",
        "binary_true",
        "pred_beta_rc_avg",
        "pred_m_rc_avg",
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
            require_columns(frame, required, path)
            frame = frame[required].copy()
            frame["probeID"] = frame["probeID"].astype(str)
            frame["chr"] = normalize_chr(frame["chr"])
            frame["pos"] = pd.to_numeric(frame["pos"], errors="raise").astype(np.int64)
            if frame["probeID"].duplicated().any():
                raise ValueError(f"Duplicate probeID values in {path}")
            frame["Seed"] = int(seed)
            frame["Model"] = str(model)
            rows.append(frame)
            hashes[path.as_posix()] = sha256_file(path)

    long = pd.concat(rows, ignore_index=True)
    reference = long[(long["Seed"] == seeds[0]) & (long["Model"] == models[0])]
    reference = reference.sort_values("probeID").reset_index(drop=True)
    for (seed, model), current in long.groupby(["Seed", "Model"], sort=False):
        current = current.sort_values("probeID").reset_index(drop=True)
        if current["probeID"].tolist() != reference["probeID"].tolist():
            raise ValueError(f"Held-out probe set differs for seed={seed}, model={model}")
        for column in ("true_beta", "true_m", "binary_true"):
            if not np.allclose(current[column], reference[column], rtol=0, atol=1e-10):
                raise ValueError(f"Truth differs for seed={seed}, model={model}, column={column}")
    return long, hashes


def ensemble_predictions(long: pd.DataFrame) -> pd.DataFrame:
    truth = long.sort_values(["Seed", "Model"]).drop_duplicates("probeID")[[
        "probeID", "chr", "pos", "true_beta", "true_m", "binary_true"
    ]]
    means = (
        long.groupby(["Model", "probeID"], as_index=False)[
            ["pred_beta_rc_avg", "pred_m_rc_avg", "class_prob_rc_avg"]
        ]
        .mean()
    )
    return means.merge(truth, on="probeID", validate="many_to_one")


def safe_auc(y: pd.Series, scores: pd.Series) -> float:
    labels = y.to_numpy(dtype=int)
    if len(labels) < 2 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores.to_numpy(dtype=float)))


def metric_rows(frame: pd.DataFrame, grouping: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (model, stratum), group in frame.groupby(["Model", grouping], sort=False):
        beta_error = group["pred_beta_rc_avg"] - group["true_beta"]
        m_error = group["pred_m_rc_avg"] - group["true_m"]
        rows.append({
            "Grouping": grouping,
            "Stratum": str(stratum),
            "Model": str(model),
            "N_CpGs": int(len(group)),
            "N_Positive": int(group["binary_true"].sum()),
            "Beta_MAE": float(beta_error.abs().mean()),
            "Beta_RMSE": float(np.sqrt(np.mean(np.square(beta_error)))),
            "M_MAE": float(m_error.abs().mean()),
            "M_RMSE": float(np.sqrt(np.mean(np.square(m_error)))),
            "ROC_AUC": safe_auc(group["binary_true"], group["class_prob_rc_avg"]),
        })
    return rows


def paired_gain_frame(annotated: pd.DataFrame) -> pd.DataFrame:
    required_models = {"sequence", "fusion"}
    available = set(annotated["Model"].unique())
    if not required_models.issubset(available):
        raise ValueError(f"Need sequence and fusion predictions; available={sorted(available)}")
    truth = annotated.drop_duplicates("probeID")[[
        "probeID", "chr", "pos", "true_beta", "CpG_Island_Context", "ATAC_Stratum"
    ]]
    wide = annotated.pivot(
        index="probeID", columns="Model", values="pred_beta_rc_avg"
    ).reset_index()
    frame = truth.merge(wide, on="probeID", validate="one_to_one")
    frame["Sequence_Absolute_Error"] = (frame["sequence"] - frame["true_beta"]).abs()
    frame["Fusion_Absolute_Error"] = (frame["fusion"] - frame["true_beta"]).abs()
    frame["Fusion_Minus_Sequence_Absolute_Error"] = (
        frame["Fusion_Absolute_Error"] - frame["Sequence_Absolute_Error"]
    )
    return frame


def block_interval(
    group: pd.DataFrame,
    block_size_bp: int,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    values = group["Fusion_Minus_Sequence_Absolute_Error"].to_numpy(dtype=float)
    block_labels = (
        group["chr"].astype(str)
        + ":"
        + (group["pos"].astype(np.int64) // block_size_bp).astype(str)
    )
    codes, unique = pd.factorize(block_labels, sort=True)
    n_blocks = len(unique)
    observed = float(np.mean(values))
    if n_blocks < 2 or replicates <= 0:
        return {
            "Fusion_Minus_Sequence_Beta_MAE": observed,
            "Bootstrap_CI_Low": float("nan"),
            "Bootstrap_CI_High": float("nan"),
            "Bootstrap_Probability_Difference_Greater_Equal_Zero": float("nan"),
            "N_Genomic_Blocks": int(n_blocks),
        }

    counts_by_block = np.bincount(codes, minlength=n_blocks).astype(float)
    sums_by_block = np.bincount(codes, weights=values, minlength=n_blocks).astype(float)
    rng = np.random.default_rng(seed)
    multiplicity = rng.multinomial(
        n_blocks,
        np.full(n_blocks, 1.0 / n_blocks),
        size=replicates,
    ).astype(float)
    denominator = multiplicity @ counts_by_block
    distribution = (multiplicity @ sums_by_block) / denominator
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "Fusion_Minus_Sequence_Beta_MAE": observed,
        "Bootstrap_CI_Low": float(low),
        "Bootstrap_CI_High": float(high),
        "Bootstrap_Probability_Difference_Greater_Equal_Zero": float(
            np.mean(distribution >= 0)
        ),
        "N_Genomic_Blocks": int(n_blocks),
    }


def gain_rows(
    frame: pd.DataFrame,
    grouping: str,
    block_size_bp: int,
    replicates: int,
    random_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, (stratum, group) in enumerate(frame.groupby(grouping, sort=False)):
        sequence_mae = float(group["Sequence_Absolute_Error"].mean())
        fusion_mae = float(group["Fusion_Absolute_Error"].mean())
        row: dict[str, object] = {
            "Grouping": grouping,
            "Stratum": str(stratum),
            "N_CpGs": int(len(group)),
            "Sequence_Beta_MAE": sequence_mae,
            "Fusion_Beta_MAE": fusion_mae,
            "Sequence_Minus_Fusion_Beta_MAE": sequence_mae - fusion_mae,
        }
        row.update(block_interval(
            group,
            block_size_bp=block_size_bp,
            replicates=replicates,
            seed=random_seed + 1000 * (index + 1),
        ))
        rows.append(row)
    return rows


def distance_bin(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-0.001, 50, 100, 250, 500],
        labels=list(DISTANCE_ORDER),
        include_lowest=True,
    )


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3 or np.unique(a[finite]).size < 2 or np.unique(b[finite]).size < 2:
        return float("nan")
    return float(spearmanr(a[finite], b[finite]).statistic)


def summarize_variant_distance(
    candidate_path: Path,
    mqtl_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    candidates = pd.read_csv(candidate_path)
    candidate_distance = next(
        (
            column
            for column in (
                "Absolute_Distance_From_Target_CpG",
                "Absolute_Distance_To_CpG",
                "absolute_distance_bp",
            )
            if column in candidates.columns
        ),
        None,
    )
    require_columns(candidates, ["Predicted_Delta_Beta"], candidate_path)
    if candidate_distance is None:
        raise ValueError(f"Could not identify candidate distance column in {candidate_path}")
    candidates["Distance_Bin"] = distance_bin(candidates[candidate_distance])

    rows: list[dict[str, object]] = []
    for label in DISTANCE_ORDER:
        group = candidates[candidates["Distance_Bin"].astype(str) == label]
        if group.empty:
            continue
        absolute = group["Predicted_Delta_Beta"].abs()
        rows.append({
            "Dataset": "TCGA synonymous candidates",
            "Model": "fusion",
            "Distance_Bin": label,
            "N_Associations": int(len(group)),
            "Median_Absolute_Predicted_Response": float(absolute.median()),
            "Mean_Absolute_Predicted_Response": float(absolute.mean()),
            "Signed_Spearman_With_Observed_Effect": float("nan"),
            "Absolute_Spearman_With_Observed_Effect": float("nan"),
            "Direction_Concordance": float("nan"),
        })

    if not mqtl_path.is_file():
        raise FileNotFoundError(mqtl_path)
    mqtl = pd.read_csv(mqtl_path)
    require_columns(
        mqtl,
        ["model", "absolute_distance_bp", "predicted_delta_m", "slope_alt_aligned"],
        mqtl_path,
    )
    mqtl = mqtl[mqtl["model"].astype(str) == "fusion"].copy()
    mqtl["Distance_Bin"] = distance_bin(mqtl["absolute_distance_bp"])
    for label in DISTANCE_ORDER:
        group = mqtl[mqtl["Distance_Bin"].astype(str) == label]
        if group.empty:
            continue
        predicted = group["predicted_delta_m"]
        observed = group["slope_alt_aligned"]
        rows.append({
            "Dataset": "eGTEx positive control",
            "Model": "fusion",
            "Distance_Bin": label,
            "N_Associations": int(len(group)),
            "Median_Absolute_Predicted_Response": float(predicted.abs().median()),
            "Mean_Absolute_Predicted_Response": float(predicted.abs().mean()),
            "Signed_Spearman_With_Observed_Effect": safe_spearman(predicted, observed),
            "Absolute_Spearman_With_Observed_Effect": safe_spearman(
                predicted.abs(), observed.abs()
            ),
            "Direction_Concordance": float(
                np.mean(np.sign(predicted.to_numpy(float)) == np.sign(observed.to_numpy(float)))
            ),
        })
    summary = pd.DataFrame(rows)
    return candidates, mqtl, summary


def plot_context_gain(gain: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=False)
    specifications = [
        ("CpG_Island_Context", list(ISLAND_ORDER), "CpG-island context"),
        ("ATAC_Stratum", list(ATAC_ORDER), "MCF-10A ATAC signal"),
    ]
    for axis, (grouping, order, title) in zip(axes, specifications):
        current = gain[gain["Grouping"] == grouping].copy()
        current["Stratum"] = pd.Categorical(current["Stratum"], categories=order, ordered=True)
        current = current.dropna(subset=["Stratum"]).sort_values("Stratum")
        x = np.arange(len(current))
        improvement = current["Sequence_Minus_Fusion_Beta_MAE"].to_numpy(float)
        ci_low = -current["Bootstrap_CI_High"].to_numpy(float)
        ci_high = -current["Bootstrap_CI_Low"].to_numpy(float)
        yerr = np.vstack([improvement - ci_low, ci_high - improvement])
        yerr = np.maximum(yerr, 0)
        axis.errorbar(
            x,
            improvement,
            yerr=yerr,
            fmt="o",
            color="#2f6f9f",
            ecolor="#7aa6c2",
            capsize=3,
            linewidth=1.2,
        )
        axis.axhline(0, color="0.35", linewidth=0.9, linestyle="--")
        axis.set_xticks(x)
        axis.set_xticklabels(current["Stratum"].astype(str), rotation=25, ha="right")
        axis.set_title(title)
        axis.set_ylabel(r"Sequence MAE $-$ fusion MAE")
        for location, (_, row) in zip(x, current.iterrows()):
            axis.annotate(
                f"n={int(row['N_CpGs']):,}",
                (location, row["Sequence_Minus_Fusion_Beta_MAE"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle("Fusion improvement over sequence-only across held-out CpG contexts")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_distance_summary(summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    candidate = summary[summary["Dataset"] == "TCGA synonymous candidates"].copy()
    mqtl = summary[summary["Dataset"] == "eGTEx positive control"].copy()
    for current in (candidate, mqtl):
        current["Distance_Bin"] = pd.Categorical(
            current["Distance_Bin"], categories=list(DISTANCE_ORDER), ordered=True
        )
        current.sort_values("Distance_Bin", inplace=True)

    x = np.arange(len(candidate))
    axes[0].plot(
        x,
        candidate["Median_Absolute_Predicted_Response"],
        marker="o",
        color="#7b3294",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(candidate["Distance_Bin"].astype(str), rotation=25, ha="right")
    axes[0].set_ylabel(r"Median absolute predicted $\Delta\hat{\beta}$")
    axes[0].set_title("TCGA synonymous candidates")
    for location, (_, row) in zip(x, candidate.iterrows()):
        axes[0].annotate(
            f"n={int(row['N_Associations'])}",
            (location, row["Median_Absolute_Predicted_Response"]),
            xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7,
        )

    x = np.arange(len(mqtl))
    axes[1].plot(
        x,
        mqtl["Direction_Concordance"],
        marker="o",
        color="#20854e",
    )
    axes[1].axhline(0.5, color="0.4", linestyle="--", linewidth=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(mqtl["Distance_Bin"].astype(str), rotation=25, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Direction concordance")
    axes[1].set_title("eGTEx positive control")
    for location, (_, row) in zip(x, mqtl.iterrows()):
        axes[1].annotate(
            f"n={int(row['N_Associations'])}",
            (location, row["Direction_Concordance"]),
            xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7,
        )
    for axis in axes:
        axis.set_xlabel("Absolute variant--CpG distance (bp)")
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle("Predicted sequence responses by variant--CpG distance")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if args.block_size_bp <= 0 or args.bootstrap_replicates < 0:
        raise ValueError("Block size must be positive and bootstrap replicates nonnegative")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    test = load_test_metadata(args.test_path)
    island = load_island_annotation(args.cpg_island_annotation)
    test = test.merge(island, on="probeID", how="left", validate="one_to_one")
    test["CpG_Island_Context"] = test["CpG_Island_Context"].fillna("Unclassified")

    long, prediction_hashes = load_predictions(
        args.prediction_template, args.seeds, args.models
    )
    ensemble = ensemble_predictions(long)
    annotated = ensemble.merge(
        test[["probeID", "Ref_ATAC_Signal", "Ref_ATAC_Signal_Missing", "ATAC_Stratum", "CpG_Island_Context"]],
        on="probeID",
        how="inner",
        validate="many_to_one",
    )
    expected = ensemble["probeID"].nunique()
    observed = annotated["probeID"].nunique()
    if observed != expected:
        raise ValueError(f"Annotated {observed} of {expected} held-out probes")

    context_rows = []
    context_rows.extend(metric_rows(annotated, "CpG_Island_Context"))
    context_rows.extend(metric_rows(annotated, "ATAC_Stratum"))
    context_metrics = pd.DataFrame(context_rows)
    atomic_csv(context_metrics, output / "locus_metrics_by_context.csv")

    paired = paired_gain_frame(annotated)
    gain = pd.DataFrame(
        gain_rows(
            paired,
            "CpG_Island_Context",
            args.block_size_bp,
            args.bootstrap_replicates,
            args.random_seed,
        )
        + gain_rows(
            paired,
            "ATAC_Stratum",
            args.block_size_bp,
            args.bootstrap_replicates,
            args.random_seed + 100_000,
        )
    )
    atomic_csv(gain, output / "fusion_gain_by_context.csv")

    locus_assignments = test[[
        "probeID", "chr", "pos", "Ref_ATAC_Signal", "Ref_ATAC_Signal_Missing",
        "ATAC_Stratum", "CpG_Island_Context"
    ]].copy()
    atomic_csv(locus_assignments, output / "heldout_cpg_context_assignments.csv")

    candidates, mqtl, distance_summary = summarize_variant_distance(
        args.candidate_path, args.mqtl_path
    )
    atomic_csv(distance_summary, output / "variant_response_by_distance.csv")

    plot_context_gain(gain, output / "plots" / "fusion_gain_by_context.png")
    plot_distance_summary(
        distance_summary,
        output / "plots" / "variant_response_by_distance.png",
    )

    payload = {
        "analysis_status": "COMPLETE",
        "interpretation": (
            "Post hoc descriptive stratification of frozen held-out predictions; "
            "not an additional training experiment or causal analysis."
        ),
        "seeds": [int(seed) for seed in args.seeds],
        "models": list(args.models),
        "heldout_cpg_count": int(expected),
        "candidate_count": int(len(candidates)),
        "mqtl_fusion_association_count": int(len(mqtl)),
        "block_size_bp": int(args.block_size_bp),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "random_seed": int(args.random_seed),
        "inputs": {
            args.test_path.as_posix(): sha256_file(args.test_path),
            args.cpg_island_annotation.as_posix(): sha256_file(args.cpg_island_annotation),
            args.candidate_path.as_posix(): sha256_file(args.candidate_path),
            args.mqtl_path.as_posix(): sha256_file(args.mqtl_path),
            **prediction_hashes,
        },
        "distance_bins_bp": list(DISTANCE_ORDER),
        "atac_strata": list(ATAC_ORDER),
        "cpg_island_strata": list(ISLAND_ORDER) + ["Unclassified"],
    }
    atomic_json(payload, output / "run_summary.json")

    print("SilentMethyl biological-context analysis")
    print(f"  held-out CpGs: {expected}")
    print(f"  candidate rows: {len(candidates)}")
    print(f"  fusion mQTL associations: {len(mqtl)}")
    print(f"  output: {output}")


if __name__ == "__main__":
    main()