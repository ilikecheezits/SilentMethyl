#!/usr/bin/env python3
"""Paired genomic-block bootstrap for held-out model-performance differences.

Reads the predictions.csv files already produced by the journal test scripts.
No model loading or retraining is performed. Regression differences are
reported as model A minus model B (negative favors A); AUROC differences are
reported as model A minus model B (positive favors A).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REQUIRED = {
    "probeID", "chr", "pos", "true_beta", "true_m", "binary_true",
    "pred_beta_rc_avg", "pred_m_rc_avg", "class_prob_rc_avg",
}

LOGGER = logging.getLogger("silentmethyl.paired_bootstrap")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired genomic-block bootstrap across held-out CpGs")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--models", nargs="+", default=["epi", "sequence", "fusion"])
    p.add_argument(
        "--prediction-template",
        default="results/journal/seed{seed}/{model}/predictions.csv",
    )
    p.add_argument("--output-dir", default="results/journal/paired_model_bootstrap")
    p.add_argument("--block-size-bp", type=int, default=1_000_000)
    p.add_argument("--bootstrap-replicates", type=int, default=5000)
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def load_predictions(path: Path, seed: int, model: str, block_size: int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Rerun the corresponding 02_test_*_journal.py script; retraining is not needed."
        )
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if df["probeID"].duplicated().any():
        raise ValueError(f"Duplicate probeID values in {path}")
    df = df.copy()
    df["probeID"] = df["probeID"].astype(str)
    df["chr"] = df["chr"].astype(str)
    df["pos"] = pd.to_numeric(df["pos"], errors="raise").astype(np.int64)
    df["Seed"] = int(seed)
    df["Model"] = model
    df["Genomic_Block"] = df["chr"] + ":" + (df["pos"] // block_size).astype(str)
    return df


def validate_alignment(frames: dict[tuple[int, str], pd.DataFrame], seeds, models) -> None:
    reference = frames[(seeds[0], models[0])].sort_values("probeID").reset_index(drop=True)
    ref_ids = reference["probeID"].tolist()
    truth_cols = ["true_beta", "true_m", "binary_true"]
    for key, frame in frames.items():
        current = frame.sort_values("probeID").reset_index(drop=True)
        if current["probeID"].tolist() != ref_ids:
            raise ValueError(f"Held-out probe set differs for seed/model {key}")
        for col in truth_cols:
            if not np.allclose(current[col], reference[col], rtol=0, atol=1e-10):
                raise ValueError(f"Held-out truth column {col} differs for seed/model {key}")


def metrics(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    beta_error = df[f"pred_beta_{prefix}"] - df["true_beta"]
    m_error = df[f"pred_m_{prefix}"] - df["true_m"]
    y = df["binary_true"].to_numpy(int)
    auc = np.nan
    if np.unique(y).size == 2:
        auc = roc_auc_score(y, df[f"class_prob_{prefix}"])
    return {
        "beta_mae": float(np.mean(np.abs(beta_error))),
        "beta_rmse": float(np.sqrt(np.mean(np.square(beta_error)))),
        "m_mae": float(np.mean(np.abs(m_error))),
        "m_rmse": float(np.sqrt(np.mean(np.square(m_error)))),
        "roc_auc": float(auc),
    }


def paired_difference(df: pd.DataFrame) -> dict[str, float]:
    a = metrics(df, "a")
    b = metrics(df, "b")
    return {name: a[name] - b[name] for name in a}


def align_pair(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["probeID", "chr", "pos", "Genomic_Block", "true_beta", "true_m", "binary_true"]
    pred_cols = ["pred_beta_rc_avg", "pred_m_rc_avg", "class_prob_rc_avg"]
    left = a[base_cols + pred_cols].copy()
    right = b[["probeID"] + pred_cols].copy()
    x = left.merge(right, on="probeID", suffixes=("_a", "_b"), validate="one_to_one")
    return x.rename(columns={
        "pred_beta_rc_avg_a": "pred_beta_a",
        "pred_m_rc_avg_a": "pred_m_a",
        "class_prob_rc_avg_a": "class_prob_a",
        "pred_beta_rc_avg_b": "pred_beta_b",
        "pred_m_rc_avg_b": "pred_m_b",
        "class_prob_rc_avg_b": "class_prob_b",
    })


def ensemble_frame(frames: dict[tuple[int, str], pd.DataFrame], seeds, model: str) -> pd.DataFrame:
    all_rows = pd.concat([frames[(seed, model)] for seed in seeds], ignore_index=True)
    truth = all_rows.sort_values("Seed").drop_duplicates("probeID")[
        ["probeID", "chr", "pos", "Genomic_Block", "true_beta", "true_m", "binary_true"]
    ]
    means = all_rows.groupby("probeID", as_index=False)[
        ["pred_beta_rc_avg", "pred_m_rc_avg", "class_prob_rc_avg"]
    ].mean()
    return truth.merge(means, on="probeID", validate="one_to_one")


def _block_sums(values: np.ndarray, block_index: np.ndarray, n_blocks: int) -> np.ndarray:
    return np.bincount(block_index, weights=values, minlength=n_blocks).astype(np.float64)


def _auc_quadratic_components(
    y: np.ndarray,
    scores: np.ndarray,
    block_index: np.ndarray,
    n_blocks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Represent weighted AUROC as c.T @ M @ c for resampled block counts c.

    This is exact, including 0.5 credit for tied positive/negative scores. The
    expensive score ordering is performed once rather than once per bootstrap
    replicate.
    """
    order = np.argsort(scores, kind="mergesort")
    score_sorted = scores[order]
    y_sorted = y[order].astype(np.int8)
    block_sorted = block_index[order]

    starts = np.r_[0, np.flatnonzero(np.diff(score_sorted) != 0) + 1]
    ends = np.r_[starts[1:], len(scores)]
    matrix = np.zeros((n_blocks, n_blocks), dtype=np.float64)
    cumulative_negative = np.zeros(n_blocks, dtype=np.float64)

    for start, end in zip(starts, ends):
        group_y = y_sorted[start:end]
        group_blocks = block_sorted[start:end]
        positive_counts = np.bincount(
            group_blocks[group_y == 1], minlength=n_blocks
        ).astype(np.float64)
        negative_counts = np.bincount(
            group_blocks[group_y == 0], minlength=n_blocks
        ).astype(np.float64)
        comparison_negative = cumulative_negative + 0.5 * negative_counts
        for positive_block in np.flatnonzero(positive_counts):
            matrix[positive_block] += (
                positive_counts[positive_block] * comparison_negative
            )
        cumulative_negative += negative_counts

    positives_by_block = np.bincount(
        block_index[y == 1], minlength=n_blocks
    ).astype(np.float64)
    negatives_by_block = np.bincount(
        block_index[y == 0], minlength=n_blocks
    ).astype(np.float64)
    return matrix, positives_by_block, negatives_by_block


def _fast_bootstrap_metrics(
    x: pd.DataFrame,
    prefix: str,
    block_counts: np.ndarray,
    block_index: np.ndarray,
    n_blocks: int,
) -> dict[str, np.ndarray]:
    true_beta = x["true_beta"].to_numpy(np.float64)
    true_m = x["true_m"].to_numpy(np.float64)
    y = x["binary_true"].to_numpy(np.int8)
    pred_beta = x[f"pred_beta_{prefix}"].to_numpy(np.float64)
    pred_m = x[f"pred_m_{prefix}"].to_numpy(np.float64)
    scores = x[f"class_prob_{prefix}"].to_numpy(np.float64)

    beta_error = pred_beta - true_beta
    m_error = pred_m - true_m
    count_by_block = np.bincount(block_index, minlength=n_blocks).astype(np.float64)
    sampled_n = block_counts @ count_by_block

    beta_abs = block_counts @ _block_sums(np.abs(beta_error), block_index, n_blocks)
    beta_sq = block_counts @ _block_sums(np.square(beta_error), block_index, n_blocks)
    m_abs = block_counts @ _block_sums(np.abs(m_error), block_index, n_blocks)
    m_sq = block_counts @ _block_sums(np.square(m_error), block_index, n_blocks)

    auc_matrix, positive_by_block, negative_by_block = _auc_quadratic_components(
        y, scores, block_index, n_blocks
    )
    auc_numerator = np.einsum(
        "ri,ij,rj->r", block_counts, auc_matrix, block_counts, optimize=True
    )
    sampled_positive = block_counts @ positive_by_block
    sampled_negative = block_counts @ negative_by_block
    auc_denominator = sampled_positive * sampled_negative
    auc = np.divide(
        auc_numerator,
        auc_denominator,
        out=np.full(len(block_counts), np.nan, dtype=np.float64),
        where=auc_denominator > 0,
    )

    return {
        "beta_mae": beta_abs / sampled_n,
        "beta_rmse": np.sqrt(beta_sq / sampled_n),
        "m_mae": m_abs / sampled_n,
        "m_rmse": np.sqrt(m_sq / sampled_n),
        "roc_auc": auc,
    }


def bootstrap_pair(
    x: pd.DataFrame,
    replicates: int,
    random_seed: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    observed = paired_difference(x)
    block_labels, block_index = np.unique(
        x["Genomic_Block"].astype(str).to_numpy(), return_inverse=True
    )
    n_blocks = len(block_labels)
    if n_blocks < 2:
        raise ValueError("Need at least two genomic blocks for block bootstrap")

    # Drawing n_blocks items with replacement is exactly equivalent to a
    # multinomial vector of block multiplicities. All 5,000 replicates can then
    # be evaluated by matrix operations.
    rng = np.random.default_rng(random_seed)
    block_counts = rng.multinomial(
        n_blocks,
        np.full(n_blocks, 1.0 / n_blocks),
        size=replicates,
    ).astype(np.float64)

    model_a = _fast_bootstrap_metrics(x, "a", block_counts, block_index, n_blocks)
    model_b = _fast_bootstrap_metrics(x, "b", block_counts, block_index, n_blocks)
    boot = pd.DataFrame({name: model_a[name] - model_b[name] for name in model_a})
    boot["Bootstrap_Replicate"] = np.arange(replicates)
    return observed, boot


def summarize(
    observed: dict[str, float],
    boot: pd.DataFrame,
    model_a: str,
    model_b: str,
    analysis: str,
    seed,
    n_loci: int,
    n_blocks: int,
) -> list[dict]:
    rows = []
    for metric, estimate in observed.items():
        values = boot[metric].dropna().to_numpy(float)
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append({
            "Analysis": analysis,
            "Seed": seed,
            "Model_A": model_a,
            "Model_B": model_b,
            "Metric": metric,
            "Difference_A_Minus_B": estimate,
            "Bootstrap_CI_Low": low,
            "Bootstrap_CI_High": high,
            "Bootstrap_Probability_Difference_Greater_Equal_Zero": float(np.mean(values >= 0)),
            "N_Loci": n_loci,
            "N_Genomic_Blocks": n_blocks,
            "Interpretation": (
                "negative favors Model_A" if metric != "roc_auc" else "positive favors Model_A"
            ),
        })
    return rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if len(args.models) < 2:
        raise ValueError("Provide at least two models")
    if args.block_size_bp <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError("Block size and bootstrap replicates must be positive")

    frames = {}
    for seed in args.seeds:
        for model in args.models:
            path = Path(args.prediction_template.format(seed=seed, model=model))
            LOGGER.info("Loading %s seed %d predictions from %s", model, seed, path)
            frames[(seed, model)] = load_predictions(path, seed, model, args.block_size_bp)
    validate_alignment(frames, args.seeds, args.models)

    comparisons = [("fusion", model) for model in args.models if model != "fusion"]
    if not comparisons:
        comparisons = [(args.models[0], model) for model in args.models[1:]]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    full_metrics_rows = []

    for pair_index, (model_a, model_b) in enumerate(comparisons):
        for seed in args.seeds:
            LOGGER.info(
                "Bootstrap comparison %s vs %s, seed %d (%d replicates)",
                model_a, model_b, seed, args.bootstrap_replicates,
            )
            x = align_pair(frames[(seed, model_a)], frames[(seed, model_b)])
            observed, boot = bootstrap_pair(
                x, args.bootstrap_replicates, args.random_seed + 1000 * pair_index + seed
            )
            summary_rows.extend(summarize(
                observed, boot, model_a, model_b, "individual_seed", seed,
                len(x), x["Genomic_Block"].nunique(),
            ))
            for model, prefix in ((model_a, "a"), (model_b, "b")):
                row = {"Analysis": "individual_seed", "Seed": seed, "Model": model}
                row.update(metrics(x, prefix))
                full_metrics_rows.append(row)

        a_ensemble = ensemble_frame(frames, args.seeds, model_a)
        b_ensemble = ensemble_frame(frames, args.seeds, model_b)
        LOGGER.info(
            "Bootstrap comparison %s vs %s, cross-seed ensemble (%d replicates)",
            model_a, model_b, args.bootstrap_replicates,
        )
        x = align_pair(a_ensemble, b_ensemble)
        observed, boot = bootstrap_pair(
            x, args.bootstrap_replicates, args.random_seed + 100_000 + pair_index
        )
        summary_rows.extend(summarize(
            observed, boot, model_a, model_b, "cross_seed_ensemble", "ensemble",
            len(x), x["Genomic_Block"].nunique(),
        ))
        for model, prefix in ((model_a, "a"), (model_b, "b")):
            row = {"Analysis": "cross_seed_ensemble", "Seed": "ensemble", "Model": model}
            row.update(metrics(x, prefix))
            full_metrics_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    metrics_df = pd.DataFrame(full_metrics_rows).drop_duplicates(
        ["Analysis", "Seed", "Model"]
    )
    atomic_csv(summary_df, out / "paired_model_difference_bootstrap.csv")
    atomic_csv(metrics_df, out / "model_metrics_recomputed.csv")
    atomic_json({
        "seeds": args.seeds,
        "models": args.models,
        "comparisons": [f"{a}_minus_{b}" for a, b in comparisons],
        "block_size_bp": args.block_size_bp,
        "bootstrap_replicates": args.bootstrap_replicates,
        "random_seed": args.random_seed,
        "difference_convention": {
            "regression_errors": "Model A minus Model B; negative favors Model A",
            "roc_auc": "Model A minus Model B; positive favors Model A",
        },
    }, out / "run_summary.json")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
