#!/usr/bin/env python3
"""Build concise manuscript figures from frozen SilentMethyl result tables.

No checkpoints are loaded and no predictions are recomputed.  The script turns
existing tabular results into three submission-facing figures:

1. incremental held-out performance for context, sequence, and fusion models;
2. signed versus absolute eGTEx mQTL rank agreement; and
3. a transparent case-study panel for the first-ranked somatic candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MODEL_ORDER = ("epi", "sequence", "fusion")
MODEL_LABELS = {
    "epi": "Context only",
    "sequence": "Sequence only",
    "fusion": "Fusion",
}
MODEL_COLORS = {
    "epi": "#5aa469",
    "sequence": "#4c78a8",
    "fusion": "#e07b39",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("results/journal/paired_model_bootstrap/model_metrics_recomputed.csv"),
    )
    parser.add_argument(
        "--mqtl-path",
        type=Path,
        default=Path(
            "results/journal/egtex_mqtl_positive_control/"
            "mqtl_predictions_seed_aggregate.csv"
        ),
    )
    parser.add_argument(
        "--candidate-path",
        type=Path,
        default=Path(
            "results/journal/candidates/"
            "candidate_matched_background_statistics.csv"
        ),
    )
    parser.add_argument(
        "--candidate-seed-path",
        type=Path,
        default=Path("results/journal/candidates/candidate_seed_scores_long.csv"),
    )
    parser.add_argument(
        "--comparator-path",
        type=Path,
        default=Path(
            "results/journal/candidates/"
            "top_candidate_matched_comparators_long.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/journal/manuscript_figures"),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


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


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    a = pd.to_numeric(x, errors="coerce").to_numpy(float)
    b = pd.to_numeric(y, errors="coerce").to_numpy(float)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 3 or np.unique(a[finite]).size < 2 or np.unique(b[finite]).size < 2:
        return float("nan")
    return float(spearmanr(a[finite], b[finite]).statistic)


def percentile_rank(values: pd.Series) -> pd.Series:
    """Return average-tie percentile ranks on a 0--100 display scale."""
    numeric = pd.to_numeric(values, errors="raise")
    return 100.0 * numeric.rank(method="average", pct=True)


def native_scalar(value):
    """Convert pandas/NumPy scalars to JSON-safe Python values."""
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    if isinstance(value, np.generic):
        return value.item()
    return value


def plot_incremental_performance(metrics_path: Path, output_path: Path) -> dict:
    frame = pd.read_csv(metrics_path)
    require_columns(
        frame,
        ["Analysis", "Seed", "Model", "beta_mae", "beta_rmse", "roc_auc"],
        metrics_path,
    )
    frame = frame[frame["Analysis"].astype(str) == "individual_seed"].copy()
    frame = frame[frame["Model"].isin(MODEL_ORDER)].copy()
    frame["Seed"] = pd.to_numeric(frame["Seed"], errors="raise").astype(int)
    if frame.groupby("Model")["Seed"].nunique().min() < 2:
        raise ValueError("Need at least two seeds per model for the performance figure")

    specifications = (
        ("beta_mae", r"Beta MAE $\downarrow$"),
        ("beta_rmse", r"Beta RMSE $\downarrow$"),
        ("roc_auc", r"ROC-AUC $\uparrow$"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    x = np.arange(len(MODEL_ORDER))
    seeds = sorted(frame["Seed"].unique())

    for axis, (metric, label) in zip(axes, specifications):
        wide = frame.pivot(index="Seed", columns="Model", values=metric)
        wide = wide.reindex(index=seeds, columns=MODEL_ORDER)
        if wide.isna().any().any():
            raise ValueError(f"Incomplete seed/model grid for {metric}")
        for seed, row in wide.iterrows():
            axis.plot(x, row.to_numpy(float), color="0.72", linewidth=1.0, zorder=1)
            axis.scatter(x, row.to_numpy(float), color="0.58", s=20, zorder=2)
        mean = wide.mean(axis=0).to_numpy(float)
        sd = wide.std(axis=0, ddof=1).to_numpy(float)
        for position, model, value, spread in zip(x, MODEL_ORDER, mean, sd):
            axis.errorbar(
                position,
                value,
                yerr=spread,
                fmt="o",
                markersize=7,
                capsize=3,
                color=MODEL_COLORS[model],
                ecolor=MODEL_COLORS[model],
                zorder=3,
            )
        axis.set_xticks(x)
        axis.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], rotation=18)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.18)
    fig.suptitle("Held-out chromosome performance across three training seeds")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "seed_count": int(len(seeds)),
        "seeds": [int(seed) for seed in seeds],
        "model_count": int(len(MODEL_ORDER)),
    }


def plot_mqtl_agreement(mqtl_path: Path, output_path: Path) -> dict:
    frame = pd.read_csv(mqtl_path)
    require_columns(
        frame,
        ["model", "predicted_delta_m", "slope_alt_aligned"],
        mqtl_path,
    )
    frame = frame[frame["model"].astype(str) == "fusion"].copy()
    frame = frame.dropna(subset=["predicted_delta_m", "slope_alt_aligned"])
    if frame.empty:
        raise ValueError("No fusion rows in the mQTL aggregate table")

    predicted = pd.to_numeric(frame["predicted_delta_m"], errors="raise")
    observed = pd.to_numeric(frame["slope_alt_aligned"], errors="raise")
    signed_rho = safe_spearman(predicted, observed)
    absolute_rho = safe_spearman(predicted.abs(), observed.abs())
    direction = float(np.mean(np.sign(predicted) == np.sign(observed)))

    observed_signed_rank = percentile_rank(observed)
    predicted_signed_rank = percentile_rank(predicted)
    observed_absolute_rank = percentile_rank(observed.abs())
    predicted_absolute_rank = percentile_rank(predicted.abs())
    direction_agreement = np.sign(predicted) == np.sign(observed)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    for agrees, label, color in (
        (True, "Direction agrees", "#2f6f9f"),
        (False, "Direction differs", "#d98c3f"),
    ):
        selected = direction_agreement == agrees
        axes[0].scatter(
            observed_signed_rank[selected],
            predicted_signed_rank[selected],
            s=28,
            alpha=0.78,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            label=label,
            zorder=3,
        )
    axes[0].plot([0, 100], [0, 100], color="0.45", linewidth=0.9, linestyle="--")
    axes[0].set_xlabel("Reported mQTL slope rank percentile")
    axes[0].set_ylabel(r"Predicted signed-response rank percentile")
    axes[0].set_title("Signed rank agreement")
    axes[0].text(
        0.04,
        0.96,
        f"Spearman $\\rho$={signed_rho:.3f}\nDirection={100 * direction:.1f}%",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
    )
    axes[0].legend(loc="lower right", frameon=False, fontsize=7.5)

    axes[1].scatter(
        observed_absolute_rank,
        predicted_absolute_rank,
        s=28,
        alpha=0.78,
        color="#b24c63",
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    axes[1].plot([0, 100], [0, 100], color="0.45", linewidth=0.9, linestyle="--")
    axes[1].set_xlabel("Absolute mQTL slope rank percentile")
    axes[1].set_ylabel(r"Absolute predicted-response rank percentile")
    axes[1].set_title("Absolute-magnitude rank agreement")
    axes[1].text(
        0.04,
        0.96,
        f"Spearman $\\rho$={absolute_rho:.3f}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
    )
    for axis in axes:
        axis.set_xlim(0, 102)
        axis.set_ylim(0, 102)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
    fig.suptitle("External breast mQTL rank agreement for the fusion ensemble")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "association_count": int(len(frame)),
        "signed_spearman": signed_rho,
        "absolute_spearman": absolute_rho,
        "direction_concordance": direction,
        "display_scale": "average-tie percentile ranks (0--100)",
    }


def candidate_title(row: pd.Series) -> str:
    gene = str(row.get("Gene", "Candidate"))
    probe = str(row.get("probeID", ""))
    coordinate = str(row.get("GDC_Genomic_DNA_Change", "")).strip()
    if not coordinate:
        uid = str(row.get("Variant_UID", ""))
        prefix = uid.split("__", 1)[0].split("_")
        if len(prefix) >= 4:
            coordinate = f"{prefix[0]}:g.{prefix[1]}{prefix[2]}>{prefix[3]}"
    pieces = [gene]
    if coordinate:
        pieces.append(coordinate)
    if probe:
        pieces.append(probe)
    return " | ".join(pieces)


def plot_top_candidate(
    candidate_path: Path,
    seed_path: Path,
    comparator_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict:
    candidates = pd.read_csv(candidate_path)
    require_columns(
        candidates,
        [
            "Variant_UID",
            "Absolute_Delta_Beta_Rank",
            "Predicted_Delta_Beta",
            "Predicted_Delta_Beta_SD",
        ],
        candidate_path,
    )
    candidates["Absolute_Delta_Beta_Rank"] = pd.to_numeric(
        candidates["Absolute_Delta_Beta_Rank"], errors="raise"
    )
    target_rows = candidates[candidates["Absolute_Delta_Beta_Rank"] == 1]
    if len(target_rows) != 1:
        raise ValueError(f"Expected one rank-1 candidate, observed {len(target_rows)}")
    target = target_rows.iloc[0]
    uid = str(target["Variant_UID"])

    seed_scores = pd.read_csv(seed_path)
    require_columns(seed_scores, ["Variant_UID", "Seed", "Predicted_Delta_Beta"], seed_path)
    seed_scores = seed_scores[seed_scores["Variant_UID"].astype(str) == uid].copy()
    seed_scores["Seed"] = pd.to_numeric(seed_scores["Seed"], errors="raise").astype(int)
    seed_scores.sort_values("Seed", inplace=True)
    if seed_scores.empty:
        raise ValueError(f"No per-seed scores for {uid}")

    comparators = pd.read_csv(comparator_path)
    require_columns(
        comparators,
        ["Target_Rank", "Target_Variant_UID", "Comparator_Delta_Beta"],
        comparator_path,
    )
    comparators = comparators[
        (pd.to_numeric(comparators["Target_Rank"], errors="coerce") == 1)
        & (comparators["Target_Variant_UID"].astype(str) == uid)
    ].copy()
    if comparators.empty:
        raise ValueError(f"No matched comparators for rank-1 candidate {uid}")
    comparator_values = pd.to_numeric(
        comparators["Comparator_Delta_Beta"], errors="raise"
    ).abs()
    target_effect = float(target["Predicted_Delta_Beta"])
    target_sd = float(target["Predicted_Delta_Beta_SD"])
    seed_mean = float(seed_scores["Predicted_Delta_Beta"].mean())
    # Candidate exports use the population SD across the prespecified seeds.
    seed_sd = float(seed_scores["Predicted_Delta_Beta"].std(ddof=0))
    if not np.isclose(seed_mean, target_effect, rtol=0, atol=1e-8):
        raise ValueError("Rank-1 ensemble effect does not match the per-seed mean")
    if not np.isclose(seed_sd, target_sd, rtol=0, atol=1e-8):
        raise ValueError("Rank-1 exported SD does not match the per-seed population SD")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    axes[0].axhline(0, color="0.45", linewidth=0.8)
    seed_positions = np.arange(len(seed_scores), dtype=float)
    axes[0].scatter(
        seed_positions,
        seed_scores["Predicted_Delta_Beta"],
        marker="o",
        s=52,
        color="#7b3294",
        label="Individual seed",
        zorder=3,
    )
    ensemble_position = float(len(seed_scores))
    axes[0].errorbar(
        ensemble_position,
        target_effect,
        yerr=target_sd,
        fmt="D",
        markersize=6.5,
        capsize=4,
        color="#d94801",
        ecolor="#d94801",
        label=r"Ensemble mean $\pm$ SD",
        zorder=4,
    )
    axes[0].axhline(target_effect, color="#d94801", linestyle="--", linewidth=1.0, alpha=0.75)
    axes[0].set_xticks(
        [*seed_positions, ensemble_position],
        [*seed_scores["Seed"].astype(str).to_list(), "Ensemble"],
    )
    axes[0].set_xlabel("Training run")
    axes[0].set_ylabel(r"RC-averaged predicted $\Delta\hat{\beta}$")
    axes[0].set_title("Cross-seed response stability")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    axes[0].grid(axis="y", alpha=0.18)

    comparator_sorted = np.sort(comparator_values.to_numpy(float))
    comparator_cdf = np.arange(1, len(comparator_sorted) + 1) / len(comparator_sorted)
    axes[1].step(
        comparator_sorted,
        comparator_cdf,
        where="post",
        color="#4c92b7",
        linewidth=1.8,
        label=f"Matched comparators ($n={len(comparator_sorted)}$)",
    )
    axes[1].scatter(
        comparator_sorted,
        comparator_cdf,
        s=16,
        alpha=0.55,
        color="#4c92b7",
        edgecolor="none",
    )
    axes[1].axvline(
        abs(target_effect),
        color="#d94801",
        linewidth=2,
        label=rf"NCOA2 $|\Delta\hat{{\beta}}|={abs(target_effect):.3f}$",
    )
    axes[1].set_xlabel(r"Absolute predicted $\Delta\hat{\beta}$")
    axes[1].set_ylabel("Cumulative fraction of matched comparators")
    axes[1].set_ylim(0, 1.04)
    axes[1].set_title("Matched-background empirical distribution")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.15)

    fig.suptitle(candidate_title(target))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    less = int(np.count_nonzero(comparator_values < abs(target_effect)))
    tied = int(np.count_nonzero(comparator_values == abs(target_effect)))
    calculated_percentile = float(100.0 * (less + 0.5 * tied) / len(comparator_values))
    percentile = float(
        target.get("Matched_Background_Absolute_Effect_Percentile", calculated_percentile)
    )
    if not np.isclose(percentile, calculated_percentile, rtol=0, atol=1e-9):
        raise ValueError(
            "Rank-1 matched-background percentile does not match the exported comparator rows"
        )
    summary = {
        "Absolute_Delta_Beta_Rank": 1,
        "Variant_UID": uid,
        "Gene": target.get("Gene", ""),
        "probeID": target.get("probeID", ""),
        "Predicted_Delta_Beta": target_effect,
        "Predicted_Delta_Beta_SD": target_sd,
        "Seed_Count": int(len(seed_scores)),
        "Seeds": ",".join(str(value) for value in seed_scores["Seed"]),
        "Per_Seed_Predicted_Delta_Beta": ",".join(
            f"{value:.10g}" for value in seed_scores["Predicted_Delta_Beta"]
        ),
        "Matched_Comparator_Count": int(len(comparator_values)),
        "Matched_Background_Tier": target.get("Matched_Background_Tier", ""),
        "Matched_Comparator_Median_Absolute_Delta_Beta": float(comparator_values.median()),
        "Empirical_Matched_Background_Percentile": percentile,
        "GDC_Genomic_DNA_Change": native_scalar(
            target.get("GDC_Genomic_DNA_Change", "")
        ),
        "Selected_Transcript_ID": native_scalar(
            target.get("Selected_Transcript_ID", "")
        ),
        "Reference_Codon": native_scalar(target.get("Reference_Codon", "")),
        "Alternate_Codon": native_scalar(target.get("Alternate_Codon", "")),
        "Amino_Acid": native_scalar(target.get("Amino_Acid", "")),
        "Absolute_Distance_From_Target_CpG": native_scalar(
            target.get("Absolute_Distance_From_Target_CpG", "")
        ),
        "CpG_Effect": native_scalar(target.get("CpG_Effect", "")),
        "GDC_Occurrence_Count": native_scalar(
            target.get("GDC_Occurrence_Count", "")
        ),
        "Delta_RC_Sign_Agreement_Fraction": native_scalar(
            target.get("Delta_RC_Sign_Agreement_Fraction", "")
        ),
        "Mean_Delta_RC_Absolute_Difference": native_scalar(
            target.get("Mean_Delta_RC_Absolute_Difference", "")
        ),
        "Interpretation": (
            "Model-derived case study for hypothesis generation; not experimental "
            "or causal validation."
        ),
    }
    atomic_csv(pd.DataFrame([summary]), summary_path)
    return summary


def main() -> None:
    args = parse_args()
    inputs = (
        args.metrics_path,
        args.mqtl_path,
        args.candidate_path,
        args.candidate_seed_path,
        args.comparator_path,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    performance = plot_incremental_performance(
        args.metrics_path, output / "model_incremental_performance.png"
    )
    mqtl = plot_mqtl_agreement(
        args.mqtl_path, output / "mqtl_signed_and_magnitude.png"
    )
    case = plot_top_candidate(
        args.candidate_path,
        args.candidate_seed_path,
        args.comparator_path,
        output / "top_candidate_case_study.png",
        output / "top_candidate_case_study.csv",
    )
    atomic_json(
        {
            "analysis_status": "COMPLETE",
            "interpretation": (
                "Figures are visual summaries of frozen model outputs and do not "
                "constitute additional model training or biological validation."
            ),
            "performance_figure": performance,
            "mqtl_figure": mqtl,
            "top_candidate": case,
            "inputs": {path.as_posix(): sha256_file(path) for path in inputs},
        },
        output / "run_summary.json",
    )
    print("SilentMethyl manuscript figures")
    print(f"  output: {output}")
    print("  figures: 3")
    

if __name__ == "__main__":
    main()