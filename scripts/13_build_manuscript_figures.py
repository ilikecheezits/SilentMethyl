from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/silentmethyl_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MODEL_ORDER = ("epi", "sequence", "fusion")
MODEL_COLORS = {"epi": "#5AA469", "sequence": "#4C78A8", "fusion": "#E07B39"}
ONE_COLUMN_WIDTH = 3.45
MANUSCRIPT_FIGURES = (
    "candidate_response_by_context.png",
    "fusion_gain_by_epigenomic_context.png",
    "model_incremental_performance.png",
    "mqtl_magnitude_rank.png",
    "mqtl_signed_rank.png",
    "top_candidate_matched_background.png",
)

plt.rcParams.update(
    {
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 6.8,
        "lines.linewidth": 1.15,
        "savefig.dpi": 400,
    }
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("results/journal/paired_model_bootstrap/model_metrics_recomputed.csv"),
    )
    parser.add_argument(
        "--context-path",
        type=Path,
        default=Path("results/journal/biological_context/fusion_gain_by_context.csv"),
    )
    parser.add_argument(
        "--variant-context-path",
        type=Path,
        default=Path("results/journal/biological_context/variant_response_by_context.csv"),
    )
    parser.add_argument(
        "--variant-distance-path",
        type=Path,
        default=Path("results/journal/biological_context/variant_response_by_distance.csv"),
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
            "results/journal/candidates/candidate_matched_background_statistics.csv"
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
            "results/journal/candidates/top_candidate_matched_comparators_long.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/journal/manuscript_figures"),
    )
    parser.add_argument(
        "--case-study-path",
        type=Path,
        default=Path("results/journal/candidates/top_candidate_case_study.csv"),
    )
    parser.add_argument(
        "--genomic-region-output",
        type=Path,
        default=Path(
            "results/journal/biological_context/plots/fusion_gain_by_genomic_region.png"
        ),
    )
    parser.add_argument(
        "--information-gains-output",
        type=Path,
        default=Path(
            "results/journal/biological_context/plots/information_source_gains.png"
        ),
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else str(value),
        )
        + "\n"
    )
    temporary.replace(path)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def keep_manuscript_outputs(output_dir: Path) -> None:
    allowed = set(MANUSCRIPT_FIGURES) | {"run_summary.json"}
    for path in output_dir.iterdir():
        if path.is_file() and path.name not in allowed:
            path.unlink()


def rank_percentile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    return 100.0 * numeric.rank(method="average", pct=True)


def spearman(x: pd.Series, y: pd.Series) -> float:
    first = pd.to_numeric(x, errors="coerce").to_numpy(float)
    second = pd.to_numeric(y, errors="coerce").to_numpy(float)
    keep = np.isfinite(first) & np.isfinite(second)
    if keep.sum() < 3:
        return float("nan")
    return float(spearmanr(first[keep], second[keep]).statistic)


def plot_performance(path: Path, output: Path) -> dict:
    frame = pd.read_csv(path)
    require_columns(
        frame,
        ["Analysis", "Seed", "Model", "m_mae", "beta_mae", "roc_auc"],
        path,
    )
    frame = frame[
        (frame["Analysis"].astype(str) == "individual_seed")
        & frame["Model"].isin(MODEL_ORDER)
    ].copy()
    frame["Seed"] = pd.to_numeric(frame["Seed"], errors="raise").astype(int)
    seeds = sorted(frame["Seed"].unique())
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required")

    # 1x3 layout to place them horizontally
    fig, axes = plt.subplots(1, 3, figsize=(ONE_COLUMN_WIDTH, 4.35))
    positions = np.arange(3)
    
    metrics = ("m_mae", "beta_mae", "roc_auc")
    labels = (r"M-value MAE $\downarrow$", r"Beta MAE $\downarrow$", r"ROC-AUC $\uparrow$")

    for axis, metric, label in zip(axes, metrics, labels):
        values = (
            frame.pivot(index="Seed", columns="Model", values=metric)
            .reindex(index=seeds, columns=MODEL_ORDER)
        )
        if values.isna().any().any():
            raise ValueError(f"Incomplete seed/model grid for {metric}")
            
        means = values.mean(axis=0).to_numpy(float)
        standard_deviations = values.std(axis=0, ddof=1).to_numpy(float)
        colors = [MODEL_COLORS[m] for m in MODEL_ORDER]

        # Draw the bar chart with error bars
        axis.bar(
            positions,
            means,
            yerr=standard_deviations,
            color=colors,
            capsize=2.5,
            edgecolor="none",
            alpha=0.9,
            error_kw=dict(lw=1.0, ecolor='0.25')
        )

        # Format axes
        axis.set_xticks(positions)
        axis.set_xticklabels(["Context", "Sequence", "Fusion"], rotation=45, ha="right")
        
        # Explicitly start the Y-axis at 0
        axis.set_ylim(bottom=0)
        axis.set_title(label, fontsize=8.0)
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.4, w_pad=0.8)
    save_figure(fig, output)
    return {"seeds": [int(seed) for seed in seeds]}


def plot_information_gains(path: Path, output: Path) -> dict:
    frame = pd.read_csv(path)
    require_columns(
        frame,
        ["Analysis", "Seed", "Model", "beta_mae", "beta_rmse", "roc_auc"],
        path,
    )
    frame = frame[
        (frame["Analysis"].astype(str) == "individual_seed")
        & frame["Model"].isin(MODEL_ORDER)
    ].copy()
    frame["Seed"] = pd.to_numeric(frame["Seed"], errors="raise").astype(int)
    seeds = sorted(frame["Seed"].unique())
    comparisons = (
        ("epi", "Add sequence\nto context"),
        ("sequence", "Add context\nto sequence"),
    )
    specifications = (
        ("beta_mae", "MAE improvement", "lower"),
        ("beta_rmse", "RMSE improvement", "lower"),
        ("roc_auc", "AUROC improvement", "higher"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(ONE_COLUMN_WIDTH, 5.65))
    summary: dict[str, dict[str, float]] = {}
    for axis, (metric, label, direction) in zip(axes, specifications):
        wide = frame.pivot(index="Seed", columns="Model", values=metric).reindex(
            index=seeds, columns=MODEL_ORDER
        )
        if wide.isna().any().any():
            raise ValueError(f"Incomplete seed/model grid for {metric}")
        values = []
        for baseline, _ in comparisons:
            if direction == "lower":
                values.append(wide[baseline] - wide["fusion"])
            else:
                values.append(wide["fusion"] - wide[baseline])
        gain = pd.concat(values, axis=1)
        gain.columns = [label_text for _, label_text in comparisons]
        for seed_index, (_, row) in enumerate(gain.iterrows()):
            jitter = (seed_index - (len(gain) - 1) / 2) * 0.055
            axis.scatter(
                np.arange(2) + jitter,
                row.to_numpy(float),
                s=17,
                color="0.48",
                alpha=0.8,
                zorder=2,
            )
        means = gain.mean(axis=0)
        sd = gain.std(axis=0, ddof=1)
        axis.errorbar(
            np.arange(2),
            means.to_numpy(float),
            yerr=sd.to_numpy(float),
            fmt="o",
            color="#D95F02",
            ecolor="#7F7F7F",
            markersize=5.5,
            capsize=2.5,
            zorder=3,
        )
        axis.axhline(0, color="0.45", linewidth=0.8, linestyle="--")
        axis.set_xticks(np.arange(2), [label_text for _, label_text in comparisons])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
        summary[metric] = {
            key.replace("\n", " "): float(value) for key, value in means.items()
        }
    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)
    axes[0].set_title("Incremental value of each information source")
    fig.tight_layout(pad=0.6, h_pad=0.7)
    save_figure(fig, output)
    return {"mean_improvements": summary}


def _draw_gain_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    group: str,
    title: str,
    order: list[str],
) -> dict[str, float]:
    subset = frame[frame["Grouping"].astype(str) == group].copy()
    subset["Stratum"] = pd.Categorical(subset["Stratum"], order, ordered=True)
    subset.dropna(subset=["Stratum"], inplace=True)
    subset.sort_values("Stratum", inplace=True)
    if len(subset) != len(order):
        raise ValueError(
            f"Unexpected strata for {group}: {subset['Stratum'].astype(str).tolist()}"
        )
    gain = pd.to_numeric(
        subset["Sequence_Minus_Fusion_Beta_MAE"], errors="raise"
    ).to_numpy(float)
    low = -pd.to_numeric(subset["Bootstrap_CI_High"], errors="raise").to_numpy(float)
    high = -pd.to_numeric(subset["Bootstrap_CI_Low"], errors="raise").to_numpy(float)
    positions = np.arange(len(subset))
    axis.errorbar(
        gain,
        positions,
        xerr=np.vstack([gain - low, high - gain]),
        fmt="o",
        color="#D95F02",
        ecolor="#7F7F7F",
        capsize=2.2,
        markersize=4.8,
    )
    axis.axvline(0, color="0.45", linewidth=0.8, linestyle="--")
    labels = [
        f"{row.Stratum}  (n={int(row.N_CpGs):,})"
        for row in subset.itertuples(index=False)
    ]
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_title(title, loc="left")
    axis.set_xlabel(r"Sequence MAE $-$ fusion MAE")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axis.grid(axis="x", alpha=0.16)
    axis.spines[["top", "right"]].set_visible(False)
    return dict(zip(order, [float(value) for value in gain]))


def plot_context(path: Path, region_output: Path, epigenomic_output: Path) -> dict:
    frame = pd.read_csv(path)
    require_columns(
        frame,
        [
            "Grouping",
            "Stratum",
            "Sequence_Minus_Fusion_Beta_MAE",
            "Bootstrap_CI_Low",
            "Bootstrap_CI_High",
        ],
        path,
    )
    output_values: dict[str, dict[str, float]] = {}

    fig, axis = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 2.65))
    output_values["Genomic_Region"] = _draw_gain_panel(
        axis,
        frame,
        "Genomic_Region",
        "Gain from context by genomic region",
        ["Promoter/TSS", "UTR", "Gene body", "Intergenic"],
    )
    fig.tight_layout(pad=0.6)
    save_figure(fig, region_output)

    panels = (
        (
            "CpG_Island_Context",
            "CpG island relation",
            ["Island", "Shore", "Shelf", "Open sea"],
        ),
        ("ATAC_Stratum", "ATAC signal", ["Q1 low", "Q2", "Q3", "Q4 high"]),
        (
            "H3K27ac_Stratum",
            "H3K27ac signal",
            ["Q1 low", "Q2", "Q3", "Q4 high"],
        ),
    )
    fig, axes = plt.subplots(3, 1, figsize=(ONE_COLUMN_WIDTH, 6.25))
    for axis, (group, title, order) in zip(axes, panels):
        output_values[group] = _draw_gain_panel(axis, frame, group, title, order)
    fig.tight_layout(pad=0.6, h_pad=0.8)
    save_figure(fig, epigenomic_output)
    return {"fusion_gain": output_values}


def plot_candidate_context(
    context_path: Path,
    distance_path: Path,
    output: Path,
) -> dict:
    context = pd.read_csv(context_path)
    distance = pd.read_csv(distance_path)
    require_columns(
        context,
        [
            "Dataset",
            "Grouping",
            "Stratum",
            "N_Associations",
            "Median_Absolute_Predicted_Response",
        ],
        context_path,
    )
    require_columns(
        distance,
        [
            "Dataset",
            "Distance_Bin",
            "N_Associations",
            "Median_Absolute_Predicted_Response",
        ],
        distance_path,
    )
    candidate_label = "TCGA synonymous candidates"
    region = context[
        (context["Dataset"].astype(str) == candidate_label)
        & (context["Grouping"].astype(str) == "Variant_Genomic_Region")
    ].copy()
    region_order = ["Promoter/TSS", "UTR", "Gene body"]
    region["Stratum"] = pd.Categorical(region["Stratum"], region_order, ordered=True)
    region.dropna(subset=["Stratum"], inplace=True)
    region.sort_values("Stratum", inplace=True)

    by_distance = distance[distance["Dataset"].astype(str) == candidate_label].copy()
    distance_order = ["0--50", "51--100", "101--250", "251--500"]
    by_distance["Distance_Bin"] = pd.Categorical(
        by_distance["Distance_Bin"], distance_order, ordered=True
    )
    by_distance.dropna(subset=["Distance_Bin"], inplace=True)
    by_distance.sort_values("Distance_Bin", inplace=True)

    fig, axes = plt.subplots(2, 1, figsize=(ONE_COLUMN_WIDTH, 4.65))
    for axis, frame, category, title in (
        (axes[0], region, "Stratum", "Variant genomic region"),
        (axes[1], by_distance, "Distance_Bin", "Variant-to-CpG distance"),
    ):
        values = pd.to_numeric(
            frame["Median_Absolute_Predicted_Response"], errors="raise"
        ).to_numpy(float)
        positions = np.arange(len(frame))
        axis.barh(positions, values, color="#7B3294", alpha=0.82)
        labels = [
            f"{getattr(row, category)}  (n={int(row.N_Associations)})"
            for row in frame.itertuples(index=False)
        ]
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_title(title, loc="left")
        axis.set_xlabel(r"Median absolute predicted $\Delta\hat{\beta}$")
        axis.grid(axis="x", alpha=0.16)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6, h_pad=0.8)
    save_figure(fig, output)
    return {
        "region_rows": int(len(region)),
        "distance_rows": int(len(by_distance)),
    }


def plot_mqtl(path: Path, signed_output: Path, magnitude_output: Path) -> dict:
    frame = pd.read_csv(path)
    require_columns(frame, ["model", "predicted_delta_m", "slope_alt_aligned"], path)
    frame = frame[frame["model"].astype(str) == "fusion"].copy()
    frame.dropna(subset=["predicted_delta_m", "slope_alt_aligned"], inplace=True)
    if frame.empty:
        raise ValueError("No fusion rows were found in the mQTL aggregate")

    predicted = pd.to_numeric(frame["predicted_delta_m"], errors="raise")
    observed = pd.to_numeric(frame["slope_alt_aligned"], errors="raise")
    signed_rho = spearman(predicted, observed)
    magnitude_rho = spearman(predicted.abs(), observed.abs())
    agreement = np.sign(predicted) == np.sign(observed)
    direction = float(agreement.mean())

    observed_rank = rank_percentile(observed)
    predicted_rank = rank_percentile(predicted)
    agrees = agreement.to_numpy(bool)
    n_agrees = int(agrees.sum())

    fig, axis = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 3.25))
    axis.scatter(
        observed_rank[agrees],
        predicted_rank[agrees],
        s=23,
        alpha=0.82,
        color="#2F6F9F",
        edgecolor="white",
        linewidth=0.35,
        marker="o",
        label=f"Direction agrees ($n={n_agrees}$)",
    )
    axis.scatter(
        observed_rank[~agrees],
        predicted_rank[~agrees],
        s=28,
        alpha=0.95,
        color="#D98C3F",
        linewidth=1.35,
        marker="x",
        label=f"Direction differs ($n={len(frame) - n_agrees}$)",
    )
    axis.plot(
        [0, 100], [0, 100], color="0.45", linewidth=0.8,
        linestyle="--", label="Identical rank"
    )
    axis.axvline(100.0 * float((observed < 0).mean()), color="0.60", linewidth=0.65, linestyle=":")
    axis.axhline(100.0 * float((predicted < 0).mean()), color="0.60", linewidth=0.65, linestyle=":")
    axis.set(
        xlabel="Reported slope rank percentile",
        ylabel="Predicted response rank percentile",
        title="Breast mQTL signed-rank agreement",
        xlim=(0, 102),
        ylim=(0, 102),
    )
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_aspect("equal", adjustable="box")
    axis.text(
        0.04,
        0.96,
        f"Spearman $\\rho$={signed_rho:.3f}\n"
        f"Direction concordance={n_agrees}/{len(frame)} ({100 * direction:.1f}%)",
        transform=axis.transAxes,
        va="top",
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.78", "alpha": 0.92},
    )
    axis.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="0.78", framealpha=0.92)
    axis.grid(alpha=0.14)
    fig.tight_layout(pad=0.55)
    save_figure(fig, signed_output)

    fig, axis = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 3.25))
    axis.scatter(
        rank_percentile(observed.abs()),
        rank_percentile(predicted.abs()),
        s=22,
        alpha=0.78,
        color="#B24C63",
        edgecolor="white",
        linewidth=0.35,
    )
    axis.plot(
        [0, 100], [0, 100], color="0.45", linewidth=0.8,
        linestyle="--", label="Identical rank"
    )
    axis.set(
        xlabel="Absolute slope rank percentile",
        ylabel="Absolute response rank percentile",
        title="Breast mQTL magnitude-rank agreement",
        xlim=(0, 102),
        ylim=(0, 102),
    )
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_aspect("equal", adjustable="box")
    axis.text(
        0.04,
        0.96,
        f"Spearman $\\rho$={magnitude_rho:.3f}",
        transform=axis.transAxes,
        va="top",
        fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.78", "alpha": 0.92},
    )
    axis.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="0.78", framealpha=0.92)
    axis.grid(alpha=0.14)
    fig.tight_layout(pad=0.55)
    save_figure(fig, magnitude_output)
    return {
        "association_count": int(len(frame)),
        "signed_spearman": signed_rho,
        "magnitude_spearman": magnitude_rho,
        "direction_concordance": direction,
        "display_scale": "average-tie percentile ranks (0--100)",
    }


def plot_candidate(
    candidate_path: Path,
    seed_path: Path,
    comparator_path: Path,
    output: Path,
    case_output: Path,
) -> dict:
    candidates = pd.read_csv(candidate_path)
    require_columns(
        candidates,
        ["Variant_UID", "Absolute_Delta_Beta_Rank", "Predicted_Delta_Beta", "Predicted_Delta_Beta_SD"],
        candidate_path,
    )
    ranks = pd.to_numeric(candidates["Absolute_Delta_Beta_Rank"], errors="raise")
    selected = candidates[ranks == 1]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one rank-1 candidate, found {len(selected)}")
    target = selected.iloc[0]
    uid = str(target["Variant_UID"])
    gene = str(target.get("Gene", "Rank-1 candidate"))
    target_effect = float(target["Predicted_Delta_Beta"])
    exported_sd = float(target["Predicted_Delta_Beta_SD"])

    seeds = pd.read_csv(seed_path)
    require_columns(seeds, ["Variant_UID", "Seed", "Predicted_Delta_Beta"], seed_path)
    seeds = seeds[seeds["Variant_UID"].astype(str) == uid].copy()
    seeds["Seed"] = pd.to_numeric(seeds["Seed"], errors="raise").astype(int)
    seeds.sort_values("Seed", inplace=True)
    if seeds.empty:
        raise ValueError(f"No seed-level scores were found for {uid}")
    seed_values = pd.to_numeric(seeds["Predicted_Delta_Beta"], errors="raise")
    if not np.isclose(seed_values.mean(), target_effect, atol=1e-8, rtol=0):
        raise ValueError("The exported ensemble effect does not equal the seed mean")
    if not np.isclose(seed_values.std(ddof=0), exported_sd, atol=1e-8, rtol=0):
        raise ValueError("The exported SD does not equal the seed population SD")

    comparators = pd.read_csv(comparator_path)
    require_columns(
        comparators,
        ["Target_Rank", "Target_Variant_UID", "Comparator_Delta_Beta"],
        comparator_path,
    )
    comparators = comparators[
        (pd.to_numeric(comparators["Target_Rank"], errors="coerce") == 1)
        & (comparators["Target_Variant_UID"].astype(str) == uid)
    ]
    comparator_values = pd.to_numeric(
        comparators["Comparator_Delta_Beta"], errors="raise"
    ).abs()
    if comparator_values.empty:
        raise ValueError(f"No matched comparators were found for {uid}")

    fig, axis = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 2.55))
    counts, _, _ = axis.hist(
        comparator_values, bins="auto", color="#9ECAE1", edgecolor="white"
    )
    x = abs(target_effect)
    peak = float(np.max(counts)) if len(counts) else 1.0
    axis.axvline(x, color="#D94801", linewidth=2.0, zorder=3, clip_on=False)
    axis.set_ylim(0, peak * 1.22)
    right = max(float(comparator_values.max()), x) * 1.06
    axis.set_xlim(0, right)
    axis.annotate(
        r"NCOA2: $-0.1798 \pm 0.0198$",
        xy=(x, peak * 1.04),
        xytext=(0, 0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.0,
        color="#9A3200",
        clip_on=False,
        annotation_clip=False,
        zorder=5,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 0.9,
            "alpha": 0.96,
        },
    )
    axis.set_xlabel(r"Absolute predicted $\Delta\hat{\beta}$")
    axis.set_ylabel("Matched comparators")
    axis.set_title("Top candidate versus matched background")
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    save_figure(fig, output)

    less = int((comparator_values < abs(target_effect)).sum())
    tied = int((comparator_values == abs(target_effect)).sum())
    percentile = 100.0 * (less + 0.5 * tied) / len(comparator_values)
    summary = {
        "Absolute_Delta_Beta_Rank": 1,
        "Variant_UID": uid,
        "Gene": target.get("Gene", ""),
        "probeID": target.get("probeID", ""),
        "Predicted_Delta_Beta": target_effect,
        "Predicted_Delta_Beta_SD": exported_sd,
        "Seeds": ",".join(str(seed) for seed in seeds["Seed"]),
        "Per_Seed_Predicted_Delta_Beta": ",".join(f"{value:.10g}" for value in seed_values),
        "Matched_Comparator_Count": int(len(comparator_values)),
        "Matched_Comparator_Median_Absolute_Delta_Beta": float(comparator_values.median()),
        "Empirical_Matched_Background_Percentile": float(percentile),
        "Interpretation": "Model-derived case study for hypothesis generation, not causal validation.",
    }
    for column in (
        "GDC_Genomic_DNA_Change",
        "Selected_Transcript_ID",
        "Reference_Codon",
        "Alternate_Codon",
        "Amino_Acid",
        "Absolute_Distance_From_Target_CpG",
        "CpG_Effect",
        "GDC_Occurrence_Count",
        "Delta_RC_Sign_Agreement_Fraction",
        "Mean_Delta_RC_Absolute_Difference",
        "Matched_Background_Tier",
    ):
        value = target.get(column, "")
        summary[column] = "" if pd.isna(value) else value
    save_csv(pd.DataFrame([summary]), case_output)
    return summary


def main() -> None:
    args = arguments()
    inputs = (
        args.metrics_path,
        args.context_path,
        args.variant_context_path,
        args.variant_distance_path,
        args.mqtl_path,
        args.candidate_path,
        args.candidate_seed_path,
        args.comparator_path,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    performance = plot_performance(
        args.metrics_path, args.output_dir / "model_incremental_performance.png"
    )
    information_gains = plot_information_gains(
        args.metrics_path, args.information_gains_output
    )
    context = plot_context(
        args.context_path,
        args.genomic_region_output,
        args.output_dir / "fusion_gain_by_epigenomic_context.png",
    )
    candidate_context = plot_candidate_context(
        args.variant_context_path,
        args.variant_distance_path,
        args.output_dir / "candidate_response_by_context.png",
    )
    mqtl = plot_mqtl(
        args.mqtl_path,
        args.output_dir / "mqtl_signed_rank.png",
        args.output_dir / "mqtl_magnitude_rank.png",
    )
    candidate = plot_candidate(
        args.candidate_path,
        args.candidate_seed_path,
        args.comparator_path,
        args.output_dir / "top_candidate_matched_background.png",
        args.case_study_path,
    )
    keep_manuscript_outputs(args.output_dir)
    save_json(
        {
            "analysis_status": "COMPLETE",
            "figure_format": "six one-column manuscript figures",
            "manuscript_figures": list(MANUSCRIPT_FIGURES),
            "performance": performance,
            "information_source_gains": information_gains,
            "context": context,
            "candidate_context": candidate_context,
            "mqtl": mqtl,
            "top_candidate": candidate,
            "input_sha256": {str(path): sha256(path) for path in inputs},
        },
        args.output_dir / "run_summary.json",
    )
    print("SilentMethyl one-column manuscript figures")
    print(f"  output: {args.output_dir}")
    print("  figures: 6")


if __name__ == "__main__":
    main()
