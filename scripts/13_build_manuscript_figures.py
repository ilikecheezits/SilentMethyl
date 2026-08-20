#!/usr/bin/env python3
"""Build the SilentMethyl one-column manuscript figures."""

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
    "mqtl_combined_rank.png",
    "top_candidate_matched_background.png",
    "stk11_variants_vs_nonsynonymous_screen.png",
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
        "--literature-variant-path",
        type=Path,
        default=Path(
            "results/journal/literature_variant_screen/"
            "literature_variant_predictions_ranked.csv"
        ),
        help=(
            "Prespecified model-target rows from the nonsynonymous literature "
            "screen produced by script 15."
        ),
    )
    parser.add_argument(
        "--stk11-case-output",
        type=Path,
        default=Path(
            "results/journal/literature_variant_screen/"
            "stk11_case_study_figure_values.csv"
        ),
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

    fig, axes = plt.subplots(1, 3, figsize=(ONE_COLUMN_WIDTH, 3))
    positions = np.arange(3)
    
    metrics = ("m_mae", "beta_mae", "roc_auc")
    labels = (r"M-value MAE $\downarrow$", r"Beta MAE $\downarrow$", r"ROC-AUC $\uparrow$")
    panel_letters = ("A", "B", "C")

    for i, (axis, metric, label, letter) in enumerate(zip(axes, metrics, labels, panel_letters)):
        values = (
            frame.pivot(index="Seed", columns="Model", values=metric)
            .reindex(index=seeds, columns=MODEL_ORDER)
        )
            
        means = values.mean(axis=0).to_numpy(float)
        standard_deviations = values.std(axis=0, ddof=1).to_numpy(float)
        colors = [MODEL_COLORS[m] for m in MODEL_ORDER]

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

        axis.set_xticks(positions)
        axis.set_xticklabels(["Context", "Sequence", "Fusion"], rotation=45, ha="right")
        axis.set_ylim(bottom=0)
        axis.set_title(label, fontsize=8.0)
        axis.grid(axis="y", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
        
                          
        axis.text(-0.34, 1.05, letter, transform=axis.transAxes, fontsize=12, fontweight='bold', va='bottom')

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
    panel_letter: str = "",
    colors: list[str] = None
) -> dict[str, float]:
    subset = frame[frame["Grouping"].astype(str) == group].copy()
    subset["Stratum"] = pd.Categorical(subset["Stratum"], order, ordered=True)
    subset.dropna(subset=["Stratum"], inplace=True)
    subset.sort_values("Stratum", inplace=True)
    
    gain = pd.to_numeric(subset["Sequence_Minus_Fusion_Beta_MAE"], errors="raise").to_numpy(float)
    low = -pd.to_numeric(subset["Bootstrap_CI_High"], errors="raise").to_numpy(float)
    high = -pd.to_numeric(subset["Bootstrap_CI_Low"], errors="raise").to_numpy(float)
    positions = np.arange(len(subset))
    
    if colors is None:
        colors = ["#D95F02"] * len(positions)
        
    for i in range(len(positions)):
        axis.errorbar(
            gain[i],
            positions[i],
            xerr=[[gain[i] - low[i]], [high[i] - gain[i]]],
            fmt="o",
            color=colors[i],
            ecolor="0.65",
            capsize=2.5,
            markersize=6.5,
            zorder=3
        )
        
    axis.axvline(0, color="0.45", linewidth=0.8, linestyle="--", zorder=1)
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
    
                      
    if panel_letter:
        axis.text(-0.35, 1.05, panel_letter, transform=axis.transAxes, fontsize=12, fontweight='bold', va='bottom')
        
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
            "A",
            ['#006d2c', '#31a354', '#74c476', '#bae4b3']                          
        ),
        (
            "ATAC_Stratum", 
            "ATAC signal", 
            ["Q1 low", "Q2", "Q3", "Q4 high"],
            "B",
            ['#08519c', '#3182bd', '#6baed6', '#bdd7e7']                         
        ),
        (
            "H3K27ac_Stratum",
            "H3K27ac signal",
            ["Q1 low", "Q2", "Q3", "Q4 high"],
            "C",
            ['#a63603', '#e6550d', '#fd8d3c', '#fdbe85']                           
        ),
    )
    fig, axes = plt.subplots(3, 1, figsize=(ONE_COLUMN_WIDTH, 6.25))
    for axis, (group, title, order, letter, colors) in zip(axes, panels):
        output_values[group] = _draw_gain_panel(axis, frame, group, title, order, letter, colors)
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
    panels = (
        (axes[0], region, "Stratum", "Variant genomic region", "A", ['#54278f', '#756bb1', '#9e9ac8']),          
        (axes[1], by_distance, "Distance_Bin", "Variant-to-CpG distance", "B", ['#a50f15', '#de2d26', '#fb6a4a', '#fcae91']),       
    )
    
    for axis, frame, category, title, letter, colors in panels:
        values = pd.to_numeric(
            frame["Median_Absolute_Predicted_Response"], errors="raise"
        ).to_numpy(float)
        positions = np.arange(len(frame))
        
        axis.barh(positions, values, color=colors, edgecolor='none', alpha=0.9)
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
        
                          
        axis.text(-0.35, 1.05, letter, transform=axis.transAxes, fontsize=12, fontweight='bold', va='bottom')
        
    fig.tight_layout(pad=0.6, h_pad=1.2)
    save_figure(fig, output)
    return {
        "region_rows": int(len(region)),
        "distance_rows": int(len(by_distance)),
    }


def plot_mqtl_combined(path: Path, output: Path) -> dict:
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

                                               
    fig, axes = plt.subplots(2, 1, figsize=(ONE_COLUMN_WIDTH, 6.1))

                                  
    ax1 = axes[0]
    ax1.scatter(
        observed_rank[agrees],
        predicted_rank[agrees],
        s=14,
        alpha=0.78,
        color="#2F6F9F",
        edgecolor="white",
        linewidth=0.25,
        marker="o",
        label=f"Direction agrees ($n={n_agrees}$)",
    )
    ax1.scatter(
        observed_rank[~agrees],
        predicted_rank[~agrees],
        s=16,
        alpha=0.9,
        color="#D98C3F",
        linewidth=1.0,
        marker="x",
        label=f"Direction differs ($n={len(frame) - n_agrees}$)",
    )
    ax1.plot(
        [0, 100], [0, 100], color="0.45", linewidth=0.8,
        linestyle="--", label="Identical rank"
    )
    ax1.axvline(100.0 * float((observed < 0).mean()), color="0.60", linewidth=0.65, linestyle=":")
    ax1.axhline(100.0 * float((predicted < 0).mean()), color="0.60", linewidth=0.65, linestyle=":")
    ax1.set(
        xlabel="Reported slope rank percentile",
        ylabel="Predicted response rank percentile",
        title="Breast mQTL signed-rank agreement",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    ax1.set_xticks([0, 25, 50, 75, 100])
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_aspect("equal", adjustable="box")
    ax1.text(
        0.02,
        0.98,
        f"Spearman $\\rho$={signed_rho:.3f}\n"
        f"Direction concordance={n_agrees}/{len(frame)} ({100 * direction:.1f}%)",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=6.2,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.78", "alpha": 0.9},
    )
    ax1.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="0.78",
        framealpha=0.9,
        fontsize=6.5,
        borderaxespad=0.2,
    )
    ax1.grid(alpha=0.14)
    ax1.text(-0.18, 1.05, 'A', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='bottom')

                                     
    ax2 = axes[1]
    ax2.scatter(
        rank_percentile(observed.abs()),
        rank_percentile(predicted.abs()),
        s=14,
        alpha=0.78,
        color="#B24C63",
        edgecolor="white",
        linewidth=0.25,
    )
    ax2.plot(
        [0, 100], [0, 100], color="0.45", linewidth=0.8,
        linestyle="--", label="Identical rank"
    )
    ax2.set(
        xlabel="Absolute slope rank percentile",
        ylabel="Absolute response rank percentile",
        title="Breast mQTL magnitude-rank agreement",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_aspect("equal", adjustable="box")
    ax2.text(
        0.02,
        0.98,
        f"Spearman $\\rho$={magnitude_rho:.3f}",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=6.2,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.78", "alpha": 0.9},
    )
    ax2.legend(
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="0.78",
        framealpha=0.9,
        fontsize=6.5,
        borderaxespad=0.2,
    )
    ax2.grid(alpha=0.14)
    ax2.text(-0.18, 1.05, 'B', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='bottom')

    fig.tight_layout(pad=0.8)
    save_figure(fig, output)

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
    ranks = pd.to_numeric(candidates["Absolute_Delta_Beta_Rank"], errors="raise")
    selected = candidates[ranks == 1]
    target = selected.iloc[0]
    uid = str(target["Variant_UID"])
    gene = str(target.get("Gene", "Rank-1 candidate"))
    target_effect = float(target["Predicted_Delta_Beta"])
    exported_sd = float(target["Predicted_Delta_Beta_SD"])

    seeds = pd.read_csv(seed_path)
    seeds = seeds[seeds["Variant_UID"].astype(str) == uid].copy()
    seeds["Seed"] = pd.to_numeric(seeds["Seed"], errors="raise").astype(int)
    seeds.sort_values("Seed", inplace=True)
    seed_values = pd.to_numeric(seeds["Predicted_Delta_Beta"], errors="raise")

    comparators = pd.read_csv(comparator_path)
    comparators = comparators[
        (pd.to_numeric(comparators["Target_Rank"], errors="coerce") == 1)
        & (comparators["Target_Variant_UID"].astype(str) == uid)
    ]
    comparator_values = pd.to_numeric(
        comparators["Comparator_Delta_Beta"], errors="raise"
    ).abs()

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
        xytext=(-4, 0),
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
        "Predicted_Delta_Beta": target_effect,
    }
    save_csv(pd.DataFrame([summary]), case_output)
    return summary


def plot_stk11_nonsynonymous_screen(
    literature_variant_path: Path,
    output: Path,
    case_output: Path,
) -> dict:
    """Place the two STK11 examples in the full protein-altering screen."""
    frame = pd.read_csv(literature_variant_path)
    require_columns(frame, ["Variant_ID", "probeID"], literature_variant_path)
    delta_column = next(
        (
            name
            for name in ("Predicted_Delta_Beta_Mean", "Predicted_Delta_Beta")
            if name in frame.columns
        ),
        None,
    )
    if delta_column is None:
        raise ValueError(
            f"{literature_variant_path} lacks a predicted delta-beta column"
        )

    # The ranked file should contain prespecified application targets only.
    # Deduplicate exact variant--CpG pairs defensively without collapsing
    # genuinely distinct CpG targets for the same variant.
    frame = frame.copy()
    frame[delta_column] = pd.to_numeric(frame[delta_column], errors="raise")
    frame = frame[np.isfinite(frame[delta_column])].copy()
    frame.drop_duplicates(["Variant_ID", "probeID"], inplace=True)
    if frame.empty:
        raise ValueError(f"No finite predictions in {literature_variant_path}")

    target_specs = (
        ("rs2145420809:A>G", "cg16601904", "#2A9D8F"),
        ("rs148928808:C>G", "cg08681293", "#D95F02"),
    )
    targets: list[tuple[str, str, str, float]] = []
    for variant_id, probe_id, color in target_specs:
        selected = frame[
            frame["Variant_ID"].astype(str).eq(variant_id)
            & frame["probeID"].astype(str).eq(probe_id)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"Expected exactly one row for {variant_id}--{probe_id} in "
                f"{literature_variant_path}, found {len(selected)}"
            )
        targets.append(
            (variant_id.split(":", 1)[0], probe_id, color, float(selected.iloc[0][delta_column]))
        )

    values = frame[delta_column].to_numpy(float)
    fig, axis = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 2.65))

    signed_min = float(np.min(values))
    signed_max = float(np.max(values))
    margin = max(0.015, 0.12 * max(abs(signed_min), abs(signed_max), 0.05))
    left = min(-margin, signed_min - 0.02)
    right = max(margin, signed_max + 0.02)
    bin_edges = np.linspace(left, right, 28)

    counts, _, _ = axis.hist(
        values,
        bins=bin_edges,
        color="#9ECAE1",
        edgecolor="white",
        linewidth=0.55,
    )
    peak = float(np.max(counts)) if len(counts) else 1.0
    axis.set_ylim(0, peak * 1.35)
    axis.set_xlim(left, right)
    axis.axvline(0, color="0.45", linewidth=0.9, linestyle="--", zorder=2)

    for rsid, probe_id, color, delta in targets:
        axis.axvline(delta, color=color, linewidth=2.0, zorder=4, clip_on=False)
        is_positive = delta > 0
        axis.annotate(
            f"{rsid}\n{delta:+.4f}",
            xy=(delta, peak * 1.05),
            xytext=(-4 if is_positive else 4, 0),
            textcoords="offset points",
            ha="right" if is_positive else "left",
            va="bottom",
            fontsize=6.9,
            color=color,
            clip_on=False,
            annotation_clip=False,
            zorder=5,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.9,
                "alpha": 0.96,
            },
        )

    pair_count = int(len(frame))
    variant_count = int(frame["Variant_ID"].astype(str).nunique())
    axis.text(
        0.98,
        0.05,
        f"{pair_count} variant--CpG pairs\n{variant_count} unique variants",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="0.25",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "0.75",
            "linewidth": 0.7,
            "alpha": 0.90,
        },
    )
    axis.set_xlabel(r"Predicted $\Delta\hat{\beta}$")
    axis.set_ylabel("Nonsynonymous variants")
    axis.set_title(r"$\mathit{STK11}$ variants within nonsynonymous background mutations")
    axis.grid(axis="y", alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.6)
    save_figure(fig, output)

    summary_rows = [
        {
            "Record": "Screened pool",
            "Variant_ID": "",
            "probeID": "",
            "Predicted_Delta_Beta": np.nan,
            "N_Variant_CpG_Pairs": pair_count,
            "N_Unique_Variants": variant_count,
        }
    ]
    summary_rows.extend(
        {
            "Record": "Highlighted target",
            "Variant_ID": rsid,
            "probeID": probe_id,
            "Predicted_Delta_Beta": delta,
            "N_Variant_CpG_Pairs": np.nan,
            "N_Unique_Variants": np.nan,
        }
        for rsid, probe_id, _, delta in targets
    )
    save_csv(pd.DataFrame(summary_rows), case_output)
    return {
        "variant_cpg_pair_count": pair_count,
        "unique_variant_count": variant_count,
        "delta_column": delta_column,
        "highlighted_targets": {
            rsid: {"probeID": probe_id, "predicted_delta_beta": delta}
            for rsid, probe_id, _, delta in targets
        },
    }


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
        args.literature_variant_path,
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
                                                                          
    mqtl = plot_mqtl_combined(
        args.mqtl_path,
        args.output_dir / "mqtl_combined_rank.png",
    )
    candidate = plot_candidate(
        args.candidate_path,
        args.candidate_seed_path,
        args.comparator_path,
        args.output_dir / "top_candidate_matched_background.png",
        args.case_study_path,
    )
    stk11 = plot_stk11_nonsynonymous_screen(
        args.literature_variant_path,
        args.output_dir / "stk11_variants_vs_nonsynonymous_screen.png",
        args.stk11_case_output,
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
            "stk11_case_study": stk11,
            "input_sha256": {str(path): sha256(path) for path in inputs},
        },
        args.output_dir / "run_summary.json",
    )
    print("SilentMethyl one-column manuscript figures")
    print(f"  output: {args.output_dir}")
    print("  figures: 6")


if __name__ == "__main__":
    main()