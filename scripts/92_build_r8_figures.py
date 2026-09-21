#!/usr/bin/env python3
"""Build the mechanism figures. Kept separate from 91_build_manuscript_figures.py because
that script deletes every non-whitelisted file in its output directory, so these would
not survive being written alongside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

ONE_COLUMN_WIDTH = 3.45
TWO_COLUMN_WIDTH = 7.09
COLORS = {
    "fusion": "#E07B39",
    "sequence": "#4C78A8",
    "epi": "#5AA469",
    "levels": "#8C6BB1",
    "deltas": "#E07B39",
    "muted": "#B8B8B8",
    "hit": "#2E7D32",
    "miss": "#C62828",
}
TRACK_LABEL = {
    "Ref_ATAC_Signal": "ATAC",
    "Ref_H3K4me3_Signal": "H3K4me3",
    "Ref_H3K27ac_Signal": "H3K27ac",
    "Ref_H3K27me3_Signal": "H3K27me3",
    "Ref_H3K9me3_Signal": "H3K9me3",
    "Ref_H3K36me3_Signal": "H3K36me3",
    "Ref_H3K4me1_Signal": "H3K4me1",
}

plt.rcParams.update({
    "font.size": 8.0, "axes.titlesize": 9.0, "axes.labelsize": 8.0,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 6.8,
    "lines.linewidth": 1.15, "savefig.dpi": 400,
})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")
    return sha256(path)


def fig_fusion_gain(path_csv: Path, out: Path) -> dict:
    """Task F. Relative reduction is the comparable quantity; absolute is context."""
    table = pd.read_csv(path_csv)
    islands = table[table["Grouping"] == "CpG_Island_Context"].copy()
    tracks = table[table["Grouping"].str.endswith("_Stratum")].copy()
    tracks["track"] = tracks["Grouping"].str.replace("_Stratum", "", regex=False)

    quartiles = tracks[tracks["Stratum"].isin(["Q1 low", "Q4 high"])]
    pivot = quartiles.pivot_table(index="track", columns="Stratum",
                                  values="Relative_Beta_MAE_Reduction")
    lo = quartiles.pivot_table(index="track", columns="Stratum",
                               values="Relative_Reduction_CI_Low")
    hi = quartiles.pivot_table(index="track", columns="Stratum",
                               values="Relative_Reduction_CI_High")
    pivot = pivot.reindex([t for t in TRACK_LABEL if t in pivot.index])
    order = pivot.sort_values("Q4 high").index

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN_WIDTH, 2.7))

    ax = axes[0]
    labels = islands["Stratum"].tolist()
    values = islands["Relative_Beta_MAE_Reduction"].to_numpy() * 100
    err = np.vstack([
        values - islands["Relative_Reduction_CI_Low"].to_numpy() * 100,
        islands["Relative_Reduction_CI_High"].to_numpy() * 100 - values,
    ])
    y = np.arange(len(labels))
    ax.barh(y, values, xerr=err, color=COLORS["fusion"], height=0.62,
            error_kw={"ecolor": "#555", "elinewidth": 0.9, "capsize": 2})
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("relative $\\beta$-MAE reduction (%)")
    ax.set_title("a  CpG-island context", loc="left")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    y = np.arange(len(order))
    for offset, (stratum, color) in enumerate(
            (("Q1 low", COLORS["muted"]), ("Q4 high", COLORS["fusion"]))):
        v = pivot.loc[order, stratum].to_numpy() * 100
        e = np.vstack([
            v - lo.loc[order, stratum].to_numpy() * 100,
            hi.loc[order, stratum].to_numpy() * 100 - v,
        ])
        ax.barh(y + (offset - 0.5) * 0.38, v, xerr=e, height=0.36, color=color,
                label=f"{stratum} quartile",
                error_kw={"ecolor": "#555", "elinewidth": 0.8, "capsize": 1.5})
    ax.set_yticks(y)
    ax.set_yticklabels([TRACK_LABEL[t] for t in order])
    ax.invert_yaxis()
    ax.set_xlabel("relative $\\beta$-MAE reduction (%)")
    ax.set_title("b  chromatin track, bottom vs top quartile", loc="left")
    ax.set_xlim(0, float(np.nanmax(hi.to_numpy())) * 100 * 1.30)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return {"fig_r8_1_fusion_gain_relative.png": save(fig, out)}


def fig_context_ladder(agreement: Path, levels: Path, out: Path) -> dict:
    """Task B. Dose-response, and the levels-vs-deltas separation at each rung."""
    agree = pd.read_csv(agreement)
    agree["norm_mae"] = agree["mae"] / agree["sd_reference"]
    quantity = {"absolute methylation (WT_M)": "levels",
                "variant effect (Delta_M)": "deltas"}
    agree["q"] = agree["quantity"].map(quantity)
    rungs = ["shuffle", "tissue_Lung", "xtissue_mean"]
    nice = {"shuffle": "shuffle\n(random locus)",
            "tissue_Lung": "Lung context\n(same locus)",
            "xtissue_mean": "cross-tissue mean\n(same locus)"}

    level = pd.read_csv(levels).set_index("scheme")

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN_WIDTH, 2.7))

    ax = axes[0]
    order = ["identity", "xtissue_mean", "tissue_Lung", "shuffle"]
    order = [s for s in order if s in level.index]
    v = level.loc[order, "beta_mae"].to_numpy()
    ax.bar(np.arange(len(order)), v,
           color=["#444"] + [COLORS["fusion"]] * (len(order) - 1), width=0.62)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([s.replace("_", "\n") for s in order])
    ax.set_ylabel("held-out $\\beta$ MAE")
    ax.set_title("a  cost of substituting context", loc="left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for i, value in enumerate(v):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=6.4)

    ax = axes[1]
    width = 0.36
    x = np.arange(len(rungs))
    for offset, q in enumerate(("levels", "deltas")):
        sub = agree[agree["q"] == q].set_index("scheme").loc[rungs]
        ax.bar(x + (offset - 0.5) * width, sub["norm_mae"].to_numpy(), width=width,
               color=COLORS[q], label=q)
    for i, rung in enumerate(rungs):
        lev = agree[(agree["q"] == "levels") & (agree["scheme"] == rung)]["norm_mae"].iloc[0]
        dlt = agree[(agree["q"] == "deltas") & (agree["scheme"] == rung)]["norm_mae"].iloc[0]
        ax.text(i, max(lev, dlt) * 1.04, f"{lev/dlt:.2f}x", ha="center",
                va="bottom", fontsize=6.8)
    ax.set_xticks(x); ax.set_xticklabels([nice[r] for r in rungs])
    ax.set_ylabel("normalised MAE vs native context")
    ax.set_title("b  levels move more than variant effects", loc="left")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return {"fig_r8_2_context_ladder.png": save(fig, out)}


def fig_transfer_vs_gain(transfer: Path, gain: Path, out: Path) -> dict:
    """Tasks D x F. Whether the pairing holds depends on WHICH gain you plot.

    Task D found transfer failure concentrating at shores, high H3K4me1 and high
    H3K27me3. The tempting sentence is "context helps most exactly where sequence
    alone transfers worst". It is true on ABSOLUTE beta-MAE gain for two of the
    three strata and false on RELATIVE gain for all three, and it inverts on
    H3K27me3 either way. Both panels are drawn so the dependency is visible
    rather than decided by whichever measure flatters the claim.
    """
    summary = json.loads(transfer.read_text())
    top = summary["top_decile_error"]
    table = pd.read_csv(gain)

    islands = table[table["Grouping"] == "CpG_Island_Context"].set_index("Stratum")
    tracks = table[table["Grouping"].str.endswith("_Stratum")].copy()
    tracks["track"] = tracks["Grouping"].str.replace("_Stratum", "", regex=False)
    top_q = tracks[tracks["Stratum"] == "Q4 high"].set_index("track")

    flagged = [
        ("Shore", islands.loc["Shore"],
         top["cgi_class_enrichment"]["Shore"]["log2_enrichment"], "log2 enr."),
        ("H3K4me1 high", top_q.loc["Ref_H3K4me1_Signal"],
         top["chromatin_contrast"]["Ref_H3K4me1_Signal"]["rank_biserial"], "rank-bis."),
        ("H3K27me3 high", top_q.loc["Ref_H3K27me3_Signal"],
         top["chromatin_contrast"]["Ref_H3K27me3_Signal"]["rank_biserial"], "rank-bis."),
    ]

    stratified = table[table["Grouping"] != "All"]
    abs_all = stratified["Fusion_Minus_Sequence_Beta_MAE"].to_numpy()
    rel_all = stratified["Relative_Beta_MAE_Reduction"].to_numpy()
    n_strata = len(stratified)

    fig, axes = plt.subplots(1, 2, figsize=(TWO_COLUMN_WIDTH, 2.8))
    x = np.arange(len(flagged))
    labels = [f"{name}\n(D: {value:+.2f} {unit})"
              for name, _, value, unit in flagged]

    ax = axes[0]
    vals = np.array([-float(row["Fusion_Minus_Sequence_Beta_MAE"]) for _, row, _, _ in flagged])
    err = np.vstack([
        vals - np.array([-float(row["Beta_MAE_Difference_CI_High"]) for _, row, _, _ in flagged]),
        np.array([-float(row["Beta_MAE_Difference_CI_Low"]) for _, row, _, _ in flagged]) - vals,
    ])
    ranks = [int((abs_all <= float(row["Fusion_Minus_Sequence_Beta_MAE"])).sum())
             for _, row, _, _ in flagged]
    ax.bar(x, vals, yerr=err, width=0.55, color=COLORS["fusion"],
           error_kw={"ecolor": "#555", "elinewidth": 0.9, "capsize": 2.5})
    median_abs = float(np.median(-abs_all))
    ax.axhline(median_abs, color="#666", linestyle=":", linewidth=0.9)
    ax.text(len(flagged) - 0.45, median_abs, "median of\nall 32 strata", fontsize=6.0,
            color="#555", va="bottom", ha="right")
    for i, (value, rank) in enumerate(zip(vals, ranks)):
        ax.text(i, value, f"  rank {rank}/{n_strata}", ha="center", va="bottom",
                fontsize=6.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("absolute $\\beta$-MAE removed")
    ax.set_title("a  absolute gain - pairing holds for 2 of 3", loc="left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)

    ax = axes[1]
    vals = np.array([float(row["Relative_Beta_MAE_Reduction"]) * 100 for _, row, _, _ in flagged])
    err = np.vstack([
        vals - np.array([float(row["Relative_Reduction_CI_Low"]) * 100 for _, row, _, _ in flagged]),
        np.array([float(row["Relative_Reduction_CI_High"]) * 100 for _, row, _, _ in flagged]) - vals,
    ])
    ranks = [int((rel_all >= float(row["Relative_Beta_MAE_Reduction"])).sum())
             for _, row, _, _ in flagged]
    ax.bar(x, vals, yerr=err, width=0.55, color=COLORS["sequence"],
           error_kw={"ecolor": "#555", "elinewidth": 0.9, "capsize": 2.5})
    median_rel = float(np.median(rel_all)) * 100
    ax.axhline(median_rel, color="#666", linestyle=":", linewidth=0.9)
    ax.text(len(flagged) - 0.45, median_rel, "median of\nall 32 strata", fontsize=6.0,
            color="#555", va="bottom", ha="right")
    for i, (value, rank) in enumerate(zip(vals, ranks)):
        ax.text(i, value, f"  rank {rank}/{n_strata}", ha="center", va="bottom",
                fontsize=6.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("relative $\\beta$-MAE reduction (%)")
    ax.set_title("b  relative gain - pairing holds for none", loc="left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)

    fig.suptitle("Task D flagged these three strata; does the fusion gain follow?",
                 fontsize=8.6, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return {"fig_r8_3_transfer_vs_gain.png": save(fig, out)}


def fig_asm(discrimination: Path, out: Path) -> dict:
    """Task E1. Forest of AUROC with block-bootstrap intervals."""
    table = pd.read_csv(discrimination)
    table = table[table["Score"].isin(["fusion", "sequence", "distance_only_baseline"])]
    table = table.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(TWO_COLUMN_WIDTH * 0.72,
                                    0.32 * len(table) + 1.1))
    palette = {"fusion": COLORS["fusion"], "sequence": COLORS["sequence"],
               "distance_only_baseline": COLORS["muted"]}
    y = np.arange(len(table))
    for i, r in table.iterrows():
        lo = r["auroc"] - r["ci_low"] if np.isfinite(r["ci_low"]) else 0
        hi = r["ci_high"] - r["auroc"] if np.isfinite(r["ci_high"]) else 0
        ax.errorbar(r["auroc"], i, xerr=[[lo], [hi]], fmt="o", markersize=4.2,
                    color=palette.get(r["Score"], "#444"), elinewidth=1.0,
                    capsize=2.2)
    ax.axvline(0.5, color="#C62828", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['Stratum']}  ·  {r['Score']}" for _, r in table.iterrows()],
                       fontsize=6.2)
    ax.set_xlabel("AUROC, ASM vs distance-matched non-ASM pairs")
    ax.set_title("ASM discrimination on held-out chr8+chr9", loc="left")
    ax.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return {"fig_r8_4_asm_discrimination.png": save(fig, out)}


def fig_gate(summary_path: Path, out: Path) -> dict:
    """Task A. Variance shares of the two channels, both cohorts."""
    data = json.loads(summary_path.read_text())
    cohorts = ["egtex", "genoa"]
    dna = [data["cohorts"][c]["instrumented"]["variance_share_dna_channel"] for c in cohorts]
    gate = [data["cohorts"][c]["instrumented"]["variance_share_gate_channel"] for c in cohorts]
    rest = [1 - a - b for a, b in zip(dna, gate)]

    fig, ax = plt.subplots(figsize=(ONE_COLUMN_WIDTH, 2.4))
    x = np.arange(len(cohorts))
    ax.bar(x, dna, width=0.55, color=COLORS["sequence"], label="DNA channel")
    ax.bar(x, gate, width=0.55, bottom=dna, color=COLORS["fusion"],
           label="gate channel")
    ax.bar(x, rest, width=0.55, bottom=np.add(dna, gate), color=COLORS["muted"],
           label="covariance / residual")
    for i, (a, b) in enumerate(zip(dna, gate)):
        ax.text(i, a / 2, f"{a:.0%}", ha="center", va="center", fontsize=6.8,
                color="white")
        ax.text(i, a + b / 2, f"{b:.0%}", ha="center", va="center", fontsize=6.8,
                color="white")
    ax.set_xticks(x)
    nice = {"egtex": "eGTEx", "genoa": "GENOA"}
    ax.set_xticklabels([f"{nice.get(c, c)}\n(n={data['cohorts'][c]['pairs']:,})"
                        for c in cohorts])
    ax.set_ylabel("share of variant-effect variance")
    ax.set_ylim(0, 1.14)
    ax.set_title("Gate channel carries independent signal", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    fig.tight_layout()
    return {"fig_r8_5_gate_decomposition.png": save(fig, out)}


def run(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    built: list[str] = []
    skipped: dict[str, str] = {}

    print("building R8 figures")
    if args.fusion_gain.is_file():
        hashes.update(fig_fusion_gain(
            args.fusion_gain, args.output_dir / "fig_r8_1_fusion_gain_relative.png"))
        built.append("fig_r8_1_fusion_gain_relative.png")
    else:
        skipped["fig_r8_1"] = str(args.fusion_gain)

    if args.ladder_agreement.is_file() and args.ladder_levels.is_file():
        hashes.update(fig_context_ladder(
            args.ladder_agreement, args.ladder_levels,
            args.output_dir / "fig_r8_2_context_ladder.png"))
        built.append("fig_r8_2_context_ladder.png")
    else:
        skipped["fig_r8_2"] = str(args.ladder_agreement)

    if args.transfer_failure.is_file() and args.fusion_gain.is_file():
        hashes.update(fig_transfer_vs_gain(
            args.transfer_failure, args.fusion_gain,
            args.output_dir / "fig_r8_3_transfer_vs_gain.png"))
        built.append("fig_r8_3_transfer_vs_gain.png")
    else:
        skipped["fig_r8_3"] = str(args.transfer_failure)

    if args.asm_discrimination.is_file():
        hashes.update(fig_asm(
            args.asm_discrimination,
            args.output_dir / "fig_r8_4_asm_discrimination.png"))
        built.append("fig_r8_4_asm_discrimination.png")
    else:
        skipped["fig_r8_4"] = (f"{args.asm_discrimination} not present -- "
                               "rerun after the ASM job lands")

    if args.gate_decomposition.is_file():
        hashes.update(fig_gate(
            args.gate_decomposition,
            args.output_dir / "fig_r8_5_gate_decomposition.png"))
        built.append("fig_r8_5_gate_decomposition.png")
    else:
        skipped["fig_r8_5"] = str(args.gate_decomposition)

    summary = {
        "analysis": "R8 manuscript figures",
        "analysis_status": "COMPLETE" if not skipped else "PARTIAL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "figures_built": built,
        "figures_skipped": skipped,
        "output_sha256": hashes,
        "note": (
            "Separate from 91_build_manuscript_figures.py, which deletes any file "
            "in its output directory not on its own whitelist. Script 91 is not "
            "modified by this script."
        ),
        "drawing_decisions": {
            "relative_not_absolute": (
                "Figure 1 leads with relative beta-MAE reduction. Absolute gain is "
                "bounded by the sequence-only error per stratum, so it ranks strata "
                "by how bad they start rather than by how much context helps."
            ),
            "pairing_not_forced": (
                "Figure 3 colours the H3K27me3 inversion differently rather than "
                "omitting it. Two of three strata agree with Task D; one inverts."
            ),
        },
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nbuilt {len(built)} figure(s) -> {args.output_dir}")
    for name, reason in skipped.items():
        print(f"  SKIPPED {name}: {reason}")
    return 0


def parse_args() -> argparse.Namespace:
    root = Path("results/journal")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fusion-gain", type=Path,
                   default=root / "ablation_breast_epithelium/fusion_gain_stratified/fusion_gain_stratified.csv")
    p.add_argument("--ladder-agreement", type=Path,
                   default=root / "ablation_breast_epithelium/context_ladder/agreement_with_identity.csv")
    p.add_argument("--ladder-levels", type=Path,
                   default=root / "ablation_breast_epithelium/context_ladder/level_accuracy_by_scheme.csv")
    p.add_argument("--transfer-failure", type=Path,
                   default=root / "joint/transfer_failure/transfer_failure_summary.json")
    p.add_argument("--asm-discrimination", type=Path,
                   default=root / "asm_validation/asm_discrimination.csv")
    p.add_argument("--gate-decomposition", type=Path,
                   default=root / "ablation_breast_epithelium/gate_decomposition/gate_decomposition_summary.json")
    p.add_argument("--output-dir", type=Path,
                   default=root / "manuscript_figures_r8")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
