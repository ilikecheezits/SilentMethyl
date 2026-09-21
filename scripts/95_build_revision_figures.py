"""Build main Figures 2-7 and Supplementary Figures S1-S3 for the expanded revision, with one source-data CSV per panel, from current breast-epithelium exports only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

MM = 1.0 / 25.4

FUSION = "#1B6CA8"
SEQUENCE = "#6A4C93"
CONTEXT = "#2A9D8F"
CPGENIE = "#E9A13B"
DEEPCPG = "#5FA855"
KMER = "#D9534F"
COMPOSITION = "#8D99AE"
DISTANCE = "#B5559B"
NULLC = "#B8C0C8"
HIGHLIGHT = "#D7642C"
TINT = "#EAF2F8"
OTHER = "#5A6673"
TEXT = "#22303C"

MODEL_STYLE = {
    "fusion": (FUSION, "D"),
    "sequence": (SEQUENCE, "o"),
    "context": (CONTEXT, "s"),
    "epi": (CONTEXT, "s"),
    "cpgenie": (CPGENIE, "^"),
    "deepcpg": (DEEPCPG, "v"),
    "kmer_ridge": (KMER, "P"),
    "composition": (COMPOSITION, "X"),
    "distance_only": (NULLC, "o"),
    "distance_only_baseline": (NULLC, "o"),
}

MODEL_LABEL = {
    "fusion": "Fusion", "sequence": "Sequence", "epi": "Context", "context": "Context",
    "cpgenie": "CpGenie", "deepcpg": "DeepCpG", "kmer_ridge": "$k$-mer ridge",
    "composition": "Composition", "distance_only": "Distance only",
    "distance_only_baseline": "Distance only",
}

HEAT_CMAP = "YlGnBu"

SOURCE_DATA: dict[str, pd.DataFrame] = {}


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.0,
        "axes.edgecolor": TEXT,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def panel_letter(ax, letter: str, xoff: float = -40.0, yoff: float = 9.0) -> None:
    ax.annotate(letter.upper(), xy=(0.0, 1.0), xycoords="axes fraction",
                textcoords="offset points", xytext=(xoff, yoff), fontsize=10,
                fontweight="bold", va="bottom", ha="left", color=TEXT,
                annotation_clip=False)


def predictions(repo, model: str, seed: int = 42) -> pd.DataFrame:
    if model in ("cpgenie", "deepcpg"):
        path = repo.j / "published_baselines" / model / f"seed{seed}" / "predictions.csv"
    else:
        path = repo.abl / f"seed{seed}" / model / "predictions.csv"
    return pd.read_csv(path)


def roc_curve(labels: np.ndarray, scores: np.ndarray, n_points: int = 200):
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = np.concatenate([[0.0], tps / max(tps[-1], 1)])
    fpr = np.concatenate([[0.0], fps / max(fps[-1], 1)])
    idx = np.unique(np.linspace(0, len(fpr) - 1, n_points).astype(int))
    return fpr[idx], tpr[idx], float(np.trapezoid(tpr, fpr))


def heatmap(ax, matrix, row_labels, col_labels, vmin, vmax, cmap=HEAT_CMAP,
            fmt="{:.0f}", fontsize=5.2, text_colour=None):
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=6.0)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6.0)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    mid = 0.5 * (vmin + vmax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isfinite(v):
                continue
            if text_colour == "auto":
                r, g, b, _ = plt.get_cmap(cmap)((v - vmin) / (vmax - vmin))
                colour_v = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.58 else TEXT
            else:
                colour_v = text_colour or ("white" if v > mid else TEXT)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                    color=colour_v)
    return im


SPECTRUM_CMAP = "viridis_r"
COOL_CMAP = "Blues"
WARM_CMAP = "Oranges"


def hexpanel(ax, x, y, extent, gridsize=44, cmap=SPECTRUM_CMAP, square=True):
    """Hexbin whose cells render as regular hexagons.

    matplotlib sizes its cells sx = (xmax - xmin) / nx wide and 2 * sy / 3 tall, so they
    are equilateral on screen exactly when the axes box has height/width = ny * sqrt(3)
    / nx and its limits equal the binning extent. Both are pinned here: ny is rounded to
    the integer nearest the panel's own shape, then the box aspect is set to the value
    that integer implies, which moves the panel by under a per cent and leaves no
    rounding error in the cells. `square` asks for a box that is also close to 1:1.
    """
    if square:
        ratio = 1.0
    else:
        box = ax.get_position()
        fw, fh = ax.figure.get_size_inches()
        ratio = (box.height * fh) / (box.width * fw)
    ny = max(2, int(round(gridsize * ratio / np.sqrt(3))))
    ax.set_box_aspect(ny * np.sqrt(3) / gridsize)
    hb = ax.hexbin(x, y, gridsize=(gridsize, ny), bins="log", cmap=cmap, mincnt=1,
                   linewidths=0, extent=extent, zorder=2)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return hb


def panel_title(ax, text: str) -> None:
    ax.set_title(text, fontsize=9, loc="left", pad=4, color=TEXT)


def record(name: str, df: pd.DataFrame) -> None:
    SOURCE_DATA[name] = df.copy()


def forest(ax, rows, xlabel, null_line=None, xlim=None, label_pad=None):
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        colour = r.get("colour", OTHER)
        marker = r.get("marker", "o")
        lo, hi = r.get("lo"), r.get("hi")
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [y, y], color=colour, lw=1.0, solid_capstyle="butt", zorder=2)
            for b in (lo, hi):
                ax.plot([b, b], [y - 0.14, y + 0.14], color=colour, lw=0.8, zorder=2)
        ax.plot([r["value"]], [y], marker=marker, ms=r.get("ms", 4.0),
                color=colour, mfc=r.get("mfc", colour), mec=colour, mew=0.8, zorder=3)
    if null_line is not None:
        ax.axvline(null_line, color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    if label_pad is not None:
        ax.tick_params(axis="y", pad=label_pad)
    ax.grid(axis="x", color="#E6EAED", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    return ys


class Repo:
    def __init__(self, root: Path):
        self.root = root
        self.j = root / "results" / "journal"
        self.abl = self.j / "ablation_breast_epithelium"

    def csv(self, rel: str) -> pd.DataFrame:
        return pd.read_csv(self.j / rel)

    def acsv(self, rel: str) -> pd.DataFrame:
        return pd.read_csv(self.abl / rel)

    def ajson(self, rel: str):
        with open(self.abl / rel) as fh:
            return json.load(fh)

    def jjson(self, rel: str):
        with open(self.j / rel) as fh:
            return json.load(fh)

    def rjson(self, rel: str):
        with open(self.root / rel) as fh:
            return json.load(fh)


def fig2_transfer_panel(repo: Repo) -> pd.DataFrame:
    summary = repo.jjson("joint/transfer_failure/transfer_failure_summary.json")
    table = summary["error_vs_variance"]["decile_table"]
    return pd.DataFrame(table)


def fig2_full(repo: Repo, out: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 173 * MM))
    gs = fig.add_gridspec(3, 6, left=0.160, right=0.965, top=0.920, bottom=0.070,
                          wspace=0.92, hspace=0.55,
                          height_ratios=[0.80, 1.0, 0.98])

    metrics = repo.acsv("paired_model_bootstrap/model_metrics_recomputed.csv")
    seeded = metrics[metrics["Analysis"] == "individual_seed"].copy()
    seeded["Seed"] = seeded["Seed"].astype(str)
    base = repo.csv("sequence_baselines/absolute_prediction_metrics.csv")

    rows = []
    for model, det in [("composition", True), ("kmer_ridge", True), ("epi", False),
                       ("cpgenie", False), ("deepcpg", False), ("sequence", False),
                       ("fusion", False)]:
        if det:
            r = base[base["model"] == model].iloc[0]
            rows.append(dict(model=model, deterministic=True,
                             mae=[float(r["beta_mae"])], auc=[float(r["auc"])]))
        elif model in ("cpgenie", "deepcpg"):
            mae, auc = [], []
            for seed in (42, 43, 44):
                m = repo.jjson(f"published_baselines/{model}/seed{seed}/metrics.json")
                mae.append(float(m["beta_mae"]))
                auc.append(float(m.get("auc", m.get("roc_auc"))))
            rows.append(dict(model=model, deterministic=False, mae=mae, auc=auc))
        else:
            sub = seeded[seeded["Model"] == model]
            rows.append(dict(model=model, deterministic=False,
                             mae=sub["beta_mae"].astype(float).tolist(),
                             auc=sub["roc_auc"].astype(float).tolist()))

    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ys = np.arange(len(rows))[::-1]
    # b is drawn from the chance baseline: bars from zero would make every ROC-AUC
    # between 0.87 and 0.98 look alike.
    for ax, key, xlabel, base, xlim in [(ax_a, "mae", r"$\beta$ MAE", 0.0, (0.0, 0.215)),
                                        (ax_b, "auc", "ROC-AUC", 0.5, (0.5, 1.0))]:
        for y, r in zip(ys, rows):
            colour, _ = MODEL_STYLE[r["model"]]
            vals = np.asarray(r[key], dtype=float)
            ax.barh(y, float(vals.mean()) - base, left=base, height=0.62, color=colour,
                    edgecolor="none", zorder=3)
            if vals.size > 1:  # the two ridge baselines are single deterministic fits
                ax.plot([vals.min(), vals.max()], [y, y], color=TEXT, lw=0.9,
                        solid_capstyle="butt", zorder=5)
                for bound in (vals.min(), vals.max()):
                    ax.plot([bound, bound], [y - 0.16, y + 0.16], color=TEXT, lw=0.9, zorder=5)
        ax.set_yticks(ys)
        ax.set_yticklabels([MODEL_LABEL[r["model"]] for r in rows], fontsize=7)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)
        ax.grid(axis="x", color="#E9EDF0", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    for lab in ax_a.get_yticklabels():
        lab.set_color(TEXT)
    ax_b.set_yticklabels([])  # same rows as a, aligned in the same grid row
    panel_letter(ax_a, "a", xoff=-72)
    panel_title(ax_a, "Held-out accuracy")
    panel_title(ax_b, "Discrimination")
    panel_letter(ax_b, "b", xoff=-14)
    record("fig2ab_heldout_accuracy", pd.DataFrame([
        dict(panel="2a/2b", model=r["model"], panel_label=MODEL_LABEL[r["model"]].replace("$", ""),
             developed_here=r["model"] in ("fusion", "sequence", "epi"),
             deterministic=r["deterministic"], n_test_cpgs=26570,
             seeds="42;43;44" if not r["deterministic"] else "single fit",
             beta_mae_values=";".join(f"{v:.6f}" for v in r["mae"]),
             beta_mae_mean=float(np.mean(r["mae"])),
             roc_auc_values=";".join(f"{v:.6f}" for v in r["auc"]),
             roc_auc_mean=float(np.mean(r["auc"])),
             uncertainty="point is the mean over the seeds listed; per-seed values are in "
                         "beta_mae_values and roc_auc_values")
        for r in rows]))

    preds = {m: predictions(repo, m) for m in ("fusion", "sequence", "epi")}

    ax_c = fig.add_subplot(gs[1, 0:2])
    pos_c = ax_c.get_position()
    ax_c.set_position([pos_c.x0, pos_c.y0, pos_c.width * 0.84, pos_c.height])
    d = preds["fusion"]
    hb = hexpanel(ax_c, d["true_beta"], d["pred_beta_rc_avg"], (0.0, 1.0, 0.0, 1.0),
                  gridsize=40, cmap=SPECTRUM_CMAP)
    ax_c.plot([0, 1], [0, 1], color=HIGHLIGHT, ls=(0, (3, 2)), lw=1.0, zorder=3)
    r_p = float(np.corrcoef(d["true_beta"], d["pred_beta_rc_avg"])[0, 1])
    mae = float(np.mean(np.abs(d["true_beta"] - d["pred_beta_rc_avg"])))
    ax_c.set_xlabel(r"observed $\beta$", labelpad=1.5)
    ax_c.set_ylabel(r"predicted $\beta$", labelpad=1.5)
    ax_c.set_xticks([0, 0.5, 1.0])
    ax_c.set_yticks([0, 0.5, 1.0])
    ax_c.text(0.04, 0.96, f"$r$ = {r_p:.3f}\nMAE = {mae:.4f}",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=5.8,
              color=TEXT, linespacing=1.4,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5))
    cb = fig.colorbar(hb, cax=ax_c.inset_axes([1.030, 0.0, 0.028, 1.0]))
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=4.8)
    cb.outline.set_visible(False)
    panel_letter(ax_c, "c", xoff=-46)
    panel_title(ax_c, "Fusion calibration")
    record("fig2c_calibration", pd.DataFrame([dict(
        panel="2c", model="fusion", seed=42, n_test_cpgs=len(d), pearson_r=r_p,
        beta_mae=mae, note="log-scaled hexbin of observed vs predicted beta; dashed line is y=x",
        source="ABL/seed42/fusion/predictions.csv")]))

    ax_d = fig.add_subplot(gs[1, 2:4])
    d_src = []
    edges = np.linspace(0, 1, 11)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for model in ("fusion", "sequence", "epi"):
        dm = preds[model]
        idx = np.clip(np.digitize(dm["true_beta"], edges) - 1, 0, 9)
        err = np.abs(dm["true_beta"] - dm["pred_beta_rc_avg"])
        vals = [float(err[idx == k].mean()) if (idx == k).sum() else np.nan for k in range(10)]
        ns = [int((idx == k).sum()) for k in range(10)]
        colour, marker = MODEL_STYLE[model]
        ax_d.plot(centres, vals, marker=marker, ms=3.4, lw=1.5, color=colour, zorder=3,
                  label=MODEL_LABEL[model])
        d_src.append(pd.DataFrame(dict(panel="2d", model=model, seed=42,
                                       beta_bin_centre=centres, beta_mae=vals, n_cpgs=ns)))
    ax_d.set_xlabel(r"observed $\beta$", labelpad=1.5)
    ax_d.set_ylabel(r"$\beta$ MAE", labelpad=1.5)
    ax_d.set_xlim(0, 1)
    ax_d.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_d.grid(color="#E9EDF0", lw=0.5, zorder=0)
    ax_d.set_axisbelow(True)
    ax_d.legend(loc="upper right", frameon=False, fontsize=5.8, handletextpad=0.4,
                borderpad=0.0, labelspacing=0.15)
    panel_letter(ax_d, "d", xoff=-46)
    panel_title(ax_d, "Error by level")
    record("fig2d_error_by_level", pd.concat(d_src, ignore_index=True))

    ax_e = fig.add_subplot(gs[2, 0:2])
    folds = []
    for f in (1, 2, 3):
        fu = repo.ajson(f"fold{f}/fusion/metrics.json")
        sq = repo.jjson(f"folds/fold{f}/sequence/metrics.json")
        folds.append(dict(fold=f, n=int(fu["n_test_loci"]),
                          fusion=float(fu["beta_mae"]), sequence=float(sq["beta_mae"])))
    f42 = seeded[(seeded["Model"] == "fusion") & (seeded["Seed"] == "42")].iloc[0]
    s42 = seeded[(seeded["Model"] == "sequence") & (seeded["Seed"] == "42")].iloc[0]
    folds.insert(0, dict(fold=0, n=26570, fusion=float(f42["beta_mae"]),
                         sequence=float(s42["beta_mae"])))
    labels = ["chr8-9 (primary)", "chr12/18", "chr4/20", "chr13/15/21"]
    split_cols = plt.get_cmap("winter")(np.linspace(0.0, 0.85, len(folds)))
    for fd, lab, cc in zip(folds, labels, split_cols):
        ax_e.plot([0, 1], [fd["sequence"], fd["fusion"]], "-", color=cc, lw=1.5, zorder=3,
                  label=lab)
        ax_e.plot([0], [fd["sequence"]], "o", ms=4.8, color=cc, mec="white", mew=0.7, zorder=4)
        ax_e.plot([1], [fd["fusion"]], "D", ms=4.8, color=cc, mec="white", mew=0.7, zorder=4)
    ax_e.set_xticks([0, 1])
    ax_e.set_xticklabels(["sequence", "fusion"], fontsize=7)
    ax_e.set_xlim(-0.18, 1.18)
    ax_e.set_ylabel(r"$\beta$ MAE (seed 42)", labelpad=1.5)
    ax_e.grid(axis="y", color="#E9EDF0", lw=0.5, zorder=0)
    ax_e.set_axisbelow(True)
    ax_e.legend(loc="upper right", frameon=False, fontsize=5.2, handlelength=1.1,
                handletextpad=0.4, borderpad=0.0, labelspacing=0.18,
                title="held-out set", title_fontsize=5.2)
    panel_letter(ax_e, "f", xoff=-44)
    panel_title(ax_e, "Chromosome splits")
    record("fig2f_chromosome_splits", pd.DataFrame([
        dict(panel="2f", held_out_set=lab, fold=fd["fold"], seed=42, n_test_cpgs=fd["n"],
             sequence_beta_mae=fd["sequence"], fusion_beta_mae=fd["fusion"],
             fusion_minus_sequence=fd["fusion"] - fd["sequence"],
             uncertainty="single seed-42 fit per split")
        for fd, lab in zip(folds, labels)]))

    gain = repo.acsv("fusion_gain_stratified/fusion_gain_stratified.csv")
    marks = [("Ref_ATAC_Signal_Stratum", "ATAC"), ("Ref_H3K4me3_Signal_Stratum", "H3K4me3"),
             ("Ref_H3K27ac_Signal_Stratum", "H3K27ac"), ("Ref_H3K4me1_Signal_Stratum", "H3K4me1"),
             ("Ref_H3K36me3_Signal_Stratum", "H3K36me3"), ("Ref_H3K9me3_Signal_Stratum", "H3K9me3"),
             ("Ref_H3K27me3_Signal_Stratum", "H3K27me3")]
    marks = [m for m in marks if m[0] in set(gain["Grouping"])]
    quarts = ["Q1 low", "Q2", "Q3", "Q4 high"]
    mat = np.full((len(marks), 4), np.nan)
    g_src = []
    for i, (grouping, short) in enumerate(marks):
        sub = gain[gain["Grouping"] == grouping].set_index("Stratum")
        for j, q in enumerate(quarts):
            if q in sub.index:
                r = sub.loc[q]
                mat[i, j] = 100 * float(r["Relative_Beta_MAE_Reduction"])
                g_src.append(dict(panel="2g", grouping=grouping, mark=short, stratum=q,
                                  n_cpgs=int(r["N_CpGs"]), n_genomic_blocks=int(r["N_Genomic_Blocks"]),
                                  relative_reduction_pct=mat[i, j],
                                  ci_low_pct=100 * float(r["Relative_Reduction_CI_Low"]),
                                  ci_high_pct=100 * float(r["Relative_Reduction_CI_High"]),
                                  uncertainty="2,000 draws over 1 Mb blocks"))
    sub_f = gs[2, 2:6].subgridspec(2, 1, height_ratios=[7.0, 1.25], hspace=0.22)
    ax_f = fig.add_subplot(sub_f[0])
    vmin, vmax = 8.0, 26.0
    im = heatmap(ax_f, mat, [m[1] for m in marks], quarts, vmin, vmax, cmap=HEAT_CMAP)
    ax_f.xaxis.set_ticks_position("top")
    ax_f.xaxis.set_label_position("top")
    panel_letter(ax_f, "g", xoff=-56, yoff=20)
    ax_f.set_title(r"Relative $\beta$ MAE reduction (%)", fontsize=9, loc="left",
                   pad=16, color=TEXT)

    cgi = gain[gain["Grouping"] == "CpG_Island_Context"]
    order = ["Island", "Shore", "Shelf", "Open sea"]
    cmat = np.full((1, len(order)), np.nan)
    for j, lab in enumerate(order):
        r = cgi[cgi["Stratum"] == lab]
        if len(r):
            r = r.iloc[0]
            cmat[0, j] = 100 * float(r["Relative_Beta_MAE_Reduction"])
            g_src.append(dict(panel="2g", grouping="CpG_Island_Context", mark="CpG class",
                              stratum=lab, n_cpgs=int(r["N_CpGs"]),
                              n_genomic_blocks=int(r["N_Genomic_Blocks"]),
                              relative_reduction_pct=cmat[0, j],
                              ci_low_pct=100 * float(r["Relative_Reduction_CI_Low"]),
                              ci_high_pct=100 * float(r["Relative_Reduction_CI_High"]),
                              uncertainty="2,000 draws over 1 Mb blocks"))
    ax_f2 = fig.add_subplot(sub_f[1])
    heatmap(ax_f2, cmat, ["CpG class"], order, vmin, vmax, cmap=HEAT_CMAP)
    cb = fig.colorbar(im, ax=[ax_f, ax_f2], fraction=0.030, pad=0.015, aspect=22)
    cb.ax.set_title("%", fontsize=5.4, pad=2)
    cb.ax.tick_params(labelsize=5.0)
    cb.outline.set_visible(False)
    record("fig2g_relative_context_benefit", pd.DataFrame(g_src))

    ax_g = fig.add_subplot(gs[1, 4:6])
    dec = fig2_transfer_panel(repo)
    x = dec["decile"].astype(int).values
    for key, model in [("mae_sequence", "sequence"), ("mae_fusion", "fusion"), ("mae_epi", "epi")]:
        if key not in dec.columns:
            continue
        vals = dec[key].astype(float).values
        ci = np.array([list(v) for v in dec[f"{key}_95ci"]], dtype=float)
        colour, marker = MODEL_STYLE[model]
        ax_g.plot(x, vals, "-", color=colour, lw=1.4, zorder=3)
        ax_g.plot(x, vals, marker, ms=3.2, color=colour, zorder=4, label=MODEL_LABEL[model])
        ax_g.fill_between(x, ci[:, 0], ci[:, 1], color=colour, alpha=0.16, lw=0, zorder=2)
    ax_g.set_xticks(x)
    ax_g.set_xlabel("cross-tissue variance decile", labelpad=1.5)
    ax_g.set_ylabel(r"$\beta$ MAE", labelpad=1.5)
    ax_g.set_ylim(0, 0.235)
    ax_g.grid(axis="y", color="#E9EDF0", lw=0.5, zorder=0)
    ax_g.set_axisbelow(True)
    ax_g.legend(loc="upper left", frameon=False, fontsize=5.8, handletextpad=0.4,
                borderpad=0.0, labelspacing=0.15)
    panel_letter(ax_g, "e", xoff=-46)
    panel_title(ax_g, "Transfer error")
    record("fig2e_transfer_error", dec.assign(panel="2e", n_probes=dec["n"],
                                              uncertainty="400 draws over 1 Mb blocks"))


    fig.savefig(out / "fig2_prediction_transfer.pdf", metadata={"CreationDate": None}, dpi=450)
    plt.close(fig)


PAIR_SEEDS = (42, 43, 44)


def _pair_scores(repo: Repo, cohort: str) -> pd.DataFrame:
    # seed-mean predicted delta M per Pair_UID, as in 30_transfer_synthesis.seed_ensemble,
    # so the figure shows the same three-seed estimate the text reports
    cols = ["Pair_UID", "beta_ref_to_alt", "Predicted_Delta_M", "pvalue",
            "creates_cpg", "destroys_cpg", "cpg_chr", "cpg_pos_hg38"]
    frames = [pd.read_csv(repo.abl / f"{cohort}_variant_scoring" / "heldout" / "fusion" /
                          f"seed{s}" / "pair_scores.csv", usecols=cols) for s in PAIR_SEEDS]
    mean = (pd.concat(frames).groupby("Pair_UID", sort=False)["Predicted_Delta_M"]
            .mean().rename("Predicted_Delta_M"))
    ps = (frames[0].drop(columns="Predicted_Delta_M")
          .merge(mean, left_on="Pair_UID", right_index=True, how="inner",
                 validate="one_to_one"))
    altering = ps["creates_cpg"].astype(bool) | ps["destroys_cpg"].astype(bool)
    return ps[~altering].dropna(subset=["beta_ref_to_alt", "Predicted_Delta_M", "pvalue"])


def _block_bootstrap_agree(frame: pd.DataFrame, n_boot: int = 2000, seed: int = 42,
                           block_bp: int = 1_000_000) -> np.ndarray:
    blocks = (frame["cpg_chr"].astype(str) + ":"
              + (frame["cpg_pos_hg38"] // block_bp).astype(int).astype(str))
    hit = (np.sign(frame["beta_ref_to_alt"]) == np.sign(frame["Predicted_Delta_M"]))
    per = hit.groupby(blocks.values).agg(["sum", "count"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(per), (n_boot, len(per)))
    return per[pick, 0].sum(axis=1) / per[pick, 1].sum(axis=1)


def fig3(repo: Repo, out: Path) -> None:
    from scipy.stats import rankdata, spearmanr

    fig = plt.figure(figsize=(180 * MM, 158 * MM))
    gs = fig.add_gridspec(3, 6, left=0.105, right=0.965, top=0.925, bottom=0.095,
                          wspace=1.6, hspace=0.72, height_ratios=[1.05, 1.0, 1.0])

    AGREE, DISAGREE, QUIET = "#2B7BBA", "#D6604D", "#DDE2E6"
    THRESH = {"genoa": 5e-8, "egtex": 1.483e-5}

    ax_a = fig.add_subplot(gs[0, 0:4])
    g = _pair_scores(repo, "genoa")
    x = g["beta_ref_to_alt"].to_numpy(dtype=float)
    p = np.clip(g["pvalue"].to_numpy(dtype=float), 1e-300, None)
    y = -np.log10(p)
    sig = p < THRESH["genoa"]
    pred_a = g["Predicted_Delta_M"].to_numpy(dtype=float)
    agree = np.sign(x) == np.sign(pred_a)
    # every pair is coloured by direction; colour, opacity and size grow with the
    # percentile of |predicted delta M|, so the model's confident calls stand out and
    # its near-zero ones fade to grey whatever their p-value
    conf = (rankdata(np.abs(pred_a)) - 0.5) / len(pred_a)
    w = np.clip((conf - 0.5) / 0.45, 0.0, 1.0) ** 1.3
    rgba = np.where(agree[:, None], np.array(to_rgba(AGREE)), np.array(to_rgba(DISAGREE)))
    rgba = rgba * w[:, None] + np.array(to_rgba(QUIET)) * (1.0 - w[:, None])
    rgba[:, 3] = 0.35 + 0.55 * w
    cap = 60.0
    order = np.argsort(conf)
    thr_y = -np.log10(THRESH["genoa"])
    ax_a.axhline(thr_y, color=TEXT, ls=(0, (3, 2)), lw=0.7, zorder=6)
    ax_a.annotate(r"$p = 5\times10^{-8}$", (-2.55, thr_y), textcoords="offset points",
                  xytext=(0, 3), ha="left", va="bottom", fontsize=5.0, color=TEXT, zorder=7)
    ax_a.axvline(0, color=TEXT, ls=(0, (3, 2)), lw=0.7, zorder=3)
    ax_a.scatter(x[order], np.minimum(y, cap * 1.025)[order], c=rgba[order],
                 s=(0.8 + 3.2 * w)[order], lw=0, zorder=4, rasterized=True)
    ax_a.set_xlabel("measured effect (study units)", labelpad=1.5)
    ax_a.set_ylabel(r"$-\log_{10}\,p$", labelpad=1.5)
    ax_a.set_ylim(0, cap * 1.06)
    ax_a.set_xlim(-2.6, 2.6)
    ax_a.grid(color="#EDF0F2", lw=0.5, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=3.2, color=AGREE, label="direction correct"),
        Line2D([], [], marker="o", ls="none", ms=3.2, color=DISAGREE, label="direction wrong"),
        Line2D([], [], marker="o", ls="none", ms=1.6, color=QUIET,
               label=r"small $|\Delta\widehat{M}|$ (faded)"),
    ], loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.92,
        fontsize=5.0, handletextpad=0.3, borderpad=0.3, labelspacing=0.18).set_zorder(8)
    panel_letter(ax_a, "a", xoff=-46)
    panel_title(ax_a, "GENOA associations")
    top = sig & (conf > 0.9)
    top_draws = _block_bootstrap_agree(g[top])
    record("fig3a_volcano", pd.DataFrame(dict(
        panel="3a", cohort="GENOA", seed="mean 42-44", measured_effect=x, neglog10_p=y,
        predicted_delta_m=pred_a, abs_pred_percentile=conf, significant=sig,
        predicted_sign_agrees=agree)).assign(
        threshold=THRESH["genoa"],
        y_axis_cap=cap,
        sig_top_decile_n=int(top.sum()),
        sig_top_decile_agreement=float(agree[top].mean()),
        sig_top_decile_ci_low=float(np.quantile(top_draws, 0.025)),
        sig_top_decile_ci_high=float(np.quantile(top_draws, 0.975)),
        note="non-CpG-altering held-out pairs; p floored at 1e-300 and points above the "
             "display cap of 60 (583) drawn at the cap; colour weight rises from 0 at the "
             "median |predicted delta M| to 1 at its 95th percentile; the top-decile "
             "agreement is among significant pairs above the 90th |predicted delta M| "
             "percentile, with 2,000 resamples of 1 Mb genomic blocks"))

    syn = repo.acsv("variant_effect_synthesis/all_strata.csv")
    syn = syn[(syn["stratum"] == "significant") & (syn["model"] == "fusion")]

    def synth(cohort_key, metric):
        r = syn[(syn["cohort"] == cohort_key) & (syn["metric"] == metric)].iloc[0]
        return float(r["value"]), float(r["ci_low"]), float(r["ci_high"])

    cohort_spec = [("genoa", "GENOA blood", FUSION, "b", (0, slice(4, 6))),
                   ("egtex", "eGTEx breast", CONTEXT, "c", (1, slice(0, 2)))]
    rank_src = []
    for cohort, tag, colour, letter, slot in cohort_spec:
        ax = fig.add_subplot(gs[slot[0], slot[1]])
        d = _pair_scores(repo, cohort)
        keep = d[d["pvalue"] < THRESH[cohort]].copy()
        obs = keep["beta_ref_to_alt"].to_numpy(dtype=float)
        pred = keep["Predicted_Delta_M"].to_numpy(dtype=float)
        u = (rankdata(obs) - 0.5) / len(obs)
        v = (rankdata(pred) - 0.5) / len(pred)
        rho = float(spearmanr(obs, pred).statistic)
        # the band is the published interval itself (scripts/30, 500 block resamples);
        # re-bootstrapping here would draw a different interval from the one in the text
        pub_rho, lo, hi = synth("GENOA" if cohort == "genoa" else "eGTEx", "signed_rho")
        if abs(pub_rho - rho) > 1e-6:
            raise SystemExit(f"fig3{letter}: seed-mean rho {rho:.6f} != published {pub_rho:.6f}")

        ax.plot(u, v, "o", ms=1.9 if len(u) > 1500 else 3.0, color=colour,
                alpha=0.08 if len(u) > 1500 else 0.18, mec="none", zorder=2, rasterized=True)
        xs = np.array([0.0, 1.0])
        band_lo = 0.5 + lo * (xs - 0.5)
        band_hi = 0.5 + hi * (xs - 0.5)
        ax.fill_between(xs, band_lo, band_hi, color=colour, alpha=0.38, lw=0, zorder=3)
        for bound in (band_lo, band_hi):
            ax.plot(xs, bound, ls=(0, (2.5, 1.8)), color=colour, lw=1.2, zorder=4)
        ax.plot(xs, 0.5 + rho * (xs - 0.5), "-", color=TEXT, lw=0.9, zorder=6)
        ax.plot([0, 1], [0.5, 0.5], color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("measured effect rank", labelpad=1.5)
        ax.set_ylabel(r"predicted $\Delta\widehat{M}$ rank", labelpad=1.5)
        ax.grid(color="#EDF0F2", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.text(0.03, 0.97, f"$\\rho$ = {rho:.3f}", transform=ax.transAxes, ha="left",
                va="top", fontsize=6.0, color=TEXT,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        panel_letter(ax, letter, xoff=-52 if letter == "b" else -44)
        panel_title(ax, f"{tag.split()[0]}: rank slope")
        rank_src.append(pd.DataFrame(dict(
            panel=f"3{letter}", cohort=tag, seed="mean 42-44", measured_effect=obs,
            predicted_delta_m=pred, percentile_rank_measured=u,
            percentile_rank_predicted=v)).assign(
            spearman_rho=rho, ci_low=lo, ci_high=hi, n_boot=500,
            note="percentile ranks of both quantities; predicted delta M is the mean over "
                 "seeds 42-44; because both axes are uniform on [0,1] the least-squares slope "
                 "of the bold line equals Spearman's rho, and the band is bounded by the lines "
                 "of slope ci_low and ci_high, the published interval from 500 resamples of "
                 "1 Mb genomic blocks (variant_effect_synthesis/all_strata.csv)"))
    record("fig3bc_rank_agreement", pd.concat(rank_src, ignore_index=True))

    ax_d = fig.add_subplot(gs[1, 2:4])
    prim = repo.acsv("genoa_variant_evaluation/primary_metrics.csv")
    matched = repo.acsv("genoa_variant_evaluation/matched_negative_auroc.csv")
    bal = repo.acsv("genoa_variant_evaluation/matching_balance.csv")
    base_prim = repo.csv("baseline_variant_evaluation/primary_metrics.csv")
    base_matched = repo.csv("baseline_variant_evaluation/matched_negative_auroc.csv")

    def ensemble(df, **eq):
        sub = df.copy()
        for k, v in eq.items():
            sub = sub[sub[k] == v]
        sub = sub[sub["seed"] == -1]
        return sub.iloc[0] if len(sub) else None

    d_src = []
    spec = [("fusion", prim, matched), ("sequence", prim, matched),
            ("kmer_ridge", base_prim, base_matched),
            ("composition", base_prim, base_matched)]
    for model, pdf_, mdf in spec:
        um = ensemble(pdf_, model=model, variant_class="non_cpg_altering", metric="auroc_marginal")
        mt = ensemble(mdf, model=model, metric="auroc_distance_matched")
        if um is None or mt is None:
            continue
        colour, marker = MODEL_STYLE[model]
        ax_d.plot([0, 1], [float(um["value"]), float(mt["value"])], "-",
                  color=colour, lw=1.5, zorder=3)
        ax_d.plot([0], [float(um["value"])], "o", ms=3.8, color=colour, mec="none", zorder=4)
        ax_d.plot([1], [float(mt["value"])], marker, ms=4.4 if marker in "PX" else 3.8,
                  color=colour, mec="none", zorder=4)
        for kind, r in [("unmatched", um), ("distance-matched", mt)]:
            d_src.append(dict(panel="3d", cohort="GENOA", model=model, comparison=kind,
                              seed="ensemble (seed -1)", n=int(r["n"]), n_blocks=int(r["n_blocks"]),
                              value=float(r["value"]), ci_low=float(r["ci_low"]),
                              ci_high=float(r["ci_high"]),
                              source=("ABL/genoa_variant_evaluation" if model in ("fusion", "sequence")
                                      else "baseline_variant_evaluation"),
                              uncertainty="2,000 draws over 1 Mb blocks; intervals in Source Data"))
    dist_um = ensemble(prim, model="fusion", variant_class="non_cpg_altering",
                       metric="auroc_distance_only_baseline")
    dist_mt = float(bal["distance_only_auroc_after_matching"].iloc[0])
    ax_d.plot([0, 1], [float(dist_um["value"]), dist_mt], "-", color=DISTANCE,
              lw=1.5, zorder=3)
    ax_d.plot([0], [float(dist_um["value"])], "o", ms=3.8, color=DISTANCE, mec="none", zorder=3.5)
    ax_d.plot([1], [dist_mt], "o", ms=3.8, color=DISTANCE, mec="none", zorder=3.5)
    d_src.append(dict(panel="3d", cohort="GENOA", model="distance_only", comparison="unmatched",
                      seed="ensemble (seed -1)", n=int(dist_um["n"]), n_blocks=int(dist_um["n_blocks"]),
                      value=float(dist_um["value"]), ci_low=float(dist_um["ci_low"]),
                      ci_high=float(dist_um["ci_high"]),
                      source="ABL/genoa_variant_evaluation/primary_metrics.csv",
                      uncertainty="2,000 draws over 1 Mb blocks"))
    d_src.append(dict(panel="3d", cohort="GENOA", model="distance_only",
                      comparison="realized after matching", seed="ensemble (seed -1)",
                      n=int(bal["matched_null_n"].iloc[0]), n_blocks=None, value=dist_mt,
                      ci_low=None, ci_high=None,
                      source="ABL/genoa_variant_evaluation/matching_balance.csv",
                      uncertainty="realized balance statistic"))
    ax_d.axhline(0.5, color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["unmatched", "matched"], fontsize=6.4)
    ax_d.set_xlim(-0.30, 1.30)
    ax_d.set_ylim(0.445, 0.690)
    ax_d.set_yticks([0.45, 0.50, 0.55, 0.60])
    ax_d.set_ylabel("AUROC", labelpad=1.5)
    ax_d.grid(axis="y", color="#EDF0F2", lw=0.5, zorder=0)
    ax_d.set_axisbelow(True)
    ax_d.legend(handles=[
        Line2D([], [], color=FUSION, lw=1.5, marker="D", ms=4.0, label="Fusion"),
        Line2D([], [], color=SEQUENCE, lw=1.5, marker="o", ms=4.0, label="Sequence"),
        Line2D([], [], color=KMER, lw=1.5, marker="P", ms=4.0, label="$k$-mer ridge"),
        Line2D([], [], color=COMPOSITION, lw=1.5, marker="X", ms=4.0, label="Composition"),
        Line2D([], [], color=DISTANCE, lw=1.5, marker="o", ms=4.0, label="Distance only"),
    ], loc="upper center", ncol=2, frameon=False, fontsize=4.8, handlelength=1.5,
        handletextpad=0.35, borderpad=0.0, labelspacing=0.22, columnspacing=0.9)
    panel_letter(ax_d, "d", xoff=-40)
    panel_title(ax_d, "Distance controls")
    record("fig3d_distance_control", pd.DataFrame(d_src))

    ax_e = fig.add_subplot(gs[1, 4:6])
    db = repo.acsv("genoa_variant_evaluation/distance_bins.csv")
    e_src, bins_lab = [], []
    for model in ("fusion", "sequence"):
        sub = db[(db["model"] == model) & (db["metric"] == "auroc")]
        if not len(sub):
            continue
        xs = np.arange(len(sub))
        colour, marker = MODEL_STYLE[model]
        ax_e.plot(xs, sub["value"].astype(float), marker=marker, ms=3.0, lw=1.4,
                  color=colour, zorder=3, label=MODEL_LABEL[model])
        ax_e.fill_between(xs, sub["ci_low"].astype(float), sub["ci_high"].astype(float),
                          color=colour, alpha=0.15, lw=0, zorder=2)
        e_src.append(sub.assign(panel="3e"))
        bins_lab = sub["distance_bin"].astype(str).tolist()
    ax_e.axhline(0.5, color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
    ax_e.set_xticks(np.arange(len(bins_lab)))
    ax_e.set_xticklabels(bins_lab, fontsize=5.2, rotation=30, ha="right")
    ax_e.set_xlabel("variant-CpG distance (bp)", labelpad=1.5)
    ax_e.set_ylabel("AUROC", labelpad=1.5)
    ax_e.grid(axis="y", color="#EDF0F2", lw=0.5, zorder=0)
    ax_e.set_axisbelow(True)
    ax_e.legend(loc="lower left", frameon=False, fontsize=5.4, handletextpad=0.4,
                borderpad=0.0, labelspacing=0.15)
    panel_letter(ax_e, "e", xoff=-42)
    panel_title(ax_e, "AUROC by distance")
    if e_src:
        record("fig3e_distance_bins", pd.concat(e_src, ignore_index=True))

    sub_f = gs[2, 0:3].subgridspec(2, 1, hspace=0.25)
    ax_f1 = fig.add_subplot(sub_f[0])
    ax_f2 = fig.add_subplot(sub_f[1])
    grad = repo.acsv("genoa_variant_evaluation/significance_gradient.csv")
    grad = grad[grad["model"] == "fusion"]
    strata = list(dict.fromkeys(grad["stratum"].tolist()))
    f_src = []
    for ax, metric, ylabel, null in [(ax_f1, "signed_rho", r"signed $\rho$", 0.0),
                                     (ax_f2, "direction_agreement", "dir. agr.", 0.5)]:
        xs, vals, los, his = [], [], [], []
        for i, st in enumerate(strata):
            sub = grad[(grad["stratum"] == st) & (grad["metric"] == metric)]
            if not len(sub):
                continue
            r = sub.iloc[0]
            xs.append(i); vals.append(float(r["value"]))
            los.append(float(r["ci_low"])); his.append(float(r["ci_high"]))
            f_src.append(dict(panel="3f", model="fusion", stratum=st, metric=metric,
                              n_pairs=int(r["n"]), n_blocks=int(r["n_blocks"]),
                              value=float(r["value"]), ci_low=float(r["ci_low"]),
                              ci_high=float(r["ci_high"]),
                              uncertainty="1 Mb block bootstrap"))
        ax.fill_between(xs, los, his, color=FUSION, alpha=0.16, lw=0, zorder=2)
        ax.plot(xs, vals, "-", color=FUSION, lw=1.4, zorder=3)
        ax.plot(xs, vals, "D", ms=3.0, color=FUSION, zorder=4)
        ax.axhline(null, color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
        ax.set_xticks(range(len(strata)))
        ax.set_ylabel(ylabel, fontsize=6.2, labelpad=1.0)
        ax.grid(axis="y", color="#EDF0F2", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.4, len(strata) - 0.6)
        ax.tick_params(axis="y", labelsize=5.8)
        if ax is ax_f1:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(strata, fontsize=5.0, rotation=38, ha="right")
    panel_letter(ax_f1, "f", xoff=-40, yoff=13)
    panel_title(ax_f1, "Agreement by association strength")
    ax_f2.set_xlabel("GENOA p-value stratum", fontsize=6.2, labelpad=1.5)
    record("fig3f_association_strength", pd.DataFrame(f_src))

    ax_g = fig.add_subplot(gs[2, 3:6])
    ros = repo.csv("asm_validation/asm_discrimination.csv")
    tyc1 = repo.csv("asm_validation_tycko/tycko_e1_discrimination.csv")
    tyc2 = repo.csv("asm_validation_tycko/tycko_e2_signed_agreement.csv")
    ros_sub = ros[ros["Stratum"] == "positive_vs_bimodal_non_asm | all tissues"]
    tyc_sub = tyc1[tyc1["Stratum"] == "all tissues"]
    tyc2_sub = tyc2[tyc2["Stratum"] == "all tissues"]

    endpoint_rows = [
        ("Rosenski\ndiscrimination", 0.5, {}),
        ("Do-Tycko\ndiscrimination", 0.5, {}),
        ("Do-Tycko\ndirectional AUROC", 0.5, {}),
        ("Do-Tycko direction\nconcordance", 0.5, {}),
        ("Do-Tycko\nsigned Spearman", 0.0, {}),
    ]
    arms = ["fusion", "sequence", "distance_only_baseline"]
    g_src = []
    for model in arms:
        r = ros_sub[ros_sub["Score"] == model]
        if len(r):
            endpoint_rows[0][2][model] = (float(r.iloc[0]["auroc"]), int(r.iloc[0]["N_Pairs"]))
        r = tyc_sub[tyc_sub["Score"] == model]
        if len(r):
            endpoint_rows[1][2][model] = (float(r.iloc[0]["AUROC_Detection"]), int(r.iloc[0]["N_Pairs"]))
    for _, r in tyc2_sub.iterrows():
        endpoint_rows[2][2][r["Arm"]] = (float(r["AUROC_Directional"]), int(r["N_SNPs"]))
        endpoint_rows[3][2][r["Arm"]] = (float(r["Direction_Concordance"]), int(r["N_SNPs"]))
        endpoint_rows[4][2][r["Arm"]] = (float(r["Signed_Spearman"]), int(r["N_SNPs"]))

    mat = np.full((len(endpoint_rows), len(arms)), np.nan)
    ann = np.empty((len(endpoint_rows), len(arms)), dtype=object)
    for i_, (label, null, vals) in enumerate(endpoint_rows):
        for j_, model in enumerate(arms):
            if model not in vals:
                ann[i_, j_] = ""
                continue
            v, n_ = vals[model]
            mat[i_, j_] = (v - null) / (1.0 - null)
            ann[i_, j_] = f"{v:.3f}"
            g_src.append(dict(panel="3g", endpoint=label.replace("\n", " "), score=model,
                              value=v, null=null, fraction_of_null_to_perfect=mat[i_, j_],
                              n=n_, seed="42",
                              note="colour is the distance travelled from the endpoint's own null "
                                   "towards a perfect score; the printed number is the raw value",
                              uncertainty="2,000 block resamples; intervals in Supplementary Table S2"))
    ax_g.set_facecolor("#F2F4F6")
    im = ax_g.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=0.30, aspect="auto")
    for i_ in range(mat.shape[0]):
        for j_ in range(mat.shape[1]):
            if ann[i_, j_]:
                ax_g.text(j_, i_, ann[i_, j_], ha="center", va="center", fontsize=5.4,
                          color="white" if mat[i_, j_] > 0.17 else TEXT)
    ax_g.set_xticks(range(len(arms)))
    ax_g.set_xticklabels([MODEL_LABEL[a_] for a_ in arms], fontsize=5.6)
    ax_g.set_yticks(range(len(endpoint_rows)))
    ax_g.set_yticklabels([e[0] for e in endpoint_rows], fontsize=5.2)
    ax_g.set_xticks(np.arange(-0.5, len(arms), 1), minor=True)
    ax_g.set_yticks(np.arange(-0.5, len(endpoint_rows), 1), minor=True)
    ax_g.grid(which="minor", color="white", lw=1.0)
    ax_g.tick_params(which="both", length=0)
    for sp in ax_g.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax_g, fraction=0.042, pad=0.02)
    cb.set_label("null $\\rightarrow$ perfect", fontsize=4.6, labelpad=2)
    cb.ax.tick_params(labelsize=4.4)
    cb.outline.set_visible(False)
    panel_letter(ax_g, "g", xoff=-62)
    panel_title(ax_g, "ASM scorecard")
    record("fig3g_asm", pd.DataFrame(g_src))

    fig.savefig(out / "fig3_external_validation.pdf", metadata={"CreationDate": None}, dpi=450)
    plt.close(fig)


def icon_seq(ax, x, y, alt=False, s=0.065):
    # four base blocks; the ALT allele recolours one of them
    cols = ["#9FB3C8", "#C3CFDB", "#9FB3C8", "#C3CFDB"]
    if alt:
        cols[2] = HIGHLIGHT
    for k, c in enumerate(cols):
        ax.add_patch(Rectangle((x + k * 1.25 * s, y - s), s, 2 * s, facecolor=c,
                               edgecolor="none", zorder=4))


def icon_ctx(ax, x, y, s=0.065):
    # three stacked context tracks of uneven length
    for k, frac in enumerate((1.0, 0.65, 0.85)):
        ax.add_patch(Rectangle((x, y + 0.75 * s * (1 - k) - 0.22 * s), 4.75 * s * frac,
                               0.44 * s, facecolor=CONTEXT, edgecolor="none", alpha=0.85,
                               zorder=4))


def icon_gate(ax, x, y, on=False, s=0.065, aspect=1.0):
    # toggle switch: knob left and grey for REF, right and orange for ALT;
    # aspect is the y-units-per-x-unit ratio that makes the knob round
    from matplotlib.patches import Ellipse, FancyBboxPatch
    w = 4.75 * s
    ax.add_patch(FancyBboxPatch((x, y - 0.55 * s * aspect), w, 1.1 * s * aspect,
                                boxstyle=f"round,pad=0,rounding_size={0.55 * s}",
                                mutation_aspect=aspect,
                                facecolor=to_rgba(HIGHLIGHT, 0.25) if on else "#E3E8ED",
                                edgecolor="none", zorder=4))
    cx = x + w - 0.8 * s if on else x + 0.8 * s
    ax.add_patch(Ellipse((cx, y), 1.45 * s, 1.45 * s * aspect,
                         facecolor=HIGHLIGHT if on else "#8A97A5", edgecolor="white",
                         lw=0.4, zorder=5))


def fig4(repo: Repo, out: Path) -> None:
    from scipy.stats import spearmanr

    fig = plt.figure(figsize=(180 * MM, 110 * MM))

    ax_a = fig.add_axes([0.030, 0.565, 0.955, 0.395])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 3.0)
    ax_a.axis("off")
    # y-units per x-unit of equal physical length, so round icons stay round
    asp = (3.0 / (0.395 * 110)) / (10 / (0.955 * 180))
    # rows per card: (kind, label, changed from REF)
    steps = [
        ("1. REF baseline", [("seq", "REF sequence", False), ("ctx", "REF context", False),
                             ("gate", "REF gates", False)]),
        ("2. ALT sequence", [("seq", "ALT sequence", True), ("ctx", "REF context", False),
                             ("gate", "REF gates", False)]),
        ("3. + sequence gate", [("seq", "ALT sequence", True), ("ctx", "REF context", False),
                                ("gate", "ALT seq. gate", True)]),
        ("4. + context gate", [("seq", "ALT sequence", True), ("ctx", "REF context", False),
                               ("gate", "both ALT gates", True)]),
    ]
    w, gap = 1.82, 0.88
    y0, h = 0.92, 1.42
    for i, (title, rows) in enumerate(steps):
        x = i * (w + gap)
        ax_a.add_patch(Rectangle((x, y0), w, h, facecolor="white", edgecolor=FUSION, lw=0.8,
                                 zorder=2))
        ax_a.text(x + w / 2, y0 + h - 0.13, title, ha="center", va="top", fontsize=6.6,
                  fontweight="bold", color=TEXT, zorder=3)
        for r, (kind, label, changed) in enumerate(rows):
            yy = y0 + h - 0.60 - 0.30 * r
            ix = x + 0.14
            if kind == "seq":
                icon_seq(ax_a, ix, yy, alt=changed)
            elif kind == "ctx":
                icon_ctx(ax_a, ix, yy)
            else:
                icon_gate(ax_a, ix, yy, on=changed, aspect=asp)
            ax_a.text(x + 0.55, yy, label, ha="left", va="center", fontsize=6.0,
                      color=HIGHLIGHT if changed else TEXT,
                      fontweight="bold" if changed else "normal", zorder=4)
        if i < 3:
            xm = x + w + gap / 2
            ya = y0 + h / 2
            ax_a.annotate("", xy=(x + w + gap - 0.12, ya), xytext=(x + w + 0.12, ya),
                          arrowprops=dict(arrowstyle="-|>", lw=0.9, color=TEXT,
                                          mutation_scale=7, shrinkA=0, shrinkB=0))
            # what changes in this step: icon above the arrow, label below it
            if i == 0:
                icon_seq(ax_a, xm - 0.18, ya + 0.22, alt=True)
            else:
                icon_gate(ax_a, xm - 0.155, ya + 0.22, on=True, aspect=asp)
            lab = [r"$\Delta$sequence", "$\\Delta$sequence\ngate", "$\\Delta$context\ngate"][i]
            ax_a.text(xm, ya - 0.11, lab, ha="center", va="top", fontsize=5.6, color=TEXT,
                      linespacing=1.15)
    xl, xr = w / 2, 3 * (w + gap) + w / 2
    yb = y0 - 0.24
    ax_a.plot([xl, xr], [yb, yb], color=NULLC, lw=0.8, zorder=1)
    for xx in (xl, xr):
        ax_a.plot([xx, xx], [yb, y0 - 0.07], color=NULLC, lw=0.8, zorder=1)
    ax_a.text((xl + xr) / 2, yb, r"step contributions sum to $\Delta$total",
              ha="center", va="center", fontsize=6.2, color=TEXT, zorder=2,
              bbox=dict(facecolor="white", edgecolor="none", pad=2.0))
    ax_a.text(0.0, 2.92, "A", fontsize=10, fontweight="bold", va="top", ha="left", color=TEXT)
    ax_a.text(0.42, 2.92, "Counterfactual decomposition",
              fontsize=9, va="top", ha="left", color=TEXT)

    gs = fig.add_gridspec(1, 3, left=0.105, right=0.975, top=0.495, bottom=0.125,
                          width_ratios=[1.70, 1.0, 1.0], wspace=0.38)

    perm = pd.read_csv(repo.abl / "context_permutation" / "pair_scores_identity.csv",
                       usecols=["WT_M_RC_Avg", "Predicted_Delta_M", "WT_Gate_Avg_DNA_Share"])
    ax_b = fig.add_subplot(gs[0, 0])
    pos_b = ax_b.get_position()
    ax_b.set_position([pos_b.x0, pos_b.y0, pos_b.width * 0.90, pos_b.height])
    xb = perm["WT_M_RC_Avg"].to_numpy(dtype=float)
    yb = perm["WT_Gate_Avg_DNA_Share"].to_numpy(dtype=float)
    hb = hexpanel(ax_b, xb, yb, (xb.min(), xb.max(), yb.min(), yb.max()), gridsize=48,
                  cmap=SPECTRUM_CMAP, square=False)
    # equal-width bins put the markers at even intervals; equal-count bins put them
    # wherever the density happened to be. A 3-point mean smooths the remaining wobble.
    edges_b = np.linspace(*np.quantile(xb, [0.002, 0.998]), 17)
    cen_b = 0.5 * (edges_b[:-1] + edges_b[1:])
    idx_b = np.digitize(xb, edges_b) - 1
    counts_b = np.array([int((idx_b == k).sum()) for k in range(len(cen_b))])
    med_b = np.array([np.median(yb[idx_b == k]) if counts_b[k] else np.nan
                      for k in range(len(cen_b))])
    keep_b = counts_b >= 100
    x_tr = cen_b[keep_b]
    y_tr = pd.Series(med_b[keep_b]).rolling(3, center=True, min_periods=1).mean().to_numpy()
    ax_b.plot(x_tr, y_tr, "-", color=HIGHLIGHT, lw=1.8, zorder=4)
    ax_b.plot(x_tr, y_tr, "o", ms=3.0, color=HIGHLIGHT, mec="white", mew=0.5, zorder=5)
    rho_gate = float(spearmanr(perm["WT_M_RC_Avg"], perm["WT_Gate_Avg_DNA_Share"]).statistic)
    ax_b.axhline(0.5, color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
    ax_b.set_xlabel(r"reference $\widehat{M}$", labelpad=1.5)
    ax_b.set_ylabel("gate share to\nsequence", labelpad=0.5)
    ax_b.text(0.03, 0.97, f"$\\rho$ = {rho_gate:.3f}, $n$ = {len(perm):,}",
              transform=ax_b.transAxes, ha="left", va="top", fontsize=5.4, color=TEXT,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    cb = fig.colorbar(hb, cax=ax_b.inset_axes([1.020, 0.0, 0.022, 1.0]))
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=4.6)
    cb.outline.set_visible(False)
    ax_b.grid(color="#EDF0F2", lw=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    panel_letter(ax_b, "b", xoff=-40)
    panel_title(ax_b, "Gate share vs level")
    record("fig4b_gate_allocation", pd.DataFrame(dict(
        panel="4b", cohort="eGTEx", seed=42,
        reference_m=perm["WT_M_RC_Avg"], dna_gate_share=perm["WT_Gate_Avg_DNA_Share"])).assign(
        spearman_rho=rho_gate,
        note="per-pair gate allocation from context_permutation/pair_scores_identity.csv; "
             "the orange line is a 3-point running mean of the median share within "
             "sixteen equally spaced bins of reference M-hat that hold 100 pairs or more"))


    sub_d = gs[0, 1:3].subgridspec(1, 2, wspace=0.42)
    ident = pd.read_csv(repo.abl / "context_permutation" / "pair_scores_identity.csv",
                        usecols=["Pair_UID", "WT_M_RC_Avg", "Predicted_Delta_M"]).set_index("Pair_UID")
    shuf = pd.read_csv(repo.abl / "context_permutation" / "pair_scores_shuffle.csv",
                       usecols=["Pair_UID", "WT_M_RC_Avg", "Predicted_Delta_M"]).set_index("Pair_UID")
    joined = ident.join(shuf, lsuffix="_ident", rsuffix="_shuf", how="inner").dropna()
    d_src = []
    for k, (col, lab, lim) in enumerate([
            ("WT_M_RC_Avg", "methylation level, $\\widehat{M}$", (-8, 8)),
            ("Predicted_Delta_M", "variant effect, $\\Delta\\widehat{M}$", (-0.5, 0.5))]):
        ax = fig.add_subplot(sub_d[k])
        pos_k = ax.get_position()
        ax.set_position([pos_k.x0, pos_k.y0, pos_k.width * 0.88, pos_k.height])
        ax.set_anchor("N")  # square boxes sit at the top, so the titles align with b
        x = joined[f"{col}_ident"].to_numpy(dtype=float)
        y = joined[f"{col}_shuf"].to_numpy(dtype=float)
        cmap_d, diag_d = (COOL_CMAP, HIGHLIGHT) if k == 0 else (WARM_CMAP, FUSION)
        hb2 = hexpanel(ax, x, y, (lim[0], lim[1], lim[0], lim[1]), gridsize=38, cmap=cmap_d)
        ax.plot(lim, lim, color=diag_d, ls=(0, (3, 2)), lw=1.0, zorder=3)
        cb_d = fig.colorbar(hb2, cax=ax.inset_axes([1.030, 0.0, 0.028, 1.0]))
        cb_d.ax.minorticks_off()
        cb_d.ax.tick_params(labelsize=4.4)
        cb_d.outline.set_visible(False)
        ax.set_xlabel(f"native context", labelpad=1.5)
        if k == 0:
            ax.set_ylabel("shuffled context", labelpad=1.5)
        rho_d = float(spearmanr(x, y).statistic)
        ax.text(0.04, 0.96, f"{lab}\n$\\rho$ = {rho_d:.3f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=5.0, linespacing=1.4, color=TEXT,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        ax.grid(color="#EDF0F2", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        if k == 0:
            panel_letter(ax, "c", xoff=-40)
            panel_title(ax, "Shuffled context")
        d_src.append(pd.DataFrame(dict(panel="4c", quantity=lab.split(",")[0], native=x,
                                       shuffled=y)).assign(spearman_rho=rho_d, seed=42))
    record("fig4c_context_swap", pd.concat(d_src, ignore_index=True))


    fig.savefig(out / "fig4_context_mechanism.pdf", metadata={"CreationDate": None}, dpi=450)
    plt.close(fig)


def fig5(repo: Repo, out: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 120 * MM))
    gs = fig.add_gridspec(2, 2, left=0.095, right=0.985, top=0.905, bottom=0.095,
                          wspace=0.32, hspace=0.62)

    bg = repo.acsv("candidates/candidate_matched_background_statistics.csv")
    ncoa2 = bg[bg["Selected_Gene_Name"] == "NCOA2"].iloc[0]
    case = repo.acsv("candidates/top_candidate_case_study.csv").iloc[0]
    delta = float(case["Predicted_Delta_Beta"])
    offset = int(ncoa2["Mutation_Offset_From_CpG"])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(-60, 60)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    ax_a.plot([-58, 58], [0.79, 0.79], color=OTHER, lw=1.2, solid_capstyle="round", zorder=2)
    for tick in range(-50, 51, 25):
        ax_a.plot([tick, tick], [0.765, 0.815], color=OTHER, lw=0.7, zorder=2)
        ax_a.text(tick, 0.752, f"{tick:+d}" if tick else "0", ha="center", va="top",
                  fontsize=5.8, color=OTHER)
    ax_a.plot([0], [0.79], "o", ms=6.0, color=CONTEXT, mec="white", mew=0.8, zorder=4)
    ax_a.text(0, 0.845, "cg20699548", ha="center", va="bottom", fontsize=6.2, color=CONTEXT)
    ax_a.plot([offset], [0.79], "v", ms=5.6, color=HIGHLIGHT, zorder=4)
    ax_a.annotate(f"chr8:g.70148394G>A\n{abs(offset)} bp from CpG",
                  xy=(offset, 0.79), xytext=(offset - 6, 1.02), fontsize=6.0, color=HIGHLIGHT,
                  ha="right", va="top", linespacing=1.3,
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=HIGHLIGHT))

    # zoom to the reference bases themselves, the way a locus panel usually does
    half = 18
    ref_allele, alt_allele = str(case["Variant_UID"]).split("__")[0].split("_")[2:4]
    cpg0 = int(ncoa2["pos"])
    seq = None
    fa_path = repo.root / "data" / "hg38.fa"
    if fa_path.exists():
        from pyfaidx import Fasta
        fa = Fasta(str(fa_path), as_raw=True, sequence_always_upper=True)
        seq = str(fa[str(ncoa2["chr"])][cpg0 - half:cpg0 + half + 1])
        if seq[half:half + 2] != "CG" or seq[half + offset] != ref_allele:
            raise ValueError(f"hg38 does not match the candidate record: "
                             f"dinucleotide {seq[half:half + 2]}, "
                             f"allele {seq[half + offset]} vs {ref_allele}")
    if seq:
        ax_a.add_patch(Rectangle((-half - 0.5, 0.762), 2 * half + 1, 0.056, facecolor=TINT,
                                 edgecolor="none", zorder=1))
        span_x = 56.0
        xs_seq = np.linspace(-span_x, span_x, len(seq))
        for xa, xb in ((-half - 0.5, -span_x - 1.2), (half + 0.5, span_x + 1.2)):
            ax_a.plot([xa, xb], [0.762, 0.575], color="#CBD3DA", lw=0.6, zorder=1)
        step = xs_seq[1] - xs_seq[0]
        ax_a.add_patch(Rectangle((xs_seq[half] - step / 2, 0.478), 2 * step, 0.084,
                                 facecolor=TINT, edgecolor=CONTEXT, lw=0.6, zorder=2))
        for k, (base, xx) in enumerate(zip(seq, xs_seq)):
            off_k = k - half
            if off_k in (0, 1):
                colour_k, weight_k = CONTEXT, "bold"
            elif off_k == offset:
                colour_k, weight_k = HIGHLIGHT, "bold"
            else:
                colour_k, weight_k = OTHER, "normal"
            ax_a.text(xx, 0.520, base, ha="center", va="center", fontsize=5.0,
                      family="monospace", color=colour_k, fontweight=weight_k, zorder=3)
        ax_a.text(xs_seq[half] + step / 2, 0.470, "CpG", ha="center", va="top",
                  fontsize=5.4, color=CONTEXT)
        ax_a.annotate(f"{ref_allele}>{alt_allele}", xy=(xs_seq[half + offset], 0.478),
                      xytext=(0, -8), textcoords="offset points", ha="center", va="top",
                      fontsize=5.8, color=HIGHLIGHT,
                      arrowprops=dict(arrowstyle="-", lw=0.7, color=HIGHLIGHT))
        ax_a.text(span_x, 0.405, f"hg38 reference, {2 * half + 1} bp", ha="right",
                  va="top", fontsize=5.2, color=OTHER)
    ax_a.text(0, 0.700, "bp from cg20699548", ha="center", va="top",
              fontsize=6.0, color=OTHER)

    ax_a.add_patch(Rectangle((14, 0.235), 44, 0.085, facecolor=TINT, edgecolor=CONTEXT,
                             lw=0.6, zorder=3))
    ax_a.text(36, 0.278, "ATAC Q4  $\\cdot$  H3K27ac Q3", ha="center", va="center",
              fontsize=5.8, color=TEXT, zorder=4)
    ax_a.text(-58, 0.278, "NCOA2 gene body, OpenSea",
              fontsize=6.0, color=TEXT, ha="left", va="center")
    ax_a.text(-58, 0.155, "transcript ENST00000452400.7 ($-$ strand)",
              fontsize=6.0, color=TEXT, ha="left", va="center")
    ax_a.text(-58, 0.030, f"current-context $\\Delta\\hat\\beta$ = {delta:.4f}  (seed 42, held-out test split)",
              fontsize=6.4, color=HIGHLIGHT, ha="left", va="center")
    panel_letter(ax_a, "a")
    panel_title(ax_a, "NCOA2 synonymous candidate")
    record("fig5a_ncoa2_locus", pd.DataFrame([dict(
        panel="5a", variant_uid=case["Variant_UID"], gene="NCOA2", probe="cg20699548",
        genome_build="hg38", chrom=ncoa2["chr"], variant_pos_1based=int(ncoa2["Variant_Position_1based"]),
        cpg_pos=int(ncoa2["pos"]), signed_offset_bp=offset, abs_distance_bp=int(ncoa2["Absolute_Distance_To_CpG"]),
        model_split=ncoa2["Model_Split"], predicted_delta_beta=delta, seed=42,
        seed_count=int(case["Seed_Count"]),
        source="ABL/candidates/top_candidate_case_study.csv + candidate_matched_background_statistics.csv",
        uncertainty="single seed-42 value; no across-seed interval available")]))

    ax_b = fig.add_subplot(gs[0, 1])
    comp = repo.acsv("candidates/top_candidate_matched_comparators_long.csv")
    comp1 = comp[comp["Target_Rank"] == 1]
    vals = np.sort(np.abs(comp1["Comparator_Delta_Beta"].astype(float).values))
    ecdf = np.arange(1, len(vals) + 1) / len(vals)
    x_right = abs(delta) * 1.13
    ax_b.step(np.append(vals, x_right), np.append(ecdf, 1.0), where="post", color=OTHER,
              lw=1.1, zorder=3)
    ax_b.axvline(abs(delta), color=HIGHLIGHT, lw=1.1, zorder=4)
    ax_b.annotate(f"NCOA2 |$\\Delta\\hat\\beta$| = {abs(delta):.4f}", (abs(delta), 0.52),
                  textcoords="offset points", xytext=(-4, 0), ha="center", va="center",
                  fontsize=6.0, color=HIGHLIGHT, rotation=90)
    ax_b.set_xlabel(r"matched comparator |$\Delta\hat\beta$|")
    ax_b.set_ylabel("cumulative fraction")
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.set_xlim(-0.006, x_right)
    ax_b.grid(color="#E6EAED", lw=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    exceed = int((vals >= abs(delta)).sum())
    panel_letter(ax_b, "b")
    panel_title(ax_b, "Background extremeness")
    record("fig5b_matched_background", comp1.assign(
        panel="5b", abs_comparator_delta_beta=np.abs(comp1["Comparator_Delta_Beta"].astype(float)),
        candidate_abs_delta_beta=abs(delta), seed=42,
        uncertainty="single seed-42 values; no across-seed interval")[[
        "panel", "Target_Variant_UID", "Target_Gene", "Matching_Tier", "Comparator_Variant_UID",
        "Comparator_Gene", "Comparator_Delta_Beta", "abs_comparator_delta_beta", "Comparator_SBS96",
        "Comparator_CpG_Effect", "Comparator_Distance_From_CpG", "candidate_abs_delta_beta", "seed",
        "uncertainty"]])

    ax_c = fig.add_subplot(gs[1, 0])
    lit = repo.acsv("literature_variant_screen/literature_variant_predictions_ranked.csv")
    d = lit["Predicted_Delta_Beta"].astype(float).values
    counts_c, _, _ = ax_c.hist(d, bins=26, color=NULLC, edgecolor="white", lw=0.3, zorder=2)
    stk = repo.acsv("literature_variant_screen/stk11_case_study_figure_values.csv")
    stk = stk[stk["Record"] == "Highlighted target"]
    styles = [(HIGHLIGHT, "-"), (SEQUENCE, (0, (4, 2)))]
    c_src = []
    pad_c = 0.055 * (d.max() - d.min())
    ax_c.set_xlim(d.min() - pad_c, d.max() + pad_c)
    ax_c.set_ylim(0, counts_c.max() * 1.42)
    for i, (_, r) in enumerate(stk.iterrows()):
        v = float(r["Predicted_Delta_Beta"])
        colour, ls = styles[i]
        ax_c.axvline(v, color=colour, ls=ls, lw=1.1, zorder=4)
        # both lines sit at the extremes of the scored range, so the labels go inward
        ax_c.annotate(f"{r['Variant_ID']}\n{r['probeID']}\n{v:+.4f}",
                      (v, counts_c.max() * (1.30 if i == 0 else 0.92)),
                      textcoords="offset points", xytext=(-5 if v > 0 else 5, 0),
                      ha="right" if v > 0 else "left", va="top", fontsize=5.4,
                      color=colour, linespacing=1.35)
        c_src.append(dict(panel="5c", variant_id=r["Variant_ID"], probe=r["probeID"],
                          predicted_delta_beta=v, seed="42-44 mean", status="training probe",
                          source="ABL/literature_variant_screen/stk11_case_study_figure_values.csv"))
    ax_c.set_xlabel(r"predicted $\Delta\hat\beta$")
    ax_c.set_ylabel("variants", fontsize=7.5)
    ax_c.grid(axis="y", color="#E6EAED", lw=0.5, zorder=0)
    ax_c.set_axisbelow(True)
    panel_letter(ax_c, "c")
    panel_title(ax_c, "STK11 training probes")
    record("fig5c_stk11", pd.concat([
        pd.DataFrame(c_src),
        pd.DataFrame([dict(panel="5c", variant_id="background pool", probe="",
                           predicted_delta_beta=np.nan, seed="42-44 mean",
                           status=f"{len(d)} scored nonsynonymous variant-CpG pairs",
                           source="ABL/literature_variant_screen/literature_variant_predictions_ranked.csv")])],
        ignore_index=True))

    ax_d = fig.add_subplot(gs[1, 1])
    d_rows, d_src = [], []
    for cohort, label, colour in [("genoa", "GENOA", FUSION), ("egtex", "eGTEx", CONTEXT)]:
        summary = repo.jjson(f"gwas_nominal_current/{cohort}/run_summary.json")
        tail = [t for t in summary["primary_tail_enrichment"]
                if abs(t["fraction"] - 0.05) < 1e-9][0]
        d_rows.append(dict(label=f"{label}\nn={tail['n_in_tail']} in tail",
                           value=float(tail["gwas_share"]), lo=float(tail["ci_low"]),
                           hi=float(tail["ci_high"]), colour=colour, marker="D"))
        d_src.append(dict(panel="5d", cohort=label,
                          context="current primary breast epithelium",
                          population="pairs with measured methylation association, cohort p < 0.05",
                          endpoint="GWAS fraction among top 5% predicted effects",
                          n_in_tail=int(tail["n_in_tail"]),
                          n_labelled_pairs=int(summary["n_gwas_pairs"]),
                          n_matched_pairs=int(summary["n_matched_pairs"]),
                          value=float(tail["gwas_share"]), ci_low=float(tail["ci_low"]),
                          ci_high=float(tail["ci_high"]), matched_null_fraction=0.5,
                          excludes_half=bool(tail["excludes_half"]), seeds="42;43;44",
                          source=f"gwas_nominal_current/{cohort}/run_summary.json",
                          uncertainty=f"{summary['n_boot']} block bootstrap draws, "
                                      f"{summary['block_size_bp']} bp blocks"))
    ys_d = np.arange(len(d_rows))[::-1]
    for y, r in zip(ys_d, d_rows):
        ax_d.barh(y, r["value"], height=0.46, color=r["colour"], edgecolor="none", zorder=3)
        ax_d.plot([r["lo"], r["hi"]], [y, y], color=TEXT, lw=1.0, solid_capstyle="butt",
                  zorder=5)
        for bound in (r["lo"], r["hi"]):
            ax_d.plot([bound, bound], [y - 0.12, y + 0.12], color=TEXT, lw=1.0, zorder=5)
        ax_d.text(0.012, y, f"{r['value']:.2f}", ha="left", va="center",
                  fontsize=6.0, color="white", zorder=6)
    ax_d.axvline(0.5, color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
    ax_d.set_yticks(ys_d)
    ax_d.set_yticklabels([r["label"] for r in d_rows], fontsize=6.2)
    ax_d.set_ylim(-0.6, len(d_rows) - 0.4)
    ax_d.set_xlim(0, 0.80)
    ax_d.set_xlabel("GWAS fraction, top 5%")
    ax_d.grid(axis="x", color="#E6EAED", lw=0.5, zorder=0)
    ax_d.set_axisbelow(True)
    ax_d.annotate("matched null 0.5", (0.5, 0.03), xycoords=("data", "axes fraction"),
                  textcoords="offset points", xytext=(3, 0), ha="left", va="bottom",
                  fontsize=5.4, color=OTHER)
    panel_letter(ax_d, "d")
    panel_title(ax_d, "GWAS control")
    record("fig5d_gwas_control", pd.DataFrame(d_src))

    fig.savefig(out / "fig5_variant_prioritization.pdf", metadata={"CreationDate": None}, dpi=450)
    plt.close(fig)


def fig6(repo: Repo, out: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 118 * MM))
    gs = fig.add_gridspec(2, 6, left=0.120, right=0.955, top=0.925, bottom=0.095,
                          wspace=1.0, hspace=0.50)

    est = repo.acsv("rc_uncertainty/estimator_comparison.csv")
    corr = repo.acsv("rc_uncertainty/disagreement_error_correlations.csv")
    curves = repo.acsv("rc_uncertainty/selective_prediction_curves.csv")

    est_order = ["rc_disagreement", "cross_seed_sd", "boundary_distance", "combined", "random"]
    est_label = {"rc_disagreement": "RC disagreement", "cross_seed_sd": "seed SD",
                 "boundary_distance": "boundary", "combined": "combined", "random": "random"}
    est_colour = dict(zip(est_order, plt.get_cmap("plasma")(np.linspace(0.05, 0.80, len(est_order)))))

    ax_a = fig.add_subplot(gs[0, 0:3])
    a_src = []
    ys_a = np.arange(len(est_order))[::-1]
    for model in ("fusion", "sequence", "epi"):
        colour, marker = MODEL_STYLE[model]
        xs, ys = [], []
        for y, e in zip(ys_a, est_order):
            sub = est[(est["model"] == model) & (est["estimator"] == e) & (est["seed"] == 42)]
            if not len(sub):
                continue
            r = sub.iloc[0]
            xs.append(float(r["aurc_beta_mae"]))
            ys.append(y)
            a_src.append(dict(panel="6a", model=model, seed=42, estimator=e,
                              aurc_beta_mae=float(r["aurc_beta_mae"]),
                              beta_mae_full_coverage=float(r["beta_mae_at_full_coverage"]),
                              beta_mae_50pct=float(r["beta_mae_at_50pct_coverage"]),
                              scale="beta",
                              uncertainty="deterministic summary; no interval exported"))
        ax_a.plot(xs, ys, marker, ms=4.0, color=colour, mec="white", mew=0.5, ls="none",
                  zorder=3, label=MODEL_LABEL[model])
    ax_a.set_yticks(ys_a)
    ax_a.set_yticklabels([est_label[e] for e in est_order], fontsize=6.2)
    ax_a.set_ylim(-0.6, len(est_order) - 0.4)
    ax_a.set_xlabel(r"AURC ($\beta$ MAE)")
    ax_a.grid(axis="x", color="#E6EAED", lw=0.5, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="center right", frameon=True, facecolor="white", edgecolor="#D5DBE0",
                framealpha=0.95, fontsize=5.4, handletextpad=0.3, borderpad=0.28,
                labelspacing=0.18)
    panel_letter(ax_a, "a", xoff=-54)
    panel_title(ax_a, "Error ranking")
    record("figS4a_estimator_comparison", pd.DataFrame(a_src))

    ax_b = fig.add_subplot(gs[0, 3:6])
    b_src = []
    for e in est_order:
        sub = curves[(curves["model"] == "fusion") & (curves["seed"] == 42) &
                     (curves["estimator"] == e)].sort_values("coverage")
        if not len(sub):
            continue
        ax_b.plot(sub["coverage"], sub["beta_mae"], color=est_colour[e], lw=1.3,
                  zorder=3, label=est_label[e])
        b_src.append(sub.assign(panel="6b"))
    ax_b.set_xlabel("coverage")
    ax_b.set_ylabel(r"retained $\beta$ MAE", labelpad=1.5)
    ax_b.grid(color="#E9EDF0", lw=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#D5DBE0",
                framealpha=0.95, fontsize=5.0, handlelength=1.0, handletextpad=0.3,
                borderpad=0.24, labelspacing=0.12)
    panel_letter(ax_b, "b", xoff=-46)
    panel_title(ax_b, "Risk-coverage")
    if b_src:
        record("figS4b_risk_coverage", pd.concat(b_src, ignore_index=True))

    ax_c = fig.add_subplot(gs[1, 0:2])
    pos_c = ax_c.get_position()
    ax_c.set_position([pos_c.x0, pos_c.y0, pos_c.width * 0.86, pos_c.height])
    ax_c.set_anchor("N")  # square box at the top of the row, so its title lines up with d
    d = predictions(repo, "fusion")
    disagree = np.abs(d["pred_beta_fwd"] - d["pred_beta_rc"]).to_numpy(dtype=float)
    abserr = np.abs(d["true_beta"] - d["pred_beta_rc_avg"]).to_numpy(dtype=float)
    hb = hexpanel(ax_c, disagree, abserr,
                  (0.0, float(np.quantile(disagree, 0.995)), 0.0,
                   float(np.quantile(abserr, 0.995))), gridsize=38, cmap=SPECTRUM_CMAP)
    lim_c = min(ax_c.get_xlim()[1], ax_c.get_ylim()[1])
    ax_c.plot([0, lim_c], [0, lim_c], color=HIGHLIGHT, ls=(0, (3, 2)), lw=1.0, zorder=3)
    ax_c.set_xlabel("|forward $-$ RC|", labelpad=1.5)
    ax_c.set_ylabel(r"|$\beta$ error|", labelpad=1.5)
    from scipy.stats import spearmanr
    rho = float(spearmanr(disagree, abserr).statistic)
    ax_c.text(0.04, 0.96, f"$\\rho$ = {rho:.3f}", transform=ax_c.transAxes, ha="left",
              va="top", fontsize=5.8, color=TEXT,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5))
    cb = fig.colorbar(hb, cax=ax_c.inset_axes([1.030, 0.0, 0.028, 1.0]))
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=4.6)
    cb.outline.set_visible(False)
    panel_letter(ax_c, "c", xoff=-31)
    panel_title(ax_c, "Disagreement vs error")
    record("figS4c_disagreement_vs_error_raw", pd.DataFrame(dict(
        panel="6c", model="fusion", seed=42, rc_disagreement=disagree, absolute_beta_error=abserr
    )).assign(spearman_rho=rho, source="ABL/seed42/fusion/predictions.csv"))

    ax_d = fig.add_subplot(gs[1, 2:6])
    rows_d, d_src = [], []
    for model in ("fusion", "sequence", "epi"):
        sub_c = corr[corr["model"] == model]
        if not len(sub_c):
            continue
        vals = sub_c["spearman_disagreement_vs_error"].astype(float)
        colour, marker = MODEL_STYLE[model]
        rows_d.append(dict(label=MODEL_LABEL[model], value=float(vals.mean()),
                           lo=float(vals.min()), hi=float(vals.max()),
                           colour=colour, marker=marker))
        for _, r in sub_c.iterrows():
            d_src.append(dict(panel="6d", model=model, seed=int(r["seed"]),
                              n_loci=int(r["n_loci"]), n_blocks=int(r["n_blocks"]),
                              spearman=float(r["spearman_disagreement_vs_error"]),
                              ci_low=float(r["spearman_ci_lo"]),
                              ci_high=float(r["spearman_ci_hi"]),
                              pearson=float(r["pearson_disagreement_vs_error"]),
                              plotted_point="mean over seeds 42-44",
                              plotted_bar="range over seeds 42-44",
                              uncertainty="per-seed 1 Mb block-bootstrap interval in "
                                          "ci_low/ci_high; not the bar drawn in the panel"))
    ys_d = np.arange(len(rows_d))[::-1]
    ax_d.barh(ys_d, [r["value"] for r in rows_d], height=0.55,
              color=[r["colour"] for r in rows_d], edgecolor="none", zorder=3)
    for y, r in zip(ys_d, rows_d):
        ax_d.text(r["value"] - 0.012, y, f"{r['value']:.2f}", ha="right", va="center",
                  fontsize=6.0, color="white", zorder=4)
    ax_d.set_yticks(ys_d)
    ax_d.set_yticklabels([r["label"] for r in rows_d], fontsize=6.2)
    ax_d.set_ylim(-0.6, len(rows_d) - 0.4)
    ax_d.set_xlim(0, 0.60)
    ax_d.set_xlabel(r"Spearman $\rho$", labelpad=1.5)
    ax_d.grid(axis="x", color="#E6EAED", lw=0.5, zorder=0)
    ax_d.set_axisbelow(True)
    panel_letter(ax_d, "d", xoff=-44)
    panel_title(ax_d, "Ranking correlation")
    record("figS4d_disagreement_vs_error", pd.DataFrame(d_src))



    fig.savefig(out / "fig6_uncertainty.pdf", metadata={"CreationDate": None}, dpi=450)
    plt.close(fig)


def fig7(repo: Repo, out: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 131 * MM))
    gs = fig.add_gridspec(2, 6, left=0.175, right=0.955, top=0.930, bottom=0.105,
                          wspace=1.0, hspace=0.58)

    cp = repo.acsv("motif_disruption/per_motif_coupling.csv")
    cp = cp.reindex(cp["coupling_spearman"].abs().sort_values(ascending=False).index)
    top = cp.head(14).copy()
    # The manuscript states Fig. 7 shows the neural and k-mer results on a shared
    # axis, so a missing table must fail rather than silently drop the comparison.
    kmer_path = repo.j / "motif_disruption_kmer_baseline" / "per_motif_coupling.csv"
    if not kmer_path.exists():
        raise SystemExit(f"STOP: {kmer_path} is missing; Fig. 7 needs the k-mer baseline.")
    kmer = pd.read_csv(kmer_path)
    ets = top["factor"].str.contains("EL|EHF|FEV|ETS|ERG|ETV|FLI|SPDEF|GABPA",
                                     regex=True, na=False)

    ax_a = fig.add_subplot(gs[0, 0:2])
    rows, src = [], []
    for (_, r), is_ets in zip(top.iterrows(), ets):
        rows.append(dict(label=f"{r['factor']} ({r['matrix_id']})",
                         value=float(r["coupling_spearman"]),
                         lo=float(r["coupling_ci_low"]), hi=float(r["coupling_ci_high"]),
                         colour=HIGHLIGHT if is_ets else FUSION, marker="D"))
        src.append(dict(panel="7a", arm="fusion", matrix_id=r["matrix_id"], factor=r["factor"],
                        ets_family=bool(is_ets), n_covered=int(r["n_covered"]),
                        n_blocks=int(r["n_blocks"]),
                        coupling_spearman=float(r["coupling_spearman"]),
                        ci_low=float(r["coupling_ci_low"]), ci_high=float(r["coupling_ci_high"]),
                        partial_gc_distance=float(r["coupling_partial_gc_distance"]),
                        partial_gc_distance_substitution=float(r["coupling_partial_gc_distance_substitution"]),
                        q_bh=float(r["coupling_q_bh"]), consensus=r["motif_consensus"],
                        uncertainty="1 Mb block bootstrap"))
    forest(ax_a, rows, r"coupling $\rho$", null_line=0.0,
           xlim=(-0.56, 0.06))
    ax_a.tick_params(axis="y", labelsize=5.4)
    ax_a.legend(handles=[
        Line2D([], [], marker="D", ls="none", ms=4.0, color=HIGHLIGHT, label="ETS family"),
        Line2D([], [], marker="D", ls="none", ms=4.0, color=FUSION, label="other"),
    ], loc="lower left", frameon=False, fontsize=5.4, handletextpad=0.4, borderpad=0.0,
        labelspacing=0.15)
    panel_letter(ax_a, "a", xoff=-74)
    panel_title(ax_a, "Fusion coupling")

    ax_b = fig.add_subplot(gs[0, 2:4], sharex=ax_a)
    rows_b = []
    if kmer is not None:
        km = kmer.set_index("matrix_id")
        for _, r in top.iterrows():
            if r["matrix_id"] in km.index:
                k = km.loc[r["matrix_id"]]
                rows_b.append(dict(label="", value=float(k["coupling_spearman"]),
                                   lo=float(k["coupling_ci_low"]), hi=float(k["coupling_ci_high"]),
                                   colour=KMER, marker="P"))
                src.append(dict(panel="7b", arm="k-mer ridge", matrix_id=r["matrix_id"],
                                factor=r["factor"], ets_family=None,
                                n_covered=int(k["n_covered"]), n_blocks=int(k["n_blocks"]),
                                coupling_spearman=float(k["coupling_spearman"]),
                                ci_low=float(k["coupling_ci_low"]), ci_high=float(k["coupling_ci_high"]),
                                partial_gc_distance=float(k.get("coupling_partial_gc_distance", np.nan)),
                                partial_gc_distance_substitution=float(
                                    k.get("coupling_partial_gc_distance_substitution", np.nan)),
                                q_bh=float(k["coupling_q_bh"]), consensus=k.get("motif_consensus", ""),
                                uncertainty="1 Mb block bootstrap"))
            else:
                rows_b.append(dict(label="", value=np.nan, lo=None, hi=None, colour=KMER, marker="P"))
    if rows_b:
        forest(ax_b, rows_b, r"coupling $\rho$", null_line=0.0,
               xlim=(-0.56, 0.06))
    ax_b.set_yticklabels([])
    panel_letter(ax_b, "b", xoff=-16)
    panel_title(ax_b, "$k$-mer ridge")

    ax_c = fig.add_subplot(gs[0, 4:6])
    cols = ["coupling_spearman", "coupling_partial_gc_distance",
            "coupling_partial_gc_distance_substitution"]
    labels_c = ["raw", "| GC,\ndistance", "| GC, dist,\nsubstitution"]
    mat = top[cols].to_numpy(dtype=float)
    im = heatmap(ax_c, mat, [r["factor"] for _, r in top.iterrows()], labels_c,
                 float(np.nanmin(mat)), float(np.nanmax(mat)), cmap="viridis",
                 fmt="{:.2f}", fontsize=4.4, text_colour="auto")
    ax_c.tick_params(axis="y", labelsize=5.0)
    ax_c.tick_params(axis="x", labelsize=4.6)
    cb = fig.colorbar(im, ax=ax_c, fraction=0.045, pad=0.03)
    cb.ax.set_title(r"$\rho$", fontsize=5.2, pad=2)
    cb.ax.tick_params(labelsize=4.6)
    cb.outline.set_visible(False)
    panel_letter(ax_c, "c", xoff=-40)
    panel_title(ax_c, "After controls")

    ax_d = fig.add_subplot(gs[1, 0:2])
    mb = repo.acsv("motif_disruption/motif_vs_background.csv").set_index("group")
    d_src = []
    groups = [("strong_disruption", "strong\ndisruption"),
              ("matched_weak_disruption", "matched weak\ndisruption")]
    xs = np.arange(len(groups))
    cols_d = [KMER, FUSION]
    for x, (key, lab), cc in zip(xs, groups, cols_d):
        r = mb.loc[key]
        ax_d.bar(x, float(r["median_abs_delta_m"]), width=0.50, color=cc, edgecolor="none",
                 zorder=2)
        ax_d.plot([x, x], [float(r["ci_low"]), float(r["ci_high"])], color=TEXT, lw=1.0,
                  solid_capstyle="butt", zorder=4)
        for bound in (float(r["ci_low"]), float(r["ci_high"])):
            ax_d.plot([x - 0.10, x + 0.10], [bound, bound], color=TEXT, lw=1.0, zorder=4)
        d_src.append(dict(panel="7d", group=key, n=int(r["n"]),
                          median_abs_delta_m=float(r["median_abs_delta_m"]),
                          ci_low=float(r["ci_low"]), ci_high=float(r["ci_high"]),
                          median_abs_distance_bp=float(r["median_abs_distance_bp"]),
                          mean_gc=float(r["mean_gc"]),
                          uncertainty="1 Mb block bootstrap"))
    cc_row = mb.loc["continuous_coupling"]
    coupling_rho = float(cc_row["median_abs_delta_m"])
    coupling_ci = (float(cc_row["ci_low"]), float(cc_row["ci_high"]))
    d_src.append(dict(panel="7d-note", group="continuous_coupling", n=int(cc_row["n"]),
                      median_abs_delta_m=coupling_rho, ci_low=coupling_ci[0],
                      ci_high=coupling_ci[1], median_abs_distance_bp=np.nan, mean_gc=np.nan,
                      uncertainty="1 Mb block bootstrap; this row is a Spearman correlation "
                                  "between disruption magnitude and |delta M|, not a median, "
                                  "and is reported in the footnote rather than plotted"))
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels([lab for _, lab in groups], fontsize=5.6)
    ax_d.set_xlim(-0.6, len(groups) - 0.4)
    ax_d.set_ylabel(r"median |$\Delta\widehat{M}$|", labelpad=1.5)
    ax_d.set_ylim(0, float(mb.loc[[g for g, _ in groups], "ci_high"].max()) * 1.18)
    ax_d.grid(axis="y", color="#E9EDF0", lw=0.5, zorder=0)
    ax_d.set_axisbelow(True)
    panel_letter(ax_d, "d", xoff=-52)
    ax_d.set_title("Disruption vs control", fontsize=9, loc="left", pad=4,
                   color=TEXT, x=-0.30)
    record("figS5d_motif_vs_background", pd.DataFrame(d_src))

    ax_e = fig.add_subplot(gs[1, 2:6])
    allm = repo.acsv("motif_disruption/per_motif_coupling.csv")
    ax_e.axhline(0, color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
    cpg_counts = allm["motif_consensus_cpg_count"]
    cpg_groups = [("0", cpg_counts == 0), ("1", cpg_counts == 1), ("2+", cpg_counts >= 2)]
    for (lab_e, mask), cc in zip(cpg_groups, plt.get_cmap("plasma")([0.10, 0.55, 0.86])):
        ax_e.scatter(allm.loc[mask, "motif_gc_content"], allm.loc[mask, "coupling_spearman"],
                     color=cc, s=5.5, alpha=0.60, linewidths=0,
                     zorder=2 + len(lab_e), label=f"{lab_e}  ($n$ = {int(mask.sum())})")
    from scipy.stats import spearmanr
    rho_gc = float(spearmanr(allm["motif_gc_content"], allm["coupling_spearman"]).statistic)
    ax_e.set_xlabel("motif GC content", labelpad=1.5)
    ax_e.set_ylabel(r"coupling $\rho$", labelpad=1.5)
    ax_e.text(0.04, 0.05, f"$\\rho$ = {rho_gc:.3f}", transform=ax_e.transAxes,
              ha="left", va="bottom", fontsize=5.4)
    ax_e.legend(title="CpGs in consensus", loc="upper right", frameon=True,
                facecolor="white", edgecolor="#D5DBE0", framealpha=0.95, fontsize=5.2,
                title_fontsize=5.2, handletextpad=0.25, borderpad=0.28, labelspacing=0.18,
                markerscale=1.8)
    ax_e.grid(color="#E9EDF0", lw=0.5, zorder=0)
    ax_e.set_axisbelow(True)
    panel_letter(ax_e, "e", xoff=-34)
    panel_title(ax_e, "GC confound")
    record("figS5e_gc_confound", allm[[
        "matrix_id", "factor", "motif_gc_content", "motif_consensus_cpg_count",
        "coupling_spearman", "coupling_partial_gc_distance", "n_covered"]].assign(
        panel="7e", spearman_gc_vs_coupling=rho_gc))


    record("figS5abc_motif_coupling", pd.DataFrame(src))
    n_ets = int(ets.sum())
    fig.savefig(out / "fig7_motif_controls.pdf", metadata={"CreationDate": None}, dpi=450)
    plt.close(fig)


def figs1(repo: Repo, out: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 100 * MM))
    gs = fig.add_gridspec(1, 3, left=0.085, right=0.985, top=0.875, bottom=0.165, wspace=0.42)

    cov = repo.acsv("target_qc/test_coverage_per_probe.csv")
    ax_a = fig.add_subplot(gs[0, 0])
    covcol = [c for c in cov.columns if "coverage" in c.lower() or "n_samples" in c.lower()]
    series = cov[covcol[0]].astype(float) if covcol else cov.iloc[:, 1].astype(float)
    ax_a.hist(series, bins=40, color=CONTEXT, edgecolor="white", lw=0.3, zorder=2)
    ax_a.set_xlabel("samples with an observed value")
    ax_a.set_ylabel("probes (log scale)")
    ax_a.grid(axis="y", color="#E6EAED", lw=0.5, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.set_yscale("log")
    panel_letter(ax_a, "a")
    panel_title(ax_a, "Coverage distribution")
    record("figS1a_coverage", cov.assign(panel="S1a"))

    ax_b = fig.add_subplot(gs[0, 1])
    binm = repo.acsv("target_qc/coverage_bin_metrics.csv")
    b_src = []
    for model, colour, marker in [("sequence", SEQUENCE, "o"), ("fusion", FUSION, "D"), ("epi", CONTEXT, "s")]:
        sub = binm[binm["model"] == model]
        if not len(sub):
            continue
        xs = np.arange(len(sub))
        ax_b.plot(xs, sub["beta_mae"].astype(float), marker, ms=3.2, ls="-", lw=1.0,
                  color=colour, zorder=3, label="context" if model == "epi" else model)
        b_src.append(sub.assign(panel="S1b"))
    ax_b.set_xticks(np.arange(len(binm[binm["model"] == "epi"])))
    ax_b.set_xticklabels(binm[binm["model"] == "epi"]["coverage_bin"], fontsize=5.8, rotation=35, ha="right")
    ax_b.set_xlabel("coverage bin (samples)")
    ax_b.set_ylabel(r"$\beta$ MAE")
    ax_b.grid(axis="y", color="#E6EAED", lw=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.legend(frameon=False, fontsize=6.2, handletextpad=0.5, labelspacing=0.2)
    panel_letter(ax_b, "b")
    panel_title(ax_b, "Accuracy by coverage")
    if b_src:
        record("figS1b_coverage_bins", pd.concat(b_src, ignore_index=True))

    ax_c = fig.add_subplot(gs[0, 2])
    splits = [("training", 345359), ("validation\nchr10-11", 46557), ("test\nchr8-9", 26570)]
    total = 418486
    xs = np.arange(len(splits))
    ax_c.bar(xs, [s[1] for s in splits], 0.6, color=[CONTEXT, NULLC, FUSION], edgecolor="none", zorder=3)
    for x, (lab, n) in zip(xs, splits):
        ax_c.annotate(f"{n:,}\n{100 * n / total:.1f}%", (x, n), textcoords="offset points",
                      xytext=(0, 3), ha="center", va="bottom", fontsize=6.0, color=TEXT, linespacing=1.25)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels([s[0] for s in splits], fontsize=6.5)
    ax_c.set_ylabel("retained CpGs")
    ax_c.set_ylim(0, total * 0.95)
    ax_c.grid(axis="y", color="#E6EAED", lw=0.5, zorder=0)
    ax_c.set_axisbelow(True)
    panel_letter(ax_c, "c")
    panel_title(ax_c, "Split counts")
    record("figS1c_splits", pd.DataFrame([
        dict(panel="S1c", split=lab.replace("\n", " "), n_retained_cpgs=n,
             fraction_of_retained=n / total, denominator_total_retained_cpgs=total,
             n_samples=97, note="split is by chromosome, not by participant")
        for lab, n in splits]))

    fig.savefig(out / "figS1_target_split_qc.pdf", metadata={"CreationDate": None}, dpi=450)
    plt.close(fig)


def figs2(repo: Repo, out: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 104 * MM))
    gs = fig.add_gridspec(2, 2, left=0.225, right=0.965, top=0.900, bottom=0.120,
                          wspace=0.80, hspace=0.95)

    ax_a = fig.add_subplot(gs[0, 0])
    pc = repo.acsv("egtex_mqtl_positive_control/mqtl_positive_control_metrics.csv")
    endpoints = [
        ("spearman_delta_m", "spearman_delta_m_cluster_bootstrap_ci_low",
         "spearman_delta_m_cluster_bootstrap_ci_high", r"signed $\rho$ ($\Delta M$)"),
        ("direction_concordance_all", "direction_concordance_all_cluster_bootstrap_ci_low",
         "direction_concordance_all_cluster_bootstrap_ci_high", "direction concordance"),
        ("auc_positive_slope_delta_m", "auc_positive_slope_delta_m_cluster_bootstrap_ci_low",
         "auc_positive_slope_delta_m_cluster_bootstrap_ci_high", "directional AUROC"),
    ]
    rows, a_src = [], []
    for value_col, lo_col, hi_col, lab in endpoints:
        for _, r in pc.iterrows():
            colour, marker = MODEL_STYLE[r["model"]]
            rows.append(dict(label=f"{lab}\n{r['model']}", value=float(r[value_col]),
                             lo=float(r[lo_col]), hi=float(r[hi_col]),
                             colour=colour, marker=marker))
            a_src.append(dict(panel="S2a", cohort="eGTEx breast lead mQTL", model=r["model"],
                              seeds=r["seeds"], endpoint=lab, n_associations=int(r["n_loci"]),
                              n_unique_variants=int(r["n_unique_variants"]),
                              value=float(r[value_col]), ci_low=float(r[lo_col]),
                              ci_high=float(r[hi_col]), bootstrap_unit=r["bootstrap_unit"],
                              source="ABL/egtex_mqtl_positive_control/mqtl_positive_control_metrics.csv",
                              uncertainty="10,000 variant-cluster bootstrap draws"))
    forest(ax_a, rows, "agreement", null_line=None, xlim=(-0.08, 1.02))
    for null in (0.0, 0.5):
        ax_a.axvline(null, color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
    ax_a.tick_params(axis="y", labelsize=5.4)
    ax_a.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_a.tick_params(axis="x", labelsize=6.2)
    panel_letter(ax_a, "a", xoff=-72)
    panel_title(ax_a, "eGTEx positive control")
    record("figS2a_positive_control", pd.DataFrame(a_src))

    ax_b = fig.add_subplot(gs[0, 1])
    mn = repo.acsv("egtex_mqtl_matched_negative/matched_negative_metrics.csv")
    rows_b, b_src = [], []
    for _, r in mn.iterrows():
        colour, marker = MODEL_STYLE[r["model"]]
        score = {"predicted_delta_m": r"$\Delta\hat{M}$",
                 "predicted_delta_beta": r"$\Delta\hat{\beta}$"}.get(str(r["score"]), str(r["score"]))
        rows_b.append(dict(label=f"{r['model']}\n{score}", value=float(r["auroc"]),
                           lo=float(r["auroc_match_set_bootstrap_ci_low"]),
                           hi=float(r["auroc_match_set_bootstrap_ci_high"]),
                           colour=colour, marker=marker))
        b_src.append(dict(panel="S2b", cohort="eGTEx matched leads", model=r["model"],
                          score=r["score"], seeds=r["seeds"], n_rows=int(r["n_rows"]),
                          n_match_sets=int(r["n_match_sets"]),
                          n_significant=int(r["n_significant"]),
                          n_nonsignificant=int(r["n_nonsignificant"]),
                          auroc=float(r["auroc"]),
                          ci_low=float(r["auroc_match_set_bootstrap_ci_low"]),
                          ci_high=float(r["auroc_match_set_bootstrap_ci_high"]),
                          within_set_permutation_p=float(r["auroc_within_set_permutation_p"]),
                          comparator_definition=r["comparator_definition"],
                          source="ABL/egtex_mqtl_matched_negative/matched_negative_metrics.csv",
                          uncertainty="5,000 match-set bootstrap draws"))
    forest(ax_b, rows_b, "AUROC (35 matched pairs)", null_line=0.5, xlim=(0.28, 0.70))
    ax_b.tick_params(axis="y", labelsize=5.4)
    ax_b.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    ax_b.tick_params(axis="x", labelsize=6.2)
    panel_letter(ax_b, "b", xoff=-52)
    panel_title(ax_b, "Matched-pair discrimination")
    record("figS2b_matched_negative", pd.DataFrame(b_src))


    ax_d = fig.add_subplot(gs[1, 0:2])
    t = repo.csv("asm_validation_tycko/tycko_e2_signed_agreement.csv")
    rows_d, d_src = [], []
    for stratum, lab in [("mammary", "mammary (n=53)"), ("|effect| >= 20.0pp", "|effect|>=20pp (post hoc)")]:
        sub = t[t["Stratum"] == stratum]
        for _, r in sub.iterrows():
            colour, marker = MODEL_STYLE[r["Arm"]]
            rows_d.append(dict(label=f"{lab}\n{r['Arm']}", value=float(r["Direction_Concordance"]),
                               lo=float(r["Direction_CI_Low"]), hi=float(r["Direction_CI_High"]),
                               colour=colour, marker=marker))
            d_src.append(dict(panel="S2c", stratum=stratum, arm=r["Arm"], n_snps=int(r["N_SNPs"]),
                              n_blocks=int(r["N_Genomic_Blocks"]),
                              direction_concordance=float(r["Direction_Concordance"]),
                              ci_low=float(r["Direction_CI_Low"]), ci_high=float(r["Direction_CI_High"]),
                              signed_spearman=float(r["Signed_Spearman"]),
                              prespecified=(stratum == "mammary"),
                              note="post-hoc effect-size restriction" if stratum != "mammary" else "tissue subset",
                              uncertainty="2,000 block resamples"))
    forest(ax_d, rows_d, "direction concordance", null_line=0.5)
    ax_d.tick_params(axis="y", labelsize=5.4)
    panel_letter(ax_d, "c")
    panel_title(ax_d, "Do-Tycko subsets")
    record("figS2c_asm_subsets", pd.DataFrame(d_src))

    fig.savefig(out / "figS2_external_controls.pdf", metadata={"CreationDate": None}, dpi=450)
    plt.close(fig)


def figs3(repo: Repo, out: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 62 * MM))
    gs = fig.add_gridspec(1, 2, left=0.150, right=0.965, top=0.830, bottom=0.215,
                          wspace=0.75, width_ratios=[1.15, 1.0])

    # a: the same estimators and colours as Fig. 5, ranked against error within
    # predicted-beta bins so the bounded-range shape of beta cannot drive the ranking
    ws = repo.acsv("rc_uncertainty_conditional/within_stratum_correlations.csv")
    ws = ws[ws["model"] == "fusion"]
    est_order = ["rc_disagreement", "cross_seed_sd", "boundary_distance", "combined", "random"]
    est_colour = dict(zip(est_order, plt.get_cmap("plasma")(np.linspace(0.05, 0.80, len(est_order)))))
    est_label = {"rc_disagreement": "RC disagreement", "cross_seed_sd": "seed SD",
                 "boundary_distance": "boundary"}
    ax_a = fig.add_subplot(gs[0, 0])
    shown = ["rc_disagreement", "cross_seed_sd", "boundary_distance"]
    ys_a = np.arange(len(shown))[::-1]
    a_src = []
    for y, e in zip(ys_a, shown):
        colour = est_colour[e]
        pts = {}
        for target, dy, filled in (("beta", 0.13, True), ("m", -0.13, False)):
            v = ws[(ws["estimator"] == e) & (ws["error_target"] == target)]
            vals = v["within_bin_spearman_pooled"].astype(float)
            mean = float(vals.mean())
            pts[target] = (mean, y + dy)
            ax_a.plot([vals.min(), vals.max()], [y + dy, y + dy], color=colour, lw=1.0,
                      solid_capstyle="butt", zorder=2)
            ax_a.plot([mean], [y + dy], "o", ms=4.4, color=colour,
                      mfc=colour if filled else "white", mec=colour, mew=1.0, zorder=4)
            for _, r in v.iterrows():
                a_src.append(dict(panel="S3a", model="fusion", seed=int(r["seed"]), estimator=e,
                                  error_scale="beta" if target == "beta" else "M",
                                  n_predicted_beta_bins=int(r["n_bins_used"]),
                                  within_bin_spearman_pooled=float(r["within_bin_spearman_pooled"]),
                                  plotted_point="mean over seeds 42-44",
                                  plotted_bar="range over seeds 42-44",
                                  source="ABL/rc_uncertainty_conditional/within_stratum_correlations.csv"))
        ax_a.plot([pts["beta"][0], pts["m"][0]], [pts["beta"][1], pts["m"][1]], color=colour,
                  lw=0.6, ls=(0, (1.5, 1.2)), zorder=3)
    ax_a.axvline(0.0, color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
    ax_a.set_yticks(ys_a)
    ax_a.set_yticklabels([est_label[e] for e in shown], fontsize=6.6)
    ax_a.set_ylim(-0.6, len(shown) - 0.4)
    ax_a.set_xlim(-0.01, 0.24)
    ax_a.set_xlabel(r"Spearman $\rho$ with error, within predicted-$\beta$ bins", labelpad=1.5)
    ax_a.grid(axis="x", color="#E6EAED", lw=0.5, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(handles=[Line2D([], [], ls="none", marker="o", ms=4.4, color=OTHER, label=r"$\beta$ error"),
                         Line2D([], [], ls="none", marker="o", ms=4.4, color=OTHER, mfc="white",
                                mew=1.0, label="$M$ error")],
                loc="lower right", frameon=False, fontsize=6.0, handletextpad=0.3,
                borderaxespad=0.2)
    panel_letter(ax_a, "a", xoff=-62)
    panel_title(ax_a, "Error ranking by scale")
    record("figS3a_scale_dependence", pd.DataFrame(a_src))

    # b: the discrimination test behind the matched groups of Fig. 7d
    md = repo.acsv("motif_disruption/meqtl_discrimination_by_motif_status.csv").set_index("group")
    ax_b = fig.add_subplot(gs[0, 1])
    colour, marker = MODEL_STYLE["fusion"]
    rows_b, b_src = [], []
    for g, lab in (("strong_disruption", "strong disruption"),
                   ("weak_disruption", "matched weak\ndisruption")):
        r = md.loc[g]
        rows_b.append(dict(label=f"{lab}\n($n$ = {int(r['n']):,})", value=float(r["auroc"]),
                           lo=float(r["ci_low"]), hi=float(r["ci_high"]),
                           colour=colour, marker=marker,
                           mfc=colour if g == "strong_disruption" else "white"))
        b_src.append(dict(panel="S3b", cohort="GENOA", model="fusion", seeds="42;43;44",
                          group=g, n_pairs=int(r["n"]), n_significant=int(r["n_significant"]),
                          auroc=float(r["auroc"]), ci_low=float(r["ci_low"]),
                          ci_high=float(r["ci_high"]),
                          source="ABL/motif_disruption/meqtl_discrimination_by_motif_status.csv",
                          uncertainty="500 resamples of 1 Mb genomic blocks"))
    forest(ax_b, rows_b, "AUROC, significant vs control", null_line=0.5, xlim=(0.48, 0.64))
    ax_b.tick_params(axis="y", labelsize=6.2)
    ax_b.set_xticks([0.50, 0.55, 0.60])
    panel_letter(ax_b, "b", xoff=-70)
    panel_title(ax_b, "Motif-stratified discrimination")
    record("figS3b_motif_discrimination", pd.DataFrame(b_src))

    fig.savefig(out / "figS3_uncertainty_motif_controls.pdf", metadata={"CreationDate": None}, dpi=450)
    plt.close(fig)


def write_source_data(out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for name, df in sorted(SOURCE_DATA.items()):
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        index.append(dict(panel_file=path.name, n_rows=len(df), n_columns=df.shape[1],
                          sha256=digest))
    idx = pd.DataFrame(index)
    idx.to_csv(out_dir / "source_data_index.csv", index=False)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".", type=Path)
    ap.add_argument("--figures-output", default="figures", type=Path)
    ap.add_argument("--source-data-output", default="figures/source_data", type=Path)
    args = ap.parse_args()

    style()
    repo = Repo(args.repo_root.resolve())
    out = args.figures_output
    out.mkdir(parents=True, exist_ok=True)

    fig2_full(repo, out)
    fig3(repo, out)
    fig4(repo, out)
    fig5(repo, out)
    fig6(repo, out)
    fig7(repo, out)
    figs1(repo, out)
    figs2(repo, out)
    figs3(repo, out)

    idx = write_source_data(args.source_data_output)
    print(idx.to_string(index=False))
    for p in sorted(out.glob("*.pdf")):
        print(f"{p} {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
