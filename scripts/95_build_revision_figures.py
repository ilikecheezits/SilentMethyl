"""Build main Figures 2-5 and Supplementary Figures S1-S4 for the expanded revision, with one source-data CSV per panel, from current breast-epithelium exports only."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

MM = 1.0 / 25.4

SEQUENCE = "#135E96"
CONTEXT = "#239DAD"
FUSION = "#168BC4"
OTHER = "#687582"
NULLC = "#AAB4BE"
HIGHLIGHT = "#D55E00"
TINT = "#EDF7FB"
TEXT = "#263238"

MODEL_STYLE = {
    "sequence": (SEQUENCE, "o"),
    "fusion": (FUSION, "D"),
    "context": (CONTEXT, "s"),
    "epi": (CONTEXT, "s"),
    "cpgenie": (OTHER, "^"),
    "deepcpg": (OTHER, "v"),
    "kmer_ridge": (OTHER, "P"),
    "composition": (OTHER, "X"),
    "distance_only": (NULLC, "o"),
}

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
    ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                textcoords="offset points", xytext=(xoff, yoff), fontsize=10,
                fontweight="bold", va="bottom", ha="left", color=TEXT,
                annotation_clip=False)


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
    fig = plt.figure(figsize=(180 * MM, 145 * MM))
    gs = fig.add_gridspec(3, 2, left=0.135, right=0.985, top=0.945, bottom=0.115,
                          wspace=0.50, hspace=1.05, height_ratios=[1.0, 1.0, 0.92])

    metrics = repo.acsv("paired_model_bootstrap/model_metrics_recomputed.csv")
    seeded = metrics[metrics["Analysis"] == "individual_seed"].copy()
    seeded["Seed"] = seeded["Seed"].astype(str)
    base = repo.csv("sequence_baselines/absolute_prediction_metrics.csv")

    rows = []
    for model, label, det in [
        ("composition", "Composition", True),
        ("kmer_ridge", "$k$-mer ridge", True),
        ("epi", "Context", False),
        ("cpgenie", "CpGenie", False),
        ("deepcpg", "DeepCpG", False),
        ("sequence", "Sequence", False),
        ("fusion", "Fusion", False),
    ]:
        if det:
            r = base[base["model"] == model].iloc[0]
            rows.append(dict(model=model, label=label, deterministic=True,
                             mae=[float(r["beta_mae"])], auc=[float(r["auc"])]))
        elif model in ("cpgenie", "deepcpg"):
            mae, auc = [], []
            for seed in (42, 43, 44):
                m = repo.jjson(f"published_baselines/{model}/seed{seed}/metrics.json")
                mae.append(float(m["beta_mae"]))
                auc.append(float(m.get("auc", m.get("roc_auc"))))
            rows.append(dict(model=model, label=label, deterministic=False, mae=mae, auc=auc))
        else:
            sub = seeded[seeded["Model"] == model]
            rows.append(dict(model=model, label=label, deterministic=False,
                             mae=sub["beta_mae"].astype(float).tolist(),
                             auc=sub["roc_auc"].astype(float).tolist()))

    ax_mae = fig.add_subplot(gs[0, 0])
    ax_auc = fig.add_subplot(gs[0, 1])
    ys = np.arange(len(rows))[::-1]
    for ax, key, xlabel, xlim in [(ax_mae, "mae", r"$\beta$ MAE", (0.07, 0.21)),
                                  (ax_auc, "auc", "ROC-AUC", (0.86, 1.0))]:
        for y, r in zip(ys, rows):
            colour, marker = MODEL_STYLE[r["model"]]
            vals = r[key]
            if not r["deterministic"]:
                ax.plot(vals, [y + 0.26] * len(vals), marker, ms=2.4, color=colour,
                        mfc="white", mec=colour, mew=0.7, ls="none", zorder=2)
            ax.plot([float(np.mean(vals))], [y], marker=marker, ms=5.0, color=colour,
                    mfc=colour if not r["deterministic"] else "white",
                    mec=colour, mew=1.0, zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([r["label"] for r in rows], fontsize=7)
        ax.set_ylim(-0.7, len(rows) - 0.1)
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)
        ax.grid(axis="x", color="#E6EAED", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    panel_letter(ax_mae, "a")
    panel_title(ax_mae, "Held-out accuracy (26,570 CpGs)")
    panel_title(ax_auc, "Held-out state discrimination")
    ax_auc.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=2.4, mfc="white", mec=TEXT, mew=0.7, label="individual seed"),
        Line2D([], [], marker="o", ls="none", ms=5.0, color=TEXT, label="seed mean"),
        Line2D([], [], marker="o", ls="none", ms=5.0, mfc="white", mec=TEXT, mew=1.0, label="single ridge fit"),
    ], loc="lower left", frameon=False, fontsize=5.8, handletextpad=0.4, borderpad=0.1, labelspacing=0.18)
    record("fig2a_heldout_accuracy", pd.DataFrame([
        dict(panel="2a", model=r["model"], panel_label=r["label"].replace("$", ""),
             deterministic=r["deterministic"], n_test_cpgs=26570,
             seeds="42;43;44" if not r["deterministic"] else "single fit",
             beta_mae_values=";".join(f"{v:.6f}" for v in r["mae"]),
             beta_mae_mean=float(np.mean(r["mae"])),
             roc_auc_values=";".join(f"{v:.6f}" for v in r["auc"]),
             roc_auc_mean=float(np.mean(r["auc"])),
             uncertainty="individual seeds shown; no bootstrap interval") for r in rows]))

    ax_b = fig.add_subplot(gs[1, 0])
    folds = []
    for f in (1, 2, 3):
        fu = repo.ajson(f"fold{f}/fusion/metrics.json")
        sq = repo.jjson(f"folds/fold{f}/sequence/metrics.json")
        folds.append(dict(fold=f, n=int(fu["n_test_loci"]),
                          fusion=float(fu["beta_mae"]), sequence=float(sq["beta_mae"])))
    f42 = seeded[(seeded["Model"] == "fusion") & (seeded["Seed"] == "42")].iloc[0]
    s42 = seeded[(seeded["Model"] == "sequence") & (seeded["Seed"] == "42")].iloc[0]
    folds.insert(0, dict(fold=0, n=26570, fusion=float(f42["beta_mae"]), sequence=float(s42["beta_mae"])))
    labels = ["chr8-9\n(primary)", "fold 1", "fold 2", "fold 3"]
    ys = np.arange(len(folds))[::-1]
    for y, fd, lab in zip(ys, folds, labels):
        ax_b.plot([fd["fusion"], fd["sequence"]], [y, y], color="#C8D2D8", lw=1.0, zorder=1)
        ax_b.plot([fd["sequence"]], [y], "o", ms=4.2, color=SEQUENCE, zorder=3)
        ax_b.plot([fd["fusion"]], [y], "D", ms=4.2, color=FUSION, zorder=3)
        ax_b.annotate(f"n={fd['n']:,}", (fd["sequence"], y), textcoords="offset points",
                      xytext=(6, 0), ha="left", va="center", fontsize=5.8, color=OTHER)
    ax_b.set_yticks(ys)
    ax_b.set_yticklabels(labels, fontsize=6.8)
    ax_b.set_ylim(-0.75, len(folds) - 0.05)
    ax_b.set_xlim(0.080, 0.121)
    ax_b.set_xticks([0.08, 0.09, 0.10, 0.11])
    ax_b.set_xlabel(r"$\beta$ MAE (seed 42)")
    ax_b.grid(axis="x", color="#E6EAED", lw=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    top_y = ys[0]
    ax_b.annotate("fusion", (folds[0]["fusion"], top_y), textcoords="offset points",
                  xytext=(0, 8), ha="center", va="bottom", fontsize=6.2, color=FUSION)
    ax_b.annotate("sequence", (folds[0]["sequence"], top_y), textcoords="offset points",
                  xytext=(0, 8), ha="center", va="bottom", fontsize=6.2, color=SEQUENCE)
    panel_letter(ax_b, "b")
    panel_title(ax_b, "Repeated chromosome splits")
    record("fig2b_chromosome_splits", pd.DataFrame([
        dict(panel="2b", held_out_set=lab.replace("\n", " "), fold=fd["fold"], seed=42, n_test_cpgs=fd["n"],
             sequence_beta_mae=fd["sequence"], fusion_beta_mae=fd["fusion"],
             fusion_minus_sequence=fd["fusion"] - fd["sequence"],
             uncertainty="single seed-42 fit per fold; no interval")
        for fd, lab in zip(folds, labels)]))

    ax_d = fig.add_subplot(gs[1, 1])
    dec = fig2_transfer_panel(repo)
    x = dec["decile"].astype(int).values
    for key, colour, marker, lab in [("mae_sequence", SEQUENCE, "o", "sequence"),
                                     ("mae_fusion", FUSION, "D", "fusion")]:
        vals = dec[key].astype(float).values
        ci = np.array([list(v) for v in dec[f"{key}_95ci"]], dtype=float)
        ax_d.plot(x, vals, "-", color=colour, lw=1.0, zorder=3)
        ax_d.plot(x, vals, marker, ms=3.0, color=colour, zorder=4, label=lab)
        ax_d.fill_between(x, ci[:, 0], ci[:, 1], color=colour, alpha=0.16, lw=0, zorder=2)
    ax_d.set_xticks(x)
    ax_d.set_xlabel("measured cross-tissue variance decile")
    ax_d.set_ylabel(r"$\beta$ MAE")
    ax_d.set_ylim(0, 0.235)
    ax_d.grid(axis="y", color="#E6EAED", lw=0.5, zorder=0)
    ax_d.set_axisbelow(True)
    ax_d.legend(loc="upper left", frameon=False, fontsize=6.2, handletextpad=0.4,
                borderpad=0.1, labelspacing=0.18)
    panel_letter(ax_d, "d")
    panel_title(ax_d, "Transfer error by measured variance")
    per_probe = pd.read_csv(repo.j / "joint" / "transfer_failure" / "transfer_failure_per_probe.csv")
    per_probe = per_probe.assign(
        decile=pd.qcut(per_probe["cross_tissue_var"].rank(method="first"), 10, labels=False) + 1)
    worst = per_probe[per_probe["decile"] == 10]["cgi_class"].value_counts(normalize=True)
    allp = per_probe["cgi_class"].value_counts(normalize=True)
    note = (f"decile 10: shore {100 * worst.get('Shore', 0):.0f}%, island {100 * worst.get('Island', 0):.0f}%\n"
            f"all probes: shore {100 * allp.get('Shore', 0):.0f}%, island {100 * allp.get('Island', 0):.0f}%")
    ax_d.text(0.97, 0.03, note, transform=ax_d.transAxes, fontsize=5.6, ha="right", va="bottom",
              color=OTHER, linespacing=1.3)
    record("fig2d_transfer_error", dec.assign(panel="2d", n_probes=dec["n"],
                                              uncertainty="400-draw 1 Mb block bootstrap"))

    ax_c = fig.add_subplot(gs[2, :])
    gain = repo.acsv("fusion_gain_stratified/fusion_gain_stratified.csv")
    wanted = [("CpG_Island_Context", "CpG class"),
              ("Ref_ATAC_Signal_Stratum", "ATAC"),
              ("Ref_H3K27ac_Signal_Stratum", "H3K27ac"),
              ("Ref_H3K27me3_Signal_Stratum", "H3K27me3")]
    wanted = [w for w in wanted if w[0] in set(gain["Grouping"])]
    xpos, ticks, tick_labels, seps, used, spans = 0.0, [], [], [], [], []
    for gi, (grouping, short) in enumerate(wanted):
        sub = gain[gain["Grouping"] == grouping]
        start = xpos
        for _, r in sub.iterrows():
            used.append((xpos, r, short))
            ticks.append(xpos)
            tick_labels.append(str(r["Stratum"]))
            xpos += 1
        spans.append((start, xpos - 1, short))
        if gi < len(wanted) - 1:
            seps.append(xpos - 0.5)
            xpos += 0.5
    for xx, r, short in used:
        val = 100 * float(r["Relative_Beta_MAE_Reduction"])
        lo = 100 * float(r["Relative_Reduction_CI_Low"])
        hi = 100 * float(r["Relative_Reduction_CI_High"])
        ax_c.plot([xx, xx], [lo, hi], color=FUSION, lw=1.0, zorder=2)
        for bnd in (lo, hi):
            ax_c.plot([xx - 0.12, xx + 0.12], [bnd, bnd], color=FUSION, lw=0.8, zorder=2)
        ax_c.plot([xx], [val], "D", ms=3.4, color=FUSION, zorder=3)
        ax_c.annotate(f"{int(r['N_CpGs']):,}", (xx, hi), textcoords="offset points",
                      xytext=(0, 3), ha="center", va="bottom", fontsize=5.0, color=OTHER)
    for s in seps:
        ax_c.axvline(s, color="#DDE3E7", lw=0.6, zorder=0)
    ax_c.axhline(0, color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
    ax_c.set_xticks(ticks)
    ax_c.set_xticklabels(tick_labels, fontsize=6.0)
    ax_c.set_xlim(-0.7, xpos - 0.8)
    ax_c.set_ylim(0, 30)
    ax_c.set_yticks([0, 10, 20, 30])
    ax_c.set_ylabel(r"relative $\beta$ MAE" "\n" "reduction (%)", fontsize=7)
    ax_c.grid(axis="y", color="#E6EAED", lw=0.5, zorder=0)
    ax_c.set_axisbelow(True)
    for a, b, short in spans:
        ax_c.annotate(short, ((a + b) / 2, -0.30), xycoords=("data", "axes fraction"),
                      ha="center", va="top", fontsize=7, color=TEXT, fontweight="bold")
    panel_letter(ax_c, "c")
    panel_title(ax_c, "Relative context benefit by stratum (n above each interval)")
    record("fig2c_relative_context_benefit", pd.DataFrame([
        dict(panel="2c", grouping=r["Grouping"], stratum=r["Stratum"], n_cpgs=int(r["N_CpGs"]),
             n_genomic_blocks=int(r["N_Genomic_Blocks"]),
             sequence_beta_mae=float(r["Sequence_Beta_MAE"]), fusion_beta_mae=float(r["Fusion_Beta_MAE"]),
             relative_reduction_pct=100 * float(r["Relative_Beta_MAE_Reduction"]),
             ci_low_pct=100 * float(r["Relative_Reduction_CI_Low"]),
             ci_high_pct=100 * float(r["Relative_Reduction_CI_High"]),
             uncertainty="1 Mb block bootstrap")
        for _, r, _ in used]))

    fig.savefig(out / "fig2_prediction_transfer.pdf")
    plt.close(fig)

def fig3(repo: Repo, out: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 145 * MM))
    gs = fig.add_gridspec(3, 2, left=0.175, right=0.955, top=0.935, bottom=0.075,
                          wspace=0.70, hspace=0.85, height_ratios=[0.8, 1.0, 1.0])

    syn = repo.acsv("variant_effect_synthesis/all_strata.csv")
    sig = syn[(syn["stratum"] == "significant") & (syn["model"] == "fusion")]
    cohorts = [("eGTEx", "eGTEx breast"), ("GENOA", "GENOA blood")]

    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    a_rows = []
    for ax, metric, xlabel, null in [(ax_a1, "signed_rho", r"signed Spearman $\rho$", 0.0),
                                     (ax_a2, "direction_agreement", "direction agreement", 0.5)]:
        rws = []
        for key, lab in cohorts:
            r = sig[(sig["cohort"] == key) & (sig["metric"] == metric)].iloc[0]
            rws.append(dict(label=f"{lab}\nn={int(r['n']):,}", value=float(r["value"]),
                            lo=float(r["ci_low"]), hi=float(r["ci_high"]),
                            colour=FUSION, marker="D"))
            a_rows.append(dict(panel="3a", cohort=key, model="fusion", metric=metric,
                               n_pairs=int(r["n"]), n_blocks=int(r["n_blocks"]),
                               value=float(r["value"]), ci_low=float(r["ci_low"]),
                               ci_high=float(r["ci_high"]),
                               source="ABL/variant_effect_synthesis/all_strata.csv (significant, non-CpG-altering)",
                               uncertainty="1 Mb block bootstrap"))
        forest(ax, rws, xlabel, null_line=null)
        ax.tick_params(axis="y", labelsize=6.2)
    panel_letter(ax_a1, "a", xoff=-58)
    panel_title(ax_a1, "Signed cohort agreement")
    panel_title(ax_a2, "Direction agreement")
    record("fig3a_cohort_agreement", pd.DataFrame(a_rows))

    ax_b = fig.add_subplot(gs[1, 0])
    prim = repo.acsv("genoa_variant_evaluation/primary_metrics.csv")
    matched = repo.acsv("genoa_variant_evaluation/matched_negative_auroc.csv")
    bal = repo.acsv("genoa_variant_evaluation/matching_balance.csv")
    base_prim = pd.read_csv(repo.root / "repro_check" / "baseline_variant_evaluation" / "primary_metrics.csv")
    base_matched = pd.read_csv(repo.root / "repro_check" / "baseline_variant_evaluation" / "matched_negative_auroc.csv")

    def ensemble(df, **eq):
        sub = df.copy()
        for k, v in eq.items():
            sub = sub[sub[k] == v]
        sub = sub[sub["seed"] == -1]
        return sub.iloc[0] if len(sub) else None

    b_spec = [("fusion", "Fusion", prim, matched), ("sequence", "Sequence", prim, matched),
              ("kmer_ridge", "$k$-mer ridge", base_prim, base_matched),
              ("composition", "Composition", base_prim, base_matched)]
    b_src, b_rows = [], []
    for model, lab, pdf_, mdf in b_spec:
        um = ensemble(pdf_, model=model, variant_class="non_cpg_altering", metric="auroc_marginal")
        mt = ensemble(mdf, model=model, metric="auroc_distance_matched")
        if um is None or mt is None:
            continue
        dup = int((mdf[(mdf["model"] == model) & (mdf["seed"] == -1)]).shape[0])
        colour, marker = MODEL_STYLE[model]
        b_rows.append(dict(label=lab, colour=colour, marker=marker,
                           unmatched=float(um["value"]), u_lo=float(um["ci_low"]), u_hi=float(um["ci_high"]),
                           matched=float(mt["value"]), m_lo=float(mt["ci_low"]), m_hi=float(mt["ci_high"])))
        for kind, r in [("unmatched", um), ("distance-matched", mt)]:
            b_src.append(dict(panel="3b", cohort="GENOA", model=model, comparison=kind,
                              seed="ensemble (seed -1)", n=int(r["n"]), n_blocks=int(r["n_blocks"]),
                              value=float(r["value"]), ci_low=float(r["ci_low"]), ci_high=float(r["ci_high"]),
                              duplicate_ensemble_rows_in_export=dup,
                              source=("ABL/genoa_variant_evaluation" if model in ("fusion", "sequence")
                                      else "repro_check/baseline_variant_evaluation (clean rerun, 19 Sep 2026)"),
                              uncertainty="2,000 draws over 1 Mb genomic blocks"))
    dist_um = ensemble(prim, model="fusion", variant_class="non_cpg_altering",
                       metric="auroc_distance_only_baseline")
    dist_mt = float(bal["distance_only_auroc_after_matching"].iloc[0])
    b_rows.append(dict(label="Distance only", colour=NULLC, marker="o",
                       unmatched=float(dist_um["value"]), u_lo=float(dist_um["ci_low"]),
                       u_hi=float(dist_um["ci_high"]), matched=dist_mt, m_lo=None, m_hi=None))
    b_src.append(dict(panel="3b", cohort="GENOA", model="distance_only", comparison="unmatched",
                      seed="ensemble (seed -1)", n=int(dist_um["n"]), n_blocks=int(dist_um["n_blocks"]),
                      value=float(dist_um["value"]), ci_low=float(dist_um["ci_low"]),
                      ci_high=float(dist_um["ci_high"]), duplicate_ensemble_rows_in_export=1,
                      source="ABL/genoa_variant_evaluation/primary_metrics.csv",
                      uncertainty="2,000 draws over 1 Mb genomic blocks"))
    b_src.append(dict(panel="3b", cohort="GENOA", model="distance_only",
                      comparison="realized after matching", seed="ensemble (seed -1)",
                      n=int(bal["matched_null_n"].iloc[0]), n_blocks=None, value=dist_mt,
                      ci_low=None, ci_high=None, duplicate_ensemble_rows_in_export=1,
                      source="ABL/genoa_variant_evaluation/matching_balance.csv",
                      uncertainty="realized balance statistic, not a bootstrap estimate"))

    ys = np.arange(len(b_rows))[::-1]
    for y, r in zip(ys, b_rows):
        ax_b.plot([r["unmatched"], r["matched"]], [y + 0.16, y - 0.16], color="#C8D2D8", lw=0.8, zorder=1)
        for val, lo, hi, dy, filled in [(r["unmatched"], r["u_lo"], r["u_hi"], 0.16, False),
                                        (r["matched"], r["m_lo"], r["m_hi"], -0.16, True)]:
            if lo is not None and hi is not None:
                ax_b.plot([lo, hi], [y + dy, y + dy], color=r["colour"], lw=1.0, zorder=2)
                for bnd in (lo, hi):
                    ax_b.plot([bnd, bnd], [y + dy - 0.09, y + dy + 0.09], color=r["colour"], lw=0.8, zorder=2)
            ax_b.plot([val], [y + dy], marker=r["marker"], ms=4.0, color=r["colour"],
                      mfc=r["colour"] if filled else "white", mec=r["colour"], mew=0.9, zorder=3)
    ax_b.axvline(0.5, color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
    ax_b.set_yticks(ys)
    ax_b.set_yticklabels([r["label"] for r in b_rows], fontsize=6.8)
    ax_b.set_ylim(-0.7, len(b_rows) - 0.3)
    ax_b.set_xlim(0.44, 0.65)
    ax_b.set_xlabel("AUROC (significant vs null pairs)")
    ax_b.grid(axis="x", color="#E6EAED", lw=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.legend(handles=[
        Line2D([], [], marker="s", ls="none", ms=4.0, mfc="white", mec=TEXT, mew=0.9, label="unmatched"),
        Line2D([], [], marker="s", ls="none", ms=4.0, color=TEXT, label="distance-matched"),
    ], loc="upper left", frameon=False, fontsize=5.8, handletextpad=0.4, borderpad=0.0, labelspacing=0.18)
    panel_letter(ax_b, "b", xoff=-58)
    panel_title(ax_b, "Distance controls")
    record("fig3b_distance_control", pd.DataFrame(b_src))

    sub_c = gs[1, 1].subgridspec(2, 1, hspace=0.50)
    ax_c1 = fig.add_subplot(sub_c[0])
    ax_c2 = fig.add_subplot(sub_c[1])
    grad = repo.acsv("genoa_variant_evaluation/significance_gradient.csv")
    grad = grad[grad["model"] == "fusion"]
    strata = list(dict.fromkeys(grad["stratum"].tolist()))
    c_src = []
    for ax, metric, ylabel, null in [(ax_c1, "signed_rho", r"signed $\rho$", 0.0),
                                     (ax_c2, "direction_agreement", "direction agr.", 0.5)]:
        xs, vals, los, his, ns = [], [], [], [], []
        for i, s in enumerate(strata):
            sub = grad[(grad["stratum"] == s) & (grad["metric"] == metric)]
            if not len(sub):
                continue
            r = sub.iloc[0]
            xs.append(i)
            vals.append(float(r["value"]))
            los.append(float(r["ci_low"]))
            his.append(float(r["ci_high"]))
            ns.append(int(r["n"]))
            c_src.append(dict(panel="3c", model="fusion", stratum=s, metric=metric,
                              n_pairs=int(r["n"]), n_blocks=int(r["n_blocks"]),
                              value=float(r["value"]), ci_low=float(r["ci_low"]),
                              ci_high=float(r["ci_high"]),
                              source="ABL/genoa_variant_evaluation/significance_gradient.csv",
                              uncertainty="1 Mb block bootstrap"))
        ax.errorbar(xs, vals, yerr=[np.array(vals) - np.array(los), np.array(his) - np.array(vals)],
                    fmt="D", ms=3.0, color=FUSION, ecolor=FUSION, elinewidth=0.9,
                    capsize=1.6, capthick=0.7, zorder=3)
        ax.axhline(null, color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
        ax.set_xticks(range(len(strata)))
        ax.set_ylabel(ylabel, fontsize=6.8)
        ax.grid(axis="y", color="#E6EAED", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.6, len(strata) - 0.4)
        ax.tick_params(axis="y", labelsize=6.2)
        if ax is ax_c1:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels(strata, fontsize=5.2, rotation=38, ha="right")
            for xx, n in zip(xs, ns):
                ax.annotate(f"{n:,}", (xx, 1.0), xycoords=("data", "axes fraction"),
                            textcoords="offset points", xytext=(0, 2.5), ha="center",
                            va="bottom", fontsize=4.6, color=OTHER)
    panel_letter(ax_c1, "c", xoff=-40)
    panel_title(ax_c1, "Agreement by association strength")
    ax_c2.set_xlabel("GENOA association p-value stratum", fontsize=6.5, labelpad=1.5)
    record("fig3c_association_strength", pd.DataFrame(c_src))

    ax_d = fig.add_subplot(gs[2, 0])
    ros = repo.csv("asm_validation/asm_discrimination.csv")
    tyc1 = repo.csv("asm_validation_tycko/tycko_e1_discrimination.csv")
    tyc2 = repo.csv("asm_validation_tycko/tycko_e2_signed_agreement.csv")
    d_rows, d_src = [], []
    for df, name, stratum in [(ros, "Rosenski", "positive_vs_bimodal_non_asm | all tissues"),
                              (tyc1, "Do-Tycko", "all tissues")]:
        sub = df[df["Stratum"] == stratum]
        auroc_col = "auroc" if "auroc" in df.columns else "AUROC_Detection"
        lo_col = "ci_low" if "ci_low" in df.columns else "CI_Low"
        hi_col = "ci_high" if "ci_high" in df.columns else "CI_High"
        nb = "n_blocks" if "n_blocks" in df.columns else "N_Genomic_Blocks"
        for model, lab in [("fusion", "fusion"), ("sequence", "sequence"),
                           ("distance_only_baseline", "distance only")]:
            row = sub[sub["Score"] == model]
            if not len(row):
                continue
            r = row.iloc[0]
            colour, marker = MODEL_STYLE.get(model, (NULLC, "o"))
            d_rows.append(dict(label=f"{name} {lab}", value=float(r[auroc_col]),
                               lo=float(r[lo_col]), hi=float(r[hi_col]), colour=colour,
                               marker=marker, mfc="white" if model == "distance_only_baseline" else colour))
            d_src.append(dict(panel="3d-left", catalogue=name, score=model,
                              endpoint="discrimination AUROC", n_rows=int(r["N_Pairs"]),
                              n_blocks=int(r[nb]), value=float(r[auroc_col]),
                              ci_low=float(r[lo_col]), ci_high=float(r[hi_col]), seed="42",
                              uncertainty="2,000 block resamples"))
    forest(ax_d, d_rows, "ASM discrimination AUROC", null_line=0.5, xlim=(0.46, 0.63))
    ax_d.tick_params(axis="y", labelsize=6.0)
    panel_letter(ax_d, "d", xoff=-58)
    panel_title(ax_d, "ASM catalogue discrimination")

    sub_d = gs[2, 1].subgridspec(2, 1, hspace=0.55)
    ax_d2 = fig.add_subplot(sub_d[0])
    ax_d3 = fig.add_subplot(sub_d[1])
    t = tyc2[tyc2["Stratum"] == "all tissues"]
    rows_d2, rows_d3 = [], []
    for _, r in t.iterrows():
        colour, marker = MODEL_STYLE[r["Arm"]]
        rows_d2.append(dict(label=str(r["Arm"]), value=float(r["Direction_Concordance"]),
                            lo=float(r["Direction_CI_Low"]), hi=float(r["Direction_CI_High"]),
                            colour=colour, marker=marker))
        rows_d3.append(dict(label=str(r["Arm"]), value=float(r["Signed_Spearman"]),
                            lo=float(r["Spearman_CI_Low"]), hi=float(r["Spearman_CI_High"]),
                            colour=colour, marker=marker))
        d_src.append(dict(panel="3d-right", catalogue="Do-Tycko", score=r["Arm"],
                          endpoint="direction concordance", n_rows=int(r["N_SNPs"]),
                          n_blocks=int(r["N_Genomic_Blocks"]),
                          value=float(r["Direction_Concordance"]),
                          ci_low=float(r["Direction_CI_Low"]), ci_high=float(r["Direction_CI_High"]),
                          seed="42", uncertainty="2,000 block resamples"))
        d_src.append(dict(panel="3d-right", catalogue="Do-Tycko", score=r["Arm"],
                          endpoint="signed Spearman", n_rows=int(r["N_SNPs"]),
                          n_blocks=int(r["N_Genomic_Blocks"]), value=float(r["Signed_Spearman"]),
                          ci_low=float(r["Spearman_CI_Low"]), ci_high=float(r["Spearman_CI_High"]),
                          seed="42", uncertainty="2,000 block resamples"))
    ax_d2.axvspan(0.60, 0.70, color=TINT, zorder=0)
    forest(ax_d2, rows_d2, "direction concordance", null_line=0.5, xlim=(0.50, 0.72))
    ax_d2.tick_params(axis="y", labelsize=6.2)
    ax_d2.annotate("recorded 0.60-0.70\nexpectation", xy=(0.685, 0.94),
                   xycoords=("data", "axes fraction"), ha="center", va="top",
                   fontsize=4.8, color=OTHER, linespacing=1.25)
    panel_title(ax_d2, "Do-Tycko signed endpoints")
    forest(ax_d3, rows_d3, r"signed Spearman $\rho$", null_line=0.0, xlim=(0.15, 0.35))
    ax_d3.tick_params(axis="y", labelsize=6.2)
    record("fig3d_asm", pd.DataFrame(d_src))

    fig.savefig(out / "fig3_external_validation.pdf")
    plt.close(fig)

def card(ax, x, y, w, h, title, lines, face=TINT, edge=CONTEXT, title_colour=TEXT):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=0.8,
                           joinstyle="round", zorder=2))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=6.8,
            fontweight="bold", color=title_colour, zorder=3)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.44 - 0.26 * i, ln, ha="center", va="top",
                fontsize=6.0, color=TEXT, zorder=3)


def fig4(repo: Repo, out: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 140 * MM))

    ax_b = fig.add_axes([0.030, 0.640, 0.955, 0.330])
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 3.0)
    ax_b.axis("off")
    steps = [
        ("1. REF baseline", ["REF sequence", "REF context", "REF gates"]),
        ("2. ALT sequence", ["ALT sequence", "REF context", "REF gates"]),
        ("3. + sequence gate", ["ALT sequence", "REF context", "ALT seq. gate"]),
        ("4. + context gate", ["ALT sequence", "REF context", "both ALT gates"]),
    ]
    w, gap = 2.0, 0.66
    for i, (title, lines) in enumerate(steps):
        x = i * (w + gap)
        card(ax_b, x, 0.90, w, 1.25, title, lines, face="white", edge=FUSION)
        ax_b.add_patch(Rectangle((x + 0.18, 1.11), w - 0.36, 0.22, facecolor=CONTEXT,
                                 edgecolor="none", alpha=0.32, zorder=3))
        ax_b.annotate("", xy=(x + w / 2, 0.85), xytext=(x + w / 2, 0.58),
                      arrowprops=dict(arrowstyle="<-", lw=0.8, color=OTHER))
        ax_b.text(x + w / 2, 0.46, "head $H$", ha="center", va="top", fontsize=6.2, color=OTHER)
        if i < 3:
            ax_b.annotate("", xy=(x + w + gap - 0.06, 1.52), xytext=(x + w + 0.06, 1.52),
                          arrowprops=dict(arrowstyle="->", lw=0.9, color=TEXT))
            lab = [r"$\Delta$sequence", r"$\Delta$sequence-gate", r"$\Delta$context-gate"][i]
            ax_b.text(x + w + gap / 2, 1.64, lab, ha="center", va="bottom", fontsize=6.0, color=TEXT)
    ax_b.plot([w / 2, 3 * (w + gap) + w / 2], [0.20, 0.20], color=NULLC, lw=0.8, zorder=1)
    for xx in (w / 2, 3 * (w + gap) + w / 2):
        ax_b.plot([xx, xx], [0.20, 0.30], color=NULLC, lw=0.8, zorder=1)
    ax_b.text((w / 2 + 3 * (w + gap) + w / 2) / 2, 0.06, r"sum $= \Delta$total", ha="center",
              va="bottom", fontsize=6.4, color=TEXT)
    ax_b.text(0.0, 2.92, "b", fontsize=10, fontweight="bold", va="top", ha="left", color=TEXT)
    ax_b.text(0.42, 2.92, "Counterfactual mechanism: one factor changes per step, same nonlinear head",
              fontsize=9, va="top", ha="left", color=TEXT)
    ax_b.text(0.42, 2.52, "ordered within-model decomposition; not unique causal attribution. "
                          "The context embedding (teal) is identical in all four cards; only its gate changes.",
              fontsize=6.0, va="top", ha="left", color=OTHER, style="italic")

    gs = fig.add_gridspec(2, 3, left=0.135, right=0.950, top=0.560, bottom=0.085,
                          wspace=0.72, hspace=0.95, width_ratios=[1.0, 1.0, 0.85])

    ax_a = fig.add_subplot(gs[0, 0])
    a_src, rows_a = [], []
    pf = repo.acsv("genoa_variant_evaluation/fusion_vs_sequence_paired.csv")
    for metric, lab in [("fusion_minus_sequence_signed_rho", r"signed $\rho$"),
                        ("fusion_minus_sequence_direction_agreement", "direction agr."),
                        ("fusion_minus_sequence_auroc_within_distance_bin", "matched AUROC")]:
        r = pf[pf["metric"] == metric].iloc[0]
        rows_a.append(dict(label=lab, value=float(r["value"]), lo=float(r["ci_low"]),
                           hi=float(r["ci_high"]), colour=FUSION, marker="D"))
        a_src.append(dict(panel="4a", cohort="GENOA", contrast="fusion minus sequence",
                          metric=metric, n=int(r["n"]), n_blocks=int(r["n_blocks"]),
                          value=float(r["value"]), ci_low=float(r["ci_low"]),
                          ci_high=float(r["ci_high"]), seed="42",
                          source="ABL/genoa_variant_evaluation/fusion_vs_sequence_paired.csv",
                          uncertainty="1 Mb block bootstrap"))
    forest(ax_a, rows_a, "fusion minus sequence", null_line=0.0)
    ax_a.tick_params(axis="y", labelsize=6.5)
    ax_a.set_xticks([-0.005, 0.0, 0.005])
    ax_a.tick_params(axis="x", labelsize=6.2)
    panel_letter(ax_a, "a", xoff=-56)
    panel_title(ax_a, "No clear added signal")
    record("fig4a_paired_differences", pd.DataFrame(a_src))

    ax_c1 = fig.add_subplot(gs[0, 1])
    ax_c2 = fig.add_subplot(gs[0, 2])
    gd = repo.ajson("gate_decomposition/gate_decomposition_summary.json")
    c_src, quantities = [], []
    for cohort in ("genoa", "egtex"):
        ins = gd["cohorts"][cohort]["instrumented"]
        gate = float(ins["variance_share_gate_channel"])
        dna = float(ins["variance_share_dna_channel"])
        gepi_of_gate = float(ins["gate_subchannels"]["variance_share_of_gate_channel_gepi"])
        quantities.append(dict(cohort={"genoa": "GENOA", "egtex": "eGTEx"}[cohort], n=int(ins["pairs"]),
                               gate_channel=gate, dna_channel=dna,
                               context_rescaling=gate * gepi_of_gate))
        for lab, val, note in [
            ("gate channel", gate, "Var(gate channel)/Var(total); covariance is part of Var(total)"),
            ("DNA channel", dna, "Var(DNA channel)/Var(total); covariance is part of Var(total)"),
            ("context rescaling (g_epi)", gate * gepi_of_gate,
             "gate-channel share x g_epi share of gate channel; the only route by which context content reaches the effect"),
        ]:
            c_src.append(dict(panel="4c-left", cohort={"genoa": "GENOA", "egtex": "eGTEx"}[cohort], quantity=lab,
                              n_pairs=int(ins["pairs"]), value=val, seed=42, note=note,
                              source="ABL/gate_decomposition/gate_decomposition_summary.json",
                              uncertainty="deterministic variance ratio; no interval exported"))
    labels_c = ["gate\nchannel", "DNA\nchannel", "context\nrescaling"]
    keys_c = ["gate_channel", "dna_channel", "context_rescaling"]
    width = 0.34
    xs = np.arange(len(labels_c))
    for i, (q, colour) in enumerate(zip(quantities, [FUSION, CONTEXT])):
        vals = [q[k] for k in keys_c]
        ax_c1.bar(xs + (i - 0.5) * width, vals, width, color=colour, edgecolor="none",
                  label=f"{q['cohort']}", zorder=3)
    ax_c1.set_xticks(xs)
    ax_c1.set_xticklabels(labels_c, fontsize=6.0, linespacing=1.2)
    ax_c1.set_ylabel("component / total\nvariance ratio", fontsize=7)
    ax_c1.set_ylim(0, 0.64)
    ax_c1.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax_c1.tick_params(axis="y", labelsize=6.2)
    ax_c1.legend(frameon=False, fontsize=5.8, loc="upper left", handletextpad=0.4,
                 borderpad=0.0, labelspacing=0.15, handlelength=1.1)
    ax_c1.grid(axis="y", color="#E6EAED", lw=0.5, zorder=0)
    ax_c1.set_axisbelow(True)
    panel_letter(ax_c1, "c", xoff=-42)
    panel_title(ax_c1, "Variance ratios (unstacked)")

    partial_rows = []
    for cohort, colour in [("genoa", FUSION), ("egtex", CONTEXT)]:
        sa = gd["cohorts"][cohort]["instrumented"]["signal_attribution"]
        val = float(sa["partial_spearman_gate_given_dna"])
        ci = sa["partial_spearman_gate_given_dna_95ci"]
        name = {"genoa": "GENOA", "egtex": "eGTEx"}[cohort]
        partial_rows.append(dict(label=name, value=val, lo=float(ci[0]), hi=float(ci[1]),
                                 colour=colour, marker="D"))
        c_src.append(dict(panel="4c-right", cohort=name,
                          quantity="partial Spearman (gate channel | DNA channel)",
                          n_pairs=int(sa["n"]), value=val, seed=42,
                          note="marginal gate %.6f, marginal DNA %.6f, measured column %s" % (
                              sa["marginal_spearman_gate_vs_measured"],
                              sa["marginal_spearman_dna_vs_measured"],
                              sa["measured_effect_column"]),
                          ci_low=float(ci[0]), ci_high=float(ci[1]),
                          source="ABL/gate_decomposition/gate_decomposition_summary.json",
                          uncertainty="1 Mb genomic-block bootstrap"))
    forest(ax_c2, partial_rows, "partial Spearman", null_line=0.0, xlim=(-0.002, 0.021))
    ax_c2.tick_params(axis="y", labelsize=6.5)
    ax_c2.set_xticks([0.0, 0.01, 0.02])
    ax_c2.tick_params(axis="x", labelsize=6.2)
    panel_title(ax_c2, "Gate | sequence")
    record("fig4c_variance_and_information", pd.DataFrame(c_src))

    ax_d = fig.add_subplot(gs[1, :])
    perm = repo.acsv("context_permutation/agreement_with_identity.csv")
    perm = perm.assign(norm_dev=perm["mae"] / perm["sd_reference"])
    schemes = [("shuffle", "shuffled context"), ("median", "median context")]
    quants = [("absolute methylation (WT_M)", "levels", SEQUENCE, "o"),
              ("variant effect (Delta_M)", "effects", FUSION, "D")]
    d_src = []
    xs = np.arange(len(schemes), dtype=float)
    for si, (scheme, _) in enumerate(schemes):
        pair = []
        for qname, qlab, colour, marker in quants:
            row = perm[(perm["scheme"] == scheme) & (perm["quantity"] == qname)]
            if not len(row):
                continue
            r = row.iloc[0]
            v = float(r["norm_dev"])
            pair.append(v)
            d_src.append(dict(panel="4d", scheme=scheme, quantity=qlab, n_pairs=int(r["n"]),
                              mae_vs_unperturbed=float(r["mae"]), sd_reference=float(r["sd_reference"]),
                              normalized_deviation=v, spearman=float(r["spearman"]),
                              pearson=float(r["pearson"]), sign_agreement=float(r["sign_agreement"]),
                              seed="42",
                              source="ABL/context_permutation/agreement_with_identity.csv",
                              uncertainty="deterministic summary; no interval"))
        if len(pair) == 2:
            ax_d.plot([xs[si] - 0.09, xs[si] + 0.09], pair, color="#C8D2D8", lw=0.9, zorder=2)
        for qi, (v, (_, qlab, colour, marker)) in enumerate(zip(pair, quants)):
            ax_d.plot([xs[si] + (qi - 0.5) * 0.18], [v], marker, ms=5.4, color=colour,
                      ls="none", zorder=3, label=qlab if si == 0 else None)
            ax_d.annotate(f"{v:.4f}", (xs[si] + (qi - 0.5) * 0.18, v), textcoords="offset points",
                          xytext=(0, 7), ha="center", fontsize=6.2, color=colour)
    ax_d.set_xticks(xs)
    ax_d.set_xticklabels([lab for _, lab in schemes], fontsize=7.5)
    ax_d.set_ylabel("deviation / original SD", fontsize=7.5)
    ax_d.set_xlim(-0.5, len(schemes) - 0.5)
    ax_d.set_ylim(0, float(perm["norm_dev"].max()) * 1.32)
    ax_d.grid(axis="y", color="#E6EAED", lw=0.5, zorder=0)
    ax_d.set_axisbelow(True)
    ax_d.legend(frameon=False, fontsize=6.5, loc="upper right", handletextpad=0.4,
                borderpad=0.0, labelspacing=0.2)
    panel_letter(ax_d, "d", xoff=-56)
    panel_title(ax_d, "Context perturbation, current breast-epithelium context (seed 42, n = 76,893)")
    record("fig4d_input_perturbation", pd.DataFrame(d_src))

    fig.savefig(out / "fig4_context_mechanism.pdf")
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
    ax_a.plot([-58, 58], [0.56, 0.56], color=OTHER, lw=1.2, solid_capstyle="round", zorder=2)
    for tick in range(-50, 51, 25):
        ax_a.plot([tick, tick], [0.53, 0.59], color=OTHER, lw=0.7, zorder=2)
        ax_a.text(tick, 0.47, f"{tick:+d}" if tick else "0", ha="center", va="top",
                  fontsize=6.0, color=OTHER)
    ax_a.text(0, 0.355, "bp relative to cg20699548 (hg38)", ha="center", va="top", fontsize=6.2, color=OTHER)
    ax_a.plot([0], [0.56], "o", ms=6.5, color=CONTEXT, mec="white", mew=0.8, zorder=4)
    ax_a.text(0, 0.68, "cg20699548", ha="center", va="bottom", fontsize=6.4, color=CONTEXT)
    ax_a.plot([offset], [0.56], "v", ms=6.0, color=HIGHLIGHT, zorder=4)
    ax_a.annotate(f"chr8:g.70148394G>A\n{abs(offset)} bp from CpG",
                  xy=(offset, 0.56), xytext=(offset - 6, 0.90), fontsize=6.2, color=HIGHLIGHT,
                  ha="right", va="top", linespacing=1.3,
                  arrowprops=dict(arrowstyle="-", lw=0.7, color=HIGHLIGHT))
    ax_a.add_patch(Rectangle((14, 0.155), 44, 0.09, facecolor=TINT, edgecolor=CONTEXT,
                             lw=0.6, zorder=3))
    ax_a.text(36, 0.20, "ATAC Q4  $\\cdot$  H3K27ac Q4", ha="center", va="center",
              fontsize=5.8, color=TEXT, zorder=4)
    ax_a.text(-58, 0.20, "NCOA2 gene body, OpenSea",
              fontsize=6.0, color=TEXT, ha="left", va="center")
    ax_a.text(-58, 0.10, "transcript ENST00000452400.7 ($-$ strand)",
              fontsize=6.0, color=TEXT, ha="left", va="center")
    ax_a.text(-58, 0.00, f"current-context $\\Delta\\hat\\beta$ = {delta:.4f}  (seed 42, held-out test split)",
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
    ax_b.step(vals, ecdf, where="post", color=OTHER, lw=1.1, zorder=3)
    ax_b.plot(vals, np.zeros_like(vals) - 0.035, "|", ms=4, color=OTHER, mew=0.6, zorder=3)
    ax_b.axvline(abs(delta), color=HIGHLIGHT, lw=1.1, zorder=4)
    ax_b.annotate(f"NCOA2 |$\\Delta\\hat\\beta$| = {abs(delta):.4f}", (abs(delta), 0.52),
                  textcoords="offset points", xytext=(-4, 0), ha="center", va="center",
                  fontsize=6.0, color=HIGHLIGHT, rotation=90)
    ax_b.set_xlabel(r"|$\Delta\hat\beta$| of matched synonymous comparators")
    ax_b.set_ylabel("cumulative fraction")
    ax_b.set_ylim(-0.08, 1.05)
    ax_b.set_xlim(-0.006, abs(delta) * 1.13)
    ax_b.grid(color="#E6EAED", lw=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    exceed = int((vals >= abs(delta)).sum())
    ax_b.text(0.33, 0.62, f"n = {len(vals)} matched comparators\n"
                          f"{exceed} at or above the candidate\n"
                          "tier T4: SBS6, CpG effect, $\\leq$100 bp",
              transform=ax_b.transAxes, ha="left", va="top", fontsize=5.2, color=OTHER, linespacing=1.45)
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
    ax_c.hist(d, bins=40, color=NULLC, edgecolor="white", lw=0.3, zorder=2)
    stk = repo.acsv("literature_variant_screen/stk11_case_study_figure_values.csv")
    stk = stk[stk["Record"] == "Highlighted target"]
    styles = [(HIGHLIGHT, "-"), (SEQUENCE, (0, (4, 2)))]
    c_src = []
    span = max(abs(d.min()), abs(d.max()), 0.10)
    ax_c.set_xlim(-span * 1.35, span * 1.35)
    ax_c.set_ylim(0, 300)
    for i, (_, r) in enumerate(stk.iterrows()):
        v = float(r["Predicted_Delta_Beta"])
        colour, ls = styles[i]
        ax_c.axvline(v, color=colour, ls=ls, lw=1.1, zorder=4)
        ax_c.annotate(f"{r['Variant_ID']}\n{r['probeID']}\n{v:+.4f}",
                      (v, 268 if i == 0 else 196),
                      textcoords="offset points", xytext=(5 if v > 0 else -5, 0),
                      ha="left" if v > 0 else "right", va="top", fontsize=5.4,
                      color=colour, linespacing=1.35)
        c_src.append(dict(panel="5c", variant_id=r["Variant_ID"], probe=r["probeID"],
                          predicted_delta_beta=v, seed=42, status="training probe",
                          source="ABL/literature_variant_screen/stk11_case_study_figure_values.csv"))
    ax_c.set_xlabel(r"signed predicted $\Delta\hat\beta$")
    ax_c.set_ylabel("variants", fontsize=7.5)
    ax_c.grid(axis="y", color="#E6EAED", lw=0.5, zorder=0)
    ax_c.set_axisbelow(True)
    panel_letter(ax_c, "c")
    panel_title(ax_c, f"STK11 training-probe examples (n = {len(d)} scored)")
    record("fig5c_stk11", pd.concat([
        pd.DataFrame(c_src),
        pd.DataFrame([dict(panel="5c", variant_id="background pool", probe="",
                           predicted_delta_beta=np.nan, seed=42,
                           status=f"{len(d)} scored nonsynonymous variant-CpG pairs",
                           source="ABL/literature_variant_screen/literature_variant_predictions_ranked.csv")])],
        ignore_index=True))

    ax_d = fig.add_subplot(gs[1, 1])
    d_rows, d_src = [], []
    for cohort, label, colour in [("genoa", "GENOA", FUSION), ("egtex", "eGTEx", CONTEXT)]:
        summary = repo.rjson(f"repro_check/gwas_nominal_current/{cohort}/run_summary.json")
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
                          source=f"repro_check/gwas_nominal_current/{cohort}/run_summary.json",
                          uncertainty=f"{summary['n_boot']} block bootstrap draws, "
                                      f"{summary['block_size_bp']} bp blocks"))
    forest(ax_d, d_rows, "GWAS fraction among top 5% predicted effects",
           null_line=0.5, xlim=(0.22, 0.80))
    ax_d.tick_params(axis="y", labelsize=6.2)
    ax_d.annotate("matched null 0.5", (0.5, 0.02), xycoords=("data", "axes fraction"),
                  textcoords="offset points", xytext=(3, 0), ha="left", va="bottom",
                  fontsize=5.4, color=OTHER)
    panel_letter(ax_d, "d")
    panel_title(ax_d, "Clinical annotation control")
    record("fig5d_gwas_control", pd.DataFrame(d_src))

    fig.savefig(out / "fig5_variant_prioritization.pdf")
    plt.close(fig)


def figs1(repo: Repo, out: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 110 * MM))
    gs = fig.add_gridspec(2, 2, left=0.235, right=0.975, top=0.915, bottom=0.135,
                          wspace=0.46, hspace=0.70)

    est = repo.acsv("rc_uncertainty/estimator_comparison.csv")
    corr = repo.acsv("rc_uncertainty/disagreement_error_correlations.csv")
    curves = repo.acsv("rc_uncertainty/selective_prediction_curves.csv")

    ax_a = fig.add_subplot(gs[0, 0])
    order = ["rc_disagreement", "cross_seed_sd", "boundary_distance", "combined"]
    names = {"rc_disagreement": "RC disagr.", "cross_seed_sd": "seed SD",
             "boundary_distance": "boundary", "combined": "combined"}
    rows = []
    a_src = []
    for model, colour, marker in [("sequence", SEQUENCE, "o"), ("fusion", FUSION, "D")]:
        for e in order:
            sub = est[(est["model"] == model) & (est["estimator"] == e) & (est["seed"] == 42)]
            if not len(sub):
                continue
            r = sub.iloc[0]
            rows.append(dict(label=f"{names[e]} - {model}", value=float(r["aurc_beta_mae"]),
                             lo=None, hi=None, colour=colour, marker=marker))
            a_src.append(dict(panel="S1a", model=model, seed=42, estimator=e,
                              aurc_beta_mae=float(r["aurc_beta_mae"]),
                              beta_mae_full_coverage=float(r["beta_mae_at_full_coverage"]),
                              beta_mae_50pct=float(r["beta_mae_at_50pct_coverage"]),
                              scale="beta"))
    forest(ax_a, rows, r"AURC ($\beta$ MAE, lower is better)")
    ax_a.tick_params(axis="y", labelsize=6.0)
    panel_letter(ax_a, "a", xoff=-62)
    panel_title(ax_a, "Error-ranking quality")
    record("figS1a_estimator_comparison", pd.DataFrame(a_src))

    ax_b = fig.add_subplot(gs[0, 1])
    b_src = []
    for model, colour, ls in [("sequence", SEQUENCE, "-"), ("fusion", FUSION, (0, (4, 2)))]:
        sub = curves[(curves["model"] == model) & (curves["seed"] == 42) &
                     (curves["estimator"] == "rc_disagreement")].sort_values("coverage")
        if not len(sub):
            continue
        ax_b.plot(sub["coverage"], sub["beta_mae"], ls=ls, color=colour, lw=1.1, zorder=3, label=model)
        b_src.append(sub.assign(panel="S1b"))
    ax_b.set_xlabel("coverage")
    ax_b.set_ylabel(r"$\beta$ MAE of retained loci")
    ax_b.grid(color="#E6EAED", lw=0.5, zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.legend(frameon=False, fontsize=6.5, loc="upper left", handletextpad=0.5)
    panel_letter(ax_b, "b")
    panel_title(ax_b, "Risk-coverage (RC disagreement)")
    if b_src:
        record("figS1b_risk_coverage", pd.concat(b_src, ignore_index=True))

    ax_c = fig.add_subplot(gs[1, 0])
    rows_c, c_src = [], []
    for _, r in corr.iterrows():
        colour, marker = MODEL_STYLE[r["model"]]
        rows_c.append(dict(label=f"{'context' if r['model'] == 'epi' else r['model']} s{int(r['seed'])}",
                           value=float(r["spearman_disagreement_vs_error"]),
                           lo=float(r["spearman_ci_lo"]), hi=float(r["spearman_ci_hi"]),
                           colour=colour, marker=marker))
        c_src.append(dict(panel="S1c", model=r["model"], seed=int(r["seed"]),
                          n_loci=int(r["n_loci"]), n_blocks=int(r["n_blocks"]),
                          spearman=float(r["spearman_disagreement_vs_error"]),
                          ci_low=float(r["spearman_ci_lo"]), ci_high=float(r["spearman_ci_hi"]),
                          pearson=float(r["pearson_disagreement_vs_error"]),
                          uncertainty="1 Mb block bootstrap"))
    forest(ax_c, rows_c, "Spearman: RC disagreement vs locus error", null_line=0.0)
    ax_c.tick_params(axis="y", labelsize=6.0)
    panel_letter(ax_c, "c")
    panel_title(ax_c, "Ranking correlation with error")
    record("figS1c_disagreement_vs_error", pd.DataFrame(c_src))

    ax_d = fig.add_subplot(gs[1, 1])
    dec = repo.acsv("rc_uncertainty/disagreement_decile_table.csv")
    d_src = []
    col = [c for c in dec.columns if "mae" in c.lower()]
    xcol = [c for c in dec.columns if "decile" in c.lower()]
    if col and xcol:
        for model, colour, marker in [("sequence", SEQUENCE, "o"), ("fusion", FUSION, "D")]:
            sub = dec[dec["model"] == model] if "model" in dec.columns else dec
            if "seed" in sub.columns:
                sub = sub[sub["seed"] == 42]
            if not len(sub):
                continue
            ax_d.plot(sub[xcol[0]], sub[col[0]], marker, ms=3.2, ls="-", lw=1.0,
                      color=colour, zorder=3, label=model)
            d_src.append(sub.assign(panel="S1d"))
    ax_d.set_xlabel("RC-disagreement decile")
    ax_d.set_ylabel(r"$\beta$ MAE")
    ax_d.grid(color="#E6EAED", lw=0.5, zorder=0)
    ax_d.set_axisbelow(True)
    ax_d.legend(frameon=False, fontsize=6.5, loc="upper left", handletextpad=0.5)
    panel_letter(ax_d, "d")
    panel_title(ax_d, "Error by disagreement decile")
    if d_src:
        record("figS1d_disagreement_deciles", pd.concat(d_src, ignore_index=True))
    fig.text(0.5, 0.022, "Ranking and scale-dependent diagnostics only: no prediction interval with verified coverage is claimed.",
             ha="center", fontsize=6.0, color=OTHER)

    fig.savefig(out / "figS1_uncertainty.pdf")
    plt.close(fig)


def figs2(repo: Repo, out: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 105 * MM))
    gs = fig.add_gridspec(1, 2, left=0.235, right=0.985, top=0.885, bottom=0.175, wspace=0.10)

    cp = repo.acsv("motif_disruption/per_motif_coupling.csv")
    cp = cp.reindex(cp["coupling_spearman"].abs().sort_values(ascending=False).index)
    top = cp.head(14).copy()
    kmer_path = repo.j / "motif_disruption_kmer_baseline" / "per_motif_coupling.csv"
    kmer = pd.read_csv(kmer_path) if kmer_path.exists() else None

    ax = fig.add_subplot(gs[0, 0])
    rows, src = [], []
    for _, r in top.iterrows():
        rows.append(dict(label=f"{r['factor']} ({r['matrix_id']})", value=float(r["coupling_spearman"]),
                         lo=float(r["coupling_ci_low"]), hi=float(r["coupling_ci_high"]),
                         colour=FUSION, marker="D"))
        src.append(dict(panel="S2a", arm="fusion", matrix_id=r["matrix_id"], factor=r["factor"],
                        n_covered=int(r["n_covered"]), n_blocks=int(r["n_blocks"]),
                        coupling_spearman=float(r["coupling_spearman"]),
                        ci_low=float(r["coupling_ci_low"]), ci_high=float(r["coupling_ci_high"]),
                        partial_gc_distance=float(r["coupling_partial_gc_distance"]),
                        partial_gc_distance_substitution=float(r["coupling_partial_gc_distance_substitution"]),
                        q_bh=float(r["coupling_q_bh"]), consensus=r["motif_consensus"],
                        uncertainty="1 Mb block bootstrap"))
    forest(ax, rows, r"coupling Spearman with $\Delta\hat M$", null_line=0.0)
    ax.tick_params(axis="y", labelsize=6.0)
    panel_letter(ax, "a")
    panel_title(ax, "Fusion: motif disruption coupling")

    ax2 = fig.add_subplot(gs[0, 1], sharex=ax)
    rows2 = []
    if kmer is not None:
        km = kmer.set_index("matrix_id")
        for _, r in top.iterrows():
            if r["matrix_id"] in km.index:
                k = km.loc[r["matrix_id"]]
                rows2.append(dict(label="", value=float(k["coupling_spearman"]),
                                  lo=float(k["coupling_ci_low"]), hi=float(k["coupling_ci_high"]),
                                  colour=OTHER, marker="P"))
                src.append(dict(panel="S2b", arm="k-mer ridge", matrix_id=r["matrix_id"],
                                factor=r["factor"], n_covered=int(k["n_covered"]),
                                n_blocks=int(k["n_blocks"]), coupling_spearman=float(k["coupling_spearman"]),
                                ci_low=float(k["coupling_ci_low"]), ci_high=float(k["coupling_ci_high"]),
                                partial_gc_distance=float(k.get("coupling_partial_gc_distance", np.nan)),
                                partial_gc_distance_substitution=float(
                                    k.get("coupling_partial_gc_distance_substitution", np.nan)),
                                q_bh=float(k["coupling_q_bh"]), consensus=k.get("motif_consensus", ""),
                                uncertainty="1 Mb block bootstrap"))
            else:
                rows2.append(dict(label="", value=np.nan, lo=None, hi=None, colour=OTHER, marker="P"))
    if rows2:
        forest(ax2, rows2, r"coupling Spearman with $\Delta\hat M$", null_line=0.0)
    ax2.set_yticklabels([])
    panel_letter(ax2, "b")
    panel_title(ax2, "Context-free $k$-mer baseline")
    ets = top[top["factor"].str.contains("EL|EHF|FEV|ETS|ERG|ETV|FLI", regex=True, na=False)]
    fig.text(0.5, 0.045,
             f"Overlapping ETS matrices sharing the GGAA core ({len(ets)} of the {len(top)} shown) are one redundant family,\n"
             "not independent mechanistic findings. Composition, substitution and distance controls are in the\n"
             "partial columns of the source data.",
             ha="center", va="bottom", fontsize=5.8, color=OTHER, linespacing=1.5)
    record("figS2_motif_controls", pd.DataFrame(src))
    fig.savefig(out / "figS2_motif_controls.pdf")
    plt.close(fig)


def figs3(repo: Repo, out: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 100 * MM))
    gs = fig.add_gridspec(1, 3, left=0.085, right=0.985, top=0.875, bottom=0.165, wspace=0.42)

    cov = repo.acsv("target_qc/test_coverage_per_probe.csv")
    ax_a = fig.add_subplot(gs[0, 0])
    covcol = [c for c in cov.columns if "coverage" in c.lower() or "n_samples" in c.lower()]
    series = cov[covcol[0]].astype(float) if covcol else cov.iloc[:, 1].astype(float)
    ax_a.hist(series, bins=40, color=CONTEXT, edgecolor="white", lw=0.3, zorder=2)
    ax_a.set_xlabel("samples with observed methylation, per test probe")
    ax_a.set_ylabel("probes (log scale)")
    ax_a.grid(axis="y", color="#E6EAED", lw=0.5, zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.set_yscale("log")
    ax_a.text(0.03, 0.95, f"n = {len(cov):,} test probes\ndenominator: retained\nheld-out CpGs",
              transform=ax_a.transAxes, ha="left", va="top", fontsize=5.8, color=OTHER, linespacing=1.4)
    panel_letter(ax_a, "a")
    panel_title(ax_a, "Coverage distribution")
    record("figS3a_coverage", cov.assign(panel="S3a"))

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
        b_src.append(sub.assign(panel="S3b"))
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
        record("figS3b_coverage_bins", pd.concat(b_src, ignore_index=True))

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
    ax_c.text(0.98, 0.55, f"denominator: {total:,}\nretained CpGs\n(not participants;\n97 samples)",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=5.8, color=OTHER, linespacing=1.4)
    panel_letter(ax_c, "c")
    panel_title(ax_c, "Chromosome split counts")
    record("figS3c_splits", pd.DataFrame([
        dict(panel="S3c", split=lab.replace("\n", " "), n_retained_cpgs=n,
             fraction_of_retained=n / total, denominator_total_retained_cpgs=total,
             n_samples=97, note="split is by chromosome, not by participant")
        for lab, n in splits]))

    fig.savefig(out / "figS3_target_split_qc.pdf")
    plt.close(fig)


def figs4(repo: Repo, out: Path) -> None:
    fig = plt.figure(figsize=(180 * MM, 120 * MM))
    gs = fig.add_gridspec(2, 2, left=0.225, right=0.965, top=0.905, bottom=0.115,
                          wspace=0.80, hspace=0.85)

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
            a_src.append(dict(panel="S4a", cohort="eGTEx breast lead mQTL", model=r["model"],
                              seeds=r["seeds"], endpoint=lab, n_associations=int(r["n_loci"]),
                              n_unique_variants=int(r["n_unique_variants"]),
                              value=float(r[value_col]), ci_low=float(r[lo_col]),
                              ci_high=float(r[hi_col]), bootstrap_unit=r["bootstrap_unit"],
                              source="ABL/egtex_mqtl_positive_control/mqtl_positive_control_metrics.csv",
                              uncertainty="10,000 variant-cluster bootstrap draws"))
    forest(ax_a, rows, "positive-control agreement", null_line=None, xlim=(-0.08, 1.02))
    for null, lab in [(0.0, r"$\rho$ null"), (0.5, "rate null")]:
        ax_a.axvline(null, color=NULLC, ls=(0, (2, 2)), lw=0.8, zorder=1)
        ax_a.annotate(lab, (null, 1.0), xycoords=("data", "axes fraction"),
                      textcoords="offset points", xytext=(1.5, 1), ha="left", va="bottom",
                      fontsize=4.8, color=OTHER)
    ax_a.tick_params(axis="y", labelsize=5.4)
    ax_a.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_a.tick_params(axis="x", labelsize=6.2)
    panel_letter(ax_a, "a", xoff=-72)
    panel_title(ax_a, "eGTEx 81-lead positive control")
    record("figS4a_positive_control", pd.DataFrame(a_src))

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
        b_src.append(dict(panel="S4b", cohort="eGTEx matched leads", model=r["model"],
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
    record("figS4b_matched_negative", pd.DataFrame(b_src))

    ax_c = fig.add_subplot(gs[1, 0])
    db = repo.acsv("genoa_variant_evaluation/distance_bins.csv")
    c_src = []
    sub = db[db["model"] == "fusion"] if "model" in db.columns else db
    mcol = [c for c in sub.columns if c.lower() in ("value", "auroc", "signed_rho")]
    bcol = [c for c in sub.columns if "bin" in c.lower() or "distance" in c.lower()]
    if mcol and bcol:
        if "metric" in sub.columns:
            sub = sub[sub["metric"] == sub["metric"].iloc[0]]
        xs = np.arange(len(sub))
        vals = sub[mcol[0]].astype(float).values
        ax_c.plot(xs, vals, "D", ms=3.2, ls="-", lw=1.0, color=FUSION, zorder=3)
        if "ci_low" in sub.columns:
            ax_c.fill_between(xs, sub["ci_low"].astype(float), sub["ci_high"].astype(float),
                              color=FUSION, alpha=0.14, lw=0, zorder=2)
        ax_c.set_xticks(xs)
        ax_c.set_xticklabels(sub[bcol[0]].astype(str), fontsize=5.5, rotation=35, ha="right")
        ax_c.set_ylabel("AUROC, fusion\n(significant vs matched)", fontsize=6.5)
        c_src.append(sub.assign(panel="S4c"))
    ax_c.set_xlabel("variant-CpG distance bin", fontsize=7)
    ax_c.grid(axis="y", color="#E6EAED", lw=0.5, zorder=0)
    ax_c.set_axisbelow(True)
    panel_letter(ax_c, "c")
    panel_title(ax_c, "Distance strata (GENOA)")
    if c_src:
        record("figS4c_distance_bins", pd.concat(c_src, ignore_index=True))

    ax_d = fig.add_subplot(gs[1, 1])
    t = repo.csv("asm_validation_tycko/tycko_e2_signed_agreement.csv")
    rows_d, d_src = [], []
    for stratum, lab in [("mammary", "mammary (n=53)"), ("|effect| >= 20.0pp", "|effect|>=20pp (post hoc)")]:
        sub = t[t["Stratum"] == stratum]
        for _, r in sub.iterrows():
            colour, marker = MODEL_STYLE[r["Arm"]]
            rows_d.append(dict(label=f"{lab}\n{r['Arm']}", value=float(r["Direction_Concordance"]),
                               lo=float(r["Direction_CI_Low"]), hi=float(r["Direction_CI_High"]),
                               colour=colour, marker=marker))
            d_src.append(dict(panel="S4d", stratum=stratum, arm=r["Arm"], n_snps=int(r["N_SNPs"]),
                              n_blocks=int(r["N_Genomic_Blocks"]),
                              direction_concordance=float(r["Direction_Concordance"]),
                              ci_low=float(r["Direction_CI_Low"]), ci_high=float(r["Direction_CI_High"]),
                              signed_spearman=float(r["Signed_Spearman"]),
                              prespecified=(stratum == "mammary"),
                              note="post-hoc effect-size restriction" if stratum != "mammary" else "tissue subset",
                              uncertainty="2,000 block resamples"))
    forest(ax_d, rows_d, "direction concordance", null_line=0.5)
    ax_d.tick_params(axis="y", labelsize=5.4)
    panel_letter(ax_d, "d")
    panel_title(ax_d, "Do-Tycko subsets")
    record("figS4d_asm_subsets", pd.DataFrame(d_src))

    fig.savefig(out / "figS4_external_controls.pdf")
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
    figs1(repo, out)
    figs2(repo, out)
    figs3(repo, out)
    figs4(repo, out)

    idx = write_source_data(args.source_data_output)
    print(idx.to_string(index=False))
    for p in sorted(out.glob("*.pdf")):
        print(f"{p} {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
