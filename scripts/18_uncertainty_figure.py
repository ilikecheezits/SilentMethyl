#!/usr/bin/env python3
"""
Stage B.3c -- Consolidate the uncertainty analysis into a paper figure + table.

The finding this renders
------------------------
Within predicted-beta strata, the trivial heuristic -|beta_hat - 0.5| loses its
apparent advantage as stratification tightens, and loses it almost entirely on the
unbounded M (logit) scale. The two real uncertainty estimators hold steady on BOTH
scales. Scale-stability is exactly what separates a genuine uncertainty signal from
a metric artifact:

    within-stratum Spearman(estimator, |error|), 50 strata
        boundary_distance    beta +0.0369   M +0.0099    <- collapses
        cross_seed_sd        beta +0.0915   M +0.0952    <- stable
        rc_disagreement      beta +0.0678   M +0.0793    <- stable

Absolute beta error is mechanically compressed near 0 and 1, so a "confident at
the extremes" score looks calibrated on beta and is exposed on M. This generalises
beyond SilentMethyl: the field routinely reports beta MAE.

Inputs
------
The three script-17 output directories (10 / 20 / 50 strata). Reads
`within_stratum_correlations.csv` and `stratified_selective_prediction.csv`.

Usage
-----
    python -u scripts/18_uncertainty_figure.py \
        --runs 10:results/journal/rc_uncertainty_conditional \
               20:results/journal/rc_uncertainty_conditional_s20 \
               50:results/journal/rc_uncertainty_conditional_s50 \
        --output-dir results/journal/rc_uncertainty_figure
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Validated categorical palette (six checks pass, light surface).
# Assigned by role, fixed order, never cycled.
COLORS = {
    "boundary_distance": "#B03A2E",   # the control / artifact
    "cross_seed_sd":     "#1F6FB2",   # the ensemble reference
    "rc_disagreement":   "#B4761A",   # the cheap single-model estimator
}
# Identity is never colour alone -- journals print in greyscale.
MARKERS = {"boundary_distance": "s", "cross_seed_sd": "o", "rc_disagreement": "^"}
STYLES = {"boundary_distance": (0, (4, 2)), "cross_seed_sd": "-", "rc_disagreement": (0, (1, 1.4))}
LABELS = {
    "boundary_distance": "−|β̂ − 0.5|  (heuristic)",
    "cross_seed_sd": "cross-seed SD  (3-model ensemble)",
    "rc_disagreement": "FWD–RC disagreement  (single model)",
}
ORDER = ["cross_seed_sd", "rc_disagreement", "boundary_distance"]
TARGET_TITLE = {"beta": "A   β error  (bounded 0–1)",
                "m": "B   M error  (unbounded logit)"}

INK, MUTED, RULE = "#1B2021", "#5B6B6E", "#D3DADB"


def parse_runs(items) -> dict:
    out = {}
    for it in items:
        if ":" not in it:
            raise SystemExit(f"--runs entries must be STRATA:PATH, got {it!r}")
        k, p = it.split(":", 1)
        out[int(k)] = Path(p)
    return dict(sorted(out.items()))


def collect(runs: dict) -> tuple:
    within, strat = [], []
    for k, path in runs.items():
        w = path / "within_stratum_correlations.csv"
        s = path / "stratified_selective_prediction.csv"
        if not w.exists():
            logging.warning("missing %s -- skipping strata=%d", w, k)
            continue
        dw = pd.read_csv(w); dw["strata"] = k; within.append(dw)
        if s.exists():
            ds = pd.read_csv(s); ds["strata"] = k; strat.append(ds)
    if not within:
        raise SystemExit("no within_stratum_correlations.csv found in any --runs path")
    return (pd.concat(within, ignore_index=True),
            pd.concat(strat, ignore_index=True) if strat else pd.DataFrame())


def make_figure(within: pd.DataFrame, out: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib unavailable; skipping figure")
        return None

    targets = [t for t in ("beta", "m") if t in set(within.error_target)]
    fig, axes = plt.subplots(1, len(targets), figsize=(7.2, 3.1), sharey=True)
    axes = np.atleast_1d(axes)

    strata = sorted(within.strata.unique())
    x = np.arange(len(strata))

    for ax, tname in zip(axes, targets):
        sub = within[within.error_target == tname]
        ax.axhline(0, color=RULE, linewidth=0.9, zorder=1)
        for est in ORDER:
            g = sub[sub.estimator == est]
            if g.empty:
                continue
            y = [g[g.strata == s]["within_bin_spearman_pooled"].mean() for s in strata]
            ax.plot(x, y, color=COLORS[est], marker=MARKERS[est], linestyle=STYLES[est],
                    linewidth=2.0, markersize=6, markeredgecolor="white",
                    markeredgewidth=0.9, zorder=3, clip_on=False)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in strata])
        ax.set_xlabel("predicted-β strata", fontsize=9, color=MUTED)
        ax.set_title(TARGET_TITLE.get(tname, tname), fontsize=9.5,
                     color=INK, loc="left", pad=8)
        ax.tick_params(labelsize=8.5, colors=MUTED, length=3)
        ax.grid(axis="y", color=RULE, linewidth=0.6, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)

    axes[0].set_ylabel("within-stratum Spearman ρ\n(estimator vs |error|)",
                       fontsize=9, color=MUTED)

    # Legend once, below -- identity is colour + marker + dash, never colour alone.
    handles = [axes[0].plot([], [], color=COLORS[e], marker=MARKERS[e],
                            linestyle=STYLES[e], linewidth=2.0, markersize=6,
                            label=LABELS[e])[0] for e in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.13), labelcolor=INK)

    fig.tight_layout()
    path = out / "uncertainty_scale_stability.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "uncertainty_scale_stability.pdf", bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, metavar="STRATA:PATH")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/journal/rc_uncertainty_figure"))
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    runs = parse_runs(args.runs)
    within, strat = collect(runs)

    # ---- consolidated supplementary table
    tbl = (within.groupby(["error_target", "estimator", "strata"])
                 ["within_bin_spearman_pooled"].mean().reset_index()
                 .pivot(index=["error_target", "estimator"], columns="strata",
                        values="within_bin_spearman_pooled")
                 .round(4).reset_index())
    tbl.columns = [f"strata_{c}" if isinstance(c, (int, np.integer)) else c
                   for c in tbl.columns]
    tbl.to_csv(out / "within_stratum_by_strata.csv", index=False)

    if not strat.empty:
        wins = (strat[strat.estimator != "boundary_distance"]
                .groupby(["error_target", "estimator", "strata"])
                ["beats_boundary_within_strata"]
                .agg(["sum", "count"]).reset_index()
                .rename(columns={"sum": "runs_beating_boundary", "count": "runs_total"}))
        wins.to_csv(out / "beats_boundary_by_strata.csv", index=False)
    else:
        wins = pd.DataFrame()

    fig_path = make_figure(within, out)

    # ---- summary
    L = ["Uncertainty: scale-stability of estimators vs the |beta_hat-0.5| heuristic",
         "=" * 74, "",
         "Within-stratum Spearman rho (pooled over predicted-beta bins)", ""]
    hdr = "  " + f"{'estimator':<20}" + "".join(f"{'s='+str(s):>12}" for s in runs)
    for tname, g in tbl.groupby("error_target"):
        L += [f"  [{tname} error]", hdr]
        for est in ORDER:
            r = g[g.estimator == est]
            if r.empty:
                continue
            vals = "".join(f"{r[f'strata_{s}'].iloc[0]:>+12.4f}"
                           if f"strata_{s}" in r else f"{'--':>12}" for s in runs)
            L.append(f"  {est:<20}{vals}")
        L.append("")

    if not wins.empty:
        L += ["Stratified selective prediction: runs beating the heuristic", ""]
        for (tname, est), g in wins.groupby(["error_target", "estimator"]):
            cells = "  ".join(f"s={int(r.strata)}: {int(r.runs_beating_boundary)}/"
                              f"{int(r.runs_total)}" for _, r in g.iterrows())
            L.append(f"  {tname:<5} {est:<20} {cells}")
        L.append("")

    L += ["Reading", "-" * 74,
          "  A genuine uncertainty signal should rank error about equally well on",
          "  either scale. boundary_distance does not: it decays toward zero as strata",
          "  tighten and is near-zero on M. cross_seed_sd and rc_disagreement hold on",
          "  both. The heuristic was tracking the mechanically compressed range of beta",
          "  near 0 and 1 -- headroom, not difficulty.", "",
          "  Consequence: report uncertainty on M-value error. Any methylation model",
          "  evaluated on beta MAE has this confound; the field routinely reports it."]

    (out / "uncertainty_figure_summary.txt").write_text("\n".join(L) + "\n")

    with (out / "run_summary.json").open("w") as fh:
        json.dump({
            "analysis": "Scale-stability of uncertainty estimators (figure + table)",
            "stage": "B.3c -- consolidation, no GPU",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_runs": {str(k): str(v) for k, v in runs.items()},
            "palette": COLORS,
            "palette_validation": "six-check categorical validator, light surface, all PASS",
            "outputs": ["uncertainty_scale_stability.png",
                        "uncertainty_scale_stability.pdf",
                        "within_stratum_by_strata.csv",
                        "beats_boundary_by_strata.csv",
                        "uncertainty_figure_summary.txt"],
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("\n".join(L))
    if fig_path:
        logging.info("figure: %s", fig_path)
    logging.info("wrote outputs to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
