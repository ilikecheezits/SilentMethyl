#!/usr/bin/env python3
"""
Evaluate SilentMethyl on TCGA-BRCA tumours as an independent evaluation domain.

Mentor requirement 1: multi-cohort testing.

The thing to understand before reading the numbers
---------------------------------------------------
**The model's predictions do not change between normal and tumour.** Its inputs
are the 1,000-bp reference sequence and MCF-10A context features; neither depends
on disease state. Scoring "the tumour cohort" therefore does not mean re-running
the model -- it means holding the existing held-out predictions fixed and
swapping the target the model is scored against.

That is not a weakness of the design, it is what the design measures: the
fraction of the tumour methylome that is determined by sequence and reference
chromatin context alone, and is therefore invariant to malignant transformation.
Where the model degrades is exactly where methylation has been reprogrammed by
something the model cannot see.

Two consequences follow, and both belong in the manuscript:

  1. This costs zero GPU. It reuses results/journal/seed*/<model>/predictions.csv.
  2. It must NOT be described as the model "generalising to tumours". The honest
     framing is domain shift in the target, with identical inputs -- a decomposition
     of the tumour methylome into a sequence-determined component and a
     state-specific remainder.

Cohorts
-------
699 unpaired primary tumours (participants absent from the 97 training normals)
are the primary result. The full 791-tumour cohort shares 91 participants with
the training normals, so it is a paired comparison and belongs in the supplement.
Both are produced by data/build_tcga_tumor_cohort.py.

Intervals are percentile bootstraps over 1 Mb genomic blocks, matching scripts
16-17 and 20: held-out probes are spatially correlated and row-level resampling
would give intervals that are far too narrow.

Usage (run from the repository root; CPU only, a few minutes)
-------------------------------------------------------------
    python -u scripts/21_tcga_tumor_domain_shift.py
    python -u scripts/21_tcga_tumor_domain_shift.py --n-boot 2000
"""

from __future__ import annotations

import argparse
import json
import logging
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
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger("silentmethyl.tumor")

COLOURS = {"normal": "#1F6FB2", "tumor": "#B03A2E", "baseline": "#B4761A"}
MARKERS = {"normal": "s", "tumor": "o", "baseline": "^"}
BLOCK_BP = 1_000_000
EPSILON = 1e-4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions-template",
                   default="results/journal/seed{seed}/{model}/predictions.csv")
    p.add_argument("--models", nargs="+", default=["fusion", "sequence", "epi"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--tumor-dir", type=Path, default=Path("data/external/tcga_tumor"))
    p.add_argument("--test-csv", type=Path, default=Path("data/datafiles/test.csv"),
                   help="Supplies probe coordinates for genomic blocking.")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/journal/tcga_tumor_domain_shift"))
    return p.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def beta_to_m(beta: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(beta, dtype=float), EPSILON, 1 - EPSILON)
    return np.log2(clipped / (1.0 - clipped))


def load_predictions(args: argparse.Namespace) -> pd.DataFrame:
    """One row per (probe, model, seed) with the RC-averaged prediction."""
    frames = []
    for model in args.models:
        for seed in args.seeds:
            path = Path(args.predictions_template.format(seed=seed, model=model))
            if not path.is_file():
                LOGGER.warning("skipping %s (not found)", path)
                continue
            frame = pd.read_csv(path)
            probe_column = next((c for c in ("probeID", "probe_id", "Probe", "cpg")
                                 if c in frame.columns), None)
            if probe_column is None:
                raise SystemExit(
                    f"{path} has no recognisable probe ID column. Columns present: "
                    f"{list(frame.columns)[:20]}")
            required = ["pred_beta_rc_avg", "true_beta"]
            absent = [c for c in required if c not in frame.columns]
            if absent:
                raise SystemExit(f"{path} is missing {absent}; columns present: "
                                 f"{list(frame.columns)[:20]}")
            keep = {probe_column: "probeID",
                    "pred_beta_rc_avg": "pred_beta",
                    "true_beta": "normal_beta"}
            if "class_prob_rc_avg" in frame.columns:
                keep["class_prob_rc_avg"] = "class_prob"
            frame = frame[list(keep)].rename(columns=keep)
            frame["Model"] = model
            frame["Seed"] = int(seed)
            frames.append(frame)
            LOGGER.info("%s seed %d: %d held-out probes", model, seed, len(frame))
    if not frames:
        raise SystemExit("no prediction files found; check --predictions-template")
    return pd.concat(frames, ignore_index=True)


def add_seed_ensemble(long: pd.DataFrame) -> pd.DataFrame:
    keys = ["Model", "probeID"]
    aggregate = {"pred_beta": "mean"}
    if "class_prob" in long.columns:
        aggregate["class_prob"] = "mean"
    means = long.groupby(keys, sort=False).agg(aggregate).reset_index()
    metadata = (long[long["Seed"] == long["Seed"].min()]
                .drop(columns=[c for c in ("pred_beta", "class_prob", "Seed")
                               if c in long.columns]))
    ensemble = metadata.merge(means, on=keys, how="inner", validate="one_to_one")
    ensemble["Seed"] = -1
    return pd.concat([long, ensemble], ignore_index=True)


def load_coordinates(path: Path) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=0)
    needed = [c for c in ("probeID", "chr", "pos") if c in head.columns]
    if len(needed) < 3:
        raise SystemExit(f"{path} needs probeID, chr and pos; found {list(head.columns)[:10]}")
    coordinates = pd.read_csv(path, usecols=needed)
    coordinates["probeID"] = coordinates["probeID"].astype(str)
    coordinates["_block"] = (coordinates["chr"].astype(str) + ":"
                             + (coordinates["pos"] // BLOCK_BP).astype(int).astype(str))
    return coordinates[["probeID", "chr", "pos", "_block"]]


# ---------------------------------------------------------------- metrics

def beta_mae(frame: pd.DataFrame, target: str) -> float:
    return float(np.mean(np.abs(frame["pred_beta"] - frame[target])))


def m_mae(frame: pd.DataFrame, target: str) -> float:
    return float(np.mean(np.abs(beta_to_m(frame["pred_beta"]) - beta_to_m(frame[target]))))


def auroc(frame: pd.DataFrame, target: str) -> float:
    labels = (frame[target] > 0.5).astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        return np.nan
    score = frame["class_prob"] if "class_prob" in frame.columns else frame["pred_beta"]
    return float(roc_auc_score(labels, score))


def rank_correlation(frame: pd.DataFrame, target: str) -> float:
    return float(spearmanr(frame["pred_beta"], frame[target]).statistic)


def block_bootstrap(frame: pd.DataFrame, metric, n_boot: int,
                    rng: np.random.Generator) -> tuple[float, float]:
    if frame.empty:
        return (np.nan, np.nan)
    unique, inverse = np.unique(frame["_block"].to_numpy(), return_inverse=True)
    if len(unique) < 5:
        return (np.nan, np.nan)
    positions = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    values = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(unique), len(unique))
        index = np.concatenate([positions[p] for p in picks])
        value = metric(frame.iloc[index])
        if np.isfinite(value):
            values.append(value)
    if len(values) < max(20, n_boot // 10):
        return (np.nan, np.nan)
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


SLIM = ["pred_beta", "class_prob", "normal_beta", "tumor_beta", "_block"]


def measure(frame: pd.DataFrame, metric, target: str, name: str, n_boot: int,
            rng: np.random.Generator, **context) -> dict:
    slim = frame[[c for c in SLIM if c in frame.columns]]
    bound = lambda f: metric(f, target)  # noqa: E731
    point = bound(slim)
    low, high = block_bootstrap(slim, bound, n_boot, rng)
    return {**context, "target": target, "metric": name, "n": int(len(slim)),
            "value": point, "ci_low": low, "ci_high": high}


# ------------------------------------------------------------------- main

def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    rng = np.random.default_rng(args.random_seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    predictions = add_seed_ensemble(load_predictions(args))
    coordinates = load_coordinates(args.test_csv)
    predictions["probeID"] = predictions["probeID"].astype(str)
    predictions = predictions.merge(coordinates, on="probeID", how="inner",
                                    validate="many_to_one")

    cohorts = {}
    for name in ("unpaired", "all"):
        path = args.tumor_dir / f"tcga_tumor_{name}.csv"
        if not path.is_file():
            LOGGER.warning("skipping %s cohort (%s not found)", name, path)
            continue
        frame = pd.read_csv(path)
        frame["probeID"] = frame["probeID"].astype(str)
        cohorts[name] = frame[["probeID", "Median_Beta", "n_samples_observed"]].rename(
            columns={"Median_Beta": "tumor_beta"})
        LOGGER.info("%s tumour cohort: %d probes", name, len(frame))
    if "unpaired" not in cohorts:
        raise SystemExit(
            "the unpaired tumour cohort is required as the primary result. Run "
            "`python -u data/build_tcga_tumor_cohort.py` first.")

    rows, shift_rows = [], []
    for cohort_name, cohort in cohorts.items():
        joined = predictions.merge(cohort, on="probeID", how="inner",
                                   validate="many_to_one")
        LOGGER.info("%s: %d probe-model-seed rows after join", cohort_name, len(joined))
        if joined.empty:
            LOGGER.warning("%s cohort shares no probes with the held-out set", cohort_name)
            continue

        # How far the target itself moved. This is the quantity the model cannot
        # see, and it is what the whole analysis is conditioning on.
        joined["target_shift"] = (joined["tumor_beta"] - joined["normal_beta"]).abs()

        for model in joined["Model"].unique():
            for seed in sorted(joined["Seed"].unique()):
                subset = joined[(joined["Model"] == model) & (joined["Seed"] == seed)]
                if subset.empty:
                    continue
                context = {"cohort": cohort_name, "model": model, "seed": int(seed)}
                for target, label in (("normal_beta", "normal"), ("tumor_beta", "tumor")):
                    rows.append(measure(subset, beta_mae, target, "beta_mae",
                                        args.n_boot, rng, **context, domain=label))
                    rows.append(measure(subset, m_mae, target, "m_mae",
                                        args.n_boot, rng, **context, domain=label))
                    rows.append(measure(subset, auroc, target, "auroc",
                                        args.n_boot, rng, **context, domain=label))
                    rows.append(measure(subset, rank_correlation, target,
                                        "spearman", args.n_boot, rng,
                                        **context, domain=label))

        # Stratify by how much the target moved. The model is blind to disease
        # state, so this curve is the interpretable result: it says what fraction
        # of the tumour methylome is sequence-determined and where that breaks.
        ensemble = joined[(joined["Seed"] == -1)]
        if not ensemble.empty:
            edges = [0.0, 0.05, 0.10, 0.20, 0.40, 1.01]
            for model in ensemble["Model"].unique():
                for low, high in zip(edges[:-1], edges[1:]):
                    stratum = ensemble[(ensemble["Model"] == model)
                                       & (ensemble["target_shift"] >= low)
                                       & (ensemble["target_shift"] < high)]
                    if len(stratum) < 50:
                        continue
                    context = {"cohort": cohort_name, "model": model,
                               "shift_bin": f"{low:.2f}-{high:.2f}"}
                    shift_rows.append(measure(stratum, beta_mae, "tumor_beta",
                                              "beta_mae", args.n_boot, rng,
                                              **context, domain="tumor"))
                    shift_rows.append(measure(stratum, beta_mae, "normal_beta",
                                              "beta_mae", args.n_boot, rng,
                                              **context, domain="normal"))

    metrics = pd.DataFrame(rows)
    atomic_csv(metrics, out / "domain_shift_metrics.csv")
    shift = pd.DataFrame(shift_rows)
    atomic_csv(shift, out / "metrics_by_target_shift.csv")

    # A single descriptive line about how far the methylome actually moved.
    unpaired = predictions.merge(cohorts["unpaired"], on="probeID", how="inner")
    unpaired = unpaired[unpaired["Seed"] == -1]
    unpaired = unpaired[unpaired["Model"] == unpaired["Model"].iloc[0]]
    shift_values = (unpaired["tumor_beta"] - unpaired["normal_beta"]).abs()
    descriptive = {
        "probes_shared": int(len(unpaired)),
        "median_absolute_target_shift": float(shift_values.median()),
        "fraction_shift_above_0.10": float((shift_values > 0.10).mean()),
        "fraction_shift_above_0.20": float((shift_values > 0.20).mean()),
        "spearman_normal_vs_tumor_target": float(
            spearmanr(unpaired["normal_beta"], unpaired["tumor_beta"]).statistic),
    }

    make_figure(shift, out, args)

    atomic_json(
        {
            "analysis": "TCGA-BRCA tumours as an independent evaluation domain",
            "purpose": "mentor requirement 1: multi-cohort testing",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "design": (
                "Model inputs (reference sequence, MCF-10A context) do not depend on "
                "disease state, so predictions are IDENTICAL for normal and tumour. "
                "This evaluation holds predictions fixed and swaps the target. It "
                "measures the sequence-and-context-determined component of the tumour "
                "methylome, not generalisation of the model to a new input domain."),
            "framing_warning": (
                "Do not describe this as the model generalising to tumours. Describe "
                "it as target-domain shift with identical inputs."),
            "primary_cohort": "unpaired (699 tumours, participants disjoint from the "
                              "97 training normals)",
            "supplementary_cohort": "all (791 tumours; 91 participants shared with the "
                                    "training normals, therefore paired)",
            "models": args.models,
            "seeds": args.seeds,
            "seed_minus_one_means": "mean predicted beta across seeds",
            "n_boot": args.n_boot,
            "block_size_bp": BLOCK_BP,
            "descriptive": descriptive,
            "gpu_hours": 0,
        },
        out / "run_summary.json",
    )

    print_report(metrics, shift, descriptive, out)
    return 0


def make_figure(shift: pd.DataFrame, out: Path, args: argparse.Namespace) -> None:
    if shift.empty:
        return
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    subset = shift[(shift["cohort"] == "unpaired") & (shift["metric"] == "beta_mae")]
    if subset.empty:
        return
    models = [m for m in args.models if m in set(subset["model"])]
    bins = sorted(subset["shift_bin"].unique(),
                  key=lambda s: float(s.split("-")[0]))

    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.8),
                             sharey=True, squeeze=False)
    for axis, model in zip(axes[0], models):
        for domain in ("normal", "tumor"):
            rows = (subset[(subset["model"] == model) & (subset["domain"] == domain)]
                    .set_index("shift_bin").reindex(bins).reset_index())
            x = np.arange(len(bins))
            axis.errorbar(
                x, rows["value"],
                yerr=[rows["value"] - rows["ci_low"], rows["ci_high"] - rows["value"]],
                marker=MARKERS[domain], linestyle="-" if domain == "tumor" else "--",
                color=COLOURS[domain], capsize=3, markersize=5, linewidth=1.6,
                label=f"vs {domain} target")
        axis.set_xticks(np.arange(len(bins)))
        axis.set_xticklabels(bins, rotation=40, ha="right", fontsize=8)
        axis.set_title(model, fontsize=10)
        axis.set_xlabel(r"$|\beta_{tumour}-\beta_{normal}|$", fontsize=9)
        axis.grid(axis="y", alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("beta MAE (95% block-bootstrap CI)", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("Predictions are fixed; error against the tumour target grows with "
                 "how far the methylome moved", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(plots / "error_by_target_shift.png", dpi=400, bbox_inches="tight",
                facecolor="white")
    fig.savefig(plots / "error_by_target_shift.pdf", bbox_inches="tight")
    plt.close(fig)


def cell(frame: pd.DataFrame, **filters) -> str:
    subset = frame
    for key, value in filters.items():
        subset = subset[subset[key] == value]
    if subset.empty or not np.isfinite(subset["value"].iloc[0]):
        return "         n/a"
    row = subset.iloc[0]
    return f"{row['value']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"


def print_report(metrics: pd.DataFrame, shift: pd.DataFrame,
                 descriptive: dict, out: Path) -> None:
    print()
    print("=" * 78)
    print("TCGA-BRCA tumour domain shift -- unpaired cohort, seed ensemble")
    print("Predictions are IDENTICAL in both columns; only the target differs.")
    print("=" * 78)
    print(f"\nprobes shared with the held-out set : {descriptive['probes_shared']:,}")
    print(f"median |beta_tumour - beta_normal|  : {descriptive['median_absolute_target_shift']:.4f}")
    print(f"probes shifted > 0.10               : {descriptive['fraction_shift_above_0.10']:.1%}")
    print(f"probes shifted > 0.20               : {descriptive['fraction_shift_above_0.20']:.1%}")
    print(f"spearman(normal target, tumour target): {descriptive['spearman_normal_vs_tumor_target']:.4f}")

    subset = metrics[(metrics["cohort"] == "unpaired") & (metrics["seed"] == -1)]
    for model in subset["model"].unique():
        print(f"\n{model.upper()}")
        for name, label in (("beta_mae", "beta MAE"), ("m_mae", "M MAE"),
                            ("auroc", "ROC-AUC"), ("spearman", "Spearman")):
            normal = cell(subset, model=model, metric=name, domain="normal")
            tumour = cell(subset, model=model, metric=name, domain="tumor")
            print(f"  {label:<10} normal {normal}    tumour {tumour}")

    if not shift.empty:
        print("\nBETA MAE AGAINST THE TUMOUR TARGET, BY HOW FAR THE TARGET MOVED")
        rows = shift[(shift["cohort"] == "unpaired") & (shift["domain"] == "tumor")
                     & (shift["metric"] == "beta_mae")]
        model = rows["model"].iloc[0] if not rows.empty else None
        for _, row in rows[rows["model"] == model].iterrows():
            print(f"  shift {row['shift_bin']:<12} n={row['n']:>7,}  "
                  f"{row['value']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
        print(f"  ({model}; the first bin is where the methylome barely moved)")

    print(f"\noutput: {out}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
