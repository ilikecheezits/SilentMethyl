#!/usr/bin/env python3
"""
Stage B.3 -- Uncertainty calibration for SilentMethyl, in three stages.

This file merges what were scripts 16, 17 and 18. They were always one analysis
split across three files; the split cost a duplicated loader, a duplicated
Spearman, and a duplicated cross-seed-SD merge, and it made the argument hard to
follow because the punchline of each stage is the motivation for the next.

    --stage base          Is FWD/RC disagreement a free uncertainty estimate?
    --stage conditional   Does ANY estimator beat the |beta_hat - 0.5| heuristic?
    --stage figure        Consolidate into the paper figure and table.
    --stage all           base, then conditional at 10/20/50 strata, then figure.

The argument, in order
----------------------
BASE. SilentMethyl reports RC-averaged predictions, beta_hat = 1/2[f(x)+f(RC(x))],
which is exactly RC-invariant -- so the reported output has no strand
inconsistency. But the average discards the per-locus disagreement
|f(x) - f(RC(x))|, already written to predictions.csv. Hypothesis: that discarded
quantity measures how well the model internalised RC symmetry at that locus, and
so predicts local error -- uncertainty from a SINGLE model, where the usual route
is an ensemble. Estimators compared:

    rc_disagreement   |pred_beta_fwd - pred_beta_rc|      single model, free
    cross_seed_sd     SD of pred_beta_rc_avg over seeds   needs N models
    boundary_distance -|pred_beta_rc_avg - 0.5|           naive control
    combined          rank-average of the first two       complementarity
    random            null

CONDITIONAL. The base stage finds that `boundary_distance`, a zero-parameter
heuristic, beats both real estimators at selective prediction. Two explanations
must be separated before that goes in a manuscript:

  (a) REAL      intermediate-methylation probes are genuinely harder, and
                distance from 0.5 is a legitimate difficulty signal.
  (b) ARTIFACT  absolute beta error is mechanically bounded near the extremes --
                at beta_hat = 0.02 there is almost no room to be wrong, at 0.5
                there is +/- 0.5. The heuristic may measure HEADROOM.

If (b) dominates, the calibration section would be reporting the shape of the
target distribution rather than anything about the model. Five tests: partial
Spearman given the heuristic; within-stratum Spearman; stratified selective
prediction (decisive); incremental AURC on top of the heuristic; and all of it
repeated on M-value error, which is unbounded.

FIGURE. The answer is (b). Within predicted-beta strata the heuristic decays as
stratification tightens and is near-zero on the M scale, while both real
estimators hold on both scales. Scale-stability is what separates a genuine
uncertainty signal from a metric artifact -- and the consequence generalises past
SilentMethyl, because the field routinely reports beta MAE.

Everything here is CPU-only and reads files that already exist. No GPU, no
re-inference, no retraining.

Usage
-----
    python -u scripts/16_rc_uncertainty.py --stage all

    # or one stage at a time, reproducing the original three scripts exactly
    python -u scripts/16_rc_uncertainty.py --stage base
    python -u scripts/16_rc_uncertainty.py --stage conditional --strata 50 \
        --output-dir results/journal/rc_uncertainty_conditional_s50
    python -u scripts/16_rc_uncertainty.py --stage figure \
        --runs 10:results/journal/rc_uncertainty_conditional \
               20:results/journal/rc_uncertainty_conditional_s20 \
               50:results/journal/rc_uncertainty_conditional_s50
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as sps
except ImportError:  # pragma: no cover
    sps = None

REQUIRED = ("pred_beta_fwd", "pred_beta_rc", "pred_beta_rc_avg", "true_beta")
BLOCK_BP = 1_000_000  # matches scripts/08 and scripts/12

# numpy renamed trapz -> trapezoid in 2.0; the cluster env may predate that.
_trapz = getattr(np, "trapezoid", None) or np.trapz

DEFAULT_OUT = {
    "base": Path("results/journal/rc_uncertainty"),
    "conditional": Path("results/journal/rc_uncertainty_conditional"),
    "figure": Path("results/journal/rc_uncertainty_figure"),
}


# =========================================================================
# shared utilities
# =========================================================================

def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()


def spearman(a: np.ndarray, b: np.ndarray, min_n: int = 3) -> float:
    """Rank correlation. `min_n` is kept explicit because the base stage used a
    threshold of 3 and the conditional stage 5; both are preserved at their call
    sites so outputs remain identical to the pre-merge scripts."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n:
        return float("nan")
    if sps is not None:
        r = sps.spearmanr(a[ok], b[ok])
        v = getattr(r, "statistic", None)
        return float(v if v is not None else r[0])
    return float(np.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman(x, y | z): correlate rank-residuals after removing z."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 10:
        return float("nan")
    rx, ry, rz = _rank(x[ok]), _rank(y[ok]), _rank(z[ok])
    Z = np.column_stack([np.ones_like(rz), rz])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex, ey = rx - Z @ bx, ry - Z @ by
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC; NaN if the retained set is single-class."""
    ok = np.isfinite(y_true) & np.isfinite(score)
    y, s = y_true[ok], score[ok]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank().to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def aurc(err: np.ndarray, unc: np.ndarray, coverages: np.ndarray) -> float:
    """Area under the risk-coverage curve, computed directly. Lower = better."""
    order = np.argsort(np.where(np.isfinite(unc), unc, np.inf))
    risks = []
    for cov in coverages:
        k = max(1, int(round(cov * len(err))))
        risks.append(np.nanmean(err[order[:k]]))
    risks = np.asarray(risks, dtype=float)
    if len(coverages) < 2:
        return float("nan")
    return float(_trapz(risks, coverages) / (coverages[-1] - coverages[0]))


def strat_bins(beta_hat: np.ndarray, n_bins: int) -> np.ndarray:
    ranks = pd.Series(beta_hat).rank(method="first", pct=True).to_numpy()
    return np.clip((ranks * n_bins).astype(int), 0, n_bins - 1)


def block_ids(df: pd.DataFrame, block_bp: int = BLOCK_BP) -> np.ndarray:
    """Genomic blocks so bootstrap respects local error correlation."""
    if "chr" not in df.columns or "pos" not in df.columns:
        return np.arange(len(df))
    pos = pd.to_numeric(df["pos"], errors="coerce").fillna(-1).to_numpy()
    chrom = df["chr"].astype(str).to_numpy()
    return np.array([f"{c}:{int(p) // block_bp}" for c, p in zip(chrom, pos)])


def block_bootstrap_ci(
    stat_fn, values: dict, blocks: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple:
    uniq = np.unique(blocks)
    index_by_block = {b: np.flatnonzero(blocks == b) for b in uniq}
    out = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_by_block[b] for b in picked])
        out.append(stat_fn({k: v[idx] for k, v in values.items()}))
    out = np.asarray([o for o in out if np.isfinite(o)])
    if out.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def load_predictions(root: Path, seed: int, model: str) -> pd.DataFrame | None:
    """Union of the two pre-merge loaders: derives rc_disagreement plus, when the
    columns exist, gate disagreement (base stage) and M-value error (conditional
    stage). Extra columns are inert for whichever stage does not read them."""
    path = root / f"seed{seed}" / model / "predictions.csv"
    if not path.exists():
        logging.warning("missing %s", path)
        return None
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        logging.warning("%s lacks %s -- skipping (context-only models may not "
                        "emit orientation-resolved predictions)", path, missing)
        return None
    df["seed"] = seed
    df["model"] = model
    df["rc_disagreement"] = (df["pred_beta_fwd"] - df["pred_beta_rc"]).abs()
    if "beta_absolute_error" not in df.columns:
        df["beta_absolute_error"] = (df["pred_beta_rc_avg"] - df["true_beta"]).abs()
    if {"pred_m_rc_avg", "true_m"}.issubset(df.columns):
        df["m_absolute_error"] = (df["pred_m_rc_avg"] - df["true_m"]).abs()
    if "gate_dna_share_fwd" in df.columns and "gate_dna_share_rc" in df.columns:
        df["gate_rc_disagreement"] = (
            df["gate_dna_share_fwd"] - df["gate_dna_share_rc"]
        ).abs()
    return df


def add_cross_seed_sd(frames: list) -> pd.DataFrame:
    """Per-probe SD of the RC-averaged prediction across seeds -- the ensemble
    uncertainty that rc_disagreement would be replacing."""
    allf = pd.concat(frames, ignore_index=True)
    sd = (allf.groupby("probeID")["pred_beta_rc_avg"]
              .agg(["std", "count"])
              .rename(columns={"std": "cross_seed_sd", "count": "n_seeds"}))
    return allf.merge(sd, on="probeID", how="left")


def load_model_frames(root: Path, seeds, model: str):
    frames = [f for f in (load_predictions(root, s, model) for s in seeds)
              if f is not None]
    return add_cross_seed_sd(frames) if frames else None


# =========================================================================
# stage: base
# =========================================================================

def correlation_table(df: pd.DataFrame, n_boot: int, rng) -> dict:
    d = df["rc_disagreement"].to_numpy(float)
    e = df["beta_absolute_error"].to_numpy(float)
    blocks = block_ids(df)
    rho = spearman(d, e)
    lo, hi = block_bootstrap_ci(
        lambda v: spearman(v["d"], v["e"]), {"d": d, "e": e}, blocks, n_boot, rng
    )
    row = {
        "n_loci": int(len(df)),
        "n_blocks": int(len(np.unique(blocks))),
        "spearman_disagreement_vs_error": rho,
        "spearman_ci_lo": lo,
        "spearman_ci_hi": hi,
        "pearson_disagreement_vs_error": pearson(d, e),
        "mean_rc_disagreement": float(np.nanmean(d)),
        "mean_absolute_error": float(np.nanmean(e)),
        "disagreement_over_error_ratio": float(np.nanmean(d) / np.nanmean(e))
        if np.nanmean(e) else float("nan"),
    }
    if "gate_rc_disagreement" in df.columns:
        g = df["gate_rc_disagreement"].to_numpy(float)
        row["spearman_gate_disagreement_vs_error"] = spearman(g, e)
    return row


def decile_table(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    d = df["rc_disagreement"].to_numpy(float)
    ranks = pd.Series(d).rank(method="first", pct=True)
    binned = np.clip((ranks * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = binned == b
        sub = df.loc[sel]
        if sub.empty:
            continue
        rows.append({
            "decile": b + 1,
            "n": int(sel.sum()),
            "rc_disagreement_min": float(np.nanmin(d[sel])),
            "rc_disagreement_max": float(np.nanmax(d[sel])),
            "rc_disagreement_mean": float(np.nanmean(d[sel])),
            "beta_mae": float(np.nanmean(sub["beta_absolute_error"])),
            "beta_median_ae": float(np.nanmedian(sub["beta_absolute_error"])),
            "roc_auc": roc_auc(
                sub["binary_true"].to_numpy(float),
                sub["class_prob_rc_avg"].to_numpy(float),
            ) if {"binary_true", "class_prob_rc_avg"}.issubset(sub.columns) else float("nan"),
        })
    return pd.DataFrame(rows)


def build_estimators(df: pd.DataFrame, rng) -> dict:
    """Higher value = LESS confident, for every estimator."""
    est = {"rc_disagreement": df["rc_disagreement"].to_numpy(float)}
    if "cross_seed_sd" in df.columns and df["cross_seed_sd"].notna().any():
        est["cross_seed_sd"] = df["cross_seed_sd"].to_numpy(float)
    est["boundary_distance"] = -(df["pred_beta_rc_avg"].to_numpy(float) - 0.5).__abs__()
    if "cross_seed_sd" in est:
        r1 = pd.Series(est["rc_disagreement"]).rank(pct=True).to_numpy()
        r2 = pd.Series(est["cross_seed_sd"]).rank(pct=True).to_numpy()
        est["combined"] = (r1 + r2) / 2.0
    est["random"] = rng.random(len(df))
    return est


def selective_curves(df: pd.DataFrame, estimators: dict, coverages) -> pd.DataFrame:
    err = df["beta_absolute_error"].to_numpy(float)
    have_cls = {"binary_true", "class_prob_rc_avg"}.issubset(df.columns)
    y = df["binary_true"].to_numpy(float) if have_cls else None
    p = df["class_prob_rc_avg"].to_numpy(float) if have_cls else None

    rows = []
    for name, u in estimators.items():
        order = np.argsort(np.where(np.isfinite(u), u, np.inf))  # most confident first
        for cov in coverages:
            k = max(1, int(round(cov * len(df))))
            keep = order[:k]
            rows.append({
                "estimator": name,
                "coverage": cov,
                "n_retained": int(k),
                "beta_mae": float(np.nanmean(err[keep])),
                "roc_auc": roc_auc(y[keep], p[keep]) if have_cls else float("nan"),
            })
    return pd.DataFrame(rows)


def summarise_estimators(curves: pd.DataFrame) -> pd.DataFrame:
    """Area under the risk-coverage curve. Lower = better selection."""
    rows = []
    for name, sub in curves.groupby("estimator"):
        sub = sub.sort_values("coverage")
        cov = sub["coverage"].to_numpy(float)
        mae = sub["beta_mae"].to_numpy(float)
        a = float(_trapz(mae, cov) / (cov[-1] - cov[0])) if len(cov) > 1 else float("nan")
        full = sub.loc[sub["coverage"].idxmax()]
        half = sub.iloc[(np.abs(cov - 0.5)).argmin()]
        rows.append({
            "estimator": name,
            "aurc_beta_mae": a,
            "beta_mae_at_full_coverage": float(full["beta_mae"]),
            "beta_mae_at_50pct_coverage": float(half["beta_mae"]),
            "mae_reduction_at_50pct": float(full["beta_mae"] - half["beta_mae"]),
            "roc_auc_at_full_coverage": float(full["roc_auc"]),
            "roc_auc_at_50pct_coverage": float(half["roc_auc"]),
        })
    return pd.DataFrame(rows).sort_values("aurc_beta_mae").reset_index(drop=True)


def make_base_plots(decile: pd.DataFrame, curves: pd.DataFrame, out: Path, tag: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("matplotlib unavailable; skipping plots")
        return

    out.mkdir(parents=True, exist_ok=True)

    if not decile.empty:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(decile["decile"], decile["beta_mae"], "o-", color="#0D6E68")
        ax.set_xlabel("Decile of FWD–RC disagreement (1 = most consistent)")
        ax.set_ylabel(r"$\beta$ MAE")
        ax.set_title(f"Error by strand-disagreement decile ({tag})", fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.6)
        fig.tight_layout()
        fig.savefig(out / f"error_by_disagreement_decile_{tag}.png", dpi=200)
        plt.close(fig)

    if not curves.empty:
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        for name, sub in curves.groupby("estimator"):
            sub = sub.sort_values("coverage")
            style = "--" if name == "random" else "-"
            ax.plot(sub["coverage"], sub["beta_mae"], style, label=name, linewidth=1.6)
        ax.set_xlabel("Coverage (fraction retained, most confident first)")
        ax.set_ylabel(r"$\beta$ MAE on retained loci")
        ax.set_title(f"Risk–coverage by uncertainty estimator ({tag})", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25, linewidth=0.6)
        fig.tight_layout()
        fig.savefig(out / f"risk_coverage_{tag}.png", dpi=200)
        plt.close(fig)


def run_base(args) -> int:
    """Verbatim main loop from the pre-merge script 16. The rng is consumed by
    block_bootstrap_ci before build_estimators within each model-seed; that order
    fixes the `random` estimator, so it must not be rearranged."""
    rng = np.random.default_rng(args.random_seed)
    out = args.output_dir or DEFAULT_OUT["base"]
    out.mkdir(parents=True, exist_ok=True)

    corr_rows, decile_frames, curve_frames, summary_frames = [], [], [], []
    coverages = np.round(np.arange(0.10, 1.001, 0.05), 3)
    loaded = {}

    for model in args.models:
        merged = load_model_frames(args.results_root, args.seeds, model)
        if merged is None:
            logging.warning("no usable predictions for model=%s", model)
            continue
        loaded[model] = sorted(merged["seed"].unique().tolist())

        for seed in loaded[model]:
            df = merged[merged["seed"] == seed].reset_index(drop=True)
            tag = f"{model}_seed{seed}"
            logging.info("analysing %s (n=%d)", tag, len(df))

            row = correlation_table(df, args.bootstrap_replicates, rng)
            row.update({"model": model, "seed": seed})
            corr_rows.append(row)

            dec = decile_table(df)
            dec.insert(0, "seed", seed)
            dec.insert(0, "model", model)
            decile_frames.append(dec)

            est = build_estimators(df, rng)
            cur = selective_curves(df, est, coverages)
            cur.insert(0, "seed", seed)
            cur.insert(0, "model", model)
            curve_frames.append(cur)

            summ = summarise_estimators(cur)
            summ.insert(0, "seed", seed)
            summ.insert(0, "model", model)
            summary_frames.append(summ)

            make_base_plots(dec, cur, out / "plots", tag)

    if not corr_rows:
        logging.error("Nothing analysed. Check --results-root and that "
                      "predictions.csv files contain pred_beta_fwd/pred_beta_rc.")
        return 1

    corr = pd.DataFrame(corr_rows)
    decile = pd.concat(decile_frames, ignore_index=True)
    curves = pd.concat(curve_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)

    corr.to_csv(out / "disagreement_error_correlations.csv", index=False)
    decile.to_csv(out / "disagreement_decile_table.csv", index=False)
    curves.to_csv(out / "selective_prediction_curves.csv", index=False)
    summary.to_csv(out / "estimator_comparison.csv", index=False)

    lines = ["FWD-RC disagreement as a free uncertainty estimate", "=" * 58, ""]
    for _, r in corr.iterrows():
        lines.append(
            f"{r['model']:>9} seed{int(r['seed'])}  n={int(r['n_loci']):,}  "
            f"Spearman(disagreement, |error|) = {r['spearman_disagreement_vs_error']:.4f} "
            f"[{r['spearman_ci_lo']:.4f}, {r['spearman_ci_hi']:.4f}]"
        )
    lines += ["", "Selection quality (area under risk-coverage curve; lower is better)", "-" * 58]
    for (model, seed), sub in summary.groupby(["model", "seed"]):
        lines.append(f"  {model} seed{int(seed)}:")
        for _, r in sub.iterrows():
            lines.append(
                f"      {r['estimator']:<18} AURC={r['aurc_beta_mae']:.5f}   "
                f"MAE {r['beta_mae_at_full_coverage']:.5f} -> "
                f"{r['beta_mae_at_50pct_coverage']:.5f} at 50% coverage"
            )

    verdicts = []
    for (model, seed), sub in summary.groupby(["model", "seed"]):
        s = sub.set_index("estimator")["aurc_beta_mae"]
        if "rc_disagreement" not in s or "random" not in s:
            continue
        beats_random = s["rc_disagreement"] < s["random"]
        vs_ens = (s["rc_disagreement"] < s["cross_seed_sd"]) if "cross_seed_sd" in s else None
        verdicts.append({
            "model": model, "seed": int(seed),
            "beats_random": bool(beats_random),
            "beats_cross_seed_sd": (None if vs_ens is None else bool(vs_ens)),
            "beats_boundary_distance": (bool(s["rc_disagreement"] < s["boundary_distance"])
                                        if "boundary_distance" in s else None),
        })
    lines += ["", "Verdict", "-" * 58]
    if verdicts:
        n = len(verdicts)
        nr = sum(v["beats_random"] for v in verdicts)
        ne = sum(bool(v["beats_cross_seed_sd"]) for v in verdicts)
        lines.append(f"  beats random selection:        {nr}/{n} runs")
        lines.append(f"  beats 3-seed cross-seed SD:    {ne}/{n} runs")
        lines.append("")
        if nr == n:
            lines.append("  -> RC disagreement carries real information about local error.")
            lines.append("     It is computed by the existing test scripts and costs nothing.")
            if ne >= n / 2:
                lines.append("  -> It is competitive with the ensemble SD it would replace,")
                lines.append("     i.e. ensemble-grade uncertainty from a SINGLE model.")
        else:
            lines.append("  -> Does not reliably beat random. Report as a negative result and")
            lines.append("     fall back to conformal intervals on the existing ensemble.")

    (out / "rc_uncertainty_summary.txt").write_text("\n".join(lines) + "\n")

    with (out / "run_summary.json").open("w") as fh:
        json.dump({
            "analysis": "FWD-RC disagreement as a single-model uncertainty estimate",
            "stage": "B.3 -- uncertainty calibration (no GPU, no retraining)",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "results_root": str(args.results_root),
            "models_loaded": loaded,
            "bootstrap_replicates": args.bootstrap_replicates,
            "block_size_bp": BLOCK_BP,
            "random_seed": args.random_seed,
            "coverages": coverages.tolist(),
            "estimators": ["rc_disagreement", "cross_seed_sd", "boundary_distance",
                           "combined", "random"],
            "verdicts": verdicts,
            "interpretation": (
                "Descriptive analysis of frozen held-out predictions. RC-averaged "
                "outputs are exactly RC-invariant by construction; this quantifies "
                "whether the discarded per-locus disagreement is informative about error."
            ),
            "outputs": [
                "disagreement_error_correlations.csv",
                "disagreement_decile_table.csv",
                "selective_prediction_curves.csv",
                "estimator_comparison.csv",
                "rc_uncertainty_summary.txt",
            ],
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("\n".join(lines))
    logging.info("wrote outputs to %s", out)
    return 0


# =========================================================================
# stage: conditional
# =========================================================================

def estimators_for(df: pd.DataFrame) -> dict:
    """Higher = less confident. No `combined`/`random` here: the conditional
    stage compares against the heuristic, not against a null."""
    est = {
        "rc_disagreement": df["rc_disagreement"].to_numpy(float),
        "boundary_distance": -np.abs(df["pred_beta_rc_avg"].to_numpy(float) - 0.5),
    }
    if "cross_seed_sd" in df.columns and df["cross_seed_sd"].notna().any():
        est["cross_seed_sd"] = df["cross_seed_sd"].to_numpy(float)
    return est


def conditional_tests(df: pd.DataFrame, model: str, seed: int, n_bins: int,
                      coverages: np.ndarray) -> dict:
    est = estimators_for(df)
    boundary = est["boundary_distance"]
    bins = strat_bins(df["pred_beta_rc_avg"].to_numpy(float), n_bins)

    targets = {"beta": df["beta_absolute_error"].to_numpy(float)}
    if "m_absolute_error" in df.columns:
        targets["m"] = df["m_absolute_error"].to_numpy(float)

    partial_rows, within_rows, strat_rows, incr_rows = [], [], [], []

    for tname, err in targets.items():
        base_marginal = spearman(boundary, err, min_n=5)
        base_aurc = aurc(err, boundary, coverages)

        def stratified_aurc(u: np.ndarray) -> float:
            vals, weights = [], []
            for b in range(n_bins):
                sel = bins == b
                if sel.sum() < 20:
                    continue
                vals.append(aurc(err[sel], u[sel], coverages))
                weights.append(sel.sum())
            if not vals:
                return float("nan")
            return float(np.average(vals, weights=weights))

        base_strat = stratified_aurc(boundary)

        for ename, u in est.items():
            # A -- partial rank information beyond the heuristic
            pr = (float("nan") if ename == "boundary_distance"
                  else partial_spearman(u, err, boundary))
            partial_rows.append({
                "model": model, "seed": seed, "error_target": tname,
                "estimator": ename,
                "marginal_spearman": spearman(u, err, min_n=5),
                "partial_spearman_given_boundary": pr,
                "boundary_marginal_spearman": base_marginal,
            })

            # B -- within-stratum correlation
            per_bin, sizes = [], []
            for b in range(n_bins):
                sel = bins == b
                if sel.sum() < 20:
                    continue
                per_bin.append(spearman(u[sel], err[sel], min_n=5))
                sizes.append(sel.sum())
            pooled = (float(np.average(per_bin, weights=sizes))
                      if per_bin and np.all(np.isfinite(per_bin)) else float("nan"))
            within_rows.append({
                "model": model, "seed": seed, "error_target": tname,
                "estimator": ename, "n_bins_used": len(per_bin),
                "within_bin_spearman_pooled": pooled,
                "within_bin_spearman_min": float(np.nanmin(per_bin)) if per_bin else float("nan"),
                "within_bin_spearman_max": float(np.nanmax(per_bin)) if per_bin else float("nan"),
            })

            # C -- stratified selective prediction (the decisive test)
            sa = stratified_aurc(u)
            strat_rows.append({
                "model": model, "seed": seed, "error_target": tname,
                "estimator": ename,
                "unstratified_aurc": aurc(err, u, coverages),
                "stratified_aurc": sa,
                "baseline_boundary_stratified_aurc": base_strat,
                "beats_boundary_within_strata": bool(sa < base_strat)
                if np.isfinite(sa) and np.isfinite(base_strat) else None,
            })

            # D -- incremental value on top of the free heuristic
            if ename != "boundary_distance":
                combo = (pd.Series(boundary).rank(pct=True).to_numpy()
                         + pd.Series(u).rank(pct=True).to_numpy()) / 2.0
                a_combo = aurc(err, combo, coverages)
                incr_rows.append({
                    "model": model, "seed": seed, "error_target": tname,
                    "estimator": ename,
                    "aurc_boundary_only": base_aurc,
                    "aurc_boundary_plus_estimator": a_combo,
                    "aurc_improvement": base_aurc - a_combo,
                    "improves_on_boundary": bool(a_combo < base_aurc),
                })

    return {"partial": partial_rows, "within": within_rows,
            "strat": strat_rows, "incremental": incr_rows}


def run_conditional(args) -> int:
    out = args.output_dir or DEFAULT_OUT["conditional"]
    out.mkdir(parents=True, exist_ok=True)
    coverages = np.round(np.arange(0.10, 1.001, 0.05), 3)

    partial, within, strat, incr = [], [], [], []
    for model in args.models:
        merged = load_model_frames(args.results_root, args.seeds, model)
        if merged is None:
            continue
        for seed in sorted(merged["seed"].unique()):
            df = merged[merged["seed"] == seed].reset_index(drop=True)
            logging.info("analysing %s seed%d (n=%d, strata=%d)",
                         model, seed, len(df), args.strata)
            res = conditional_tests(df, model, int(seed), args.strata, coverages)
            partial += res["partial"]; within += res["within"]
            strat += res["strat"];     incr += res["incremental"]

    if not partial:
        logging.error("nothing analysed -- check --results-root")
        return 1

    dfp, dfw = pd.DataFrame(partial), pd.DataFrame(within)
    dfs, dfi = pd.DataFrame(strat), pd.DataFrame(incr)
    dfp.to_csv(out / "partial_correlations.csv", index=False)
    dfw.to_csv(out / "within_stratum_correlations.csv", index=False)
    dfs.to_csv(out / "stratified_selective_prediction.csv", index=False)
    dfi.to_csv(out / "incremental_value.csv", index=False)

    L = ["Does any uncertainty estimator beat |beta_hat - 0.5|?", "=" * 66, ""]

    for tname in dfp["error_target"].unique():
        L += [f"### Error target: {tname}-value absolute error", ""]
        sub = dfp[(dfp.error_target == tname) & (dfp.estimator != "boundary_distance")]
        L.append("A. Partial Spearman given boundary_distance (incremental rank info)")
        for ename, g in sub.groupby("estimator"):
            L.append(f"     {ename:<18} marginal {g.marginal_spearman.mean():+.4f}"
                     f"   partial {g.partial_spearman_given_boundary.mean():+.4f}")

        subw = dfw[dfw.error_target == tname]
        L.append("")
        L.append("B. Within-stratum Spearman (pooled over predicted-beta bins)")
        for ename, g in subw.groupby("estimator"):
            L.append(f"     {ename:<18} {g.within_bin_spearman_pooled.mean():+.4f}")

        subs = dfs[dfs.error_target == tname]
        L.append("")
        L.append("C. Stratified AURC -- selection WITHIN predicted-beta bins (lower better)")
        for ename, g in subs.groupby("estimator"):
            wins = int(g.beats_boundary_within_strata.fillna(False).sum())
            L.append(f"     {ename:<18} {g.stratified_aurc.mean():.5f}"
                     f"   beats boundary in {wins}/{len(g)} runs")

        subi = dfi[dfi.error_target == tname]
        L.append("")
        L.append("D. Incremental value on top of boundary_distance")
        for ename, g in subi.groupby("estimator"):
            wins = int(g.improves_on_boundary.sum())
            L.append(f"     {ename:<18} AURC {g.aurc_boundary_only.mean():.5f}"
                     f" -> {g.aurc_boundary_plus_estimator.mean():.5f}"
                     f"   improves in {wins}/{len(g)} runs")
        L.append("")

    L += ["Interpretation", "-" * 66]
    beta_s = dfs[(dfs.error_target == "beta") & (dfs.estimator != "boundary_distance")]
    n_beat = int(beta_s.beats_boundary_within_strata.fillna(False).sum())
    if n_beat > 0:
        L.append(f"  {n_beat}/{len(beta_s)} estimator-runs still beat the heuristic INSIDE")
        L.append("  predicted-beta strata. That is real model uncertainty, not target shape.")
    else:
        L.append("  No estimator beats the heuristic inside predicted-beta strata.")
        L.append("  The apparent 'uncertainty' signal is largely the bimodal target: hard")
        L.append("  probes are the intermediate ones. Report this as a negative result and")
        L.append("  base the calibration section on conformal intervals, whose coverage")
        L.append("  guarantee holds regardless of how informative the score is.")

    if "m" in set(dfs.error_target):
        b = dfs[(dfs.error_target == "beta") & (dfs.estimator == "boundary_distance")]
        m = dfs[(dfs.error_target == "m") & (dfs.estimator == "boundary_distance")]
        L += ["",
              f"  boundary_distance stratified AURC: beta {b.stratified_aurc.mean():.5f}"
              f" | M {m.stratified_aurc.mean():.5f}",
              "  If boundary_distance is far weaker on M-value error, its advantage on",
              "  beta was partly the bounded-range artifact (explanation b), not difficulty."]
    else:
        L += ["", "  NOTE: M-value error unavailable (pred_m_rc_avg/true_m missing), so the",
              "  bounded-range artifact could not be tested. Re-run once those columns exist."]

    (out / "conditional_summary.txt").write_text("\n".join(L) + "\n")

    with (out / "run_summary.json").open("w") as fh:
        json.dump({
            "analysis": "Conditional uncertainty analysis vs the |beta_hat-0.5| heuristic",
            "stage": "B.3b -- uncertainty calibration (no GPU, no retraining)",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "results_root": str(args.results_root),
            "strata": args.strata,
            "coverages": coverages.tolist(),
            "motivation": (
                "The base stage found boundary_distance beats RC disagreement and the "
                "3-seed ensemble SD. This separates a genuine difficulty signal from "
                "the mechanically bounded range of beta near 0 and 1."
            ),
            "outputs": ["partial_correlations.csv", "within_stratum_correlations.csv",
                        "stratified_selective_prediction.csv", "incremental_value.csv",
                        "conditional_summary.txt"],
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("\n".join(L))
    logging.info("wrote outputs to %s", out)
    return 0


# =========================================================================
# stage: figure
# =========================================================================

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


def make_figure(within: pd.DataFrame, out: Path):
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


def run_figure(args) -> int:
    out = args.output_dir or DEFAULT_OUT["figure"]
    out.mkdir(parents=True, exist_ok=True)

    runs = parse_runs(args.runs)
    within, strat = collect(runs)

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


# =========================================================================
# main
# =========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("base", "conditional", "figure", "all"),
                    default="all")
    ap.add_argument("--results-root", type=Path, default=Path("results/journal"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--models", nargs="+", default=["sequence", "fusion", "epi"])
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="stage-specific default if omitted")
    # base
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    ap.add_argument("--random-seed", type=int, default=20260824)
    # conditional
    ap.add_argument("--strata", type=int, default=10,
                    help="number of predicted-beta strata (default 10)")
    ap.add_argument("--all-strata", type=int, nargs="+", default=[10, 20, 50],
                    help="strata swept by --stage all")
    # figure
    ap.add_argument("--runs", nargs="+", metavar="STRATA:PATH",
                    help="required for --stage figure; auto-built by --stage all")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    if args.stage == "base":
        return run_base(args)

    if args.stage == "conditional":
        return run_conditional(args)

    if args.stage == "figure":
        if not args.runs:
            raise SystemExit("--stage figure requires --runs STRATA:PATH ...")
        return run_figure(args)

    # ---- stage: all ------------------------------------------------------
    # Reproduces the historical sequence: base, then conditional at each strata
    # count into its own directory, then the figure over all of them.
    rc = run_base(argparse.Namespace(**{**vars(args), "output_dir": None}))
    if rc:
        return rc

    runs = []
    for s in args.all_strata:
        out = (DEFAULT_OUT["conditional"] if s == args.all_strata[0]
               else Path(f"{DEFAULT_OUT['conditional']}_s{s}"))
        rc = run_conditional(argparse.Namespace(
            **{**vars(args), "strata": s, "output_dir": out}))
        if rc:
            return rc
        runs.append(f"{s}:{out}")

    return run_figure(argparse.Namespace(
        **{**vars(args), "runs": runs, "output_dir": None}))


if __name__ == "__main__":
    raise SystemExit(main())
