#!/usr/bin/env python3
"""
Stage B.3 -- Is forward/reverse-complement disagreement a free uncertainty estimate?

Motivation
----------
SilentMethyl reports RC-averaged predictions:  beta_hat = 1/2 [f(x) + f(RC(x))].
That average is *exactly* RC-invariant, so the reported output has no strand
inconsistency. But the averaging discards the per-locus disagreement
|f(x) - f(RC(x))|, which is already computed at every locus and written to
predictions.csv as pred_beta_fwd / pred_beta_rc.

Hypothesis: that discarded disagreement measures how well the model has
internalised the reverse-complement symmetry AT THAT LOCUS, and therefore
predicts local error -- giving an uncertainty estimate from a SINGLE model at
zero additional cost, where the usual route is an ensemble.

The test that matters is not "does it correlate with error" (weak, easy) but
"does it select well, and is it competitive with the 3-seed ensemble SD it would
replace." This script answers both, against two controls.

Estimators compared
-------------------
    rc_disagreement   |pred_beta_fwd - pred_beta_rc|      single model, free
    cross_seed_sd     SD of pred_beta_rc_avg over seeds   needs N models
    boundary_distance -|pred_beta_rc_avg - 0.5|           naive control
    combined          rank-average of the first two       complementarity check
    random            null

Everything here is CPU-only and reads files that already exist. No GPU, no
re-inference, no retraining.

Usage
-----
    python -u scripts/16_rc_disagreement_uncertainty.py \
        --results-root results/journal \
        --seeds 42 43 44 \
        --models sequence fusion epi \
        --output-dir results/journal/rc_uncertainty
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


# ---------------------------------------------------------------- utilities

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    if sps is not None:
        r = sps.spearmanr(a[ok], b[ok])
        return float(getattr(r, "statistic", None) if getattr(r, "statistic", None) is not None else r[0])
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC; NaN if the retained set is single-class."""
    ok = np.isfinite(y_true) & np.isfinite(score)
    y, s = y_true[ok], score[ok]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank().to_numpy()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


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


# ------------------------------------------------------------------ loading

def load_predictions(root: Path, seed: int, model: str) -> pd.DataFrame | None:
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


# ---------------------------------------------------------------- analyses

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
        aurc = float(_trapz(mae, cov) / (cov[-1] - cov[0])) if len(cov) > 1 else float("nan")
        full = sub.loc[sub["coverage"].idxmax()]
        half = sub.iloc[(np.abs(cov - 0.5)).argmin()]
        rows.append({
            "estimator": name,
            "aurc_beta_mae": aurc,
            "beta_mae_at_full_coverage": float(full["beta_mae"]),
            "beta_mae_at_50pct_coverage": float(half["beta_mae"]),
            "mae_reduction_at_50pct": float(full["beta_mae"] - half["beta_mae"]),
            "roc_auc_at_full_coverage": float(full["roc_auc"]),
            "roc_auc_at_50pct_coverage": float(half["roc_auc"]),
        })
    return pd.DataFrame(rows).sort_values("aurc_beta_mae").reset_index(drop=True)


# -------------------------------------------------------------------- plots

def make_plots(decile: pd.DataFrame, curves: pd.DataFrame, out: Path, tag: str) -> None:
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


# --------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=Path("results/journal"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--models", nargs="+", default=["sequence", "fusion", "epi"])
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/journal/rc_uncertainty"))
    ap.add_argument("--bootstrap-replicates", type=int, default=2000)
    ap.add_argument("--random-seed", type=int, default=20260824)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    rng = np.random.default_rng(args.random_seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    corr_rows, decile_frames, curve_frames, summary_frames = [], [], [], []
    coverages = np.round(np.arange(0.10, 1.001, 0.05), 3)
    loaded = {}

    for model in args.models:
        frames = [f for f in (load_predictions(args.results_root, s, model)
                              for s in args.seeds) if f is not None]
        if not frames:
            logging.warning("no usable predictions for model=%s", model)
            continue
        merged = add_cross_seed_sd(frames)
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

            make_plots(dec, cur, out / "plots", tag)

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

    # ---- plain-language verdict -------------------------------------------
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

    run_summary = {
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
    }
    with (out / "run_summary.json").open("w") as fh:
        json.dump(run_summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("\n".join(lines))
    logging.info("wrote outputs to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
