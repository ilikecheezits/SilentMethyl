#!/usr/bin/env python3
"""
Stage B.3b -- Does ANY uncertainty estimator add information beyond |beta_hat - 0.5|?

Why this exists
---------------
Script 16 found that `boundary_distance` (= -|beta_hat - 0.5|), a zero-parameter
heuristic, beats both RC disagreement and the 3-seed ensemble SD at selective
prediction, in essentially every run. Before any of this goes in a manuscript,
two explanations must be separated:

  (a) REAL      Intermediate-methylation probes are genuinely harder. Distance
                from 0.5 is a legitimate difficulty signal and a good uncertainty
                estimator ought to capture it.

  (b) ARTIFACT  Absolute beta error is mechanically bounded near the extremes.
                At beta_hat = 0.02 there is almost no room to be wrong; at 0.5
                there is +/- 0.5. Boundary distance may be measuring HEADROOM,
                not uncertainty.

If (b) dominates, an entire calibration section would be reporting the shape of
the target distribution rather than anything about the model -- and a reviewer
at any of the target journals will say so.

Five tests, all CPU-only on existing predictions.csv files
----------------------------------------------------------
  A  Partial Spearman(estimator, error | boundary_distance)
       Incremental rank information, controlling for the heuristic.

  B  Within-stratum Spearman: bin by beta_hat, correlate inside each bin.
       Removes the between-bin effect entirely.

  C  Stratified selective prediction: select within each beta_hat bin, then pool.
       The decisive test. An estimator that selects well HERE is doing real work.

  D  Incremental AURC: boundary_distance alone vs boundary_distance + estimator.
       "Is it worth anything on top of the free heuristic?"

  E  Everything repeated on M-value error, which is unbounded (logit scale).
       If boundary_distance's advantage collapses on M-error, explanation (b)
       was driving it.

Usage
-----
    python -u scripts/17_uncertainty_conditional_analysis.py \
        --results-root results/journal \
        --seeds 42 43 44 \
        --models sequence fusion epi \
        --output-dir results/journal/rc_uncertainty_conditional
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
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------- utilities

def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return float("nan")
    if sps is not None:
        r = sps.spearmanr(a[ok], b[ok])
        v = getattr(r, "statistic", None)
        return float(v if v is not None else r[0])
    return float(np.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1])


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


def aurc(err: np.ndarray, unc: np.ndarray, coverages: np.ndarray) -> float:
    """Area under the risk-coverage curve. Lower = better selection."""
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


# ------------------------------------------------------------------ loading

def load_predictions(root: Path, seed: int, model: str):
    path = root / f"seed{seed}" / model / "predictions.csv"
    if not path.exists():
        logging.warning("missing %s", path)
        return None
    df = pd.read_csv(path)
    if [c for c in REQUIRED if c not in df.columns]:
        logging.warning("%s lacks orientation columns -- skipping", path)
        return None
    df["seed"], df["model"] = seed, model
    df["rc_disagreement"] = (df["pred_beta_fwd"] - df["pred_beta_rc"]).abs()
    if "beta_absolute_error" not in df.columns:
        df["beta_absolute_error"] = (df["pred_beta_rc_avg"] - df["true_beta"]).abs()
    if {"pred_m_rc_avg", "true_m"}.issubset(df.columns):
        df["m_absolute_error"] = (df["pred_m_rc_avg"] - df["true_m"]).abs()
    return df


def with_cross_seed_sd(frames: list) -> pd.DataFrame:
    allf = pd.concat(frames, ignore_index=True)
    sd = (allf.groupby("probeID")["pred_beta_rc_avg"].agg(["std"])
              .rename(columns={"std": "cross_seed_sd"}))
    return allf.merge(sd, on="probeID", how="left")


def estimators_for(df: pd.DataFrame) -> dict:
    """Higher = less confident."""
    est = {
        "rc_disagreement": df["rc_disagreement"].to_numpy(float),
        "boundary_distance": -np.abs(df["pred_beta_rc_avg"].to_numpy(float) - 0.5),
    }
    if "cross_seed_sd" in df.columns and df["cross_seed_sd"].notna().any():
        est["cross_seed_sd"] = df["cross_seed_sd"].to_numpy(float)
    return est


# ------------------------------------------------------------------- tests

def run_all(df: pd.DataFrame, model: str, seed: int, n_bins: int,
            coverages: np.ndarray) -> dict:
    est = estimators_for(df)
    boundary = est["boundary_distance"]
    bins = strat_bins(df["pred_beta_rc_avg"].to_numpy(float), n_bins)

    targets = {"beta": df["beta_absolute_error"].to_numpy(float)}
    if "m_absolute_error" in df.columns:
        targets["m"] = df["m_absolute_error"].to_numpy(float)

    partial_rows, within_rows, strat_rows, incr_rows = [], [], [], []

    for tname, err in targets.items():
        base_marginal = spearman(boundary, err)
        base_aurc = aurc(err, boundary, coverages)

        # ---- C: stratified AURC for the baseline, pooled over bins
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
                "marginal_spearman": spearman(u, err),
                "partial_spearman_given_boundary": pr,
                "boundary_marginal_spearman": base_marginal,
            })

            # B -- within-stratum correlation
            per_bin, sizes = [], []
            for b in range(n_bins):
                sel = bins == b
                if sel.sum() < 20:
                    continue
                per_bin.append(spearman(u[sel], err[sel]))
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


# -------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=Path("results/journal"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--models", nargs="+", default=["sequence", "fusion", "epi"])
    ap.add_argument("--output-dir", type=Path,
                    default=Path("results/journal/rc_uncertainty_conditional"))
    ap.add_argument("--strata", type=int, default=10,
                    help="number of predicted-beta strata (default 10)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    coverages = np.round(np.arange(0.10, 1.001, 0.05), 3)

    partial, within, strat, incr = [], [], [], []
    for model in args.models:
        frames = [f for f in (load_predictions(args.results_root, s, model)
                              for s in args.seeds) if f is not None]
        if not frames:
            continue
        merged = with_cross_seed_sd(frames)
        for seed in sorted(merged["seed"].unique()):
            df = merged[merged["seed"] == seed].reset_index(drop=True)
            logging.info("analysing %s seed%d (n=%d)", model, seed, len(df))
            res = run_all(df, model, int(seed), args.strata, coverages)
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

    # ------------------------------------------------------------- verdict
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

    # headline reads
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
                "Script 16 found boundary_distance beats RC disagreement and the "
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


if __name__ == "__main__":
    raise SystemExit(main())
