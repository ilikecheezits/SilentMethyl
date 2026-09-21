#!/usr/bin/env python3
"""Test whether the learned fusion gate tracks measured cross-tissue methylation variance.
Nothing in training identifies tissue-variable loci, so a correlation would make the
gate an unsupervised readout of tissue plasticity. Reports the association before and
after partialling out methylation level, and keeps every statement at the distribution
level.
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

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger("silentmethyl.gateplasticity")

TISSUE_VAL = {
    "BreastEpithelium": "data/datafiles_breast_epithelium/val.csv",
    "ColonTransverse": "data/datafiles_multitissue/ColonTransverse/val.csv",
    "KidneyCortex": "data/datafiles_multitissue/KidneyCortex/val.csv",
    "Lung": "data/datafiles_multitissue/Lung/val.csv",
}


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 10:
        return float("nan"), float("nan")
    r = stats.spearmanr(x[finite], y[finite])
    return float(r.statistic), float(r.pvalue)


def _block_bootstrap_spearman(x, y, blocks, rng, draws: int) -> dict:
    """Percentile CI for Spearman rho, resampling 1 Mb blocks.

    Neighbouring CpGs are correlated in both methylation and gate value, so
    resampling probes individually would give a badly over-tight interval.
    """
    finite = np.isfinite(x) & np.isfinite(y)
    x, y, blocks = x[finite], y[finite], blocks[finite]
    uniq = np.unique(blocks)
    index = {b: np.flatnonzero(blocks == b) for b in uniq}
    out = []
    for _ in range(draws):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[b] for b in picked])
        if len(idx) < 10:
            continue
        out.append(stats.spearmanr(x[idx], y[idx]).statistic)
    out = np.asarray([v for v in out if np.isfinite(v)], dtype=float)
    if not len(out):
        return {}
    lo, hi = np.quantile(out, [0.025, 0.975])
    return {"block_bootstrap_95ci": [float(lo), float(hi)],
            "block_bootstrap_draws": int(len(out)),
            "n_blocks": int(len(uniq))}


def load_frame(args) -> pd.DataFrame:
    gates = pd.read_csv(args.gates_csv)
    LOGGER.info("gates: %d rows from %s", len(gates), args.gates_csv)

    betas = []
    for tissue, path in TISSUE_VAL.items():
        d = pd.read_csv(path, usecols=["probeID", "Median_Beta"])
        d = d.rename(columns={"Median_Beta": tissue}).set_index("probeID")
        betas.append(d)
        LOGGER.info("%s: %d probes", tissue, len(d))
    beta = pd.concat(betas, axis=1, join="inner").reset_index()
    LOGGER.info("probes measured in all four tissues: %d", len(beta))

    assign = pd.read_csv(args.joint_val_csv, usecols=["probeID", "Tissue"])
    df = gates.merge(beta, on="probeID", how="inner", validate="one_to_one")
    df = df.merge(assign, on="probeID", how="left", validate="one_to_one")
    LOGGER.info("matched gate+4-tissue probes: %d", len(df))

    tis = list(TISSUE_VAL)
    mat = df[tis].to_numpy(float)
    df["cross_tissue_var"] = mat.var(axis=1, ddof=0)
    df["cross_tissue_sd"] = np.sqrt(df["cross_tissue_var"])
    df["cross_tissue_range"] = mat.max(axis=1) - mat.min(axis=1)
    df["cross_tissue_mean_beta"] = mat.mean(axis=1)
    df["block"] = (df["chr"].astype(str) + ":"
                   + (df["pos"].to_numpy() // 1_000_000).astype(str))
    return df


def run(args) -> int:
    df = load_frame(args)
    rng = np.random.default_rng(args.seed)
    blocks = df["block"].to_numpy()
    var = df["cross_tissue_var"].to_numpy(float)

    share_mae = float(np.mean(np.abs(df["gate_dna_share_fwd"] - df["gate_dna_share_rc"])))
    share_q = np.quantile(df["gate_dna_share_avg"], [0.10, 0.90])
    noise = {
        "gate_dna_share_fwd_rc_mae": share_mae,
        "gate_dna_share_q10_q90": [float(share_q[0]), float(share_q[1])],
        "gate_dna_share_q10_q90_range": float(share_q[1] - share_q[0]),
        "noise_to_range_ratio": float(share_mae / (share_q[1] - share_q[0])),
        "interpretation": (
            "Forward/RC disagreement at the same locus is pure measurement noise "
            "in the gate. It attenuates every correlation below toward zero, so "
            "each is a lower bound. Not disattenuated by design."),
    }

    df["gate_total_avg"] = df["gate_dna_avg"] + df["gate_epi_avg"]
    def _rank(a):
        return stats.rankdata(np.asarray(a, dtype=float))

    def _resid(y, x):
        A = np.column_stack([x, np.ones_like(x)])
        return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]

    r_var_g = _rank(df["cross_tissue_var"])
    r_lvl_g = _rank(df["cross_tissue_mean_beta"])
    r_var_resid = _resid(r_var_g, r_lvl_g)

    def _partial_vs_var_given_level(x):
        return float(np.corrcoef(_resid(_rank(x), r_lvl_g), r_var_resid)[0, 1])

    predictors = {
        "gate_dna_share_avg": "learned DNA/context share (primary)",
        "gate_dna_avg": "raw DNA gate alone",
        "gate_epi_avg": "raw context gate alone",
        "gate_total_avg": "gate_dna + gate_epi, total fused-vector magnitude",
    }

    results = {}
    for col, desc in predictors.items():
        x = df[col].to_numpy(float)
        rho, p = _spearman(x, var)
        entry = {"description": desc, "spearman_rho": rho, "p_value": p,
                 "n": int(np.isfinite(x).sum())}
        entry.update(_block_bootstrap_spearman(x, var, blocks, rng, args.bootstrap_draws))
        entry["spearman_rho_vs_sd"] = _spearman(x, df["cross_tissue_sd"].to_numpy(float))[0]
        entry["spearman_rho_vs_range"] = _spearman(x, df["cross_tissue_range"].to_numpy(float))[0]
        entry["spearman_rho_vs_mean_beta"] = _spearman(
            x, df["cross_tissue_mean_beta"].to_numpy(float))[0]
        entry["partial_spearman_vs_var_given_mean_beta"] = _partial_vs_var_given_level(x)
        results[col] = entry
        LOGGER.info("%-20s rho=%+.4f  p=%.3g  CI=%s", col, rho, p,
                    entry.get("block_bootstrap_95ci"))

    stratified = {}
    for tissue, sub in df.groupby("Tissue"):
        x = sub["gate_dna_share_avg"].to_numpy(float)
        v = sub["cross_tissue_var"].to_numpy(float)
        rho, p = _spearman(x, v)
        entry = {"n": int(len(sub)), "spearman_rho": rho, "p_value": p}
        entry.update(_block_bootstrap_spearman(x, v, sub["block"].to_numpy(),
                                               rng, args.bootstrap_draws))
        stratified[str(tissue)] = entry
        LOGGER.info("  [%s] n=%d rho=%+.4f", tissue, len(sub), rho)

    df["share_decile"] = pd.qcut(df["gate_dna_share_avg"], 10,
                                  labels=False, duplicates="drop")
    decile = (df.groupby("share_decile")
                .agg(n=("probeID", "size"),
                     gate_dna_share_median=("gate_dna_share_avg", "median"),
                     cross_tissue_var_median=("cross_tissue_var", "median"),
                     cross_tissue_var_mean=("cross_tissue_var", "mean"),
                     cross_tissue_sd_median=("cross_tissue_sd", "median"),
                     mean_beta_median=("cross_tissue_mean_beta", "median"))
                .reset_index())

    rho_level, _ = _spearman(df["gate_dna_share_avg"].to_numpy(float),
                             df["cross_tissue_mean_beta"].to_numpy(float))
    partial_rho = _partial_vs_var_given_level(df["gate_dna_share_avg"])

    payload = {
        "analysis": "gate share vs measured cross-tissue methylation plasticity",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "single_seed_note": ("Seed 42 only, by design -- this characterises one "
                             "trained model's learned gate, not run-to-run spread."),
        "gates_csv": str(args.gates_csv),
        "gate_split": "joint all4 validation loci (chr10, chr11)",
        "tissue_sources": TISSUE_VAL,
        "probes": int(len(df)),
        "bootstrap": f"{args.bootstrap_draws} draws over 1 Mb genomic blocks",
        "cross_tissue_variance_definition": (
            "population variance (ddof=0) of Median_Beta across the four tissue "
            "builds at the same probe; measured targets, not predictions"),
        "gate_measurement_noise": noise,
        "correlations": results,
        "stratified_by_assigned_training_tissue": stratified,
        "level_confound": {
            "spearman_gate_share_vs_mean_beta": rho_level,
            "partial_spearman_gate_share_vs_variance_given_mean_beta": partial_rho,
            "why": ("Cross-tissue variance is mechanically compressed at beta~0 and "
                    "beta~1. If the gate merely tracks methylation level, the raw "
                    "correlation would be a level effect wearing a plasticity "
                    "costume. The partial coefficient removes rank-linear level."),
        },
        "decile_table": json.loads(decile.to_json(orient="records")),
        "distribution_level_only": (
            "Per-locus gate share is too noisy to quote for individual loci; every "
            "statement here is a distribution-level summary."),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "gate_plasticity_summary.json"
    tmp = out_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(out_json)

    keep = ["probeID", "chr", "pos", "Tissue", "gate_dna_avg", "gate_epi_avg",
            "gate_total_avg",
            "gate_dna_share_avg", "gate_dna_share_fwd", "gate_dna_share_rc",
            *TISSUE_VAL, "cross_tissue_var", "cross_tissue_sd",
            "cross_tissue_range", "cross_tissue_mean_beta"]
    out_csv = args.output_dir / "gate_plasticity_per_probe.csv"
    tmp_csv = out_csv.with_suffix(".csv.tmp")
    df[keep].to_csv(tmp_csv, index=False)
    tmp_csv.replace(out_csv)

    print()
    print("=" * 78)
    print(f"gate share vs cross-tissue plasticity   ({len(df):,} probes, chr10+chr11)")
    print("-" * 78)
    for col, e in results.items():
        ci = e.get("block_bootstrap_95ci")
        ci_s = f"  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else ""
        print(f"  rho {e['spearman_rho']:+.4f}{ci_s}   "
              f"partial(|beta) {e['partial_spearman_vs_var_given_mean_beta']:+.4f}   {col}")
    print(f"\n  gate share vs mean beta (confound):   rho {rho_level:+.4f}")
    print(f"  partial rho (variance | mean beta):   {partial_rho:+.4f}")
    print(f"\n  attenuation floor: fwd/RC share MAE {share_mae:.4f} "
          f"vs q10-q90 range {share_q[1]-share_q[0]:.4f}")
    print("\n  by assigned training tissue:")
    for t, e in stratified.items():
        ci = e.get("block_bootstrap_95ci")
        ci_s = f"  [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        print(f"    {t:18s} n={e['n']:6d}  rho {e['spearman_rho']:+.4f}{ci_s}")
    print("\n  decile of gate DNA share -> median cross-tissue variance:")
    for r in decile.itertuples():
        print(f"    d{int(r.share_decile)+1:<2d} share~{r.gate_dna_share_median:.3f}  "
              f"var {r.cross_tissue_var_median:.5f}  n={r.n}")
    print("=" * 78)
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gates-csv", type=Path,
                   default=Path("checkpoints_joint/all4/seed42/fusion/best_validation_gates.csv"))
    p.add_argument("--joint-val-csv", type=Path,
                   default=Path("data/datafiles_joint/all4/val.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-draws", type=int, default=1000)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/journal/joint/gate_plasticity"))
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    sys.exit(run(parse_args()))
