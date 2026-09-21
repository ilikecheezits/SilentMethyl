#!/usr/bin/env python3
"""Derive eGTEx's own significance threshold instead of applying the GWAS constant 5e-8, which
is the genome-wide correction for a GWAS rather than a cis scan on 50-100 donors. Applies
the study's own standard: a probe is an mCpG if its q-value passes --fdr, then within those
probes a pair is significant if its nominal p is at or below a cutoff calibrated from the
permutation results. The per-probe beta-distribution parameters are absent from the file, so
a cohort-wide fallback is used and labelled as an approximation, and counts are reported at
every FDR level so the choice is visible.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("silentmethyl.egtex_significance")

DEFAULT_PERM = Path("data/external/egtex_breast/BreastMammaryTissue.regular.perm.fdr.txt")
FALLBACK_PERM = Path("data/BreastMammaryTissue.regular.perm.fdr.txt")
DEFAULT_SCORING = Path("data/external/egtex_breast/scoring")
FDR_LEVELS = (0.01, 0.05, 0.10)
NOMINAL_GRID = (5e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def load_perm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    expected = ["cpg_id", "variant_id", "maf", "slope", "slope_se",
                "pval_nominal", "pval_permuted", "qval"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing {missing}; columns found: {list(df.columns)}")
    df = df.rename(columns={"cpg_id": "probeID"})
    df["probeID"] = df["probeID"].astype(str)
    for col in ("maf", "slope", "slope_se", "pval_nominal", "pval_permuted", "qval"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["probeID"].duplicated().any():
        n = int(df["probeID"].duplicated().sum())
        raise SystemExit(f"{path} has {n} duplicate probe rows; expected one per probe")
    return df


def nominal_threshold(perm: pd.DataFrame, fdr: float) -> tuple[float, int]:
    """Largest lead nominal p among probes passing FDR -- the standard fallback."""
    passing = perm[perm["qval"] <= fdr]
    if passing.empty:
        return float("nan"), 0
    return float(passing["pval_nominal"].max()), int(len(passing))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perm", type=Path, default=None)
    ap.add_argument("--scoring-dir", type=Path, default=DEFAULT_SCORING)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--write-probe-table", action="store_true",
                    help="Write egtex_probe_significance.csv for merging at "
                         "evaluation time. Does not modify the scoring inputs.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_SCORING)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    perm_path = args.perm
    if perm_path is None:
        perm_path = DEFAULT_PERM if DEFAULT_PERM.is_file() else FALLBACK_PERM
    if not perm_path.is_file():
        raise SystemExit(f"permutation file not found at {DEFAULT_PERM} or {FALLBACK_PERM}")

    LOGGER.info("reading %s", perm_path)
    perm = load_perm(perm_path)
    LOGGER.info("%d probes with permutation results", len(perm))

    summary: dict = {
        "analysis": "eGTEx Breast Mammary significance definition",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "permutation_file": str(perm_path),
        "probes_tested": int(len(perm)),
        "method": ("two-stage FastQTL/GTEx: probe-level FDR on permutation "
                   "p-values, then a nominal cutoff calibrated from the probes "
                   "that pass. The per-probe beta-distribution shape parameters "
                   "are not in this file, so the cohort-wide fallback is used: "
                   "the nominal p of the least-significant probe passing FDR."),
        "caveat": ("The stage-2 cutoff is an approximation to GTEx's per-probe "
                   "threshold. State it as such; do not describe these as "
                   "'eGTEx-significant pairs' without the qualifier."),
    }

    print()
    print("=" * 74)
    print("STAGE 1 -- probe-level (how many CpGs have an mQTL at all)")
    print(f"{'FDR':>8}  {'probes':>10}  {'% of tested':>12}  {'nominal cutoff':>16}")
    stage1 = {}
    for level in FDR_LEVELS:
        thresh, n = nominal_threshold(perm, level)
        stage1[str(level)] = {"probes": n, "fraction": n / len(perm),
                              "nominal_cutoff": thresh}
        print(f"{level:>8.2f}  {n:>10,}  {100*n/len(perm):>11.2f}%  {thresh:>16.3e}")
    summary["stage1_probe_level"] = stage1

    chosen_cutoff, n_probes = nominal_threshold(perm, args.fdr)
    summary["chosen"] = {"fdr": args.fdr, "significant_probes": n_probes,
                         "nominal_cutoff": chosen_cutoff}
    print(f"\nchosen: FDR <= {args.fdr} -> {n_probes:,} mCpG probes, "
          f"nominal cutoff p <= {chosen_cutoff:.3e}")
    print(f"  for comparison, the GWAS constant we had been using is 5.0e-08 "
          f"({chosen_cutoff/5e-8:.0f}x stricter than the cohort's own calibration)")

    sig_probes = set(perm.loc[perm["qval"] <= args.fdr, "probeID"])
    lead = perm.set_index("probeID")

    print()
    print("=" * 74)
    print("STAGE 2 -- pair-level counts in the SilentMethyl scoring inputs")
    strata = {}
    for stratum in ("heldout", "model_visible"):
        path = args.scoring_dir / f"egtex_scoring_input_{stratum}.csv"
        if not path.is_file():
            LOGGER.warning("%s not found; skipping", path)
            continue
        pairs = pd.read_csv(path, usecols=["probeID", "pvalue", "Variant_ID",
                                           "abs_distance_bp"])
        pairs["probeID"] = pairs["probeID"].astype(str)
        on_sig_probe = pairs["probeID"].isin(sig_probes)

        row = {
            "pairs": int(len(pairs)),
            "probes": int(pairs["probeID"].nunique()),
            "probes_that_are_mcpg": int(pairs.loc[on_sig_probe, "probeID"].nunique()),
            "pairs_on_mcpg_probes": int(on_sig_probe.sum()),
            "by_nominal": {},
            "two_stage": {},
        }
        for cut in NOMINAL_GRID:
            row["by_nominal"][f"{cut:.0e}"] = int((pairs["pvalue"] <= cut).sum())
        row["by_nominal"][f"{chosen_cutoff:.3e} (calibrated)"] = int(
            (pairs["pvalue"] <= chosen_cutoff).sum())
        row["two_stage"]["mcpg_probe_and_calibrated_nominal"] = int(
            (on_sig_probe & (pairs["pvalue"] <= chosen_cutoff)).sum())

        joined = pairs[on_sig_probe].join(
            lead[["variant_id"]], on="probeID", rsuffix="_lead")
        row["lead_variant_in_window"] = int(
            (joined["Variant_ID"].astype(str) == joined["variant_id"].astype(str)).sum())
        strata[stratum] = row

        print(f"\n{stratum}: {row['pairs']:,} pairs on {row['probes']:,} probes")
        print(f"  probes that are mCpGs at FDR {args.fdr}: "
              f"{row['probes_that_are_mcpg']:,} "
              f"({100*row['probes_that_are_mcpg']/max(row['probes'],1):.1f}%)")
        print(f"  pairs on those probes              : {row['pairs_on_mcpg_probes']:,}")
        print("  pairs by nominal p:")
        for key, value in row["by_nominal"].items():
            print(f"    p <= {key:<26} {value:>9,}")
        print(f"  TWO-STAGE (mCpG probe AND p <= calibrated): "
              f"{row['two_stage']['mcpg_probe_and_calibrated_nominal']:,}")
        print(f"  of which the probe's own lead variant     : "
              f"{row['lead_variant_in_window']:,}")
    summary["strata"] = strata

    if args.write_probe_table:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        table = perm[["probeID", "variant_id", "maf", "slope", "slope_se",
                      "pval_nominal", "pval_permuted", "qval"]].copy()
        table = table.rename(columns={"variant_id": "egtex_lead_variant_id",
                                      "slope": "egtex_lead_slope",
                                      "pval_nominal": "egtex_lead_pval_nominal"})
        table["is_mcpg_probe"] = table["qval"] <= args.fdr
        out = args.output_dir / "egtex_probe_significance.csv"
        table.to_csv(out, index=False)
        summary["probe_table"] = str(out)
        print(f"\nwrote {out}")

    with (args.output_dir / "egtex_significance_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    print()
    print("=" * 74)
    print("Declare the threshold BEFORE scoring, and report the significance")
    print("gradient regardless. The gradient is threshold-free and is what")
    print("actually demonstrates the signal is real rather than fitted noise.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
