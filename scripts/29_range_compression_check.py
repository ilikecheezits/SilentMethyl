#!/usr/bin/env python
"""Is the MASK_snp5_common oddity range compression?

scripts/27 found that on the MASK_snp5_common stratum, beta MAE *improves* while
ROC-AUC *falls* -- for all three architectures. Lower error with worse
discrimination is the signature of a less bimodal target distribution: when
targets cluster near the middle, a model that regresses toward the mean earns a
low absolute error while the binary labels it is scored against sit close to the
decision boundary and become hard to separate.

That was recorded in REQUIREMENTS.md E.1 as a HYPOTHESIS, explicitly not to be
written into the manuscript until checked. This checks it, and the check is
decisive rather than suggestive: it takes the retained stratum, subsamples it to
match the excluded stratum's beta histogram, and recomputes both metrics on
probes that were never flagged. If range compression is the explanation, a
distribution-matched retained sample reproduces the excluded stratum's numbers,
and the flag is doing none of the work.

If it does, the same mechanism is the leading explanation for the open
tumour-domain question (context-only does not degrade on TCGA and its M MAE
improves), and that connection may be stated. If it does not, the tumour
question stays open and this must NOT be written up as compression.

No GPU, no model, no retraining. Re-reads the same predictions.csv files that
scripts/27 reads, with the same column definitions.

    python -u scripts/29_range_compression_check.py

Exit codes:  0 verdict reached (either way)   1 could not run   2 inconclusive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROBE_ID_CANDIDATES = ("probeID", "IlmnID", "Name")
# Must match scripts/27_mask_snp_sensitivity.py and scripts/02_*.
REQUIRED_PREDICTION_COLUMNS = ("probeID", "true_beta", "pred_beta_rc_avg",
                               "binary_true", "class_prob_rc_avg")

INTERMEDIATE = (0.3, 0.7)      # the band that makes a binary label ambiguous
N_MATCHED_DRAWS = 200          # repeats of the distribution-matched subsample

# Agreement tolerances for the verdict. The matched sample cannot reproduce the
# excluded histogram exactly -- binning is finite and some bins run short -- so
# these are set wide enough to absorb that residual and narrow enough that a
# real difference in model quality (which is an order of magnitude larger; see
# the negative control in the docstring of the fixture) still fails.
AUC_TOLERANCE = 0.010
MAE_TOLERANCE = 0.005
# If the retained pool cannot supply this fraction of the excluded stratum's
# probes at the right beta values, the matching is not honest and no verdict is
# issued.
MAX_BIN_SHORTFALL_FRACTION = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results/journal")
    p.add_argument("--manifest", default="data/HM450.hg38.manifest.tsv.gz")
    p.add_argument("--mask-column", default="MASK_snp5_common")
    p.add_argument("--models", nargs="+", default=["fusion", "sequence", "epi"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/journal/range_compression")
    return p.parse_args()


def coerce_boolean(series: pd.Series, name: str) -> pd.Series:
    """HM450 manifests ship these as TRUE/FALSE, True/False, or 1/0."""
    if series.dtype == bool:
        return series
    lowered = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False,
               "yes": True, "no": False}
    unknown = sorted(set(lowered) - set(mapping))
    if unknown:
        raise ValueError(f"{name} has unmappable values: {unknown[:5]}")
    return lowered.map(mapping)


def metrics(frame: pd.DataFrame) -> dict:
    """Definitions copied from scripts/27 so the numbers are comparable."""
    err = (frame["pred_beta_rc_avg"].to_numpy(float)
           - frame["true_beta"].to_numpy(float))
    y = frame["binary_true"].to_numpy(int)
    auc = (float(roc_auc_score(y, frame["class_prob_rc_avg"].to_numpy(float)))
           if len(np.unique(y)) == 2 else float("nan"))
    return {"n": int(len(frame)), "beta_mae": float(np.mean(np.abs(err))),
            "auc": auc}


def shape(betas: np.ndarray) -> dict:
    """How bimodal is this target distribution?"""
    lo, hi = INTERMEDIATE
    return {
        "sd": float(np.std(betas)),
        "frac_intermediate": float(np.mean((betas > lo) & (betas < hi))),
        "frac_extreme": float(np.mean((betas <= 0.1) | (betas >= 0.9))),
        "mean_distance_from_half": float(np.mean(np.abs(betas - 0.5))),
    }


def matched_subsample(retained: pd.DataFrame, excluded: pd.DataFrame,
                      bins: int, rng: np.random.Generator) -> tuple[list[dict], int]:
    """Draw retained probes so their beta histogram matches the excluded one.

    Matching on the TARGET distribution only -- no feature, no annotation, no
    model output enters the matching. If the excluded stratum's metrics are
    reproduced by unflagged probes with the same target distribution, the flag
    is irrelevant and the distribution is doing the work.

    Returns the per-draw metrics and the number of probes the retained pool
    could not supply at the right beta values.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    ex_counts, _ = np.histogram(excluded["true_beta"].to_numpy(float), bins=edges)
    ret_bin = np.digitize(retained["true_beta"].to_numpy(float), edges[1:-1])
    pools = {b: retained.index[ret_bin == b].to_numpy() for b in range(bins)}
    shortfall = sum(max(0, int(c) - len(pools.get(b, [])))
                    for b, c in enumerate(ex_counts))
    out: list[dict] = []
    for _ in range(N_MATCHED_DRAWS):
        picks = []
        for b, c in enumerate(ex_counts):
            pool = pools.get(b, np.array([], dtype=int))
            if len(pool) == 0 or c == 0:
                continue
            picks.append(rng.choice(pool, size=min(int(c), len(pool)),
                                    replace=False))
        if picks:
            out.append(metrics(retained.loc[np.concatenate(picks)]))
    return out, shortfall


def main() -> int:
    args = parse_args()
    root = Path(args.results_root)
    rng = np.random.default_rng(args.random_seed)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"STOP: manifest not found at {manifest_path}")
        return 1
    manifest = pd.read_csv(manifest_path, sep="\t", compression="infer",
                           low_memory=False)
    probe_col = next((c for c in PROBE_ID_CANDIDATES if c in manifest.columns), None)
    if probe_col is None:
        print(f"STOP: {manifest_path} has no probe ID column "
              f"(looked for {PROBE_ID_CANDIDATES})")
        return 1
    if args.mask_column not in manifest.columns:
        available = [c for c in manifest.columns if c.startswith("MASK")]
        print(f"STOP: {args.mask_column} not in manifest. "
              f"MASK columns present: {available}")
        return 1
    mask = manifest[[probe_col, args.mask_column]].rename(
        columns={probe_col: "probeID"})
    mask["probeID"] = mask["probeID"].astype(str)
    mask[args.mask_column] = coerce_boolean(mask[args.mask_column], args.mask_column)
    if mask["probeID"].duplicated().any():
        print(f"STOP: duplicate probe IDs in {manifest_path}")
        return 1
    print(f"manifest: {len(mask)} probes, {args.mask_column} true for "
          f"{int(mask[args.mask_column].sum())} "
          f"({100.0 * mask[args.mask_column].mean():.2f}%)")

    rows, shapes, notes = [], [], []
    for model in args.models:
        for seed in args.seeds:
            path = root / f"seed{seed}" / model / "predictions.csv"
            if not path.exists():
                print(f"missing {path} -- skipped")
                continue
            preds = pd.read_csv(path)
            absent = [c for c in REQUIRED_PREDICTION_COLUMNS
                      if c not in preds.columns]
            if absent:
                print(f"STOP: {path} is missing {absent}. "
                      f"Columns present: {list(preds.columns)}")
                return 1
            preds["probeID"] = preds["probeID"].astype(str)
            m = preds.merge(mask, on="probeID", how="left", validate="many_to_one")
            unmatched = int(m[args.mask_column].isna().sum())
            if unmatched:
                # Same handling as scripts/27: not fatal, but it must be visible.
                print(f"{model} seed {seed}: {unmatched}/{len(m)} probes absent "
                      f"from the manifest; excluded from both strata")
            m = m[m[args.mask_column].notna()].reset_index(drop=True)
            flag = m[args.mask_column].astype(bool)
            retained, excluded = m[~flag], m[flag]
            if retained.empty or excluded.empty:
                notes.append(f"{model} seed {seed}: one stratum is empty")
                continue

            r_m, e_m = metrics(retained), metrics(excluded)
            matched, shortfall = matched_subsample(
                retained.reset_index(drop=True), excluded, args.bins, rng)
            if not matched:
                notes.append(f"{model} seed {seed}: no matched draw was possible")
                continue
            mm = pd.DataFrame(matched)
            rows.append({
                "model": model, "seed": seed,
                "retained_n": r_m["n"], "excluded_n": e_m["n"],
                "retained_mae": r_m["beta_mae"], "retained_auc": r_m["auc"],
                "excluded_mae": e_m["beta_mae"], "excluded_auc": e_m["auc"],
                "matched_retained_mae": float(mm["beta_mae"].mean()),
                "matched_retained_auc": float(mm["auc"].mean()),
                "matched_mae_sd": float(mm["beta_mae"].std(ddof=0)),
                "matched_auc_sd": float(mm["auc"].std(ddof=0)),
                "matched_n": float(mm["n"].mean()),
                "bin_shortfall": int(shortfall),
                "bin_shortfall_fraction": float(shortfall / max(e_m["n"], 1)),
            })
            if seed == args.seeds[0]:
                shapes.append({"model": model, "stratum": "retained",
                               **shape(retained["true_beta"].to_numpy(float))})
                shapes.append({"model": model, "stratum": "excluded",
                               **shape(excluded["true_beta"].to_numpy(float))})

    if not rows:
        print("no usable model-seed combinations found")
        for n in notes:
            print(f"  {n}")
        return 1
    table = pd.DataFrame(rows)
    shape_tab = pd.DataFrame(shapes).drop_duplicates(["model", "stratum"])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "range_compression.csv", index=False)
    shape_tab.to_csv(out / "target_distribution_shape.csv", index=False)

    print("\n" + "=" * 78)
    print("TARGET DISTRIBUTION SHAPE (a property of the probes, not of any model)")
    print("=" * 78)
    s = shape_tab[shape_tab["model"] == shape_tab["model"].iloc[0]]
    for _, r in s.iterrows():
        print(f"  {r['stratum']:<9} sd {r['sd']:.4f}   intermediate "
              f"{100 * r['frac_intermediate']:5.1f}%   extreme "
              f"{100 * r['frac_extreme']:5.1f}%   mean |beta-0.5| "
              f"{r['mean_distance_from_half']:.4f}")
    ret_row = s[s["stratum"] == "retained"].iloc[0]
    exc_row = s[s["stratum"] == "excluded"].iloc[0]
    less_bimodal = bool(exc_row["frac_intermediate"] > ret_row["frac_intermediate"])
    print(f"\n  the excluded stratum is {'LESS' if less_bimodal else 'NOT less'} "
          f"bimodal than the retained one -- this is the premise of the test")

    print("\n" + "=" * 78)
    print("THE DECISIVE COMPARISON")
    print("=" * 78)
    print("'matched-retained' is unflagged probes resampled to the excluded")
    print("stratum's beta histogram. If range compression explains the pattern,")
    print("it reproduces the excluded stratum's metrics.\n")
    agg = table.groupby("model")[["retained_mae", "excluded_mae",
                                  "matched_retained_mae", "retained_auc",
                                  "excluded_auc", "matched_retained_auc",
                                  "bin_shortfall_fraction"]].mean()
    for model, r in agg.iterrows():
        print(f"{model}")
        print(f"  beta MAE   retained {r['retained_mae']:.4f}   "
              f"excluded {r['excluded_mae']:.4f}   "
              f"matched-retained {r['matched_retained_mae']:.4f}   "
              f"|gap| {abs(r['matched_retained_mae'] - r['excluded_mae']):.4f}")
        print(f"  ROC-AUC    retained {r['retained_auc']:.4f}   "
              f"excluded {r['excluded_auc']:.4f}   "
              f"matched-retained {r['matched_retained_auc']:.4f}   "
              f"|gap| {abs(r['matched_retained_auc'] - r['excluded_auc']):.4f}")
    print(f"\n  matched draws per model-seed {N_MATCHED_DRAWS}, "
          f"bins {args.bins}, tolerances AUC {AUC_TOLERANCE} / MAE {MAE_TOLERANCE}")
    print(f"  worst bin shortfall {100 * agg['bin_shortfall_fraction'].max():.2f}% "
          f"of the excluded stratum "
          f"(limit {100 * MAX_BIN_SHORTFALL_FRACTION:.0f}%)")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    close_auc = (agg["matched_retained_auc"] - agg["excluded_auc"]).abs()
    close_mae = (agg["matched_retained_mae"] - agg["excluded_mae"]).abs()
    verdict = "explained"
    if not np.isfinite(close_auc.to_numpy()).all() or \
            not np.isfinite(close_mae.to_numpy()).all():
        verdict = "inconclusive_nan"
    elif agg["bin_shortfall_fraction"].max() > MAX_BIN_SHORTFALL_FRACTION:
        verdict = "inconclusive_shortfall"
    elif not ((close_auc < AUC_TOLERANCE).all()
              and (close_mae < MAE_TOLERANCE).all()):
        verdict = "not_explained"

    if verdict == "explained":
        print("RANGE COMPRESSION CONFIRMED. Unflagged probes with the same target")
        print("distribution reproduce the excluded stratum's metrics, so the flag")
        print("is not doing the work -- the target distribution is. The same")
        print("mechanism is then the leading explanation for the open tumour-domain")
        print("result, and that connection may be stated in the manuscript.")
    elif verdict == "not_explained":
        print("NOT EXPLAINED by the target distribution alone. Matching on beta")
        print("does not reproduce the excluded stratum's metrics, so something")
        print("other than range compression is involved. The tumour-domain")
        print("question stays open and this must NOT be written up as compression.")
        print(f"  max |AUC gap| {close_auc.max():.4f} "
              f"(tolerance {AUC_TOLERANCE}), "
              f"max |MAE gap| {close_mae.max():.4f} "
              f"(tolerance {MAE_TOLERANCE})")
    elif verdict == "inconclusive_shortfall":
        print("INCONCLUSIVE. The retained pool cannot supply enough probes at the")
        print("excluded stratum's beta values, so the 'matched' sample is not")
        print("actually matched and neither verdict is earned. Nothing about range")
        print("compression may be written up from this run.")
        print(f"  worst shortfall {100 * agg['bin_shortfall_fraction'].max():.2f}%")
    else:
        print("INCONCLUSIVE. At least one AUC is undefined (a stratum or a matched")
        print("draw contained a single class), so the comparison cannot be made.")
        print("Nothing about range compression may be written up from this run.")
    for n in notes:
        print(f"  note: {n}")

    (out / "run_summary.json").write_text(json.dumps({
        "mask_column": args.mask_column,
        "manifest": str(args.manifest),
        "models": args.models,
        "seeds": args.seeds,
        "bins": args.bins,
        "matched_draws": N_MATCHED_DRAWS,
        "intermediate_band": list(INTERMEDIATE),
        "tolerances": {"auc": AUC_TOLERANCE, "beta_mae": MAE_TOLERANCE,
                       "max_bin_shortfall_fraction": MAX_BIN_SHORTFALL_FRACTION},
        "excluded_stratum_is_less_bimodal": less_bimodal,
        "verdict": verdict,
        "range_compression_confirmed": verdict == "explained",
        "max_abs_auc_gap": (None if not np.isfinite(close_auc.to_numpy()).all()
                            else float(close_auc.max())),
        "max_abs_mae_gap": (None if not np.isfinite(close_mae.to_numpy()).all()
                            else float(close_mae.max())),
        "notes": notes,
        "per_model": json.loads(agg.to_json()),
        "interpretation": (
            "If matched-retained reproduces excluded, the MASK_snp5_common "
            "pattern is a property of where those probes' targets sit, not of "
            "the genotype flag, and the same mechanism explains the tumour "
            "domain-shift result. If it does not, neither claim may be made."
        ),
    }, indent=2) + "\n")
    print(f"\nwrote {out}/range_compression.csv, target_distribution_shape.csv "
          f"and run_summary.json")
    print("=" * 78)
    return 0 if verdict in ("explained", "not_explained") else 2


if __name__ == "__main__":
    raise SystemExit(main())
