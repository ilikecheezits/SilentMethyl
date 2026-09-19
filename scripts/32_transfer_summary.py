#!/usr/bin/env python3
"""Aggregate the per-tissue transfer results into the single R3 table.
 
Why this exists
---------------
scripts/31_transfer_discrimination.py runs ONE tissue at a time and writes
 
    results/journal/transfer_discrimination/<Tissue>/
        run_summary.json                cohort sizes and the matching audit
        per_model_on_shared_cohort.csv  marginal metrics with block-bootstrap CIs
        tail_enrichment.csv             model vs distance, paired CIs
 
Nine directories is not a Results section. This reads all of them and emits one
table answering the R3 question directly: does a model trained only on breast
tissue prioritise variants above a distance-matched null in each tissue, and how
does that track the tissue's statistical power.
 
Nothing is recomputed. Every value is read from what 31 already wrote, so this
cannot disagree with the per-tissue outputs.
 
Two audits run automatically, because both are silent failure modes:
 
1. **Matching integrity.** `distance_only_auroc_after_matching` must be 0.5.
   Anything else means the distance-matched cohort is not distance-matched, and
   every AUROC in the column is confounded. Aborts.
 
2. **Cohort retention.** The pairs 31 actually used are compared against the
   harmonised cohort recorded in split_summary.json. A large or uneven drop is
   not necessarily wrong -- but it must be explained in Methods rather than
   noticed by a reviewer, so it is printed as its own column.
 
Usage (run from the repository root)
------------------------------------
    python -u scripts/32_transfer_summary.py
    python -u scripts/32_transfer_summary.py --model sequence
    python -u scripts/32_transfer_summary.py --fraction 0.005
"""
 
from __future__ import annotations
 
import argparse
import json
import logging
import sys
from pathlib import Path
 
import pandas as pd
 
ROOT = Path("results/journal/transfer_discrimination")
SPLIT_SUMMARY = Path("results/journal/egtex_multitissue_scoring/by_tissue/split_summary.json")
 
AUROC = "AUROC, distance-matched"
RHO = "signed rho (significant)"
AGREE = "direction agreement"
 
MATCHING_TOL = 1e-6
 
 
def read_tissue(path: Path, model: str, fraction: float) -> dict | None:
    """Pull one tissue's row. Returns None if the tissue did not finish."""
    summary = path / "run_summary.json"
    per_model = path / "per_model_on_shared_cohort.csv"
    tails = path / "tail_enrichment.csv"
    if not (summary.is_file() and per_model.is_file() and tails.is_file()):
        logging.warning("%-22s incomplete, skipped", path.name)
        return None
 
    with summary.open() as fh:
        meta = json.load(fh)
 
    matching = meta.get("matching", {})
    dist_auroc = matching.get("distance_only_auroc_after_matching")
    if dist_auroc is None:
        raise SystemExit(f"STOP: {summary} has no distance_only_auroc_after_matching")
    if abs(dist_auroc - 0.5) > MATCHING_TOL:
        raise SystemExit(
            f"STOP: {path.name} distance-only AUROC after matching is "
            f"{dist_auroc:.6f}, not 0.5. The matched null is not "
            f"distance-matched and its AUROC is confounded by distance.")
 
    marg = pd.read_csv(per_model)
    marg = marg[marg["model"] == model]
    if marg.empty:
        raise SystemExit(f"STOP: {per_model} has no rows for model '{model}'")
 
    def metric(name: str) -> tuple[float, float, float]:
        hit = marg[marg["metric"] == name]
        if len(hit) != 1:
            raise SystemExit(f"STOP: {per_model} has {len(hit)} rows for '{name}'")
        r = hit.iloc[0]
        return float(r["value"]), float(r["ci_low"]), float(r["ci_high"])
 
    auroc, auroc_lo, auroc_hi = metric(AUROC)
    rho, rho_lo, rho_hi = metric(RHO)
    agree, _, _ = metric(AGREE)
 
    tail = pd.read_csv(tails)
    hit = tail[(tail["model"] == model) & (tail["fraction"] == fraction)]
    if len(hit) != 1:
        raise SystemExit(
            f"STOP: {tails} has {len(hit)} rows for model '{model}' at "
            f"fraction {fraction}. Available: "
            f"{sorted(tail['fraction'].unique().tolist())}")
    t = hit.iloc[0]
    base = float(t["base_rate"])
 
    return {
        "tissue": path.name,
        "pairs": int(meta["shared_cohort_pairs"]),
        "significant": int(meta["significant_pairs"]),
        "base_rate": base,
        "median_distance_bp": matching.get("median_distance_significant"),
        "unmatched_slots": matching.get("unmatched_slots"),
        "distance_only_auroc": float(dist_auroc),
        "auroc": auroc, "auroc_lo": auroc_lo, "auroc_hi": auroc_hi,
        "signed_rho": rho, "rho_lo": rho_lo, "rho_hi": rho_hi,
        "direction_agreement": agree,
        "tail_precision": float(t["value"]),
        "tail_fold": float(t["value"]) / base if base > 0 else float("nan"),
        "tail_diff": float(t["difference_vs_distance"]),
        "tail_diff_lo": float(t["diff_ci_low"]),
        "tail_diff_hi": float(t["diff_ci_high"]),
        "beats_distance": bool(t["interval_excludes_zero"]),
    }
 
 
def attach_retention(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare the pairs 31 used against the harmonised cohort it drew from."""
    if not SPLIT_SUMMARY.is_file():
        logging.warning("%s not found -- retention column omitted", SPLIT_SUMMARY)
        frame["cohort_pairs"] = pd.NA
        frame["retention"] = pd.NA
        return frame
    with SPLIT_SUMMARY.open() as fh:
        rows = json.load(fh)["per_tissue"]
    lookup = {r["tissue"]: (r["rows"], r["significant"]) for r in rows}
    frame["cohort_pairs"] = frame["tissue"].map(lambda t: lookup.get(t, (pd.NA,))[0])
    frame["cohort_significant"] = frame["tissue"].map(
        lambda t: lookup.get(t, (pd.NA, pd.NA))[1])
    frame["retention"] = frame["pairs"] / frame["cohort_pairs"]
    frame["sig_retention"] = frame["significant"] / frame["cohort_significant"]
    return frame
 
 
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--model", default="fusion",
                    help="which arm to tabulate (default: fusion)")
    ap.add_argument("--fraction", type=float, default=0.001,
                    help="tail fraction for the enrichment column (default: 0.001)")
    ap.add_argument("--out", type=Path, default=None,
                    help="CSV path (default: <root>/nine_tissue_summary.csv)")
    args = ap.parse_args(argv)
 
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
 
    if not args.root.is_dir():
        raise SystemExit(f"STOP: {args.root} not found -- run from the repo root")
 
    dirs = sorted(p for p in args.root.iterdir() if p.is_dir())
    rows = [r for r in (read_tissue(p, args.model, args.fraction) for p in dirs)
            if r is not None]
    if not rows:
        raise SystemExit(f"STOP: no completed tissues under {args.root}")
 
    frame = attach_retention(pd.DataFrame(rows))
    frame = frame.sort_values("significant", ascending=False).reset_index(drop=True)
 
    pct = int(round(args.fraction * 100_000)) / 1000
    print()
    print("=" * 104)
    print(f"R3  TRANSFER FROM BREAST  --  model: {args.model}   tail: top {pct:g}%")
    print("=" * 104)
    print(f"{'tissue':<22}{'sig':>6}{'pairs':>8}{'base':>8}"
          f"{'AUROC [95% CI]':>24}{'tail':>7}{'fold':>7}"
          f"{'  vs distance [95% CI]':<26}")
    print("-" * 104)
    for _, r in frame.iterrows():
        mark = "*" if r["beats_distance"] else " "
        print(f"{r['tissue']:<22}{r['significant']:>6,}{r['pairs']:>8,}"
              f"{r['base_rate']:>8.4f}"
              f"{r['auroc']:>10.4f} [{r['auroc_lo']:.4f},{r['auroc_hi']:.4f}]"
              f"{r['tail_precision']:>7.3f}{r['tail_fold']:>6.1f}x"
              f"  {r['tail_diff']:>+7.4f} [{r['tail_diff_lo']:+.4f},"
              f"{r['tail_diff_hi']:+.4f}] {mark}")
    print("-" * 104)
    print("* = paired block-bootstrap interval on (model - distance) excludes zero")
    print("AUROC is on the distance-matched cohort, where distance alone gives "
          "exactly 0.5000 by construction.")
 
    if frame["retention"].notna().any():
        print()
        print("COHORT RETENTION  (pairs 31 used / pairs the harmoniser produced)")
        print("-" * 104)
        print(f"{'tissue':<22}{'used':>9}{'harmonised':>12}{'kept':>8}"
              f"{'sig used':>10}{'sig harm':>10}{'sig kept':>10}")
        for _, r in frame.iterrows():
            print(f"{r['tissue']:<22}{r['pairs']:>9,}{r['cohort_pairs']:>12,}"
                  f"{r['retention']:>7.1%}{r['significant']:>10,}"
                  f"{r['cohort_significant']:>10,}{r['sig_retention']:>10.1%}")
        lo, hi = frame["retention"].min(), frame["retention"].max()
        print()
        if hi - lo > 0.05:
            print(f"NOTE: retention ranges {lo:.1%} to {hi:.1%} across tissues. "
                  f"Uneven retention changes what each tissue's cohort IS, so the\n"
                  f"      cross-tissue comparison is only fair once the reason is "
                  f"identified. Explain it in Methods.")
        else:
            print(f"Retention is uniform ({lo:.1%}-{hi:.1%}), so the filter applies "
                  f"equally across tissues and the comparison is like-for-like.\n"
                  f"State the filter and this range in Methods.")
 
    out = args.out or (args.root / "nine_tissue_summary.csv")
    frame.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(frame)} tissues)")
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())