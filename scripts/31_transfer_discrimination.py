#!/usr/bin/env python
"""Paired, LD-aware intervals on the DIFFERENCE between two models' variant-effect
performance.

scripts/20 reports each model separately, and each model's distance-matched
cohort is drawn from its own call to `build_matched_cohort`. Two consequences:

  1. The reported intervals are marginal. Overlapping marginal intervals do NOT
     mean the difference is not significant -- the two estimates are computed on
     the same variants and are strongly positively correlated, so the interval on
     the difference is much narrower than either marginal interval suggests.
  2. Two models' matched AUROCs are not even measured against the same negative
     pairs, so part of any gap between them is matching noise.

This script fixes both. It builds ONE distance-matched cohort (matching depends
only on `significant` and `abs_distance_bp`, which are properties of the variant,
not of the model), scores every model on that identical cohort, and block
bootstraps the DIFFERENCE -- resampling the same 1-Mb blocks for both models
inside each resample, so the pairing is preserved.

Every definition is imported from scripts/20 rather than re-implemented, so these
numbers cannot drift from the ones the paper already quotes.

    python -u scripts/31_transfer_discrimination.py \
        --reference results/journal/genoa_variant_scoring::fusion::42,43,44 \
        --compare   results/journal/published_baselines/variant_scoring::deepcpg::42,43,44 \
        --compare   results/journal/published_baselines/variant_scoring::cpgenie::42,43,44 \
        --output-dir results/journal/paired_model_comparison_genoa

A spec is DIR::MODEL::SEEDS. Use SEEDS=-1 for the deterministic scripts/23
baselines, which have no seeds to average.

Exit codes:  0 ran   1 could not run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EVALUATOR = HERE / "21_variant_evaluation.py"


def load_evaluator():
    """Import scripts/20 by path -- its module name starts with a digit."""
    if not EVALUATOR.is_file():
        raise SystemExit(f"STOP: cannot find the evaluator at {EVALUATOR}")
    spec = importlib.util.spec_from_file_location("genoa_evaluator", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules["genoa_evaluator"] = module
    spec.loader.exec_module(module)
    missing = [n for n in ("load_scores", "add_seed_ensemble",
                           "build_matched_cohort", "block_bootstrap",
                           "marginal_auroc", "signed_rho", "direction_agreement",
                           "distance_only_auroc", "GENOME_WIDE")
               if not hasattr(module, n)]
    if missing:
        raise SystemExit(f"STOP: scripts/20 no longer exports {missing}; "
                         f"this script imports its definitions on purpose and "
                         f"must be updated rather than made to re-implement them")
    return module


def parse_spec(spec: str) -> tuple[Path, str, list[int]]:
    parts = spec.split("::")
    if len(parts) != 3:
        raise SystemExit(f"STOP: bad spec {spec!r}; expected DIR::MODEL::SEEDS")
    directory, model, seeds = parts
    try:
        seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    except ValueError:
        raise SystemExit(f"STOP: bad seeds in {spec!r}")
    if not seed_list:
        raise SystemExit(f"STOP: no seeds in {spec!r}")
    return Path(directory), model, seed_list


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True,
                   help="DIR::MODEL::SEEDS for the model everything is compared against.")
    p.add_argument("--compare", action="append", default=[],
                   help="DIR::MODEL::SEEDS. Repeatable.")
    p.add_argument("--stratum", default="heldout",
                   choices=("heldout", "model_visible"))
    p.add_argument("--significance", type=float, default=None,
                   help="Cohort p-value cutoff. Defaults to genome-wide 5e-8. "
                        "The eGTEx runs use the calibrated 1.483e-5.")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--match-tolerance", type=int, default=10)
    p.add_argument("--match-ratio", type=int, default=1)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/journal/paired_model_comparison"))
    return p.parse_args()


def ensemble_frame(ev, directory: Path, model: str, seeds: list[int],
                   stratum: str, significance: float) -> pd.DataFrame:
    """One row per pair for a single model: the seed ensemble, via scripts/20."""
    args = SimpleNamespace(scores_dir=directory, stratum=stratum,
                           models=[model], seeds=seeds,
                           significance=significance)
    long = ev.add_seed_ensemble(ev.load_scores(args))
    frame = long[long["Seed"] == -1]
    if frame.empty:
        raise SystemExit(f"STOP: no ensemble rows for {model} in {directory}")
    frame = frame[~frame["cpg_altering"].astype(bool)]
    if frame["Pair_UID"].duplicated().any():
        raise SystemExit(f"STOP: duplicate Pair_UIDs for {model}")
    return frame


def paired_metric(metric, col_a: str, col_b: str):
    """metric(model A) - metric(model B) on the SAME rows."""
    def difference(frame: pd.DataFrame) -> float:
        a = metric(frame.assign(Predicted_Delta_M=frame[col_a]))
        b = metric(frame.assign(Predicted_Delta_M=frame[col_b]))
        if not (np.isfinite(a) and np.isfinite(b)):
            return np.nan
        return float(a - b)
    return difference


def single_metric(metric, col: str):
    def value(frame: pd.DataFrame) -> float:
        return metric(frame.assign(Predicted_Delta_M=frame[col]))
    return value


TAIL_FRACTIONS = (0.001, 0.005, 0.01, 0.05)
TAIL_MIN_N = 25


def tail_enrichment(fraction: float, min_n: int = TAIL_MIN_N):
    """Share of the top |predicted effect| that is a real association.

    AUROC weights every pair equally, including the great majority whose
    predicted effect is near zero and whose ordering is model noise -- so it
    dilutes exactly the signal a prioritisation claim depends on. This measures
    the top of the ranking instead. The null value is the cohort's base rate,
    NOT 0.5, so it must always be reported alongside.
    """
    def metric(frame):
        if "significant" not in frame.columns:
            return float("nan")
        y = frame["significant"].to_numpy(dtype=float)
        s = np.abs(frame["Predicted_Delta_M"].to_numpy(dtype=float))
        n = max(min_n, int(round(fraction * len(frame))))
        if n > len(frame) or not np.isfinite(s).any() or y.sum() == 0:
            return float("nan")
        top = np.argsort(-s, kind="stable")[:n]
        return float(y[top].mean())
    return metric


def distance_tail(fraction: float, min_n: int = TAIL_MIN_N):
    """The same statistic with the model discarded, ranking by proximity alone.

    True meQTLs sit closer to their CpG than tested-but-null pairs, so distance
    is a free predictor and the confound that matters. On the unmatched cohort
    this is what the model has to beat; on the distance-matched cohort it should
    sit at the base rate by construction, which is a useful check that matching
    worked.
    """
    def metric(frame):
        if "significant" not in frame.columns or "abs_distance_bp" not in frame.columns:
            return float("nan")
        y = frame["significant"].to_numpy(dtype=float)
        s = -np.abs(frame["abs_distance_bp"].to_numpy(dtype=float))
        n = max(min_n, int(round(fraction * len(frame))))
        if n > len(frame) or not np.isfinite(s).any() or y.sum() == 0:
            return float("nan")
        top = np.argsort(-s, kind="stable")[:n]
        return float(y[top].mean())
    return metric


def base_rate(frame) -> float:
    if "significant" not in frame.columns or frame.empty:
        return float("nan")
    return float(frame["significant"].to_numpy(dtype=float).mean())


def main() -> int:

    args = parse_args()
    ev = load_evaluator()
    significance = (ev.GENOME_WIDE if args.significance is None
                    else args.significance)
    rng = np.random.default_rng(args.random_seed)

    specs = [parse_spec(args.reference)] + [parse_spec(s) for s in args.compare]
    if len(specs) < 2:
        print("STOP: give at least one --compare")
        return 1
    names = [s[1] for s in specs]
    if len(set(names)) != len(names):
        print(f"STOP: model names must be distinct; got {names}")
        return 1
    reference = names[0]

    frames = {}
    for directory, model, seeds in specs:
        frames[model] = ensemble_frame(ev, directory, model, seeds,
                                       args.stratum, significance)
        print(f"{model:<10} {len(frames[model]):>7,} non-CpG-altering pairs "
              f"from {directory}")

    common = set.intersection(*(set(f["Pair_UID"]) for f in frames.values()))
    if not common:
        print("STOP: the models share no Pair_UIDs -- different cohorts or "
              "different scoring runs")
        return 1
    for model, frame in frames.items():
        dropped = len(frame) - len(common)
        if dropped:
            print(f"  {model}: {dropped:,} pairs not shared by every model, dropped")
    common_list = sorted(common)

    base_columns = ["Pair_UID", "significant", "abs_distance_bp", "_block",
                    "beta_ref_to_alt"]
    wide = (frames[reference][frames[reference]["Pair_UID"].isin(common)]
            .set_index("Pair_UID").loc[common_list, base_columns[1:]]
            .reset_index())
    for model, frame in frames.items():
        column = f"dm__{model}"
        series = (frame[frame["Pair_UID"].isin(common)]
                  .set_index("Pair_UID")["Predicted_Delta_M"].loc[common_list])
        wide[column] = series.to_numpy(float)
    if wide[[f"dm__{m}" for m in names]].isna().any().any():
        print("STOP: missing predictions after alignment")
        return 1

    n_sig = int(wide["significant"].sum())
    print(f"\nshared cohort {len(wide):,} pairs, {n_sig:,} significant "
          f"at p < {significance:g}, {wide['_block'].nunique():,} 1-Mb blocks")

    matched, balance = ev.build_matched_cohort(
        wide, args.match_tolerance, args.match_ratio, rng)
    if matched.empty:
        print("STOP: distance matching produced an empty cohort")
        return 1
    print(f"matched cohort {len(matched):,} pairs "
          f"({balance['significant_n']:,} significant + "
          f"{balance['matched_null_n']:,} matched null), "
          f"distance-only AUROC after matching "
          f"{balance['distance_only_auroc_after_matching']:.4f}")
    if balance["unmatched_slots"]:
        print(f"  {balance['unmatched_slots']:,} negative slots could not be "
              f"filled at the requested distance tolerance")

    significant_only = wide[wide["significant"] == 1]

    plan = [
        (ev.marginal_auroc, matched, "AUROC, distance-matched"),
        (ev.signed_rho, significant_only, "signed rho (significant)"),
        (ev.direction_agreement, significant_only, "direction agreement"),
    ]
    for frac in TAIL_FRACTIONS:
        plan.append((tail_enrichment(frac), wide,
                     f"tail enrichment, top {frac*100:g}%"))

    per_model, differences = [], []
    for metric, frame, label in plan:
        for model in names:
            point = single_metric(metric, f"dm__{model}")(frame)
            low, high = ev.block_bootstrap(frame, single_metric(metric, f"dm__{model}"),
                                           args.n_boot, rng)
            per_model.append({"model": model, "metric": label, "n": int(len(frame)),
                              "value": point, "ci_low": low, "ci_high": high})
        for model in names[1:]:
            fn = paired_metric(metric, f"dm__{reference}", f"dm__{model}")
            point = fn(frame)
            low, high = ev.block_bootstrap(frame, fn, args.n_boot, rng)
            excludes_zero = bool(np.isfinite(low) and np.isfinite(high)
                                 and (low > 0 or high < 0))
            differences.append({
                "reference": reference, "comparison": model, "metric": label,
                "n": int(len(frame)), "difference": point,
                "ci_low": low, "ci_high": high,
                "interval_excludes_zero": excludes_zero,
            })

    tails = []
    for frac in TAIL_FRACTIONS:
        d_fn = distance_tail(frac)
        d_point = d_fn(wide)
        d_low, d_high = ev.block_bootstrap(wide, d_fn, args.n_boot, rng)
        tails.append({"model": "distance_only", "fraction": frac,
                      "n": int(len(wide)), "base_rate": base_rate(wide),
                      "value": d_point, "ci_low": d_low, "ci_high": d_high,
                      "difference_vs_distance": float("nan"),
                      "diff_ci_low": float("nan"), "diff_ci_high": float("nan"),
                      "interval_excludes_zero": None})
        for model in names:
            m_fn = single_metric(tail_enrichment(frac), f"dm__{model}")
            m_point = m_fn(wide)
            m_low, m_high = ev.block_bootstrap(wide, m_fn, args.n_boot, rng)

            def paired(frame, _m=model, _f=frac):
                a = single_metric(tail_enrichment(_f), f"dm__{_m}")(frame)
                b = distance_tail(_f)(frame)
                return float(a - b)

            p_point = paired(wide)
            p_low, p_high = ev.block_bootstrap(wide, paired, args.n_boot, rng)
            tails.append({
                "model": model, "fraction": frac, "n": int(len(wide)),
                "base_rate": base_rate(wide),
                "value": m_point, "ci_low": m_low, "ci_high": m_high,
                "difference_vs_distance": p_point,
                "diff_ci_low": p_low, "diff_ci_high": p_high,
                "interval_excludes_zero": bool(
                    np.isfinite(p_low) and np.isfinite(p_high)
                    and (p_low > 0 or p_high < 0)),
            })
    tail_tab = pd.DataFrame(tails)

    per_model_tab = pd.DataFrame(per_model)
    diff_tab = pd.DataFrame(differences)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    per_model_tab.to_csv(out / "per_model_on_shared_cohort.csv", index=False)
    diff_tab.to_csv(out / "paired_differences.csv", index=False)
    tail_tab.to_csv(out / "tail_enrichment.csv", index=False)

    print("\n" + "=" * 78)
    print("TAIL ENRICHMENT vs THE DISTANCE BASELINE (unmatched cohort)")
    print("A model that only rediscovers proximity shows no difference here.")
    print("=" * 78)
    for frac, grp in tail_tab.groupby("fraction"):
        br = grp["base_rate"].iloc[0]
        print(f"\n  top {frac*100:g}%   (base rate {br:.4f})")
        for _, r in grp.iterrows():
            diff = ("" if not np.isfinite(r["difference_vs_distance"])
                    else f"   vs distance {r['difference_vs_distance']:+.4f} "
                         f"[{r['diff_ci_low']:+.4f}, {r['diff_ci_high']:+.4f}]"
                         f"{'  *' if r['interval_excludes_zero'] else ''}")
            print(f"    {r['model']:<16} {r['value']:.4f} "
                  f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
                  f"   {r['value']/br:>5.2f}x base{diff}")

    print("\n" + "=" * 78)
    print("EACH MODEL ON THE SHARED COHORT (marginal intervals, for reference)")
    print("=" * 78)
    for _, _, label in plan:
        print(f"\n{label}")
        for _, r in per_model_tab[per_model_tab["metric"] == label].iterrows():
            print(f"  {r['model']:<10} {r['value']:+.4f} "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")

    print("\n" + "=" * 78)
    print(f"PAIRED DIFFERENCE: {reference} MINUS each comparison")
    print("=" * 78)
    print("Same pairs, same matched negatives, same blocks resampled together.")
    print("This is the interval that decides whether a gap is real -- not the")
    print("overlap of the marginal intervals above.\n")
    for _, _, label in plan:
        print(f"{label}")
        for _, r in diff_tab[diff_tab["metric"] == label].iterrows():
            verdict = ("DIFFERENT" if r["interval_excludes_zero"]
                       else "not distinguishable")
            print(f"  {reference} - {r['comparison']:<10} {r['difference']:+.4f} "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  {verdict}")
        print()

    payload = {
        "purpose": ("paired, LD-aware intervals on the difference between models' "
                    "variant-effect performance; scripts/20 reports marginal "
                    "intervals only"),
        "reference": reference,
        "specs": [{"dir": str(d), "model": m, "seeds": s} for d, m, s in specs],
        "stratum": args.stratum,
        "significance_threshold": significance,
        "n_boot": args.n_boot,
        "block_size_bp": ev.BLOCK_BP,
        "shared_cohort_pairs": int(len(wide)),
        "significant_pairs": n_sig,
        "matching": {"tolerance_bp": args.match_tolerance,
                     "negatives_per_positive": args.match_ratio,
                     "replacement": False, **{k: float(v) if isinstance(v, float)
                                              else int(v)
                                              for k, v in balance.items()}},
        "per_model_on_shared_cohort": json.loads(per_model_tab.to_json(orient="records")),
        "paired_differences": json.loads(diff_tab.to_json(orient="records")),
        "caveats": {
            "shared_cohort": ("All models are scored on one distance-matched "
                              "cohort, so the difference contains no matching "
                              "noise. Each model's marginal value here can differ "
                              "slightly from its scripts/20 value, which used its "
                              "own draw of negatives."),
            "dependence": ("Pairs are in LD. Intervals are 1-Mb block bootstraps "
                           "with both models resampled on the same blocks."),
            "direction": ("A positive difference favours the reference model. An "
                          "interval spanning zero means the models are not "
                          "distinguishable on this metric at this n, not that "
                          "they are equal."),
        },
    }
    (out / "run_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"output: {out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
