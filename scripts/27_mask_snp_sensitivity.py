#!/usr/bin/env python
"""Probe-QC sensitivity: does excluding common-SNP-affected probes change anything?

The held-out test set is scored on every probe that survived construction, and
roughly 11% of HM450 probes carry `MASK_snp5_common` -- a common SNP within 5 bp
of the interrogated CpG. A reviewer will ask whether the headline numbers are
propped up by probes whose measured beta is partly a genotype artefact rather
than methylation. This answers that.

No model is re-run. Everything here is a re-scoring of predictions already on
disk (`results/journal/seed<seed>/<model>/predictions.csv`), so it is CPU-only
and takes minutes.

Three numbers per model-seed, not one:

    all       every held-out probe -- reproduces the published metric
    retained  MASK_snp5_common == False -- the sensitivity analysis proper
    excluded  MASK_snp5_common == True  -- the probes we would have dropped

Reporting `excluded` matters. If the flagged probes score *worse*, the mask is
removing noise and the retained number is the honest one. If they score the
same, the mask is irrelevant here and we say so. If they score *better*, the
model is partly reading genotype artefact and that is a finding we would have to
report against ourselves. Only printing `retained` would hide the third case.

Self-check: the `all` stratum is recomputed from predictions.csv and compared
against the metrics.json written at test time. If those disagree, this script's
metric definitions have drifted from the ones the paper quotes and it exits
non-zero rather than emitting numbers that look plausible.

    python -u scripts/27_mask_snp_sensitivity.py
    python -u scripts/27_mask_snp_sensitivity.py --mask-column MASK_general
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger("mask_sensitivity")

PROBE_ID_CANDIDATES = ("probeID", "IlmnID", "Name")
# Tolerance for the self-check against metrics.json. Tight enough to catch a
# definitional change, loose enough to absorb float32/float64 round-tripping
# through the CSV.
METRIC_TOLERANCE = 5e-4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", default="results/journal")
    p.add_argument("--manifest", default="data/HM450.hg38.manifest.tsv.gz")
    p.add_argument("--mask-column", default="MASK_snp5_common")
    p.add_argument("--models", nargs="+", default=["fusion", "sequence", "epi"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--output-dir", default="results/journal/mask_sensitivity")
    p.add_argument("--bootstrap", type=int, default=2000,
                   help="Resamples for the retained-minus-excluded interval. 0 disables.")
    p.add_argument("--random-seed", type=int, default=42)
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


def load_mask(manifest_path: Path, mask_column: str) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t", compression="infer",
                           low_memory=False)
    probe_col = next((c for c in PROBE_ID_CANDIDATES if c in manifest.columns), None)
    if probe_col is None:
        raise ValueError(f"{manifest_path} has no probe ID column "
                         f"(looked for {PROBE_ID_CANDIDATES})")
    if mask_column not in manifest.columns:
        available = [c for c in manifest.columns if c.startswith("MASK")]
        raise ValueError(f"{mask_column} not in manifest. MASK columns present: "
                         f"{available}")
    out = manifest[[probe_col, mask_column]].rename(columns={probe_col: "probeID"})
    out["probeID"] = out["probeID"].astype(str)
    out[mask_column] = coerce_boolean(out[mask_column], mask_column)
    if out["probeID"].duplicated().any():
        raise ValueError(f"Duplicate probe IDs in {manifest_path}")
    return out


def metrics_for(frame: pd.DataFrame) -> dict:
    """Definitions must match scripts/02_*: beta MAE on the RC-averaged
    prediction, AUC on the RC-averaged class probability."""
    if frame.empty:
        return {"n": 0, "beta_mae": None, "beta_rmse": None, "auc": None,
                "positive_rate": None}
    err = frame["pred_beta_rc_avg"].to_numpy(float) - frame["true_beta"].to_numpy(float)
    y = frame["binary_true"].to_numpy(int)
    auc = (float(roc_auc_score(y, frame["class_prob_rc_avg"].to_numpy(float)))
           if len(np.unique(y)) == 2 else None)
    return {
        "n": int(len(frame)),
        "beta_mae": float(np.mean(np.abs(err))),
        "beta_rmse": float(np.sqrt(np.mean(err ** 2))),
        "auc": auc,
        "positive_rate": float(np.mean(y)),
    }


def bootstrap_difference(retained: pd.DataFrame, excluded: pd.DataFrame,
                         n_boot: int, rng: np.random.Generator) -> dict:
    """Retained minus excluded, resampling probes independently within each
    stratum. These are disjoint probe sets, so an unpaired bootstrap is the
    right one -- there is no pairing to preserve."""
    if n_boot <= 0 or retained.empty or excluded.empty:
        return {}
    diffs_mae, diffs_auc = [], []
    r_err = np.abs(retained["pred_beta_rc_avg"].to_numpy(float)
                   - retained["true_beta"].to_numpy(float))
    e_err = np.abs(excluded["pred_beta_rc_avg"].to_numpy(float)
                   - excluded["true_beta"].to_numpy(float))
    r_y = retained["binary_true"].to_numpy(int)
    r_p = retained["class_prob_rc_avg"].to_numpy(float)
    e_y = excluded["binary_true"].to_numpy(int)
    e_p = excluded["class_prob_rc_avg"].to_numpy(float)
    for _ in range(n_boot):
        ri = rng.integers(0, len(r_err), len(r_err))
        ei = rng.integers(0, len(e_err), len(e_err))
        diffs_mae.append(r_err[ri].mean() - e_err[ei].mean())
        ry, ey = r_y[ri], e_y[ei]
        if len(np.unique(ry)) == 2 and len(np.unique(ey)) == 2:
            diffs_auc.append(roc_auc_score(ry, r_p[ri]) - roc_auc_score(ey, e_p[ei]))
    out = {
        "beta_mae_difference_ci": [float(np.percentile(diffs_mae, 2.5)),
                                   float(np.percentile(diffs_mae, 97.5))],
    }
    if diffs_auc:
        out["auc_difference_ci"] = [float(np.percentile(diffs_auc, 2.5)),
                                    float(np.percentile(diffs_auc, 97.5))]
        out["auc_difference_resamples"] = len(diffs_auc)
    return out


def check_against_published(model: str, seed: int, recomputed: dict,
                            metrics_path: Path) -> dict:
    """The published metric is the contract. If we cannot reproduce it from the
    same predictions.csv, the sensitivity numbers below are not comparable to
    anything in the paper."""
    if not metrics_path.exists():
        LOGGER.warning("%s seed %d: no metrics.json at %s -- self-check skipped",
                       model, seed, metrics_path)
        return {"checked": False, "reason": "metrics.json absent"}
    published = json.loads(metrics_path.read_text())
    report = {"checked": True, "fields": {}}
    for field in ("beta_mae", "auc"):
        if field not in published or recomputed.get(field) is None:
            continue
        delta = abs(float(published[field]) - float(recomputed[field]))
        report["fields"][field] = {
            "published": float(published[field]),
            "recomputed": float(recomputed[field]),
            "abs_difference": delta,
            "within_tolerance": bool(delta <= METRIC_TOLERANCE),
        }
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(message)s")
    args = parse_args()
    root = Path(args.results_root)
    out_dir = Path(args.output_dir)
    rng = np.random.default_rng(args.random_seed)

    mask = load_mask(Path(args.manifest), args.mask_column)
    LOGGER.info("manifest: %d probes, %s true for %d (%.2f%%)",
                len(mask), args.mask_column, int(mask[args.mask_column].sum()),
                100.0 * mask[args.mask_column].mean())

    rows: list[dict] = []
    checks: list[dict] = []
    failures: list[str] = []

    for model in args.models:
        for seed in args.seeds:
            pred_path = root / f"seed{seed}" / model / "predictions.csv"
            if not pred_path.exists():
                LOGGER.warning("missing %s -- skipped", pred_path)
                continue
            preds = pd.read_csv(pred_path)
            preds["probeID"] = preds["probeID"].astype(str)
            merged = preds.merge(mask, on="probeID", how="left",
                                 validate="many_to_one")
            unmatched = int(merged[args.mask_column].isna().sum())
            if unmatched:
                # Not fatal, but it must be visible: a probe absent from the
                # manifest cannot be assigned to either stratum, and silently
                # dropping it would shift the denominators.
                LOGGER.warning("%s seed %d: %d/%d probes absent from manifest; "
                               "excluded from both strata",
                               model, seed, unmatched, len(merged))
                merged = merged[merged[args.mask_column].notna()]
            flag = merged[args.mask_column].astype(bool)

            strata = {
                "all": merged,
                "retained": merged[~flag],
                "excluded": merged[flag],
            }
            for name, frame in strata.items():
                rows.append({"model": model, "seed": seed, "stratum": name,
                             **metrics_for(frame)})

            check = check_against_published(
                model, seed, metrics_for(merged),
                root / f"seed{seed}" / model / "metrics.json")
            check.update({"model": model, "seed": seed})
            checks.append(check)
            for field, detail in check.get("fields", {}).items():
                if not detail["within_tolerance"]:
                    failures.append(
                        f"{model} seed {seed} {field}: published "
                        f"{detail['published']:.4f} vs recomputed "
                        f"{detail['recomputed']:.4f}")

            diff = bootstrap_difference(strata["retained"], strata["excluded"],
                                        args.bootstrap, rng)
            if diff:
                rows.append({"model": model, "seed": seed,
                             "stratum": "retained_minus_excluded", **diff})

    if not rows:
        LOGGER.error("no predictions found under %s", root)
        return 1

    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "mask_sensitivity.csv", index=False)

    per_model = (table[table["stratum"].isin(["all", "retained", "excluded"])]
                 .groupby(["model", "stratum"])[["n", "beta_mae", "auc"]]
                 .agg(["mean", "std"]))

    print("=" * 74)
    print(f"PROBE-QC SENSITIVITY -- {args.mask_column}")
    print("=" * 74)
    for model in args.models:
        sub = table[(table["model"] == model)
                    & table["stratum"].isin(["all", "retained", "excluded"])]
        if sub.empty:
            continue
        print(f"\n{model}  (mean +/- sd over {sub['seed'].nunique()} seeds)")
        for stratum in ("all", "retained", "excluded"):
            s = sub[sub["stratum"] == stratum]
            if s.empty:
                continue
            mae, auc = s["beta_mae"], s["auc"]
            print(f"  {stratum:<9} n={int(s['n'].mean()):>6}   "
                  f"beta MAE {mae.mean():.4f} +/- {mae.std(ddof=0):.4f}   "
                  f"AUC {auc.mean():.4f} +/- {auc.std(ddof=0):.4f}")

    print("\nSelf-check against the metrics.json quoted in the paper:")
    if failures:
        print("  FAILED -- metric definitions have drifted:")
        for line in failures:
            print(f"    {line}")
    else:
        n_ok = sum(len(c.get("fields", {})) for c in checks)
        print(f"  {n_ok} published values reproduced within {METRIC_TOLERANCE}")

    payload = {
        "mask_column": args.mask_column,
        "manifest": str(args.manifest),
        "mask_fraction_in_manifest": float(mask[args.mask_column].mean()),
        "models": args.models,
        "seeds": args.seeds,
        "bootstrap_resamples": args.bootstrap,
        "self_checks": checks,
        "self_check_failures": failures,
        "per_model_summary": json.loads(per_model.to_json()),
        "interpretation": (
            "retained is the sensitivity analysis; excluded is the stratum the "
            "mask would remove. If excluded scores no better than retained, the "
            "headline metrics are not propped up by genotype-affected probes."
        ),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir}/mask_sensitivity.csv and run_summary.json")
    print("=" * 74)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
