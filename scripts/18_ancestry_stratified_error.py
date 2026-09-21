#!/usr/bin/env python3
"""Prediction error stratified by donor genetic ancestry on the project's own TCGA cohort.
This isolates ancestry, unlike the GENOA replication where ancestry, tissue and platform
all differ at once. Group sizes are reported precisely, since the smaller strata do not
support stable per-group estimates.
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

LOGGER = logging.getLogger("ancestry")

BLOCK_BP = 1_000_000
PARTICIPANT_CANDIDATES = ("participant", "Participant", "submitter_id",
                          "case_submitter_id", "bcr_patient_barcode")
ANCESTRY_CANDIDATES = ("ancestry_call", "ancestry", "consensus_ancestry",
                       "ancestry_label")


def pick(frame: pd.DataFrame, candidates, what: str, path) -> str:
    for c in candidates:
        if c in frame.columns:
            return c
    raise SystemExit(f"STOP: {path} has no {what} column among {list(candidates)}.\n"
                     f"Columns present: {list(frame.columns)[:25]}")


def to_participant(sample_id: str) -> str:
    """TCGA-XX-XXXX-01A-... -> TCGA-XX-XXXX. Non-TCGA ids pass through."""
    parts = str(sample_id).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(sample_id)


def block_ids(chrom: pd.Series, pos: pd.Series) -> pd.Series:
    return chrom.astype(str) + ":" + (pos.astype(float) // BLOCK_BP).astype(int).astype(str)


def block_bootstrap_ci(values: np.ndarray, blocks: np.ndarray, n_boot: int,
                       rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean of `values` with a 1 Mb block-bootstrap interval.

    CpGs within a block are correlated; resampling CpGs independently would give
    intervals several times too narrow.
    """
    uniq = np.unique(blocks)
    idx_by_block = {b: np.flatnonzero(blocks == b) for b in uniq}
    draws = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        take = np.concatenate([idx_by_block[b] for b in chosen])
        draws[i] = float(np.mean(values[take]))
    return float(np.mean(values)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, required=False,
                    help="per-participant beta matrix: rows are probes, columns "
                         "are sample or participant ids (CSV or .parquet)")
    ap.add_argument("--ancestry", type=Path,
                    default=Path("data/tcga_ancestry_labels.csv"))
    ap.add_argument("--predictions", type=Path,
                    default=Path("results/journal/seed42/fusion/predictions.csv"))
    ap.add_argument("--test-csv", type=Path, default=Path("data/datafiles/test.csv"),
                    help="supplies chromosome and position for the block bootstrap")
    ap.add_argument("--min-group-n", type=int, default=20,
                    help="groups smaller than this are reported but not analysed")
    ap.add_argument("--n-subsample", type=int, default=200,
                    help="matched-n donor resamples")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("results/journal/ancestry_stratified"))
    ap.add_argument("--inspect", action="store_true",
                    help="report group sizes and overlap, compute nothing")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    rng = np.random.default_rng(args.seed)

    if not args.ancestry.is_file():
        raise SystemExit(
            f"STOP: {args.ancestry} not found. Run:\n"
            f"    python -u data/build_tcga_ancestry_labels.py")
    labels = pd.read_csv(args.ancestry, dtype=str)
    pcol = pick(labels, PARTICIPANT_CANDIDATES, "participant", args.ancestry)
    acol = pick(labels, ANCESTRY_CANDIDATES, "ancestry", args.ancestry)
    labels = labels[[pcol, acol]].dropna().drop_duplicates(subset=[pcol])
    LOGGER.info("%d participants with an ancestry call", len(labels))

    if args.matrix is None:
        raise SystemExit(
            "STOP: --matrix is required.\n"
            "This analysis needs the PER-PARTICIPANT methylation matrix, not the\n"
            "cohort medians in data/datafiles/*.csv -- a median computed over all\n"
            "donors carries no ancestry information, so stratifying it is not\n"
            "possible. Point --matrix at the matrix build_training_data.py read\n"
            "from (see reproducibility/tcga_participant_matrix_audit.json).")
    if not args.matrix.is_file():
        raise SystemExit(f"STOP: {args.matrix} not found")

    LOGGER.info("reading %s", args.matrix)
    matrix = (pd.read_parquet(args.matrix) if args.matrix.suffix == ".parquet"
              else pd.read_csv(args.matrix, index_col=0))
    if matrix.index.name is None:
        matrix.index.name = "probeID"

    preds = pd.read_csv(args.predictions)
    if "probeID" not in preds.columns:
        raise SystemExit(f"STOP: {args.predictions} has no probeID column")
    pred_col = next((c for c in ("Predicted_Beta", "predicted_beta", "Prediction",
                                 "y_pred", "pred_beta") if c in preds.columns), None)
    if pred_col is None:
        raise SystemExit(
            f"STOP: {args.predictions} has no recognisable predicted-beta column. "
            f"Columns: {list(preds.columns)}")

    sample_to_participant = {c: to_participant(c) for c in matrix.columns}
    part_to_anc = dict(zip(labels[pcol], labels[acol]))
    cols_by_group: dict[str, list] = {}
    unlabelled = 0
    for col, part in sample_to_participant.items():
        grp = part_to_anc.get(part)
        if grp is None:
            unlabelled += 1
            continue
        cols_by_group.setdefault(grp, []).append(col)

    sizes = {g: len(c) for g, c in sorted(cols_by_group.items(),
                                          key=lambda kv: -len(kv[1]))}
    print()
    print("=" * 68)
    print("ANCESTRY GROUP SIZES  (TCGA-BRCA normals, same tissue and platform)")
    print("=" * 68)
    for g, n in sizes.items():
        flag = "" if n >= args.min_group_n else f"  << below --min-group-n={args.min_group_n}, not analysed"
        print(f"  {g:<28}{n:>5} donors{flag}")
    print(f"  {'(no ancestry call)':<28}{unlabelled:>5} donors")

    usable = {g: c for g, c in cols_by_group.items() if len(c) >= args.min_group_n}
    if len(usable) < 2:
        print()
        print("Fewer than two groups meet --min-group-n. An ancestry CONTRAST is")
        print("not possible with this cohort; report the group sizes as a")
        print("limitation rather than computing a comparison that cannot support")
        print("a claim. The GENOA cross-cohort replication remains the ancestry")
        print("evidence, with its tissue/platform confound stated.")
        return 0

    matched_n = min(len(c) for c in usable.values())
    print()
    print(f"matched donor count per group: {matched_n} "
          f"(every group subsampled to this, {args.n_subsample} resamples)")
    print("Matching is not optional: a smaller group yields a noisier median and")
    print("therefore higher apparent error with no model involvement at all.")

    if args.inspect:
        print("\nINSPECT ONLY -- nothing computed.")
        return 0

    probes = [p for p in preds["probeID"] if p in matrix.index]
    if not probes:
        raise SystemExit("STOP: no held-out probe appears in the matrix; check "
                         "that --matrix is the same probe universe as training")
    LOGGER.info("%d of %d held-out probes present in the matrix",
                len(probes), len(preds))
    pred_by_probe = preds.set_index("probeID")[pred_col]

    test = pd.read_csv(args.test_csv, usecols=lambda c: c in
                       ("probeID", "chr", "chrom", "pos", "position", "CpG_beg"))
    chrom_col = next((c for c in ("chr", "chrom") if c in test.columns), None)
    pos_col = next((c for c in ("pos", "position", "CpG_beg") if c in test.columns), None)
    if chrom_col is None or pos_col is None:
        raise SystemExit(f"STOP: {args.test_csv} lacks chromosome/position columns "
                         f"needed for the 1 Mb block bootstrap")
    coords = test.set_index("probeID").loc[[p for p in probes if p in set(test["probeID"])]]
    probes = list(coords.index)
    blocks = block_ids(coords[chrom_col], coords[pos_col]).to_numpy()
    yhat = pred_by_probe.loc[probes].to_numpy(dtype=float)

    sub = matrix.loc[probes]
    groups = sorted(usable)
    per_resample = {g: [] for g in groups}
    for _ in range(args.n_subsample):
        for g in groups:
            take = rng.choice(usable[g], size=matched_n, replace=False)
            observed = sub[list(take)].median(axis=1).to_numpy(dtype=float)
            per_resample[g].append(np.abs(observed - yhat))

    rows = []
    mean_abs = {g: np.mean(np.stack(per_resample[g]), axis=0) for g in groups}
    for g in groups:
        m, lo, hi = block_bootstrap_ci(mean_abs[g], blocks, args.n_boot, rng)
        rows.append({"group": g, "n_donors_available": len(usable[g]),
                     "n_donors_used": matched_n, "metric": "beta_mae",
                     "value": m, "ci_low": lo, "ci_high": hi})

    paired = []
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            diff = mean_abs[a] - mean_abs[b]
            m, lo, hi = block_bootstrap_ci(diff, blocks, args.n_boot, rng)
            verdict = "DIFFERENT" if (lo > 0 or hi < 0) else "not distinguishable"
            paired.append({"group_a": a, "group_b": b, "metric": "beta_mae",
                           "difference": m, "ci_low": lo, "ci_high": hi,
                           "verdict": verdict})

    print()
    print("=" * 68)
    print("PREDICTION ERROR BY ANCESTRY  (matched donor count, 1 Mb block bootstrap)")
    print("=" * 68)
    for r in rows:
        print(f"  {r['group']:<28}beta MAE {r['value']:.4f} "
              f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]  "
              f"({r['n_donors_used']} of {r['n_donors_available']} donors)")
    print()
    print("PAIRED DIFFERENCE, same CpGs, same donor count")
    for r in paired:
        print(f"  {r['group_a']} - {r['group_b']:<24}{r['difference']:+.4f} "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  {r['verdict']}")
    print()
    print("An interval spanning zero means prediction accuracy is not detectably")
    print("ancestry-dependent at this cohort size -- which is the result to hope")
    print("for, and is NOT the same as proving equivalence. Report the interval.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "error_by_ancestry.csv", index=False)
    pd.DataFrame(paired).to_csv(args.out_dir / "paired_differences.csv", index=False)
    with (args.out_dir / "run_summary.json").open("w") as fh:
        json.dump({
            "analysis": "ancestry-stratified prediction error, tissue and platform fixed",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "why": ("GENOA confounds ancestry with tissue and platform; this holds "
                    "both fixed so only donor ancestry varies"),
            "matrix": str(args.matrix),
            "ancestry_labels": str(args.ancestry),
            "ancestry_source": "GDC open ancestry calls (CCG-AIM-2020); not controlled access",
            "predictions": str(args.predictions),
            "group_sizes": sizes,
            "donors_without_call": unlabelled,
            "min_group_n": args.min_group_n,
            "matched_donors_per_group": matched_n,
            "n_subsample": args.n_subsample,
            "n_boot": args.n_boot,
            "block_size_bp": BLOCK_BP,
            "probes_analysed": len(probes),
            "caveats": {
                "matching": ("groups are subsampled to a common donor count; without "
                             "this, a smaller group's noisier median inflates its "
                             "apparent error and manufactures a disparity"),
                "labels": ("genetic ancestry is continuous and these discrete calls "
                           "are a summary, not a biological category"),
                "equivalence": ("an interval spanning zero is absence of detectable "
                                "difference at this cohort size, not proof of "
                                "equivalence"),
                "scope": ("methylation LEVEL prediction only; variant-effect "
                          "stratification would need per-donor genotypes, which are "
                          "controlled-access and deliberately not used"),
            },
            "error_by_ancestry": rows,
            "paired_differences": paired,
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
