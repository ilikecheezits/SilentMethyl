#!/usr/bin/env python3
"""
Build a TCGA-BRCA tumour methylation cohort as an independent evaluation domain.

Why this exists
---------------
SilentMethyl was trained on the median of 97 solid-tissue-normal (sample-type-11)
columns. The same matrix already on disk also holds ~780 primary-tumour
(sample-type-01) columns that have never been used. Scoring those is a genuine
domain shift -- same platform, same probes, different biological state -- at zero
download cost.

Serves the mentor's **multi-cohort testing** requirement.

The confound this script handles for you
----------------------------------------
TCGA normal-adjacent samples are usually taken from the SAME participants as the
tumours. So "tumour cohort" is not automatically an independent sample set: if the
same person contributed both, the comparison is paired, not independent, and a
reviewer will say so.

This script therefore writes TWO cohorts and reports the overlap:

  tcga_tumor_all.csv        every sample-type-01 column
  tcga_tumor_unpaired.csv   only tumours from participants absent from the
                            97-normal training set -- the genuinely independent one

Report the unpaired cohort as the primary result and the full one as a supplement.

Coordinate/target conventions match data/build_training_data.py exactly: median
beta across available samples, beta clipped to [1e-4, 1-1e-4], M = log2(b/(1-b)),
binary = beta > 0.5.

Usage (run from the repository root)
------------------------------------
    python -u data/build_tcga_tumor_cohort.py --inspect
    python -u data/build_tcga_tumor_cohort.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MATRIX = Path("data/TCGA-BRCA.methylation450.tsv.gz")
DEFAULT_NORMAL_IDS = Path("data/datafiles/tcga_normal_sample_ids.json")
DEFAULT_PROBES = Path("data/datafiles/test.csv")
DEFAULT_OUT = Path("data/external/tcga_tumor")

# Mirrors NORMAL_SAMPLE_RE in data/build_training_data.py, but for sample type 01
# (primary solid tumour) instead of 11 (solid tissue normal).
TUMOR_SAMPLE_RE = re.compile(
    r"^TCGA[-.][A-Z0-9]{2}[-.][A-Z0-9]{4}[-.]01[A-Z0-9](?:[-.]|$)",
    re.IGNORECASE,
)
NORMAL_SAMPLE_RE = re.compile(
    r"^TCGA[-.][A-Z0-9]{2}[-.][A-Z0-9]{4}[-.]11[A-Z0-9](?:[-.]|$)",
    re.IGNORECASE,
)


def participant_of(barcode: str) -> str:
    """TCGA-XX-XXXX-01A-... -> TCGA-XX-XXXX (normalising . to -)."""
    parts = re.split(r"[-.]", str(barcode))
    return "-".join(parts[:3]).upper() if len(parts) >= 3 else str(barcode).upper()


def read_header(matrix: Path) -> list:
    with gzip.open(matrix, "rt") as fh:
        return fh.readline().rstrip("\n").split("\t")


def load_normal_participants(path: Path) -> set:
    if not path.exists():
        logging.warning("%s not found -- cannot identify paired participants; "
                        "the unpaired cohort will be skipped", path)
        return set()
    ids = json.loads(path.read_text())
    if isinstance(ids, dict):
        ids = ids.get("normal_columns") or ids.get("samples") or list(ids.values())
    parts = {participant_of(s) for s in ids}
    logging.info("loaded %d normal sample IDs -> %d participants", len(ids), len(parts))
    return parts


def load_probe_subset(path: Path) -> set | None:
    if not path.exists():
        logging.warning("%s not found -- keeping all probes", path)
        return None
    head = pd.read_csv(path, nrows=0)
    col = "probeID" if "probeID" in head.columns else head.columns[0]
    probes = set(pd.read_csv(path, usecols=[col])[col].astype(str))
    logging.info("restricting to %d probes from %s", len(probes), path.name)
    return probes


def summarise(matrix: Path, columns: list, probe_subset: set | None,
              chunksize: int) -> pd.DataFrame:
    header = read_header(matrix)
    probe_column = header[0]
    frames = []
    reader = pd.read_csv(matrix, sep="\t", usecols=[probe_column, *columns],
                         chunksize=chunksize,
                         dtype={c: "float32" for c in columns})
    for i, chunk in enumerate(reader, 1):
        chunk = chunk.rename(columns={probe_column: "probeID"})
        if probe_subset is not None:
            chunk = chunk[chunk["probeID"].astype(str).isin(probe_subset)]
        if chunk.empty:
            continue
        vals = chunk[columns].to_numpy(dtype=np.float32)
        n_obs = np.isfinite(vals).sum(axis=1)
        keep = n_obs > 0
        if not keep.any():
            continue
        chunk = chunk.loc[keep].copy()
        vals = vals[keep]
        with np.errstate(invalid="ignore"):
            median = np.nanmedian(vals, axis=1)
        chunk["Median_Beta"] = median
        clipped = np.clip(median.astype(float), 1e-4, 1 - 1e-4)
        chunk["M_Value_Target"] = np.log2(clipped / (1.0 - clipped))
        chunk["Binary_State_Target"] = (chunk["Median_Beta"] > 0.5).astype(np.int8)
        chunk["n_samples_observed"] = n_obs[keep]
        frames.append(chunk[["probeID", "Median_Beta", "M_Value_Target",
                             "Binary_State_Target", "n_samples_observed"]])
        if i % 20 == 0:
            logging.info("  chunk %d, %d probes so far", i, sum(len(f) for f in frames))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def cmd_inspect(args) -> int:
    header = read_header(args.matrix)
    sample_cols = header[1:]
    tumour = [c for c in sample_cols if TUMOR_SAMPLE_RE.search(str(c))]
    normal = [c for c in sample_cols if NORMAL_SAMPLE_RE.search(str(c))]
    other = len(sample_cols) - len(tumour) - len(normal)

    print(f"matrix           : {args.matrix}")
    print(f"probe column     : {header[0]}")
    print(f"total samples    : {len(sample_cols)}")
    print(f"  type-01 tumour : {len(tumour)}")
    print(f"  type-11 normal : {len(normal)}")
    print(f"  other types    : {other}")

    normal_parts = load_normal_participants(args.normal_ids)
    if normal_parts:
        t_parts = {participant_of(c) for c in tumour}
        shared = t_parts & normal_parts
        print(f"\nparticipant overlap")
        print(f"  tumour participants        : {len(t_parts)}")
        print(f"  training-normal participants: {len(normal_parts)}")
        print(f"  SHARED (paired)            : {len(shared)}")
        print(f"  tumour-only (independent)  : {len(t_parts - normal_parts)}")
        if shared:
            print("\n  -> The shared participants make the full tumour cohort a PAIRED")
            print("     comparison. The unpaired cohort is the one to report as the")
            print("     independent multi-cohort test.")
    print(f"\nexample tumour columns: {tumour[:3]}")
    return 0


def cmd_run(args) -> int:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    header = read_header(args.matrix)
    sample_cols = header[1:]
    tumour = [c for c in sample_cols if TUMOR_SAMPLE_RE.search(str(c))]
    if not tumour:
        raise SystemExit("no sample-type-01 columns found in the matrix")
    logging.info("%d tumour columns of %d samples", len(tumour), len(sample_cols))

    normal_parts = load_normal_participants(args.normal_ids)
    probe_subset = load_probe_subset(args.probe_source)

    unpaired = [c for c in tumour if participant_of(c) not in normal_parts] \
        if normal_parts else []
    shared_parts = ({participant_of(c) for c in tumour} & normal_parts) \
        if normal_parts else set()

    cohorts = {"all": tumour}
    if unpaired and len(unpaired) < len(tumour):
        cohorts["unpaired"] = unpaired
    elif normal_parts and not shared_parts:
        logging.info("no participant overlap -- the full cohort is already independent")

    written = {}
    for name, cols in cohorts.items():
        logging.info("summarising %s cohort (%d samples)", name, len(cols))
        df = summarise(args.matrix, cols, probe_subset, args.chunksize)
        if df.empty:
            logging.warning("%s cohort produced no probes", name)
            continue
        target = out / f"tcga_tumor_{name}.csv"
        df.to_csv(target, index=False)
        written[name] = {"file": str(target), "samples": len(cols),
                         "probes": int(len(df)),
                         "bytes": int(target.stat().st_size)}
        logging.info("  -> %s (%d probes)", target, len(df))

    summary = {
        "analysis": "TCGA-BRCA tumour cohort as an independent evaluation domain",
        "purpose": "mentor requirement: multi-cohort testing",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "matrix": str(args.matrix),
        "matrix_note": "public GDC/Xena open tier; no controlled access",
        "total_sample_columns": len(sample_cols),
        "tumor_sample_columns": len(tumour),
        "tumor_participants": len({participant_of(c) for c in tumour}),
        "training_normal_participants": len(normal_parts),
        "shared_participants_paired": len(shared_parts),
        "unpaired_tumor_columns": len(unpaired),
        "probe_subset_source": str(args.probe_source) if probe_subset else None,
        "target_convention": ("median beta across observed samples; beta clipped to "
                              "[1e-4, 1-1e-4]; M = log2(b/(1-b)); binary = beta > 0.5 "
                              "-- identical to data/build_training_data.py"),
        "cohorts": written,
        "reporting_guidance": (
            "Report the unpaired cohort as the primary independent-cohort result. "
            "The full cohort shares participants with the training normals, so it is "
            "a paired comparison and belongs in the supplement."
        ),
    }
    with (out / "tcga_tumor_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print()
    print("=" * 62)
    for name, info in written.items():
        print(f"{name:<10} {info['samples']:>4} samples  {info['probes']:>7,} probes  "
              f"{info['bytes']/1e6:>6.1f} MB")
    if shared_parts:
        print(f"\n{len(shared_parts)} participants appear in BOTH the training normals and")
        print("the tumour set. Use the unpaired cohort for the independent claim.")
    print("=" * 62)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    ap.add_argument("--normal-ids", type=Path, default=DEFAULT_NORMAL_IDS,
                    help="JSON list of the training normal sample IDs")
    ap.add_argument("--probe-source", type=Path, default=DEFAULT_PROBES,
                    help="CSV whose probeID column defines the probe subset "
                         "(default: the held-out test split)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--chunksize", type=int, default=50_000)
    ap.add_argument("--inspect", action="store_true",
                    help="report sample counts and participant overlap; write nothing")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    return cmd_inspect(args) if args.inspect else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
