#!/usr/bin/env python3
"""
Audit the training data before retraining. CPU only, reads nothing new.

Three questions, in descending order of how much damage a bad answer does.

1. SEQUENCE LEAKAGE ACROSS SPLITS  (the one nobody has checked)
   Chromosome-blocked splits stop *positional* leakage. They do nothing about
   *sequence-similarity* leakage: a probe in a segmental duplication on chr8
   can sit in near-identical sequence to a probe on chr1, and paralogues,
   repeat families and recent duplications are common in CpG-island promoters.
   If that happens at scale, held-out performance is partly memorisation and
   every headline number is soft. A reviewer will ask. We should know first.

   Measured by MinHash over canonical 31-mers of each 1,000-bp model window:
   estimated Jaccard between every test/val probe and its nearest train probe.
   This is an estimate with a known error (~1/sqrt(sketch size)), reported as
   such -- exact confirmation is available for the flagged pairs with --exact.

2. PROBE QC CONSISTENCY
   data/build_training_data.py does NOT filter on the HM450 manifest masks,
   but scripts/05 and scripts/19 DO filter on MASK_general when scoring. The
   model is therefore trained on a probe population it is never evaluated on.
   That is not automatically wrong -- more training signal can be worth some
   noise -- but it is currently an accident rather than a decision, and the
   size of the discrepancy has never been reported.

3. SPLIT COMPARABILITY
   Chromosome-blocked splits are not random samples. If chr8-9 differ
   systematically from the training chromosomes in CpG-island content or
   methylation distribution, some of the train/test gap is composition rather
   than generalisation. Reported so it can be stated, not discovered.

Nothing here modifies data. It writes one JSON and three CSVs.

Usage (run from the repository root)
------------------------------------
    python -u data/audit_training_data.py
    python -u data/audit_training_data.py --sketch-size 256 --exact
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

LOGGER = logging.getLogger("silentmethyl.audit")

FULL_SEQUENCE_LENGTH = 5000
MODEL_WINDOW_SIZE = 1000
KMER = 31
SPLITS = ("train", "val", "test")

BASE_CODE = np.full(256, 255, dtype=np.uint8)
for _i, _b in enumerate("ACGT"):
    BASE_CODE[ord(_b)] = _i
    BASE_CODE[ord(_b.lower())] = _i

MASK_COLUMNS = ["MASK_general", "MASK_snp5_common", "MASK_snp5_GMAF1p",
                "MASK_mapping", "MASK_typeINextBaseSwitch", "MASK_rmsk15"]


def centered_crop(seq: str, window: int) -> str:
    seq = seq.upper()
    start = len(seq) // 2 - window // 2
    return seq[start:start + window]


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Cheap, well-mixed 64-bit hash. MinHash needs mixing, not cryptography."""
    x = x.astype(np.uint64, copy=True)
    x += np.uint64(0x9E3779B97F4A7C15)
    z = x
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def sketch_windows(windows: list[str], sketch_size: int) -> np.ndarray:
    """Bottom-s MinHash of canonical 31-mers, for a batch of equal-length windows.

    Vectorised across the whole batch: the naive per-probe rolling loop is
    ~330M Python iterations over the training split and is not viable.
    """
    n = len(windows)
    length = len(windows[0])
    arr = np.frombuffer("".join(windows).encode("ascii"), dtype=np.uint8)
    arr = BASE_CODE[arr].reshape(n, length)
    positions = length - KMER + 1

    fwd = np.zeros((n, positions), dtype=np.uint64)
    rev = np.zeros((n, positions), dtype=np.uint64)
    valid = np.ones((n, positions), dtype=bool)
    for j in range(KMER):
        col = arr[:, j:j + positions]
        good = col < 4
        valid &= good
        safe = np.where(good, col, 0).astype(np.uint64)
        fwd = fwd * np.uint64(4) + safe
        rev = rev + (np.uint64(3) - safe) * (np.uint64(4) ** np.uint64(j))

    canonical = np.minimum(fwd, rev)
    hashed = splitmix64(canonical)
    hashed[~valid] = np.iinfo(np.uint64).max

    take = min(sketch_size, positions)
    part = np.partition(hashed, take - 1, axis=1)[:, :take]
    part.sort(axis=1)
    return part


def load_split(path: Path, chunk_size: int, sketch_size: int, limit: int):
    """Stream one split, returning per-probe metadata and MinHash sketches."""
    needed = ["probeID", "Healthy_5000bp_DNA", "M_Value_Target", "Median_Beta"]
    header = pd.read_csv(path, nrows=0)
    optional = [c for c in ("Binary_State_Target", "chr", "pos") if c in header.columns]
    missing_cols = [c for c in header.columns if c.endswith("_Missing")]
    absent = [c for c in needed if c not in header.columns]
    if absent:
        raise SystemExit(f"{path} is missing {absent}")

    meta, sketches, seen = [], [], 0
    for chunk in pd.read_csv(path, usecols=needed + optional + missing_cols,
                             chunksize=chunk_size):
        if limit and seen >= limit:
            break
        if limit:
            chunk = chunk.head(limit - seen)
        seqs = chunk["Healthy_5000bp_DNA"].astype(str)
        if (seqs.str.len() != FULL_SEQUENCE_LENGTH).any():
            raise SystemExit(f"{path}: sequences are not {FULL_SEQUENCE_LENGTH} bp")
        windows = [centered_crop(s, MODEL_WINDOW_SIZE) for s in seqs]
        sketches.append(sketch_windows(windows, sketch_size))
        keep = ["probeID", "M_Value_Target", "Median_Beta"] + optional + missing_cols
        meta.append(chunk[keep].reset_index(drop=True))
        seen += len(chunk)
        LOGGER.info("  %s: %d probes", path.name, seen)
    return pd.concat(meta, ignore_index=True), np.vstack(sketches)


def nearest_train_jaccard(query: np.ndarray, reference: np.ndarray,
                          max_postings: int) -> tuple[np.ndarray, np.ndarray]:
    """For each query row, the best estimated Jaccard against any reference row.

    Inverted index over sketch values. Hashes shared by an implausible number of
    reference probes are dropped: those are repeat-family k-mers, which would
    otherwise dominate the counts without indicating probe-level similarity.
    """
    s = reference.shape[1]
    ref_ids = np.repeat(np.arange(reference.shape[0], dtype=np.int64),
                        reference.shape[1])
    ref_vals = reference.ravel()
    order = np.argsort(ref_vals, kind="stable")
    ref_vals = ref_vals[order]
    ref_ids = ref_ids[order]

    uniq, starts, counts = np.unique(ref_vals, return_index=True, return_counts=True)
    common = counts > max_postings
    if common.any():
        drop = np.zeros(len(ref_vals), dtype=bool)
        for start, count in zip(starts[common], counts[common]):
            drop[start:start + count] = True
        LOGGER.info("  dropped %d postings on %d repeat-like hashes",
                    int(drop.sum()), int(common.sum()))
        ref_vals = ref_vals[~drop]
        ref_ids = ref_ids[~drop]

    best = np.zeros(len(query), dtype=np.float64)
    partner = np.full(len(query), -1, dtype=np.int64)
    for i in range(len(query)):
        left = np.searchsorted(ref_vals, query[i], side="left")
        right = np.searchsorted(ref_vals, query[i], side="right")
        hits = [ref_ids[a:b] for a, b in zip(left, right) if b > a]
        if not hits:
            continue
        joined = np.concatenate(hits)
        ids, counts_ = np.unique(joined, return_counts=True)
        k = int(counts_.argmax())
        best[i] = counts_[k] / s
        partner[i] = ids[k]
    return best, partner


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Create the parent directory at write time, not once at startup.

    A directory made at startup and written to minutes later can be removed in
    between -- by a cleanup pass, a quota sweep, or a stale scratch policy --
    and the analysis then dies at the final line having done all the work. That
    happened. Each output is now written the moment it exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def exact_jaccard(a: str, b: str) -> float:
    def kmers(seq):
        out = set()
        for i in range(len(seq) - KMER + 1):
            km = seq[i:i + KMER]
            if "N" in km:
                continue
            rc = km.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            out.add(min(km, rc))
        return out
    ka, kb = kmers(a), kmers(b)
    union = len(ka | kb)
    return (len(ka & kb) / union) if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-template", default="data/datafiles/{split}.csv")
    ap.add_argument("--manifest", type=Path, default=Path("data/HM450.hg38.manifest.tsv.gz"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/journal/training_data_audit"))
    ap.add_argument("--sketch-size", type=int, default=128,
                    help="MinHash sketch size. Jaccard error is roughly 1/sqrt(s).")
    ap.add_argument("--max-postings", type=int, default=2000,
                    help="Drop sketch hashes held by more reference probes than this.")
    ap.add_argument("--flag-jaccard", type=float, default=0.30,
                    help="Report held-out probes above this estimated Jaccard.")
    ap.add_argument("--exact", action="store_true",
                    help="Recompute exact Jaccard for flagged pairs. Slower, definitive.")
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="Smoke test.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    error = 1.0 / np.sqrt(args.sketch_size)
    LOGGER.info("MinHash sketch=%d -> Jaccard standard error ~%.3f",
                args.sketch_size, error)

    data = {}
    for split in SPLITS:
        path = Path(args.split_template.format(split=split))
        if not path.is_file():
            raise SystemExit(f"{path} not found")
        LOGGER.info("reading %s", path)
        data[split] = load_split(path, args.chunk_size, args.sketch_size, args.limit)

    summary: dict = {
        "analysis": "training data audit before retraining",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kmer": KMER,
        "sketch_size": args.sketch_size,
        "jaccard_standard_error": float(error),
        "splits": {s: int(len(m)) for s, (m, _) in data.items()},
    }

    LOGGER.info("building inverted index over %d train probes", len(data["train"][0]))
    train_meta, train_sketch = data["train"]
    leak_rows = []
    for split in ("val", "test"):
        meta, sketch = data[split]
        LOGGER.info("scoring %s against train", split)
        best, partner = nearest_train_jaccard(sketch, train_sketch, args.max_postings)
        frame = pd.DataFrame({
            "probeID": meta["probeID"].to_numpy(),
            "split": split,
            "nearest_train_probeID": np.where(
                partner >= 0, train_meta["probeID"].to_numpy()[np.maximum(partner, 0)], ""),
            "estimated_jaccard": best,
        })
        leak_rows.append(frame)
        summary.setdefault("leakage", {})[split] = {
            "n": int(len(frame)),
            "median_estimated_jaccard": float(np.median(best)),
            "p99_estimated_jaccard": float(np.percentile(best, 99)),
            "n_above_0.30": int((best > 0.30).sum()),
            "n_above_0.50": int((best > 0.50).sum()),
            "n_above_0.80": int((best > 0.80).sum()),
            "fraction_above_0.50": float((best > 0.50).mean()),
        }
    leakage = pd.concat(leak_rows, ignore_index=True)
    flagged = leakage[leakage["estimated_jaccard"] >= args.flag_jaccard].copy()
    flagged = flagged.sort_values("estimated_jaccard", ascending=False)

    if args.exact and not flagged.empty:
        LOGGER.info("recomputing exact Jaccard for %d flagged pairs", len(flagged))
        seqs = {}
        for split in SPLITS:
            path = Path(args.split_template.format(split=split))
            wanted = set(flagged["probeID"]) | set(flagged["nearest_train_probeID"])
            for chunk in pd.read_csv(path, usecols=["probeID", "Healthy_5000bp_DNA"],
                                     chunksize=20_000):
                chunk = chunk[chunk["probeID"].astype(str).isin(wanted)]
                for p, s in zip(chunk["probeID"].astype(str),
                                chunk["Healthy_5000bp_DNA"].astype(str)):
                    seqs[p] = centered_crop(s, MODEL_WINDOW_SIZE)
        flagged["exact_jaccard"] = [
            exact_jaccard(seqs[a], seqs[b])
            if a in seqs and b in seqs else np.nan
            for a, b in zip(flagged["probeID"], flagged["nearest_train_probeID"])
        ]
        summary["leakage"]["exact_check"] = {
            "n_pairs": int(len(flagged)),
            "median_exact_jaccard": float(flagged["exact_jaccard"].median()),
            "n_exact_above_0.50": int((flagged["exact_jaccard"] > 0.50).sum()),
        }
    write_csv(flagged, args.output_dir / "cross_split_sequence_similarity.csv")

    if args.manifest.is_file():
        head = pd.read_csv(args.manifest, sep="\t", nrows=0)
        probe_col = next((c for c in ("probeID", "IlmnID", "Name")
                          if c in head.columns), None)
        present = [c for c in MASK_COLUMNS if c in head.columns]
        manifest = pd.read_csv(args.manifest, sep="\t",
                               usecols=[probe_col] + present, low_memory=False)
        manifest = manifest.rename(columns={probe_col: "probeID"})
        manifest["probeID"] = manifest["probeID"].astype(str)
        for col in present:
            manifest[col] = (manifest[col].astype(str).str.strip().str.lower()
                             .isin({"true", "t", "1", "yes"}))
        qc_rows = []
        for split in SPLITS:
            meta, _ = data[split]
            joined = meta[["probeID"]].astype(str).merge(
                manifest, on="probeID", how="left")
            row = {"split": split, "n": int(len(joined)),
                   "not_in_manifest": int(joined[present[0]].isna().sum()) if present else 0}
            for col in present:
                row[col] = int(joined[col].fillna(False).sum())
                row[f"{col}_pct"] = round(100 * joined[col].fillna(False).mean(), 3)
            qc_rows.append(row)
        qc = pd.DataFrame(qc_rows)
        write_csv(qc, args.output_dir / "probe_qc_composition.csv")
        summary["probe_qc"] = qc.to_dict(orient="records")
        summary["probe_qc_note"] = (
            "data/build_training_data.py applies none of these masks; scripts/05 "
            "and scripts/19 exclude MASK_general when scoring. The model is "
            "therefore trained on a probe population it is never evaluated on. "
            "Decide this deliberately before retraining rather than inheriting it.")

    comp_rows = []
    for split in SPLITS:
        meta, _ = data[split]
        missing_cols = [c for c in meta.columns if c.endswith("_Missing")]
        row = {
            "split": split, "n": int(len(meta)),
            "median_beta": float(meta["Median_Beta"].median()),
            "mean_beta": float(meta["Median_Beta"].mean()),
            "sd_m_value": float(meta["M_Value_Target"].std()),
            "fraction_beta_below_0.3": float((meta["Median_Beta"] < 0.3).mean()),
            "fraction_beta_above_0.7": float((meta["Median_Beta"] > 0.7).mean()),
        }
        for col in missing_cols:
            row[f"missing_{col.replace('_Missing','')}"] = round(
                float(pd.to_numeric(meta[col], errors="coerce").fillna(0).mean()), 4)
        if "chr" in meta.columns:
            row["chrX"] = int((meta["chr"].astype(str) == "chrX").sum())
            row["chrY"] = int((meta["chr"].astype(str) == "chrY").sum())
        comp_rows.append(row)
    comparability = pd.DataFrame(comp_rows)
    write_csv(comparability, args.output_dir / "split_comparability.csv")
    summary["split_comparability"] = comparability.to_dict(orient="records")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    print()
    print("=" * 76)
    print("1. CROSS-SPLIT SEQUENCE SIMILARITY  (estimated Jaccard, 31-mers)")
    print(f"   sketch={args.sketch_size}, standard error ~{error:.3f}")
    for split in ("val", "test"):
        s = summary["leakage"][split]
        print(f"   {split:<5} n={s['n']:>7,}  median={s['median_estimated_jaccard']:.3f}  "
              f"p99={s['p99_estimated_jaccard']:.3f}  "
              f">0.5: {s['n_above_0.50']:,} ({s['fraction_above_0.50']:.2%})  "
              f">0.8: {s['n_above_0.80']:,}")
    print("   A median near zero is the expected, healthy result. What matters is")
    print("   the tail: probes above 0.5 share most of their 31-mers with a training")
    print("   probe and are not independent tests of generalisation.")
    if "probe_qc" in summary:
        print()
        print("=" * 76)
        print("2. PROBE QC COMPOSITION  (% of each split failing each manifest mask)")
        cols = ["split", "n"] + [c for c in qc.columns if c.endswith("_pct")]
        print(qc[cols].to_string(index=False))
    print()
    print("=" * 76)
    print("3. SPLIT COMPARABILITY")
    keep = ["split", "n", "median_beta", "sd_m_value",
            "fraction_beta_below_0.3", "fraction_beta_above_0.7"]
    print(comparability[[c for c in keep if c in comparability.columns]].to_string(index=False))
    print("=" * 76)
    print(f"\nwrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
