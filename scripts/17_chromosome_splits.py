#!/usr/bin/env python3
"""Generate repeated chromosome-blocked train/val/test folds. The published model uses one
split (chr10 validation, chr8+chr9 test), which cannot separate a general result from
one that happens to suit those chromosomes. Requires both --datafiles and --out-root,
and refuses an --out-root that already holds folds unless --overwrite.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("splits")

CHROM_CANDIDATES = ("chr", "chrom", "CHR", "Chromosome", "chromosome",
                    "CpG_chrm", "cpg_chrom", "seqnames")

PUBLISHED_VAL = ["chr10", "chr11"]
PUBLISHED_TEST = ["chr8", "chr9"]

NON_EVAL_CHROMS = ("chrX", "chrY", "chrM", "chrMT")


def norm_chrom(value) -> str:
    """'8' / 'chr8' / 'Chr8' -> 'chr8'. Keeps X and Y as-is."""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    if not text.lower().startswith("chr"):
        text = "chr" + text
    return "chr" + text[3:].upper().replace("CHR", "")


def resolve_chromosomes(frame: pd.DataFrame, manifest: Path | None) -> pd.Series:
    """Return a chromosome Series aligned to `frame`, or abort explaining why not."""
    for col in CHROM_CANDIDATES:
        if col in frame.columns:
            LOGGER.info("using existing chromosome column %r", col)
            return frame[col].map(norm_chrom)

    if manifest is None:
        raise SystemExit(
            "STOP: no chromosome column found among "
            f"{list(CHROM_CANDIDATES)}.\n"
            "Chromosome-blocked splits cannot be built without knowing which "
            "chromosome each probe is on. Pass --manifest with a CSV holding "
            "probeID and a chromosome column.")

    man = pd.read_csv(manifest, dtype=str)
    if "probeID" not in man.columns:
        raise SystemExit(f"STOP: {manifest} has no probeID column")
    mcol = next((c for c in CHROM_CANDIDATES if c in man.columns), None)
    if mcol is None:
        raise SystemExit(f"STOP: {manifest} has no chromosome column among "
                         f"{list(CHROM_CANDIDATES)}")
    if man["probeID"].duplicated().any():
        raise SystemExit(f"STOP: {manifest} has duplicate probeIDs; the join "
                         f"would multiply rows")
    lookup = dict(zip(man["probeID"].astype(str), man[mcol].map(norm_chrom)))
    mapped = frame["probeID"].astype(str).map(lookup)
    missing = int(mapped.isna().sum())
    if missing:
        raise SystemExit(
            f"STOP: {missing:,} of {len(frame):,} probes are absent from "
            f"{manifest}. A partial mapping would silently drop probes from "
            f"every fold.")
    LOGGER.info("chromosomes joined from %s", manifest)
    return mapped


def build_folds(counts: "pd.Series", n_folds: int) -> list[dict]:
    """Fold 0 is the published split; later folds match its test size.

    `counts` maps chromosome -> number of probes. Chromosomes are assigned whole,
    and no chromosome is used as test or val in more than one fold, so the folds
    together give near-complete coverage of the genome as held-out data.
    """
    available = set(counts.index)
    for chrom in PUBLISHED_VAL + PUBLISHED_TEST:
        if chrom not in available:
            raise SystemExit(f"STOP: {chrom} is absent from the data, so fold 0 "
                             f"cannot reproduce the published split")

    excluded = [c for c in counts.index if c in NON_EVAL_CHROMS]
    if excluded:
        LOGGER.info("held out of every val/test block (train only): %s (%d probes)",
                    ",".join(excluded), int(counts[excluded].sum()))

    target = int(counts[PUBLISHED_TEST].sum())
    LOGGER.info("fold 0 test size %d probes; later folds will match it", target)

    folds = [{"fold": 0, "val": list(PUBLISHED_VAL), "test": list(PUBLISHED_TEST)}]
    used = set(PUBLISHED_VAL + PUBLISHED_TEST)

    for k in range(1, n_folds):
        pool = [c for c in counts.index
                if c not in used and c not in NON_EVAL_CHROMS]
        if not pool:
            LOGGER.warning("ran out of unused chromosomes at fold %d; stopping "
                           "with %d folds", k, k)
            break
        best, test = None, None
        for size in (1, 2, 3):
            for combo in itertools.combinations(pool, size):
                if len(pool) - size < 1:
                    continue
                gap = abs(int(counts[list(combo)].sum()) - target)
                if best is None or gap < best:
                    best, test = gap, list(combo)
        if test is None:
            LOGGER.warning("fold %d could not form a test set; stopping with "
                           "%d folds", k, k)
            break
        leftover = [c for c in pool if c not in test]
        if not leftover:
            LOGGER.warning("fold %d has no chromosome left for validation; "
                           "stopping with %d folds", k, k)
            break
        val_target = int(counts[PUBLISHED_VAL].sum())
        val = [min(leftover, key=lambda c: abs(counts[c] - val_target))]
        folds.append({"fold": k, "val": val, "test": test})
        used |= set(test) | set(val)

    return folds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datafiles", type=Path, required=True,
                    help="context build to re-split, e.g. data/datafiles_breast_epithelium")
    ap.add_argument("--sources", default="train.csv,val.csv,test.csv",
                    help="CSVs under --datafiles to pool before re-splitting")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="probeID -> chromosome CSV, if the sources lack one")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--out-root", type=Path, required=True,
                    help="where fold{N}/ are written; refused if it already holds "
                         "a fold unless --overwrite is given")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing folds already present under --out-root")
    ap.add_argument("--fold0-hardcoded", action="store_true",
                    help="use the built-in published-split constants instead of "
                         "reading the chromosome sets out of val.csv/test.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if args.folds < 2:
        raise SystemExit("STOP: --folds must be at least 2 to be 'repeated'")
    existing = sorted(args.out_root.glob("fold*/train.csv")) if args.out_root.is_dir() else []
    if existing and not args.dry_run and not args.overwrite:
        raise SystemExit(
            f"STOP: {args.out_root} already holds {len(existing)} fold(s). These may be "
            "published splits that trained checkpoints depend on. Write to a new "
            "--out-root, or pass --overwrite if replacing them is intended.")

    global PUBLISHED_VAL, PUBLISHED_TEST
    if not args.fold0_hardcoded:
        derived = {}
        for arm, name in (("val", "val.csv"), ("test", "test.csv")):
            path = args.datafiles / name
            if not path.is_file():
                raise SystemExit(f"STOP: {path} not found; cannot derive fold 0 "
                                 f"from the sources (pass --fold0-hardcoded to "
                                 f"use the built-in fallback instead)")
            part = pd.read_csv(path, usecols=lambda c: c in CHROM_CANDIDATES)
            col = next((c for c in CHROM_CANDIDATES if c in part.columns), None)
            if col is None:
                raise SystemExit(f"STOP: {path} has no chromosome column, so "
                                 f"fold 0 cannot be derived from it")
            derived[arm] = sorted(set(part[col].map(norm_chrom)) - {""})
        PUBLISHED_VAL, PUBLISHED_TEST = derived["val"], derived["test"]
        LOGGER.info("fold 0 derived from the source files: val=%s test=%s",
                    ",".join(PUBLISHED_VAL), ",".join(PUBLISHED_TEST))

    frames = []
    for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
        path = args.datafiles / name
        if not path.is_file():
            raise SystemExit(f"STOP: {path} not found")
        part = pd.read_csv(path)
        LOGGER.info("%-12s %7d rows", name, len(part))
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)

    if "probeID" not in frame.columns:
        raise SystemExit("STOP: pooled data has no probeID column")
    if frame["probeID"].duplicated().any():
        dupes = int(frame["probeID"].duplicated().sum())
        raise SystemExit(
            f"STOP: {dupes:,} duplicate probeIDs across the pooled sources. "
            f"The published splits are supposed to be disjoint; investigate "
            f"before re-splitting, because a duplicated probe would land in two "
            f"arms of every new fold.")

    frame["_chrom"] = resolve_chromosomes(frame, args.manifest)
    blank = int((frame["_chrom"] == "").sum())
    if blank:
        raise SystemExit(f"STOP: {blank:,} probes have an unusable chromosome value")

    counts = frame["_chrom"].value_counts().sort_index()
    LOGGER.info("%d probes across %d chromosomes", len(frame), len(counts))

    folds = build_folds(counts, args.folds)

    report = []
    for spec in folds:
        val_set, test_set = set(spec["val"]), set(spec["test"])
        overlap = val_set & test_set
        if overlap:
            raise SystemExit(f"STOP: fold {spec['fold']} has {overlap} in both "
                             f"validation and test")

        assign = frame["_chrom"].map(
            lambda c: "test" if c in test_set else ("val" if c in val_set else "train"))
        parts = {arm: frame[assign == arm].copy() for arm in ("train", "val", "test")}

        ids = {arm: set(part["probeID"]) for arm, part in parts.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = ids[a] & ids[b]
            if shared:
                raise SystemExit(f"STOP: fold {spec['fold']} shares "
                                 f"{len(shared):,} probes between {a} and {b}")
        total = sum(len(p) for p in parts.values())
        if total != len(frame):
            raise SystemExit(f"STOP: fold {spec['fold']} assigned {total:,} of "
                             f"{len(frame):,} probes")
        for arm, part in parts.items():
            chroms = set(part["_chrom"])
            for other in ("train", "val", "test"):
                if other != arm and chroms & set(parts[other]["_chrom"]):
                    raise SystemExit(
                        f"STOP: fold {spec['fold']} chromosome appears in both "
                        f"{arm} and {other}")

        if spec["fold"] == 0:
            published = {}
            for name in [s.strip() for s in args.sources.split(",") if s.strip()]:
                arm = Path(name).stem
                if arm in ("train", "val", "test"):
                    published[arm] = set(
                        pd.read_csv(args.datafiles / name, usecols=["probeID"])["probeID"])
            spec["reproduces_published"] = all(
                published.get(arm) == ids[arm] for arm in published)
            if spec["reproduces_published"]:
                LOGGER.info("fold 0 reproduces the published split EXACTLY -- the "
                            "existing trained seeds can serve as fold 0, no retrain")
            else:
                for arm in published:
                    LOGGER.warning("fold 0 %s differs from %s.csv by %d probes",
                                   arm, arm,
                                   len(published[arm] ^ ids[arm]))
                LOGGER.warning(
                    "fold 0 does NOT reproduce the published split. If only the "
                    "VAL arm differs while test matches exactly, the published "
                    "validation set was not purely chromosome-blocked: fold 0 is "
                    "then a STRICTER setup, not a reproduction, and must be "
                    "trained (submit --array=0-3, not 1-3).")

        row = {"fold": spec["fold"],
               "val_chroms": ",".join(sorted(val_set)),
               "test_chroms": ",".join(sorted(test_set))}
        row.update({f"n_{arm}": len(part) for arm, part in parts.items()})
        if "reproduces_published" in spec:
            row["reproduces_published"] = spec["reproduces_published"]
        report.append(row)

        if not args.dry_run:
            dst = args.out_root / f"fold{spec['fold']}"
            dst.mkdir(parents=True, exist_ok=True)
            for arm, part in parts.items():
                out = part.drop(columns="_chrom").copy()
                out["Split"] = arm
                out.to_csv(dst / f"{arm}.csv", index=False)

    table = pd.DataFrame(report)
    print()
    print("=" * 78)
    print("REPEATED CHROMOSOME-BLOCKED SPLITS")
    print("fold 0 is the published split and must reproduce the published numbers")
    print("=" * 78)
    print(f"{'fold':<6}{'val':<10}{'test':<22}{'train':>9}{'val':>8}{'test':>8}")
    for _, r in table.iterrows():
        print(f"{r['fold']:<6}{r['val_chroms']:<10}{r['test_chroms']:<22}"
              f"{r['n_train']:>9,}{r['n_val']:>8,}{r['n_test']:>8,}")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "splits_summary.json").open("w") as fh:
        json.dump({
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": args.sources,
            "manifest": str(args.manifest) if args.manifest else None,
            "published_split": {"val": PUBLISHED_VAL, "test": PUBLISHED_TEST},
            "note": ("fold 0 reproduces the published split; later folds hold "
                     "out a probe count as close as possible to fold 0's so "
                     "that folds are comparable"),
            "guarantees_checked": [
                "no chromosome appears in two arms of a fold",
                "no probeID appears in two arms of a fold",
                "every pooled probe is assigned to exactly one arm",
            ],
            "probes_total": int(len(frame)),
            "per_chromosome": {k: int(v) for k, v in counts.items()},
            "folds": table.to_dict("records"),
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {len(table)} folds under {args.out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
