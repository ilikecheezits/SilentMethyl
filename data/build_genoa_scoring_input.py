#!/usr/bin/env python3
"""
Turn harmonized GENOA pairs into scoring input SilentMethyl can actually run.

The gap this closes
-------------------
Harmonization gives you variant-CpG pairs with hg38 coordinates, but GENOA reports
`allele1`/`allele0` (minor/major) -- NOT reference/alternate. SilentMethyl's
paired WT/MUT scoring needs REF and ALT. This script resolves REF from hg38.fa at
each position, assigns ALT as the other allele, and writes the
`Variant_ID, Gene, chr, Position_1based, Ref, Alt` lead columns that
scripts/20_variant_scoring.py consumes.

Four things it does that matter for the analysis
-------------------------------------------------
1. **Splits by what the model has seen.** Pairs are labelled by the split their
   probe belongs to (read from your actual train/val/test CSVs, not assumed from
   chromosome numbers). Scoring a variant at a probe the model trained on is not
   independent validation. Two files are written and only the held-out one
   supports the external-validation claim -- mirroring the heldout /
   model_visible distinction you already use for eGTEx.

2. **Flags CpG-altering variants.** A variant that creates or destroys a CG
   dinucleotide has a near-deterministic methylation direction that needs no
   model. Without this flag, direction-agreement numbers have an untested trivial
   explanation. Columns: `alters_target_cpg`, `creates_cpg`, `destroys_cpg`.

3. **Reports allele mismatches instead of hiding them.** If neither GENOA allele
   matches the hg38 reference base, the pair is dropped and counted. A high
   mismatch rate means the liftover or the strand convention is wrong, and you
   want that surfaced, not silently absorbed.

4. **Aligns the effect allele.** GENOA's `beta` is the effect of the MINOR allele
   (GEMMA convention). The model's delta is the effect of ALT over hg38 REF. Those
   agree only when ALT happens to be the minor allele. `beta_genoa_ref_to_alt`
   re-signs the published effect into the REF->ALT direction; compare against that
   column, never raw `beta_genoa`. Mixing the two conventions does not raise -- it
   just quietly deflates signed correlation and direction agreement, which are the
   two headline numbers for independent variant evaluation.

Reads hg38.fa directly through its .fai index -- no pyfaidx dependency.

Usage (run from the repository root)
------------------------------------
    python -u data/build_genoa_scoring_input.py --inspect
    python -u data/build_genoa_scoring_input.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PAIRS = Path("data/external/genoa_meqtl/harmonized/genoa_model_visible_pairs.csv.gz")
DEFAULT_FASTA = Path("data/hg38.fa")
DEFAULT_SPLIT_DIR = Path("data/datafiles")
DEFAULT_OUT = Path("data/external/genoa_meqtl/scoring")

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

MODEL_TARGET_C_INDEX = 499
MODEL_WINDOW_SIZE = 1000
MIN_SCOREABLE_OFFSET = -MODEL_TARGET_C_INDEX
MAX_SCOREABLE_OFFSET = MODEL_WINDOW_SIZE - 1 - MODEL_TARGET_C_INDEX


class FastaReader:
    """Random access to a FASTA using its .fai index. No external dependency."""

    def __init__(self, fasta: Path):
        fai = Path(str(fasta) + ".fai")
        if not fasta.exists():
            raise SystemExit(f"reference FASTA not found: {fasta}")
        if not fai.exists():
            raise SystemExit(f"FASTA index not found: {fai} (run: samtools faidx {fasta})")
        self.path = fasta
        self.index = {}
        for line in fai.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) >= 5:
                name, length, offset, linebases, linewidth = parts[:5]
                self.index[name] = (int(length), int(offset), int(linebases), int(linewidth))
        self._fh = open(fasta, "rb")

    def fetch(self, chrom: str, start0: int, end0: int) -> str:
        """0-based half-open [start0, end0)."""
        if chrom not in self.index:
            alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
            if alt not in self.index:
                return ""
            chrom = alt
        length, offset, linebases, linewidth = self.index[chrom]
        start0 = max(0, start0)
        end0 = min(length, end0)
        if end0 <= start0:
            return ""
        def byte_of(p):
            return offset + (p // linebases) * linewidth + (p % linebases)
        self._fh.seek(byte_of(start0))
        raw = self._fh.read(byte_of(end0 - 1) - byte_of(start0) + 1)
        return raw.decode("ascii", errors="replace").replace("\n", "").replace("\r", "").upper()

    def close(self):
        self._fh.close()


def load_split_labels(split_dir: Path) -> dict:
    """probeID -> 'train' | 'val' | 'test', read from the real split CSVs."""
    labels = {}
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.csv"
        if not path.exists():
            logging.warning("%s not found -- split labelling will be incomplete", path)
            continue
        head = pd.read_csv(path, nrows=0)
        col = "probeID" if "probeID" in head.columns else head.columns[0]
        ids = pd.read_csv(path, usecols=[col])[col].astype(str)
        for p in ids:
            labels[p] = name
        logging.info("%-5s split: %d probes", name, len(ids))
    return labels


def count_cg(seq: str) -> int:
    return sum(1 for i in range(len(seq) - 1) if seq[i:i + 2] == "CG")


def annotate(df: pd.DataFrame, fasta: FastaReader, stats: Counter) -> pd.DataFrame:
    """Resolve REF/ALT against hg38 and flag CpG alteration."""
    refs, alts, keep = [], [], []
    alters, creates, destroys, alt_is_minor = [], [], [], []

    for row in df.itertuples(index=False):
        pos0 = int(row.snp_pos_hg38)
        chrom = str(row.snp_chr)
        ctx = fasta.fetch(chrom, pos0 - 1, pos0 + 2)
        if len(ctx) < 3:
            stats["dropped_no_reference_sequence"] += 1
            keep.append(False); refs.append(""); alts.append("")
            alters.append(False); creates.append(False); destroys.append(False)
            alt_is_minor.append(False)
            continue

        ref_base = ctx[1]
        a1 = str(row.allele_minor).strip().upper()
        a0 = str(row.allele_major).strip().upper()

        if ref_base == a0:
            alt_base, alt_minor = a1, True
        elif ref_base == a1:
            alt_base, alt_minor = a0, False
        else:
            stats["dropped_allele_mismatch"] += 1
            keep.append(False); refs.append(ref_base); alts.append("")
            alters.append(False); creates.append(False); destroys.append(False)
            alt_is_minor.append(False)
            continue

        if len(alt_base) != 1 or alt_base not in "ACGT":
            stats["dropped_non_snv"] += 1
            keep.append(False); refs.append(ref_base); alts.append(alt_base)
            alters.append(False); creates.append(False); destroys.append(False)
            alt_is_minor.append(False)
            continue

        mut_ctx = ctx[0] + alt_base + ctx[2]
        delta = count_cg(mut_ctx) - count_cg(ctx)
        cpg0 = int(row.cpg_pos_hg38)
        hits_target = pos0 in (cpg0, cpg0 + 1)

        refs.append(ref_base); alts.append(alt_base); keep.append(True)
        alters.append(bool(hits_target))
        creates.append(delta > 0)
        destroys.append(delta < 0)
        alt_is_minor.append(alt_minor)
        stats["kept"] += 1
        if not alt_minor:
            stats["ref_is_minor_allele_effect_resigned"] += 1

    df = df.copy()
    df["Ref"] = refs
    df["Alt"] = alts
    df["alters_target_cpg"] = alters
    df["creates_cpg"] = creates
    df["destroys_cpg"] = destroys
    df["alt_is_minor_allele"] = alt_is_minor
    df["_keep"] = keep
    df = df[df["_keep"]].drop(columns="_keep").reset_index(drop=True)

    if "beta_genoa" in df.columns:
        sign = np.where(df["alt_is_minor_allele"].to_numpy(dtype=bool), 1.0, -1.0)
        df["beta_genoa_ref_to_alt"] = (
            pd.to_numeric(df["beta_genoa"], errors="coerce") * sign
        )
    return df


def cmd_inspect(args) -> int:
    if not args.pairs.exists():
        print(f"harmonized pairs not found: {args.pairs}")
        print("Run the harmonization array job and merge step first.")
        return 1
    df = pd.read_csv(args.pairs, nrows=5000)
    print(f"pairs file : {args.pairs} ({args.pairs.stat().st_size/1e6:.1f} MB)")
    print(f"columns    : {list(df.columns)}")
    print(f"\nfirst rows:\n{df.head(3).to_string(index=False)}")

    labels = load_split_labels(args.split_dir)
    if labels:
        got = df["probeID"].astype(str).map(labels).value_counts(dropna=False)
        print(f"\nsplit membership in the first {len(df):,} rows:")
        for k, v in got.items():
            print(f"  {str(k):<10} {v:>8,}")

    fasta = FastaReader(args.fasta)
    print(f"\nreference  : {args.fasta} ({len(fasta.index)} sequences indexed)")
    sub = df.head(200)
    stats: Counter = Counter()
    out = annotate(sub, fasta, stats)
    fasta.close()
    print(f"\nallele resolution on a 200-row probe:")
    print(f"  resolved            : {stats['kept']}")
    print(f"  allele mismatch     : {stats['dropped_allele_mismatch']}")
    print(f"  non-SNV             : {stats['dropped_non_snv']}")
    print(f"  no reference seq    : {stats['dropped_no_reference_sequence']}")
    if stats["dropped_allele_mismatch"] > 0.2 * len(sub):
        print("\n  WARNING: high mismatch rate. Check the liftover and whether GENOA")
        print("  alleles are reported on the forward strand.")
    if not out.empty:
        print(f"\n  alters target CpG   : {int(out['alters_target_cpg'].sum())}")
        print(f"  creates a CpG       : {int(out['creates_cpg'].sum())}")
        print(f"  destroys a CpG      : {int(out['destroys_cpg'].sum())}")
    return 0


def cmd_run(args) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.pairs.exists():
        raise SystemExit(f"harmonized pairs not found: {args.pairs}")
    logging.info("reading %s", args.pairs)
    df = pd.read_csv(args.pairs)
    logging.info("%d harmonized pairs", len(df))

    labels = load_split_labels(args.split_dir)
    df["probe_split"] = df["probeID"].astype(str).map(labels).fillna("unknown")

    fasta = FastaReader(args.fasta)
    stats: Counter = Counter()
    logging.info("resolving REF/ALT against %s", args.fasta)
    df = annotate(df, fasta, stats)
    fasta.close()
    logging.info("%d pairs with resolved alleles", len(df))

    offset = pd.to_numeric(df["distance_bp"], errors="coerce")
    in_window = offset.between(MIN_SCOREABLE_OFFSET, MAX_SCOREABLE_OFFSET)
    hits_cpg = df["alters_target_cpg"].astype(bool)
    stats["dropped_outside_model_window"] = int((~in_window).sum())
    stats["dropped_alters_target_cpg"] = int((in_window & hits_cpg).sum())
    df = df[in_window & ~hits_cpg].reset_index(drop=True)
    logging.info("%d pairs scoreable inside the %d-bp model window",
                 len(df), MODEL_WINDOW_SIZE)

    df["Variant_ID"] = df["rsid"].astype(str)
    df["Gene"] = "NA"
    df["chr"] = df["snp_chr"]
    df["Position_1based"] = df["snp_pos_hg38"].astype("int64") + 1

    lead_cols = ["Variant_ID", "Gene", "chr", "Position_1based", "Ref", "Alt"]
    extra = [c for c in ("probeID", "probe_split", "cpg_chr", "cpg_pos_hg38",
                         "distance_bp", "abs_distance_bp", "alters_target_cpg",
                         "creates_cpg", "destroys_cpg", "allele_minor",
                         "allele_major", "alt_is_minor_allele", "af_genoa",
                         "beta_genoa", "beta_genoa_ref_to_alt", "se_genoa",
                         "p_wald") if c in df.columns]
    df = df[lead_cols + extra]

    written = {}
    for name, subset in (("heldout", df[df["probe_split"] == "test"]),
                         ("model_visible", df[df["probe_split"].isin(["train", "val"])])):
        if subset.empty:
            logging.warning("%s stratum is empty", name)
            continue
        target = out_dir / f"genoa_scoring_input_{name}.csv"
        subset.to_csv(target, index=False)
        written[name] = {
            "file": str(target), "rows": int(len(subset)),
            "unique_variants": int(subset["Variant_ID"].nunique()),
            "unique_probes": int(subset["probeID"].nunique()),
            "alters_target_cpg": int(subset["alters_target_cpg"].sum()),
            "cpg_altering_any": int((subset["creates_cpg"] | subset["destroys_cpg"]).sum()),
            "bytes": int(target.stat().st_size),
        }

    total = sum(stats.values())
    summary = {
        "analysis": "GENOA harmonized pairs -> SilentMethyl scoring input",
        "purpose": "mentor requirements: independent variant evaluation, ancestry, multi-cohort",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_pairs": str(args.pairs),
        "reference": str(args.fasta),
        "allele_resolution": dict(stats),
        "allele_mismatch_rate": (stats["dropped_allele_mismatch"] / total) if total else 0.0,
        "strata": written,
        "split_labelling": ("probe_split read from data/datafiles/{train,val,test}.csv, "
                            "not inferred from chromosome number"),
        "effect_allele_convention": (
            "GENOA is GEMMA output: beta_genoa is the effect per copy of allele1 = "
            "allele_minor. Ref/Alt here come from hg38, so Alt is the minor allele "
            "only when the reference base is the major allele. Use "
            "beta_genoa_ref_to_alt -- beta_genoa re-signed to the REF->ALT direction "
            "-- for every comparison against the model's MUT-minus-WT delta. Raw "
            "beta_genoa mixes two sign conventions."
        ),
        "excluded_from_scoring": {
            "outside_model_window": (
                "offset from the target C outside "
                f"[{MIN_SCOREABLE_OFFSET}, {MAX_SCOREABLE_OFFSET}]; invisible to the "
                "1000-bp crop, so the model would return a delta of exactly zero"
            ),
            "alters_target_cpg": (
                "offset 0 or +1 hits the target C or G; the model is defined around a "
                "CpG at indices 499:501 and the array reading is a SNP-under-probe "
                "artifact"
            ),
        },
        "reporting_guidance": (
            "Only the heldout stratum supports an independent-validation claim -- the "
            "model trained on probes in the model_visible stratum. Report heldout as "
            "primary. Within either stratum, report CpG-altering and non-CpG-altering "
            "variants separately: a variant that creates or destroys a CG has a "
            "near-deterministic direction that requires no model."
        ),
    }
    with (out_dir / "genoa_scoring_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print()
    print("=" * 68)
    print("allele resolution")
    for k, v in sorted(stats.items()):
        print(f"  {k:<32} {v:>10,}")
    rate = summary["allele_mismatch_rate"]
    print(f"  mismatch rate                    {rate:>10.1%}")
    if rate > 0.2:
        print("\n  WARNING: >20% of pairs had neither allele matching hg38.")
        print("  Check the liftover and the GENOA strand convention before scoring.")
    print("\nstrata")
    for name, info in written.items():
        print(f"  {name:<14} {info['rows']:>9,} pairs  "
              f"{info['unique_variants']:>8,} variants  "
              f"{info['unique_probes']:>7,} probes  "
              f"({info['cpg_altering_any']:,} CpG-altering)")
    resigned = stats.get("ref_is_minor_allele_effect_resigned", 0)
    kept = stats.get("kept", 0)
    print("\neffect-allele alignment")
    print(f"  pairs where hg38 REF is the MINOR allele  {resigned:>10,}"
          f"  ({(resigned / kept if kept else 0):.1%})")
    print("  -> for those, ALT is the MAJOR allele and the model's MUT-minus-WT")
    print("     delta runs OPPOSITE to beta_genoa. Compare against")
    print("     `beta_genoa_ref_to_alt`, never raw `beta_genoa`.")
    print("\nOnly `heldout` supports the independent-validation claim.")
    print("=" * 68)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    ap.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--inspect", action="store_true",
                    help="probe 200 rows and report allele-resolution rates; write nothing")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    return cmd_inspect(args) if args.inspect else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
