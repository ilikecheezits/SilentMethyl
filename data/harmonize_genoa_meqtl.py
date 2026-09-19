#!/usr/bin/env python3
"""
Turn the raw GENOA meQTL release into a compact, model-visible scoring cohort.

What this does, in one sentence
-------------------------------
Takes 5.3 GB of genome-wide African-American meQTL summary statistics and keeps
only the variant-CpG pairs SilentMethyl can actually score -- pairs whose CpG is a
QC-passing HM450 probe and whose SNP sits inside the model's 1 kb window -- writing
well under 200 MB.

Which mentor requirements this serves
-------------------------------------
  * Independent variant evaluation -- takes the external variant set from 81
    associations to potentially hundreds of thousands.
  * Ancestry analyses -- GENOA is African American; contrasting it with the
    European-dominant eGTEx set is the cross-ancestry comparison.
  * Multi-cohort testing -- GENOA is blood on EPIC, so it shifts tissue, platform
    and ancestry all at once relative to TCGA-BRCA breast on HM450.

Two design decisions worth knowing
-----------------------------------
1. **The CpG is never lifted over.** Probe IDs are platform-stable, so GENOA's
   `CpG` column joins directly to the hg38 manifest and inherits hg38 coordinates.
   Only the SNP position needs hg19 -> hg38. This halves the liftover work and
   removes a whole class of coordinate error.

2. **Raw files stay untouched.** This writes new files. The 5.3 GB release remains
   byte-identical and checksum-verifiable against Zenodo 10.5281/zenodo.7697509
   until you choose to delete it.

Coordinate conventions (stated loudly because this is where silent bugs live)
----------------------------------------------------------------------------
  GENOA `ps`      : 1-based SNP position, hg19
  manifest CpG_beg: 0-based CpG start, hg38
Both are converted to 0-based hg38 before the distance is computed. The summary
prints a distance histogram -- if the convention were wrong you would see a
systematic offset rather than a clean peak at short distances.

Usage (run from the repository root)
------------------------------------
    # See what columns were detected and how big things are. Reads headers only.
    python -u data/harmonize_genoa_meqtl.py --inspect

    # Try one chromosome first. chr22 is the smallest.
    python -u data/harmonize_genoa_meqtl.py --chrom 22

    # Then all 22.
    python -u data/harmonize_genoa_meqtl.py --all
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_GENOA = Path("data/external/genoa_meqtl")
DEFAULT_MANIFEST = Path("data/HM450.hg38.manifest.tsv.gz")
DEFAULT_CHAIN = Path("data/reference/hg19ToHg38.over.chain.gz")
DEFAULT_OUT = Path("data/external/genoa_meqtl/harmonized")

VALID_CHROMS = {f"chr{c}" for c in range(1, 23)}
CHUNK = 2_000_000


def norm_chrom(value) -> str:
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    return s if s.startswith("chr") else f"chr{s}"


def detect_probe_column(columns) -> str | None:
    for candidate in ("probeID", "Probe_ID", "probe_id", "IlmnID", "Name"):
        if candidate in columns:
            return candidate
    return None


def detect_genoa_columns(path: Path) -> dict:
    """GENOA README: chr, CpG, cpgstart, cpgend, rs, ps, allele1, allele0, af, beta, se, p_wald"""
    with gzip.open(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    if len(header) == 1:
        header = header[0].split()
    lower = {c.lower(): c for c in header}

    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None

    return {
        "header": header,
        "chr": pick("chr", "chrom", "chromosome"),
        "cpg": pick("cpg", "cpg_site", "probe", "probeid"),
        "cpgstart": pick("cpgstart", "cpg_start", "cpg_beg"),
        "rs": pick("rs", "rsid", "snp"),
        "ps": pick("ps", "pos", "position", "bp"),
        "allele1": pick("allele1", "minor_allele", "a1"),
        "allele0": pick("allele0", "major_allele", "a0"),
        "af": pick("af", "maf", "allele_frequency"),
        "beta": pick("beta", "effect"),
        "se": pick("se", "stderr"),
        "p": pick("p_wald", "p", "pval", "pvalue"),
    }


def load_probe_universe(manifest: Path, drop_masked: bool) -> pd.DataFrame:
    head = pd.read_csv(manifest, sep="\t", nrows=0)
    probe_col = detect_probe_column(head.columns)
    if probe_col is None:
        raise SystemExit(f"{manifest}: could not find a probe ID column")
    usecols = [probe_col, "CpG_chrm", "CpG_beg"]
    if "MASK_general" in head.columns:
        usecols.append("MASK_general")

    df = pd.read_csv(manifest, sep="\t", usecols=usecols, low_memory=False)
    df = df.rename(columns={probe_col: "probeID", "CpG_chrm": "cpg_chr",
                            "CpG_beg": "cpg_pos_hg38"})
    df["cpg_chr"] = df["cpg_chr"].map(norm_chrom)
    df = df[df["cpg_chr"].isin(VALID_CHROMS)]
    df = df.dropna(subset=["probeID", "cpg_chr", "cpg_pos_hg38"])
    df["cpg_pos_hg38"] = df["cpg_pos_hg38"].astype("int64")

    n_all = len(df)
    if drop_masked and "MASK_general" in df.columns:
        mask = df["MASK_general"].astype(str).str.lower().isin({"true", "1", "yes"})
        df = df[~mask]
    df = df.drop_duplicates(subset="probeID", keep="first").reset_index(drop=True)
    logging.info("probe universe: %d probes (%d before MASK_general filter)",
                 len(df), n_all)
    return df[["probeID", "cpg_chr", "cpg_pos_hg38"]]


def make_lifter(chain: Path):
    try:
        from pyliftover import LiftOver
    except ImportError:
        raise SystemExit(
            "pyliftover not installed. Run:  pip install pyliftover\n"
            "(CrossMap also works but this script uses pyliftover.)"
        )
    if not chain.exists():
        raise SystemExit(f"chain file not found: {chain}")
    logging.info("loading chain %s", chain)
    return LiftOver(str(chain))


def lift_positions(lifter, chrom: str, positions_hg19_0based: np.ndarray) -> dict:
    """Lift each unique 0-based position once. Returns {hg19_pos: hg38_pos or None}."""
    out = {}
    for p in positions_hg19_0based:
        res = lifter.convert_coordinate(chrom, int(p))
        if res and res[0][0] == chrom:
            out[int(p)] = int(res[0][1])
        else:
            out[int(p)] = None
    return out


def harmonize_chromosome(chrom_num: int, cols: dict, genoa_dir: Path,
                         probes: pd.DataFrame, lifter, half_window: int,
                         max_p: float | None, prefilter_margin: int,
                         stats: Counter) -> pd.DataFrame:
    path = genoa_dir / f"meQTL_summarystat_chr{chrom_num}.txt.gz"
    if not path.exists():
        logging.warning("missing %s", path)
        return pd.DataFrame()

    chrom = f"chr{chrom_num}"
    probe_index = probes.set_index("probeID")
    probe_ids = set(probe_index.index)

    usecols = [c for c in (cols["chr"], cols["cpg"], cols["cpgstart"], cols["rs"], cols["ps"],
                           cols["allele1"], cols["allele0"], cols["af"],
                           cols["beta"], cols["se"], cols["p"]) if c]

    kept_chunks = []
    logging.info("[%s] streaming %s", chrom, path.name)
    reader = pd.read_csv(path, sep="\t", usecols=usecols,
                         chunksize=CHUNK, low_memory=True)

    for i, chunk in enumerate(reader, 1):
        stats[f"{chrom}_rows_in"] += len(chunk)
        chunk = chunk.rename(columns={
            cols["cpg"]: "probeID", cols["rs"]: "rsid", cols["ps"]: "snp_pos_hg19",
            **({cols["cpgstart"]: "cpg_pos_hg19"} if cols["cpgstart"] else {}),
            cols["allele1"]: "allele_minor", cols["allele0"]: "allele_major",
            cols["af"]: "af_genoa", cols["beta"]: "beta_genoa",
            cols["se"]: "se_genoa", cols["p"]: "p_wald",
        })
        chunk = chunk[chunk["probeID"].astype(str).isin(probe_ids)]
        stats[f"{chrom}_rows_probe_ok"] += len(chunk)
        if chunk.empty:
            continue
        if "cpg_pos_hg19" in chunk.columns:
            snp0 = pd.to_numeric(chunk["snp_pos_hg19"], errors="coerce") - 1
            cpg0 = pd.to_numeric(chunk["cpg_pos_hg19"], errors="coerce")
            approx = (snp0 - cpg0).abs()
            n_pre = len(chunk)
            chunk = chunk[approx <= (half_window + prefilter_margin)]
            stats[f"{chrom}_rows_dropped_hg19_prefilter"] += n_pre - len(chunk)
            stats[f"{chrom}_rows_prefilter_ok"] += len(chunk)
            if chunk.empty:
                continue

        if max_p is not None and "p_wald" in chunk.columns:
            chunk = chunk[pd.to_numeric(chunk["p_wald"], errors="coerce") <= max_p]
            stats[f"{chrom}_rows_p_ok"] += len(chunk)
        if not chunk.empty:
            kept_chunks.append(chunk)
        if i % 5 == 0:
            logging.info("[%s]   chunk %d, kept %d so far", chrom, i,
                         sum(len(c) for c in kept_chunks))

    if not kept_chunks:
        logging.info("[%s] nothing survived probe/p filters", chrom)
        return pd.DataFrame()

    df = pd.concat(kept_chunks, ignore_index=True)
    del kept_chunks

    df["snp_pos_hg19"] = pd.to_numeric(df["snp_pos_hg19"], errors="coerce")
    df = df.dropna(subset=["snp_pos_hg19"])
    df["snp_pos_hg19"] = df["snp_pos_hg19"].astype("int64")
    hg19_0 = (df["snp_pos_hg19"] - 1).to_numpy()
    uniq = np.unique(hg19_0)
    logging.info("[%s] lifting %d unique SNP positions (%d rows)",
                 chrom, len(uniq), len(df))
    mapping = lift_positions(lifter, chrom, uniq)
    df["snp_pos_hg38"] = [mapping.get(int(p)) for p in hg19_0]
    n_before = len(df)
    df = df.dropna(subset=["snp_pos_hg38"])
    stats[f"{chrom}_liftover_failed"] += n_before - len(df)
    df["snp_pos_hg38"] = df["snp_pos_hg38"].astype("int64")

    df = df.merge(probe_index.reset_index(), on="probeID", how="inner")
    df = df[df["cpg_chr"] == chrom]
    df["distance_bp"] = (df["snp_pos_hg38"] - df["cpg_pos_hg38"]).astype("int64")
    df["abs_distance_bp"] = df["distance_bp"].abs()

    n_before = len(df)
    df = df[df["abs_distance_bp"] <= half_window]
    stats[f"{chrom}_rows_outside_window"] += n_before - len(df)
    stats[f"{chrom}_rows_out"] += len(df)

    df["snp_chr"] = chrom
    keep = ["probeID", "cpg_chr", "cpg_pos_hg38", "rsid", "snp_chr",
            "snp_pos_hg19", "snp_pos_hg38", "distance_bp", "abs_distance_bp",
            "allele_minor", "allele_major", "af_genoa", "beta_genoa",
            "se_genoa", "p_wald"]
    df = df[[c for c in keep if c in df.columns]]
    logging.info("[%s] kept %d model-visible pairs", chrom, len(df))
    return df


def cmd_inspect(args) -> int:
    print("=== probe universe ===")
    probes = load_probe_universe(args.manifest, drop_masked=not args.keep_masked)
    print(f"  {len(probes):,} usable HM450 probes on chr1-22")
    print(f"  example: {probes.iloc[0].to_dict()}")

    print("\n=== GENOA files ===")
    total = 0
    for c in range(1, 23):
        p = args.genoa_dir / f"meQTL_summarystat_chr{c}.txt.gz"
        if p.exists():
            mb = p.stat().st_size / 1e6
            total += mb
            print(f"  chr{c:<2} {mb:8.1f} MB")
        else:
            print(f"  chr{c:<2} MISSING")
    print(f"  total {total/1000:.1f} GB")

    first = next((args.genoa_dir / f"meQTL_summarystat_chr{c}.txt.gz"
                  for c in range(22, 0, -1)
                  if (args.genoa_dir / f"meQTL_summarystat_chr{c}.txt.gz").exists()), None)
    if first is None:
        print("\nNo GENOA files found -- nothing to inspect.")
        return 1

    print(f"\n=== detected columns in {first.name} ===")
    cols = detect_genoa_columns(first)
    print(f"  raw header: {cols['header']}")
    for k, v in cols.items():
        if k == "header":
            continue
        flag = "ok " if v else "NOT FOUND"
        print(f"  {k:<9} -> {str(v):<20} {flag}")
    missing = [k for k, v in cols.items() if k != "header" and v is None]
    if missing:
        print(f"\n  WARNING: unmapped fields {missing} -- edit detect_genoa_columns()")

    print("\n=== first 3 data rows ===")
    preview = pd.read_csv(first, sep="\t", nrows=3)
    print(preview.to_string(index=False))

    print("\n=== coordinate assumptions ===")
    print("  GENOA ps       : 1-based, hg19  -> converted to 0-based before liftover")
    print("  manifest CpG_beg: 0-based, hg38 -> used as-is")
    print("  If these are wrong the distance histogram will show a systematic offset.")
    return 0


def cmd_run(args) -> int:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    probes = load_probe_universe(args.manifest, drop_masked=not args.keep_masked)
    first = next((args.genoa_dir / f"meQTL_summarystat_chr{c}.txt.gz"
                  for c in range(22, 0, -1)
                  if (args.genoa_dir / f"meQTL_summarystat_chr{c}.txt.gz").exists()), None)
    if first is None:
        raise SystemExit(f"no GENOA files under {args.genoa_dir}")
    cols = detect_genoa_columns(first)
    required = ["chr", "cpg", "ps"]
    if any(cols[r] is None for r in required):
        raise SystemExit(f"required columns not detected: "
                         f"{[r for r in required if cols[r] is None]}. Run --inspect.")

    lifter = make_lifter(args.chain)
    chroms = [args.chrom] if args.chrom else list(range(1, 23))
    stats: Counter = Counter()
    frames = []

    for c in chroms:
        df = harmonize_chromosome(c, cols, args.genoa_dir, probes, lifter,
                                  args.half_window, args.max_p,
                                  args.prefilter_margin, stats)
        if not df.empty:
            frames.append(df)

    if not frames:
        logging.error("no pairs survived. Check --inspect output and coordinate bases.")
        return 1

    result = pd.concat(frames, ignore_index=True)
    suffix = f"_chr{args.chrom}" if args.chrom else ""
    target = out / f"genoa_model_visible_pairs{suffix}.csv.gz"
    result.to_csv(target, index=False, compression="gzip")

    hist_edges = [0, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
    hist = {}
    for lo, hi in zip(hist_edges[:-1], hist_edges[1:]):
        hist[f"{lo}-{hi}"] = int(((result["abs_distance_bp"] >= lo) &
                                  (result["abs_distance_bp"] < hi)).sum())

    n_probes_genoa = result["probeID"].nunique()
    summary = {
        "analysis": "GENOA meQTL harmonization to hg38 model-visible pairs",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Shang et al. Nat Commun 2023; Zenodo 10.5281/zenodo.7697509",
        "source_build": "hg19",
        "output_build": "hg38",
        "chromosomes": chroms,
        "half_window_bp": args.half_window,
        "hg19_prefilter_margin_bp": args.prefilter_margin,
        "max_p_filter": args.max_p,
        "masked_probes_dropped": not args.keep_masked,
        "probe_universe_size": int(len(probes)),
        "output_rows": int(len(result)),
        "output_unique_probes": int(n_probes_genoa),
        "output_unique_variants": int(result["rsid"].nunique()) if "rsid" in result else None,
        "output_file": str(target),
        "output_bytes": int(target.stat().st_size),
        "abs_distance_histogram": hist,
        "median_abs_distance_bp": float(result["abs_distance_bp"].median()),
        "counters": dict(stats),
        "coordinate_note": (
            "GENOA ps is 1-based hg19 and was converted to 0-based before liftover; "
            "manifest CpG_beg is 0-based hg38 and used as-is. A clean short-distance "
            "peak in abs_distance_histogram indicates the conventions align."
        ),
        "cpg_liftover_note": (
            "CpG coordinates were NOT lifted. Probe IDs are platform-stable, so GENOA "
            "probes inherit hg38 coordinates from the manifest join. Only SNP positions "
            "were lifted."
        ),
    }
    with (out / f"harmonization_summary{suffix}.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print()
    print("=" * 62)
    print(f"model-visible pairs : {len(result):,}")
    print(f"unique probes       : {n_probes_genoa:,}")
    if "rsid" in result:
        print(f"unique variants     : {result['rsid'].nunique():,}")
    print(f"output              : {target}  ({target.stat().st_size/1e6:.1f} MB)")
    print(f"median |distance|   : {result['abs_distance_bp'].median():.0f} bp")
    print("\nabs distance histogram (sanity check on coordinate bases):")
    for k, v in hist.items():
        bar = "#" * min(50, int(50 * v / max(hist.values()))) if max(hist.values()) else ""
        print(f"  {k:>9} bp  {v:>10,}  {bar}")
    print("=" * 62)
    logging.info("wrote %s", out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genoa-dir", type=Path, default=DEFAULT_GENOA)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--half-window", type=int, default=500,
                    help="max |SNP - CpG| distance in bp (default 500 = the 1 kb model window)")
    ap.add_argument("--prefilter-margin", type=int, default=2000,
                    help="extra bp allowed in the hg19 pre-filter to absorb liftover "
                         "indel differences (default 2000)")
    ap.add_argument("--max-p", type=float, default=None,
                    help="optional p_wald cutoff applied before liftover")
    ap.add_argument("--keep-masked", action="store_true",
                    help="keep probes flagged by MASK_general (default: drop them)")
    ap.add_argument("--inspect", action="store_true",
                    help="print detected columns and file sizes, process nothing")
    ap.add_argument("--chrom", type=int, default=None, help="process one chromosome")
    ap.add_argument("--all", action="store_true", help="process all 22 autosomes")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    if args.inspect:
        return cmd_inspect(args)
    if not (args.all or args.chrom):
        ap.error("choose --inspect, --chrom N, or --all")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
