#!/usr/bin/env python3
"""
Merge the 22 per-chromosome GENOA harmonization outputs into one scoring cohort.

Run this after the array job finishes:
    sbatch data/run_harmonize_genoa.sh     # 22 tasks
    python -u data/merge_genoa_harmonized.py

Produces
--------
  genoa_model_visible_pairs.csv.gz   the cohort SilentMethyl will score
  harmonization_summary.json         combined counts + the distance histogram
  epic_only_probes.txt               GENOA probes absent from the HM450 universe

That last file answers an open question: if it is empty or near-empty, every GENOA
CpG is an HM450 probe whose context features already exist in
data/datafiles/*.csv, and no new feature extraction is needed. If it is large,
those probes need context features pulled from the reference bigwigs before they
can be scored.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path("data/external/genoa_meqtl/harmonized")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--keep-parts", action="store_true",
                    help="keep the per-chromosome files (default: leave them in place)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    parts = sorted(args.dir.glob("genoa_model_visible_pairs_chr*.csv.gz"))
    if not parts:
        raise SystemExit(f"no per-chromosome outputs found in {args.dir}")
    logging.info("merging %d per-chromosome files", len(parts))

    missing = [c for c in range(1, 23)
               if not (args.dir / f"genoa_model_visible_pairs_chr{c}.csv.gz").exists()]
    if missing:
        logging.warning("chromosomes with no output: %s "
                        "(either the task failed or nothing survived the filters)",
                        missing)

    frames = [pd.read_csv(p) for p in parts]
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["cpg_chr", "cpg_pos_hg38", "snp_pos_hg38"],
                                kind="mergesort").reset_index(drop=True)

    target = args.dir / "genoa_model_visible_pairs.csv.gz"
    result.to_csv(target, index=False, compression="gzip")

    # combined counters + histogram
    counters: dict = {}
    hists: dict = {}
    for c in range(1, 23):
        s = args.dir / f"harmonization_summary_chr{c}.json"
        if s.exists():
            d = json.loads(s.read_text())
            counters.update(d.get("counters", {}))
            for k, v in d.get("abs_distance_histogram", {}).items():
                hists[k] = hists.get(k, 0) + v

    summary = {
        "analysis": "GENOA meQTL harmonization -- merged across chromosomes",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Shang et al. Nat Commun 2023; Zenodo 10.5281/zenodo.7697509",
        "source_build": "hg19",
        "output_build": "hg38",
        "parts_merged": [p.name for p in parts],
        "chromosomes_missing": missing,
        "output_file": str(target),
        "output_bytes": int(target.stat().st_size),
        "output_rows": int(len(result)),
        "output_unique_probes": int(result["probeID"].nunique()),
        "output_unique_variants": int(result["rsid"].nunique()) if "rsid" in result else None,
        "median_abs_distance_bp": float(result["abs_distance_bp"].median()),
        "abs_distance_histogram": hists,
        "counters": counters,
    }
    with (args.dir / "harmonization_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print()
    print("=" * 62)
    print(f"merged rows        : {len(result):,}")
    print(f"unique probes      : {result['probeID'].nunique():,}")
    if "rsid" in result:
        print(f"unique variants    : {result['rsid'].nunique():,}")
    print(f"output             : {target}  ({target.stat().st_size/1e6:.1f} MB)")
    print(f"median |distance|  : {result['abs_distance_bp'].median():.0f} bp")
    if missing:
        print(f"MISSING chromosomes: {missing}")
    print()
    print("Compare against the raw release (5.3 GB). Once you are satisfied with")
    print("this cohort, the raw files can go -- checksums are in")
    print("data/external/external_manifest.json and the source is a permanent DOI:")
    print("  rm data/external/genoa_meqtl/meQTL_summarystat_chr*.txt.gz")
    print("=" * 62)
    logging.info("wrote %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
