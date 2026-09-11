#!/usr/bin/env python3
"""Infer how a bigWig was made, and compare tracks side by side.

Why
---
The published breast accessibility track, data/reference/ATAC_seq.bw, was
converted from a BAM with parameters nobody recorded. That matters because the
joint multi-tissue model needs one accessibility recipe across every tissue, and
"match the existing file" is impossible without knowing what it was.

Some of the recipe IS recoverable from the file itself:

  bin size        a binned track has fixed-width intervals; the modal width is
                  the bin size directly.
  normalisation   raw coverage is integer-valued. Any normalisation -- CPM, BPM,
                  RPGC -- produces floats. If every value is a whole number the
                  track was almost certainly written with --normalizeUsing None.
  exclusion list  blacklisted regions are empty in a filtered track while their
                  flanks carry signal. Sampling a few known regions shows
                  whether a blacklist was applied.

Some is NOT recoverable: duplicate handling, MAPQ threshold, read extension.
Two tracks can therefore agree on everything this script measures and still
differ systematically. Treat a match here as necessary, not sufficient.

Usage (run from the repository root, in an environment with pyBigWig)
---------------------------------------------------------------------
    python -u data/profile_bigwig.py data/reference/ATAC_seq.bw
    python -u data/profile_bigwig.py data/reference/ATAC_seq.bw \\
                                     data/reference/Lung/ATAC_seq.bw
"""

from __future__ import annotations

import argparse
import gzip
import sys
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    import pyBigWig
except ImportError as exc:
    raise SystemExit(f"STOP: {exc}. Run in an environment with pyBigWig "
                     f"(the silentmethyl or deeptools env).") from exc

# Sampling windows spread across a few chromosomes, in gene-dense and gene-poor
# territory, so the value distribution is not read off one unusual locus.
SAMPLE_REGIONS = [
    ("chr1", 1_000_000, 1_400_000),
    ("chr1", 150_000_000, 150_400_000),
    ("chr7", 5_000_000, 5_400_000),
    ("chr12", 6_000_000, 6_400_000),
    ("chr17", 7_000_000, 7_400_000),
    ("chr19", 1_000_000, 1_400_000),
]


def profile(path: Path, blacklist: Path | None) -> dict:
    bw = pyBigWig.open(str(path))
    try:
        chroms = bw.chroms()
        hdr = bw.header() or {}
        out = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "n_chroms": len(chroms),
            "minVal": hdr.get("minVal"),
            "maxVal": hdr.get("maxVal"),
            "nBasesCovered": hdr.get("nBasesCovered"),
        }

        # --- bin size, from the intervals themselves ---------------------
        widths: Counter = Counter()
        for chrom, start, end in SAMPLE_REGIONS:
            if chrom not in chroms or chroms[chrom] < end:
                continue
            for iv in (bw.intervals(chrom, start, end) or [])[:20000]:
                widths[iv[1] - iv[0]] += 1
        out["interval_widths"] = widths.most_common(5)
        out["bin_size"] = widths.most_common(1)[0][0] if widths else None
        # A single dominant width means a binned track; a spread of widths means
        # run-length encoded base-resolution signal.
        if widths:
            top = widths.most_common(1)[0][1]
            out["binned"] = top / sum(widths.values()) > 0.9
        else:
            out["binned"] = None

        # --- integer vs float, i.e. normalised or not --------------------
        vals = []
        for chrom, start, end in SAMPLE_REGIONS:
            if chrom not in chroms or chroms[chrom] < end:
                continue
            v = np.asarray(bw.values(chrom, start, end), dtype=float)
            vals.append(v[np.isfinite(v)])
        allv = np.concatenate(vals) if vals else np.array([])
        out["n_sampled"] = int(allv.size)
        if allv.size:
            nz = allv[allv > 0]
            out["frac_zero"] = float((allv == 0).mean())
            integral = bool(np.all(np.isclose(allv, np.round(allv), atol=1e-9)))
            out["integer_valued"] = integral
            out["normalisation_guess"] = ("none (raw counts)" if integral
                                          else "normalised (CPM/BPM/RPGC)")
            if nz.size:
                qs = [50, 75, 90, 99, 99.9]
                out["nonzero_quantiles"] = {
                    f"p{q}": round(float(np.percentile(nz, q)), 4) for q in qs}
                out["nonzero_mean"] = round(float(nz.mean()), 4)
                # Smallest positive value: for raw counts this is 1; for CPM it
                # is the per-read scale factor, which gives the library size.
                out["min_positive"] = round(float(nz.min()), 6)
                if not integral and nz.min() > 0:
                    out["implied_reads_if_cpm"] = int(round(1e6 / nz.min()))

        # --- was an exclusion list applied? -------------------------------
        if blacklist and blacklist.is_file():
            inside, flank = [], []
            opener = gzip.open if blacklist.suffix == ".gz" else open
            with opener(blacklist, "rt") as fh:
                for i, line in enumerate(fh):
                    if i >= 40:
                        break
                    f = line.split()
                    if len(f) < 3 or f[0] not in chroms:
                        continue
                    s, e = int(f[1]), int(f[2])
                    if e - s < 1000 or e + 20000 > chroms[f[0]]:
                        continue
                    a = np.asarray(bw.values(f[0], s, e), dtype=float)
                    b = np.asarray(bw.values(f[0], e + 10000, e + 20000), dtype=float)
                    inside.append(np.nan_to_num(a).mean())
                    flank.append(np.nan_to_num(b).mean())
            if inside:
                mi, mf = float(np.mean(inside)), float(np.mean(flank))
                out["blacklist_mean_inside"] = round(mi, 4)
                out["blacklist_mean_flank"] = round(mf, 4)
                out["blacklist_applied"] = bool(mi < 0.05 * max(mf, 1e-9))
        return out
    finally:
        bw.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", nargs="+", type=Path)
    ap.add_argument("--blacklist", type=Path,
                    default=Path("data/reference/ENCFF356LFX.bed.gz"))
    args = ap.parse_args(argv)

    profiles = []
    for t in args.tracks:
        if not t.is_file():
            raise SystemExit(f"STOP: {t} not found")
        print(f"profiling {t} ...", flush=True)
        profiles.append(profile(t, args.blacklist))

    names = [Path(p["path"]).parent.name + "/" + Path(p["path"]).name
             for p in profiles]
    width = max(24, max(len(n) for n in names) + 2)

    def row(label, key, fmt=str):
        cells = []
        for p in profiles:
            v = p.get(key)
            cells.append("--" if v is None else fmt(v))
        print(f"  {label:<26}" + "".join(f"{c:<{width}}" for c in cells))

    print()
    print("=" * (28 + width * len(profiles)))
    print(f"  {'':<26}" + "".join(f"{n:<{width}}" for n in names))
    print("=" * (28 + width * len(profiles)))
    row("size (MB)", "bytes", lambda v: f"{v/1e6:.0f}")
    row("bin size (bp)", "bin_size")
    row("binned (vs base-res)", "binned")
    row("integer valued", "integer_valued")
    row("normalisation", "normalisation_guess")
    row("smallest positive value", "min_positive")
    row("implied library size", "implied_reads_if_cpm", lambda v: f"{v:,}")
    row("fraction zero", "frac_zero", lambda v: f"{v:.3f}")
    row("non-zero mean", "nonzero_mean")
    for q in ("p50", "p75", "p90", "p99", "p99.9"):
        print(f"  {'  non-zero ' + q:<26}"
              + "".join(f"{str((p.get('nonzero_quantiles') or {}).get(q, '--')):<{width}}"
                        for p in profiles))
    row("max (header)", "maxVal")
    row("blacklist applied", "blacklist_applied")
    row("  mean inside blacklist", "blacklist_mean_inside")
    row("  mean in flank", "blacklist_mean_flank")

    print()
    print("Recoverable from the file: bin size, whether values were normalised,")
    print("and whether an exclusion list was applied.")
    print("NOT recoverable: duplicate handling, MAPQ threshold, read extension.")
    print("Two tracks can match on everything above and still differ in those,")
    print("so treat agreement here as necessary rather than sufficient.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
