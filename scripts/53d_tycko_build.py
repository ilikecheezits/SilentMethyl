#!/usr/bin/env python3
"""
Build the ASM scoring input from Do & Tycko 2020 (Tasks E1 + E2). CPU only.

Why this replaces the Nat Commun atlas build
---------------------------------------------
`53_asm_build.py` used the Rosenski/Loyfer atlas, which publishes ASM
significance but NOT a signed allelic methylation difference. That capped the
analysis at discrimination (E1) and made direction concordance and signed
Spearman -- the two statistics the mentor named first -- impossible.

Do & Tycko (Genome Biology 21:153, 2020) publish, per ASM index SNP:

    snp avg avg meth read ref    % methylation on REF-allele reads
    snp avg avg meth read alt    % methylation on ALT-allele reads
    snp avg diff alt ref         signed ALT - REF, percentage points

keyed REF->ALT, the same convention as the model's MUT-minus-WT delta. So this
one catalogue answers BOTH questions and is used as the sole ASM source.

Coordinates, which are mixed builds -- read this before trusting a position
--------------------------------------------------------------------------
    `dmr` column             hg19, needs liftOver
    SNP position             NOT in the table; rsID only, resolved through
                             Ensembl REST, which returns GRCh38 already

Verified on rs67165842: its DMR reads 1:100015940 while GRCh38 is 99550369.
Getting this backwards would silently mis-place every variant.

The REF allele is re-derived from hg38 rather than trusted
-----------------------------------------------------------
Their REF/ALT assignment comes from their own pipeline. If a site's alleles are
swapped relative to hg38, the sign of `diff alt ref` must flip with them, and a
silent error there inverts the entire direction-concordance result -- the one
failure this analysis cannot survive. So hg38's base is read directly and:

    hg38 REF == their REF   -> keep the sign
    hg38 REF == their ALT   -> swap alleles AND negate the observed effect
    neither matches         -> drop, counted

Multi-allelic sites are excluded. Ensembl gives REF plus several ALTs and the
table does not say which ALT its methylation refers to; guessing would score a
different substitution than the one measured.

Two labelled sets are emitted
-----------------------------
    asm_label 1   CpG inside the SNP's own ASM DMR      (E1 positive, E2 unit)
    asm_label 0   CpG outside every ASM DMR, distance-matched   (E1 negative)

E2 uses the positives only: per SNP, the mean predicted delta over its DMR CpGs
is compared against that SNP's observed signed effect.

Usage
-----
    python -u scripts/53d_tycko_build.py --output-dir data/external/asm_tycko_gb2020/scoring
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
from pyliftover import LiftOver

LOGGER = logging.getLogger("tycko_build")

WINDOW_SIZE = 5000
CENTER_C_INDEX_FULL = 2499
MODEL_WINDOW_SIZE = 1000
CENTER_C_INDEX = 499
PROTECTED_OFFSETS = frozenset({0, 1})
MIN_OFFSET, MAX_OFFSET = -CENTER_C_INDEX, MODEL_WINDOW_SIZE - CENTER_C_INDEX - 1
TEST_CHROMS = ("chr8", "chr9")

TRACKS = ["Ref_ATAC_Signal", "Ref_H3K4me3_Signal", "Ref_H3K27ac_Signal",
          "Ref_H3K27me3_Signal", "Ref_H3K9me3_Signal", "Ref_H3K36me3_Signal",
          "Ref_H3K4me1_Signal"]
PHYLOP_1 = "Target_Base_PhyloP_100way_1"
PHYLOP_2 = "Target_Base_PhyloP_100way_2"
FEATURES = TRACKS + [PHYLOP_1, PHYLOP_2]

EFFECT = "snp avg diff alt ref"
METH_REF = "snp avg avg meth read ref"
METH_ALT = "snp avg avg meth read alt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def read_chromosomes(fasta: Path, wanted: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    current, buffer = None, []
    with fasta.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if current in wanted:
                    out[current] = "".join(buffer).upper()
                current = line[1:].split()[0]
                buffer = []
                continue
            if current in wanted:
                buffer.append(line.strip())
    if current in wanted and current not in out:
        out[current] = "".join(buffer).upper()
    missing = wanted - set(out)
    if missing:
        raise ValueError(f"{fasta} missing {sorted(missing)}")
    return out


def load_table(xlsx: Path, cache: Path) -> tuple[pd.DataFrame, dict]:
    """Table S2 joined to the cached Ensembl GRCh38 resolution."""
    used = ["snpid", "dmr", METH_REF, METH_ALT, EFFECT, "snp avg fdr",
            "all asm tissues", "n asm samples", "n asm individuals"]
    table = pd.read_excel(xlsx, sheet_name="Table S2", header=1, usecols=used)
    counters = {"total_rows": int(len(table))}

    resolved = pd.read_csv(cache)[["snpid", "chr", "pos1", "alleles"]]
    resolved["n_alleles"] = resolved["alleles"].astype(str).str.count("/") + 1
    counters["resolved_chr8_chr9"] = int(len(resolved))

    biallelic = resolved[resolved["n_alleles"] == 2].copy()
    counters["multiallelic_excluded"] = int(len(resolved) - len(biallelic))

    joined = table.merge(biallelic, on="snpid", how="inner", validate="one_to_one")
    counters["biallelic_joined"] = int(len(joined))

    region = joined["dmr"].astype(str).str.extract(r"^([\dXY]+):(\d+)-(\d+)$")
    joined["dmr_chrom"] = "chr" + region[0]
    joined["dmr_start_hg19"] = pd.to_numeric(region[1], errors="coerce")
    joined["dmr_end_hg19"] = pd.to_numeric(region[2], errors="coerce")
    joined = joined.dropna(subset=["dmr_start_hg19", "dmr_end_hg19"])
    counters["with_parsable_dmr"] = int(len(joined))
    # Rename once. The published column names contain spaces, which itertuples
    # mangles into positional _N attributes -- a reliable source of silent
    # off-by-one column reads.
    joined = joined.rename(columns={
        EFFECT: "observed_diff", METH_REF: "meth_ref", METH_ALT: "meth_alt",
        "snp avg fdr": "fdr", "all asm tissues": "asm_tissues",
        "n asm samples": "n_asm_samples", "n asm individuals": "n_asm_individuals",
    })
    return joined.reset_index(drop=True), counters


def lift_dmrs(frame: pd.DataFrame, chain: Path) -> tuple[pd.DataFrame, dict]:
    """DMR bounds hg19 -> hg38. SNP positions are already GRCh38 from Ensembl."""
    lifter = LiftOver(str(chain))
    counters = {"dmr_unmapped": 0, "dmr_inverted": 0}
    starts, ends, keep = [], [], []
    for row in frame.itertuples(index=False):
        chrom = str(row.dmr_chrom)
        a = lifter.convert_coordinate(chrom, int(row.dmr_start_hg19))
        b = lifter.convert_coordinate(chrom, int(row.dmr_end_hg19))
        if not a or not b or a[0][0] != chrom or b[0][0] != chrom:
            counters["dmr_unmapped"] += 1
            starts.append(-1); ends.append(-1); keep.append(False); continue
        s, e = int(a[0][1]), int(b[0][1])
        if e < s:
            counters["dmr_inverted"] += 1
            s, e = e, s
        starts.append(s); ends.append(e); keep.append(True)
    out = frame.copy()
    out["dmr_start_hg38"] = starts
    out["dmr_end_hg38"] = ends
    out = out[keep].reset_index(drop=True)
    return out, counters


def merged_intervals(frame: pd.DataFrame) -> dict[str, tuple[list[int], list[int]]]:
    index: dict[str, tuple[list[int], list[int]]] = {}
    for chrom, group in frame.groupby("dmr_chrom"):
        spans = sorted(zip(group["dmr_start_hg38"].astype(int),
                           group["dmr_end_hg38"].astype(int)))
        starts, ends = [], []
        for a, b in spans:
            if starts and a <= ends[-1] + 1:
                ends[-1] = max(ends[-1], b)
            else:
                starts.append(a); ends.append(b)
        index[chrom] = (starts, ends)
    return index


def build_pairs(frame: pd.DataFrame, sequences: dict[str, str],
                cpgs: dict[str, list[int]],
                dmr_index: dict[str, tuple[list[int], list[int]]]):
    counters = {"ref_matches_hg38": 0, "alleles_swapped_sign_flipped": 0,
                "ref_matches_neither_dropped": 0, "snp_out_of_bounds": 0,
                "cpg_out_of_bounds": 0, "protected_offset": 0,
                "positive": 0, "negative": 0, "snp_with_no_cpg": 0}

    def in_any_dmr(chrom: str, position: int) -> bool:
        starts, ends = dmr_index.get(chrom, ([], []))
        i = bisect.bisect_right(starts, position) - 1
        return i >= 0 and position <= ends[i]

    rows = []
    for row in frame.itertuples(index=False):
        chrom = str(row.chr)
        sequence = sequences[chrom]
        snp0 = int(row.pos1) - 1                    # Ensembl start is 1-based
        if not 0 <= snp0 < len(sequence):
            counters["snp_out_of_bounds"] += 1
            continue

        a1, a2 = str(row.alleles).upper().split("/")
        genome_ref = sequence[snp0]
        observed = float(row.observed_diff)

        if genome_ref == a1:
            ref, alt, signed = a1, a2, observed
            counters["ref_matches_hg38"] += 1
        elif genome_ref == a2:
            # hg38 reference is what they called ALT: swap, and negate, because
            # the published effect is keyed to THEIR ref.
            ref, alt, signed = a2, a1, -observed
            counters["alleles_swapped_sign_flipped"] += 1
        else:
            counters["ref_matches_neither_dropped"] += 1
            continue

        positions = cpgs[chrom]
        lo = bisect.bisect_left(positions, snp0 - MAX_OFFSET)
        hi = bisect.bisect_right(positions, snp0 - MIN_OFFSET)
        found = 0
        for cpg_c in positions[lo:hi]:
            offset = snp0 - cpg_c
            if offset in PROTECTED_OFFSETS:
                counters["protected_offset"] += 1
                continue
            start = cpg_c - CENTER_C_INDEX_FULL
            if start < 0 or start + WINDOW_SIZE > len(sequence):
                counters["cpg_out_of_bounds"] += 1
                continue
            if sequence[cpg_c:cpg_c + 2] != "CG":
                continue
            inside_own = int(row.dmr_start_hg38) <= cpg_c <= int(row.dmr_end_hg38)
            if inside_own:
                label = 1
                counters["positive"] += 1
                found += 1
            elif not in_any_dmr(chrom, cpg_c):
                label = 0
                counters["negative"] += 1
            else:
                continue
            rows.append({
                "Variant_ID": f"{chrom}_{snp0 + 1}_{ref}_{alt}_b38",
                "rsid": str(row.snpid),
                "chr": chrom,
                "Position_1based": snp0 + 1,
                "Ref": ref, "Alt": alt,
                "cpg_id": f"{chrom}:{cpg_c}",
                "cpg_chr": chrom, "cpg_pos0": cpg_c,
                "distance_bp": offset,
                "abs_distance_bp": abs(offset),
                "asm_label": label,
                "observed_diff_alt_ref": signed,
                "observed_abs_diff": abs(signed),
                "dmr_hg38": f"{chrom}:{int(row.dmr_start_hg38)}-{int(row.dmr_end_hg38)}",
                "asm_tissues": str(row.asm_tissues),
                "mammary": 0,
                "fdr": float(row.fdr),
                "meth_ref_pct": float(row.meth_ref),
                "meth_alt_pct": float(row.meth_alt),
            })
        if found == 0:
            counters["snp_with_no_cpg"] += 1

    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs["mammary"] = pairs["asm_tissues"].str.contains("mammary", case=False,
                                                             na=False).astype(int)
    return pairs, counters


def match_negatives(pairs: pd.DataFrame, tolerance: int, seed: int) -> pd.DataFrame:
    """1:1 on |distance|, same variant preferred then same chromosome."""
    rng = np.random.default_rng(seed)
    positives = pairs[pairs["asm_label"] == 1]
    negatives = pairs[pairs["asm_label"] == 0]
    if positives.empty or negatives.empty:
        LOGGER.warning("cannot match: %d pos, %d neg -- returning positives only",
                       len(positives), len(negatives))
        return positives.reset_index(drop=True)

    by_variant: dict[tuple, list[int]] = {}
    by_chrom: dict[tuple, list[int]] = {}
    for idx, row in zip(negatives.index, negatives.itertuples(index=False)):
        by_variant.setdefault((row.Variant_ID, int(row.abs_distance_bp)), []).append(idx)
        by_chrom.setdefault((row.cpg_chr, int(row.abs_distance_bp)), []).append(idx)

    used: set[int] = set()
    keep_pos, keep_neg = [], []
    index = positives.index.to_numpy()
    for i in rng.permutation(len(positives)):
        idx = index[i]
        row = positives.loc[idx]
        target = int(row["abs_distance_bp"])
        pick = None
        for table, key in ((by_variant, row["Variant_ID"]), (by_chrom, row["cpg_chr"])):
            candidates = [j for delta in range(-tolerance, tolerance + 1)
                          for j in table.get((key, target + delta), ())
                          if j not in used]
            if candidates:
                pick = candidates[int(rng.integers(len(candidates)))]
                break
        if pick is None:
            continue
        used.add(pick)
        keep_pos.append(idx)
        keep_neg.append(pick)

    # E2 needs EVERY positive, not only the matched ones, so positives are kept
    # whole and a `matched_for_e1` flag marks the balanced E1 subset.
    out = pd.concat([positives, negatives.loc[keep_neg]]).reset_index(drop=True)
    matched = set(keep_pos) | set(keep_neg)
    out["matched_for_e1"] = 0
    positions = pd.concat([positives, negatives.loc[keep_neg]]).index.to_numpy()
    out.loc[[i for i, p in enumerate(positions) if p in matched], "matched_for_e1"] = 1
    return out


def extract_features(cpgs: pd.DataFrame, reference_dir: Path, phylop: Path,
                     imputation: dict) -> pd.DataFrame:
    paths = {
        "Ref_ATAC_Signal": reference_dir / "ATAC_seq.bw",
        "Ref_H3K4me3_Signal": reference_dir / "H3K4me3.bw",
        "Ref_H3K27ac_Signal": reference_dir / "H3K27ac.bw",
        "Ref_H3K27me3_Signal": reference_dir / "H3K27me3.bw",
        "Ref_H3K9me3_Signal": reference_dir / "H3K9me3.bw",
        "Ref_H3K36me3_Signal": reference_dir / "H3K36me3.bw",
        "Ref_H3K4me1_Signal": reference_dir / "H3K4me1.bw",
        PHYLOP_1: phylop, PHYLOP_2: phylop,
    }
    handles = {n: pyBigWig.open(str(p)) for n, p in paths.items()}
    values = {n: [] for n in paths}
    try:
        for row in cpgs.itertuples(index=False):
            chrom, pos = str(row.cpg_chr), int(row.cpg_pos0)
            for name, handle in handles.items():
                if name == PHYLOP_1:
                    v = _bw(handle, chrom, pos, pos + 1)
                elif name == PHYLOP_2:
                    v = _bw(handle, chrom, pos + 1, pos + 2)
                else:
                    v = _bw(handle, chrom, pos - 49, pos + 51)
                values[name].append(v)
    finally:
        for handle in handles.values():
            handle.close()
    out = cpgs.copy()
    for name in FEATURES:
        arr = np.asarray(values[name], dtype=np.float32)
        out[f"{name}_Missing"] = np.isnan(arr).astype(np.int8)
        out[name] = np.where(np.isnan(arr), np.float32(imputation[name]), arr).astype(np.float32)
    return out


def _bw(handle, chrom: str, start: int, end: int) -> float:
    try:
        chroms = handle.chroms()
        query = chrom if chrom in chroms else (
            chrom[3:] if chrom.startswith("chr") else f"chr{chrom}")
        if query not in chroms:
            return float("nan")
        s, e = max(0, int(start)), min(int(chroms[query]), int(end))
        if s >= e:
            return float("nan")
        stats = handle.stats(query, s, e, type="mean")
        return float(stats[0]) if stats and stats[0] is not None else float("nan")
    except (RuntimeError, ValueError):
        return float("nan")


def run(args: argparse.Namespace) -> int:
    table, counters = load_table(args.table, args.ensembl_cache)
    LOGGER.info("table: %s", counters)

    lifted, lift_counters = lift_dmrs(table, args.chain)
    LOGGER.info("DMR liftOver: %d kept, %s", len(lifted), lift_counters)

    LOGGER.info("reading genome")
    sequences = read_chromosomes(args.genome, set(TEST_CHROMS))
    cpg_positions = {c: [m.start() for m in re.finditer("CG", sequences[c])]
                     for c in TEST_CHROMS}

    dmr_index = merged_intervals(lifted)
    pairs, pair_counters = build_pairs(lifted, sequences, cpg_positions, dmr_index)
    LOGGER.info("pairs: %s", pair_counters)
    if pairs.empty:
        raise RuntimeError("no pairs built")

    matched = match_negatives(pairs, args.distance_tolerance_bp, args.seed)
    LOGGER.info("after matching: %d rows (%d pos, %d neg)", len(matched),
                int((matched.asm_label == 1).sum()), int((matched.asm_label == 0).sum()))

    imputation = json.loads(args.imputation.read_text())["values"]
    unique = matched[["cpg_id", "cpg_chr", "cpg_pos0"]].drop_duplicates("cpg_id").reset_index(drop=True)
    LOGGER.info("extracting context for %d CpGs", len(unique))
    records = extract_features(unique, args.reference_dir, args.phylop, imputation)
    records["Healthy_5000bp_DNA"] = [
        sequences[str(r.cpg_chr)][int(r.cpg_pos0) - CENTER_C_INDEX_FULL:
                                  int(r.cpg_pos0) - CENTER_C_INDEX_FULL + WINDOW_SIZE]
        for r in records.itertuples(index=False)]
    bad = sum(1 for s in records["Healthy_5000bp_DNA"]
              if len(s) != WINDOW_SIZE or s[CENTER_C_INDEX_FULL:CENTER_C_INDEX_FULL + 2] != "CG")
    if bad:
        raise RuntimeError(f"{bad} sequences not CpG-centred")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "tycko_scoring_pairs.csv"
    records_path = args.output_dir / "tycko_cpg_records.csv"
    atomic_csv(matched, pairs_path)
    atomic_csv(records, records_path)

    positives = matched[matched.asm_label == 1]
    snps = positives.drop_duplicates("Variant_ID")
    summary = {
        "analysis": "ASM scoring input from Do & Tycko 2020 (Tasks E1 + E2)",
        "analysis_status": "COMPLETE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Do C et al., Genome Biology 21:153 (2020), Additional file 3 Table S2",
        "held_out_by": "chromosome split (chr8+chr9), not array membership",
        "coordinate_builds": {
            "dmr": "hg19, lifted to hg38 here",
            "snp_position": "GRCh38 direct from Ensembl REST (cached)",
        },
        "table_counters": counters,
        "liftover_counters": lift_counters,
        "pair_counters": pair_counters,
        "distance_tolerance_bp": int(args.distance_tolerance_bp),
        "seed": int(args.seed),
        "rows": int(len(matched)),
        "positives": int(len(positives)),
        "negatives": int((matched.asm_label == 0).sum()),
        "unique_asm_snps_scored": int(len(snps)),
        "unique_asm_snps_mammary": int((snps.mammary == 1).sum()),
        "observed_sign_split": {
            "positive": int((snps.observed_diff_alt_ref > 0).sum()),
            "negative": int((snps.observed_diff_alt_ref < 0).sum()),
        },
        "unique_cpgs": int(len(records)),
        "sign_convention": (
            "observed_diff_alt_ref is ALT minus REF in percentage points, keyed to "
            "the hg38 reference base. Where hg38's reference matched their ALT the "
            "alleles were swapped AND the sign negated; that count is "
            "`alleles_swapped_sign_flipped`."
        ),
        "out_of_distribution_caveat": (
            "Trained only at HM450 positions, enriched at promoters and islands. "
            "Every CpG here is an arbitrary genomic CpG -- out of distribution with "
            "respect to training POSITIONS."
        ),
        "inputs": {p.as_posix(): sha256_file(p)
                   for p in (args.table, args.ensembl_cache, args.imputation)},
    }
    (args.output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("=" * 74)
    print("Tycko ASM scoring input (E1 + E2)")
    print("=" * 74)
    for k, v in counters.items():
        print(f"  {k:<34}{v:>8,}")
    for k, v in pair_counters.items():
        print(f"  {k:<34}{v:>8,}")
    print(f"  {'unique ASM SNPs scored':<34}{len(snps):>8,}")
    print(f"  {'... annotated mammary':<34}{int((snps.mammary == 1).sum()):>8,}")
    print(f"  {'... observed sign pos/neg':<34}"
          f"{int((snps.observed_diff_alt_ref > 0).sum()):>4,} /"
          f"{int((snps.observed_diff_alt_ref < 0).sum()):>4,}")
    print(f"  {'unique CpGs':<34}{len(records):>8,}")
    print("=" * 74)
    print(f"wrote {pairs_path}\nwrote {records_path}")
    return 0


def parse_args() -> argparse.Namespace:
    base = Path("data/external/asm_tycko_gb2020")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", type=Path,
                   default=base / "13059_2020_2059_MOESM3_ESM.xlsx")
    p.add_argument("--ensembl-cache", type=Path,
                   default=base / "ensembl_grch38_chr8_chr9_cache.csv")
    p.add_argument("--chain", type=Path,
                   default=Path("data/reference/hg19ToHg38.over.chain.gz"))
    p.add_argument("--genome", type=Path, default=Path("data/hg38.fa"))
    p.add_argument("--reference-dir", type=Path,
                   default=Path("data/reference/BreastEpithelium"))
    p.add_argument("--phylop", type=Path,
                   default=Path("data/reference/hg38.phyloP100way.bw"))
    p.add_argument("--imputation", type=Path,
                   default=Path("data/datafiles_breast_epithelium/feature_imputation.json"))
    p.add_argument("--distance-tolerance-bp", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=base / "scoring")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    sys.exit(run(parse_args()))
