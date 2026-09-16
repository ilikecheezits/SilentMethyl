#!/usr/bin/env python3
"""
Build the ASM validation scoring input (Task E1). CPU only, no GPU, no model.

What this is
------------
Slot 53, reserved for allele-specific methylation since the R2 framework was
written. Stage 1 of two: turn the published ASM catalogue into SNV-CpG pairs the
frozen checkpoints can score, plus the context features every scored CpG needs.

Source
------
Rosenski et al., "Atlas of imprinted and allele-specific DNA methylation in the
human body", Nat Commun 16:2141 (2025), Supplementary Table `S3. ASM SNPs`:
55,271 ASM SNP loci with their bimodal region, Fisher exact p per sample, and the
sample names each call fired in. Underlying WGBS is GEO GSE186458 (Loyfer atlas).
See data/external/asm_atlas_natcommun2025/SOURCE.txt.

Why this script exists instead of 20_variant_scoring.py
--------------------------------------------------------
Script 20 reads its target CpGs out of the prebuilt split CSVs and then requires
every probeID to be present in the HM450 manifest. Both assumptions are exactly
what this analysis is built to escape: ASM CpGs are arbitrary genomic CpGs, not
array probes. Nothing in the model requires an array probe -- sequence, the seven
context tracks and phyloP are all genome-wide, and held-out-ness is preserved by
the CHROMOSOME split. So the target CpGs here are built from scratch, using the
same conventions build_training_data.py uses, and script 20 is left untouched.

Conventions replicated exactly from data/build_training_data.py
----------------------------------------------------------------
    pos                      0-based coordinate of the C in the target CpG
    Healthy_5000bp_DNA       genome[chr][pos-2499 : pos+2501], seq[2499:2501]=="CG"
    Ref_<mark>_Signal        bigWig mean over [pos-49, pos+51)   (100 bp)
    Target_Base_PhyloP_1     phyloP over [pos,   pos+1)          (the C)
    Target_Base_PhyloP_2     phyloP over [pos+1, pos+2)          (the G)
    imputation               train-split medians from
                             data/datafiles_breast_epithelium/feature_imputation.json

Window arithmetic, which is easy to get wrong
----------------------------------------------
The model window is 1,000 bp taken as seq[2000:3000], so the target C sits at
index 499 and a variant at signed offset `d` from the C lands at 499 + d. The
variant is therefore scoreable only for **d in [-499, +500]** -- an asymmetric
+/-500 bp window, NOT +/-1000. Offsets 0 and 1 are the target CpG itself and are
excluded (PROTECTED_CPG_INDICES).

Positives and two tiers of negative
------------------------------------
positive            CpG inside the ASM SNP's own ASM region
negative/background CpG inside no bimodal region at all
negative/bimodal    CpG inside a BIMODAL region that is not an ASM region

Using the same variant on both sides is the tightest available control: variant
identity, allele, MAF, local sequence and the whole 1,000-bp window are shared,
and only the CpG's status differs. Matching on |offset| then removes the distance
confound, the one the mQTL work showed matters most.

**Why the second tier exists, and why it is the one to believe.** ASM regions are
a subset of the atlas's bimodal methylation regions -- CpG-dense, intermediate-
methylation, enhancer-like. Against plain background CpGs the model could
separate them merely by recognising that regional character, with no
allele-specific information involved, and a reviewer will say so. The
bimodal-but-not-ASM tier holds that character fixed and asks the sharper
question: among regions that all look like this, does the model pick out the ones
where methylation is actually allele-specific? Report both; lead with the
bimodal contrast.

Usage
-----
    python -u scripts/53_asm_build.py --output-dir data/external/asm_atlas/scoring
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

LOGGER = logging.getLogger("asm_build")

WINDOW_SIZE = 5000
CENTER_C_INDEX_FULL = 2499
MODEL_WINDOW_SIZE = 1000
CENTER_C_INDEX = 499          # index of the target C inside the 1,000-bp window
PROTECTED_OFFSETS = frozenset({0, 1})
MIN_OFFSET, MAX_OFFSET = -CENTER_C_INDEX, MODEL_WINDOW_SIZE - CENTER_C_INDEX - 1
TEST_CHROMS = ("chr8", "chr9")

TRACKS = [
    "Ref_ATAC_Signal",
    "Ref_H3K4me3_Signal",
    "Ref_H3K27ac_Signal",
    "Ref_H3K27me3_Signal",
    "Ref_H3K9me3_Signal",
    "Ref_H3K36me3_Signal",
    "Ref_H3K4me1_Signal",
]
PHYLOP_1 = "Target_Base_PhyloP_100way_1"
PHYLOP_2 = "Target_Base_PhyloP_100way_2"
FEATURES = TRACKS + [PHYLOP_1, PHYLOP_2]
COMPLEMENT = str.maketrans("ACGT", "TGCA")


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


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_asm_table(path: Path) -> pd.DataFrame:
    """Sheet `S3. ASM SNPs`. Row 0 is a title, row 1 the real header."""
    frame = pd.read_excel(path, sheet_name="S3. ASM SNPs", header=2)
    frame.columns = [
        "chrom", "snp_pos_hg19", "rsid", "alleles",
        "asm_region_hg19", "adj_p", "fisher_p", "samples",
    ]
    frame = frame[frame["chrom"].astype(str).str.match(r"^chr[\dXY]+$")].copy()
    region = frame["asm_region_hg19"].astype(str).str.extract(
        r"^(chr[\dXY]+):(\d+)-(\d+)$"
    )
    frame["region_chrom"] = region[0]
    frame["region_start_hg19"] = pd.to_numeric(region[1], errors="coerce")
    frame["region_end_hg19"] = pd.to_numeric(region[2], errors="coerce")
    frame["snp_pos_hg19"] = pd.to_numeric(frame["snp_pos_hg19"], errors="coerce")
    frame = frame.dropna(
        subset=["snp_pos_hg19", "region_start_hg19", "region_end_hg19"]
    )
    mismatched = int((frame["region_chrom"] != frame["chrom"]).sum())
    if mismatched:
        raise ValueError(f"{mismatched} rows have region on a different chromosome")
    return frame.reset_index(drop=True)


def lift_frame(frame: pd.DataFrame, chain: Path) -> tuple[pd.DataFrame, dict]:
    """hg19 -> hg38 for the SNP and both region bounds; drop anything unclean."""
    lifter = LiftOver(str(chain))
    counters = {"snp_unmapped": 0, "region_unmapped": 0, "chrom_changed": 0,
                "inverted": 0}

    def convert(chrom: str, position: float) -> int | None:
        hits = lifter.convert_coordinate(chrom, int(position))
        if not hits:
            return None
        new_chrom, new_pos = hits[0][0], hits[0][1]
        return None if new_chrom != chrom else int(new_pos)

    snp, start, end, keep = [], [], [], []
    for row in frame.itertuples(index=False):
        s = convert(row.chrom, row.snp_pos_hg19)
        a = convert(row.chrom, row.region_start_hg19)
        b = convert(row.chrom, row.region_end_hg19)
        if s is None:
            counters["snp_unmapped"] += 1
            keep.append(False); snp.append(-1); start.append(-1); end.append(-1); continue
        if a is None or b is None:
            counters["region_unmapped"] += 1
            keep.append(False); snp.append(-1); start.append(-1); end.append(-1); continue
        if b < a:
            counters["inverted"] += 1
            a, b = b, a
        snp.append(s); start.append(a); end.append(b); keep.append(True)

    out = frame.copy()
    out["snp_pos_hg38"] = snp
    out["region_start_hg38"] = start
    out["region_end_hg38"] = end
    out = out[keep].reset_index(drop=True)
    return out, counters


def read_chromosome(fasta: Path, wanted: set[str]) -> dict[str, str]:
    """Plain FASTA reader; hg38.fa is line-wrapped and pyfaidx is not needed here."""
    sequences: dict[str, str] = {}
    current, buffer = None, []
    with fasta.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if current in wanted:
                    sequences[current] = "".join(buffer).upper()
                current = line[1:].split()[0]
                buffer = []
                continue
            if current in wanted:
                buffer.append(line.strip())
    if current in wanted and current not in sequences:
        sequences[current] = "".join(buffer).upper()
    missing = wanted - set(sequences)
    if missing:
        raise ValueError(f"{fasta} is missing {sorted(missing)}")
    return sequences


def cpg_index(sequence: str) -> list[int]:
    return [m.start() for m in re.finditer("CG", sequence)]


def resolve_alleles(sequence: str, position: int, alleles: str) -> tuple[str, str] | None:
    """Key the atlas's unordered allele pair to hg38 REF->ALT.

    The table stores e.g. 'G/A' without saying which is reference, and the call
    may have been made on either strand. Try the pair as given, then its
    complement; accept only if exactly one member matches the hg38 base.
    """
    parts = [p.strip().upper() for p in str(alleles).split("/")]
    if len(parts) != 2 or any(p not in {"A", "C", "G", "T"} for p in parts):
        return None
    reference = sequence[position]
    for candidate in (parts, [p.translate(COMPLEMENT) for p in parts]):
        if candidate[0] == candidate[1]:
            continue
        if candidate[0] == reference:
            return reference, candidate[1]
        if candidate[1] == reference:
            return reference, candidate[0]
    return None


def build_pairs(asm: pd.DataFrame, sequences: dict[str, str],
                cpgs: dict[str, list[int]],
                region_index: dict[str, tuple[list[int], list[int]]],
                bimodal_index: dict[str, tuple[list[int], list[int]]]) -> tuple[pd.DataFrame, dict]:
    counters = {
        "allele_unresolved": 0, "snp_out_of_bounds": 0, "cpg_out_of_bounds": 0,
        "sequence_not_cpg_centred": 0, "protected_offset": 0,
        "positive": 0, "negative_bimodal": 0, "negative_background": 0,
        "other_asm_region_dropped": 0,
    }

    def _hit(index: dict[str, tuple[list[int], list[int]]], chrom: str, position: int) -> bool:
        starts, ends = index.get(chrom, ([], []))
        i = bisect.bisect_right(starts, position) - 1
        return i >= 0 and position <= ends[i]

    def in_any_region(chrom: str, position: int) -> bool:
        return _hit(region_index, chrom, position)

    def in_bimodal(chrom: str, position: int) -> bool:
        return _hit(bimodal_index, chrom, position)

    rows = []
    for row in asm.itertuples(index=False):
        chrom = str(row.chrom)
        sequence = sequences[chrom]
        snp = int(row.snp_pos_hg38)
        if not 0 <= snp < len(sequence):
            counters["snp_out_of_bounds"] += 1
            continue
        resolved = resolve_alleles(sequence, snp, row.alleles)
        if resolved is None:
            counters["allele_unresolved"] += 1
            continue
        ref, alt = resolved

        # Candidate CpGs whose 1,000-bp model window would contain this variant.
        # variant offset d = snp - cpg_c, and d must lie in [-499, +500], so the
        # CpG's C lies in [snp - 500, snp + 499].
        positions = cpgs[chrom]
        lo = bisect.bisect_left(positions, snp - MAX_OFFSET)
        hi = bisect.bisect_right(positions, snp - MIN_OFFSET)
        for cpg_c in positions[lo:hi]:
            offset = snp - cpg_c
            if offset in PROTECTED_OFFSETS:
                counters["protected_offset"] += 1
                continue
            start = cpg_c - CENTER_C_INDEX_FULL
            if start < 0 or start + WINDOW_SIZE > len(sequence):
                counters["cpg_out_of_bounds"] += 1
                continue
            if sequence[cpg_c:cpg_c + 2] != "CG":
                counters["sequence_not_cpg_centred"] += 1
                continue

            inside_own = (int(row.region_start_hg38) <= cpg_c <= int(row.region_end_hg38))
            if inside_own:
                label, tier = 1, "positive"
                counters["positive"] += 1
            elif in_any_region(chrom, cpg_c):
                # Inside some OTHER ASM region: neither a clean positive for this
                # variant nor a clean negative. Dropped rather than guessed at.
                counters["other_asm_region_dropped"] += 1
                continue
            elif in_bimodal(chrom, cpg_c):
                label, tier = 0, "bimodal_non_asm"
                counters["negative_bimodal"] += 1
            else:
                label, tier = 0, "background"
                counters["negative_background"] += 1

            rows.append({
                "Variant_ID": f"{chrom}_{snp + 1}_{ref}_{alt}_b38",
                "chr": chrom,
                "Position_1based": snp + 1,
                "Position_0based": snp,
                "Ref": ref,
                "Alt": alt,
                "rsid": str(row.rsid),
                "cpg_id": f"{chrom}:{cpg_c}",
                "cpg_chr": chrom,
                "cpg_pos0": cpg_c,
                "distance_bp": offset,
                "abs_distance_bp": abs(offset),
                "asm_label": label,
                "tier": tier,
                "asm_region_hg38": f"{chrom}:{int(row.region_start_hg38)}-{int(row.region_end_hg38)}",
                "samples": str(row.samples),
                "breast": int("Breast" in str(row.samples)),
                "min_fisher_p": _min_p(row.fisher_p),
                "min_adj_p": _min_p(row.adj_p),
            })
    return pd.DataFrame(rows), counters


def _min_p(value) -> float:
    try:
        return float(min(float(v) for v in str(value).split(",") if v.strip()))
    except (ValueError, TypeError):
        return float("nan")


def build_region_index(asm_all: pd.DataFrame) -> dict[str, tuple[list[int], list[int]]]:
    """Merged ASM region spans per chromosome, for the negative-set exclusion."""
    index: dict[str, tuple[list[int], list[int]]] = {}
    for chrom, group in asm_all.groupby("chrom"):
        spans = sorted(
            (int(a), int(b))
            for a, b in zip(group["region_start_hg38"], group["region_end_hg38"])
        )
        starts, ends = [], []
        for a, b in spans:
            if starts and a <= ends[-1] + 1:
                ends[-1] = max(ends[-1], b)
            else:
                starts.append(a); ends.append(b)
        index[chrom] = (starts, ends)
    return index


def load_bed_index(path: Path, chroms: tuple[str, ...]) -> dict[str, tuple[list[int], list[int]]]:
    """Merged intervals per chromosome from a 3-column BED."""
    spans: dict[str, list[tuple[int, int]]] = {c: [] for c in chroms}
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split("\t")
            if len(fields) < 3 or fields[0] not in spans:
                continue
            spans[fields[0]].append((int(fields[1]), int(fields[2])))
    index: dict[str, tuple[list[int], list[int]]] = {}
    for chrom, items in spans.items():
        items.sort()
        starts, ends = [], []
        for a, b in items:
            if starts and a <= ends[-1] + 1:
                ends[-1] = max(ends[-1], b)
            else:
                starts.append(a); ends.append(b)
        index[chrom] = (starts, ends)
    return index


def extract_features(cpg_table: pd.DataFrame, reference_dir: Path,
                     phylop: Path, imputation: dict) -> pd.DataFrame:
    paths = {
        "Ref_ATAC_Signal": reference_dir / "ATAC_seq.bw",
        "Ref_H3K4me3_Signal": reference_dir / "H3K4me3.bw",
        "Ref_H3K27ac_Signal": reference_dir / "H3K27ac.bw",
        "Ref_H3K27me3_Signal": reference_dir / "H3K27me3.bw",
        "Ref_H3K9me3_Signal": reference_dir / "H3K9me3.bw",
        "Ref_H3K36me3_Signal": reference_dir / "H3K36me3.bw",
        "Ref_H3K4me1_Signal": reference_dir / "H3K4me1.bw",
        PHYLOP_1: phylop,
        PHYLOP_2: phylop,
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing BigWig files:\n" + "\n".join(sorted(set(missing))))

    handles = {name: pyBigWig.open(str(path)) for name, path in paths.items()}
    values: dict[str, list[float]] = {name: [] for name in paths}
    try:
        for row in cpg_table.itertuples(index=False):
            chrom, pos = str(row.cpg_chr), int(row.cpg_pos0)
            for name, handle in handles.items():
                if name == PHYLOP_1:
                    value = _bw(handle, chrom, pos, pos + 1)
                elif name == PHYLOP_2:
                    value = _bw(handle, chrom, pos + 1, pos + 2)
                else:
                    value = _bw(handle, chrom, pos - 49, pos + 51)
                values[name].append(value)
    finally:
        for handle in handles.values():
            handle.close()

    out = cpg_table.copy()
    for name in FEATURES:
        arr = np.asarray(values[name], dtype=np.float32)
        out[f"{name}_Missing"] = np.isnan(arr).astype(np.int8)
        out[name] = np.where(np.isnan(arr), np.float32(imputation[name]), arr).astype(np.float32)
    return out


def _bw(handle, chrom: str, start: int, end: int) -> float:
    """Mean over [start, end), matching build_training_data.get_bw_signal."""
    try:
        chroms = handle.chroms()
        query = chrom if chrom in chroms else (
            chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        )
        if query not in chroms:
            return float("nan")
        length = int(chroms[query])
        s, e = max(0, int(start)), min(length, int(end))
        if s >= e:
            return float("nan")
        stats = handle.stats(query, s, e, type="mean")
        if not stats or stats[0] is None:
            return float("nan")
        return float(stats[0])
    except (RuntimeError, ValueError):
        return float("nan")


def match_negatives(pairs: pd.DataFrame, tolerance: int, seed: int) -> pd.DataFrame:
    """1:1 distance matching on |variant-to-CpG distance|.

    Same protocol as the GENOA/eGTEx work: a hard tolerance on |offset|, so the
    distance-only baseline is pinned near 0.5 by construction and the headline
    AUROC cannot be distance leaking back in.

    Matching is greedy with a two-tier preference -- a negative from the SAME
    variant first, then any negative on the same chromosome. Same-variant
    matches hold variant identity, allele, MAF and most of the 1,000-bp window
    fixed, so only the CpG's ASM status differs. Each negative is consumed at
    most once, and **positives that find no partner are dropped**: keeping them
    would reintroduce the imbalance the matching exists to remove.
    """
    rng = np.random.default_rng(seed)
    positives = pairs[pairs["asm_label"] == 1]
    negatives = pairs[pairs["asm_label"] == 0]
    if positives.empty or negatives.empty:
        raise RuntimeError(
            f"cannot match: {len(positives)} positives, {len(negatives)} negatives")

    # Bucket negatives by |distance| so a candidate lookup is a small scan over
    # the tolerance band rather than a pass over the whole pool.
    by_variant: dict[tuple[str, int], list[int]] = {}
    by_chrom: dict[tuple[str, int], list[int]] = {}
    for idx, row in zip(negatives.index, negatives.itertuples(index=False)):
        by_variant.setdefault((row.Variant_ID, int(row.abs_distance_bp)), []).append(idx)
        by_chrom.setdefault((row.cpg_chr, int(row.abs_distance_bp)), []).append(idx)

    used: set[int] = set()
    keep_positive: list[int] = []
    keep_negative: list[int] = []

    order = rng.permutation(len(positives))
    positive_index = positives.index.to_numpy()
    for i in order:
        idx = positive_index[i]
        row = positives.loc[idx]
        target = int(row["abs_distance_bp"])
        pick = None
        for table, key_prefix in ((by_variant, row["Variant_ID"]), (by_chrom, row["cpg_chr"])):
            candidates = []
            for delta in range(-tolerance, tolerance + 1):
                candidates.extend(
                    j for j in table.get((key_prefix, target + delta), ())
                    if j not in used
                )
            if candidates:
                pick = candidates[int(rng.integers(len(candidates)))]
                break
        if pick is None:
            continue
        used.add(pick)
        keep_positive.append(idx)
        keep_negative.append(pick)

    matched = pd.concat([
        pairs.loc[keep_positive],
        pairs.loc[keep_negative],
    ]).reset_index(drop=True)
    return matched


def match_all_tiers(pairs: pd.DataFrame, tolerance: int, seed: int) -> pd.DataFrame:
    """One independent 1:1 matched cohort per negative tier.

    The positive pool is shared, so a positive pair can appear in both contrasts.
    They are separate analyses and are tagged by `contrast`; never pool them.
    """
    frames = []
    for offset, tier in enumerate(("bimodal_non_asm", "background")):
        subset = pairs[(pairs["asm_label"] == 1) | (pairs["tier"] == tier)]
        if int((subset["asm_label"] == 0).sum()) == 0:
            LOGGER.warning("tier %s has no negatives; contrast skipped", tier)
            continue
        cohort = match_negatives(subset, tolerance, seed + 17 * offset)
        cohort = cohort.copy()
        cohort["contrast"] = f"positive_vs_{tier}"
        LOGGER.info("contrast positive_vs_%s: %d positives, %d negatives",
                    tier,
                    int((cohort["asm_label"] == 1).sum()),
                    int((cohort["asm_label"] == 0).sum()))
        frames.append(cohort)
    if not frames:
        raise RuntimeError("no contrast could be matched")
    return pd.concat(frames, ignore_index=True)


def run(args: argparse.Namespace) -> int:
    LOGGER.info("reading ASM table: %s", args.asm_table)
    asm = load_asm_table(args.asm_table)
    LOGGER.info("  %d ASM SNP loci, all chromosomes", len(asm))

    lifted_all, lift_counters = lift_frame(asm, args.chain)
    LOGGER.info("  lifted hg19->hg38: %d (%.1f%%), %s",
                len(lifted_all), 100 * len(lifted_all) / len(asm), lift_counters)

    region_index = build_region_index(lifted_all)
    selected = lifted_all[lifted_all["chrom"].isin(TEST_CHROMS)].reset_index(drop=True)
    LOGGER.info("  %s: %d ASM SNP loci", "+".join(TEST_CHROMS), len(selected))

    LOGGER.info("reading genome: %s", args.genome)
    sequences = read_chromosome(args.genome, set(TEST_CHROMS))
    cpgs = {c: cpg_index(sequences[c]) for c in TEST_CHROMS}
    for c in TEST_CHROMS:
        LOGGER.info("  %s: %d bp, %d CpGs", c, len(sequences[c]), len(cpgs[c]))

    bimodal_index = load_bed_index(args.bimodal_bed, TEST_CHROMS)
    LOGGER.info("bimodal regions: %s",
                {c: len(bimodal_index[c][0]) for c in TEST_CHROMS})
    pairs, counters = build_pairs(selected, sequences, cpgs, region_index, bimodal_index)
    LOGGER.info("pair construction: %s", counters)
    if pairs.empty:
        raise RuntimeError("no scoreable pairs built")

    imputation = json.loads(args.imputation.read_text())["values"]
    absent = [f for f in FEATURES if f not in imputation]
    if absent:
        raise ValueError(f"{args.imputation} lacks imputation values for {absent}")

    matched = match_all_tiers(pairs, args.distance_tolerance_bp, args.seed)
    LOGGER.info("after distance matching: %d rows across %d contrasts",
                len(matched), matched["contrast"].nunique())

    cpg_table = (
        matched[["cpg_id", "cpg_chr", "cpg_pos0"]]
        .drop_duplicates("cpg_id")
        .reset_index(drop=True)
    )
    LOGGER.info("extracting context for %d unique CpGs", len(cpg_table))
    features = extract_features(cpg_table, args.reference_dir, args.phylop, imputation)

    sequences_out = []
    for row in features.itertuples(index=False):
        start = int(row.cpg_pos0) - CENTER_C_INDEX_FULL
        sequences_out.append(sequences[str(row.cpg_chr)][start:start + WINDOW_SIZE])
    features["Healthy_5000bp_DNA"] = sequences_out
    bad = [s for s in sequences_out
           if len(s) != WINDOW_SIZE or s[CENTER_C_INDEX_FULL:CENTER_C_INDEX_FULL + 2] != "CG"]
    if bad:
        raise RuntimeError(f"{len(bad)} built sequences are not CpG-centred")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = args.output_dir / "asm_scoring_pairs.csv"
    probes_path = args.output_dir / "asm_cpg_records.csv"
    atomic_csv(matched, pairs_path)
    atomic_csv(features, probes_path)

    breast = matched[matched["breast"] == 1]
    summary = {
        "analysis": "ASM validation (Task E1) scoring input",
        "analysis_status": "COMPLETE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": (
            "Rosenski et al. Nat Commun 16:2141 (2025), Supplementary Table "
            "S3 'ASM SNPs'; underlying WGBS GEO GSE186458"
        ),
        "test_chromosomes": list(TEST_CHROMS),
        "held_out_by": "chromosome split (chr8+chr9), NOT array membership",
        "model_window_offsets": [MIN_OFFSET, MAX_OFFSET],
        "protected_offsets": sorted(PROTECTED_OFFSETS),
        "distance_tolerance_bp": int(args.distance_tolerance_bp),
        "seed": int(args.seed),
        "liftover_counters": lift_counters,
        "pair_counters": counters,
        "asm_snp_loci_all_chroms": int(len(asm)),
        "asm_snp_loci_lifted": int(len(lifted_all)),
        "asm_snp_loci_test_chroms": int(len(selected)),
        "pairs_before_matching": int(len(pairs)),
        "pairs_after_matching": int(len(matched)),
        "contrasts": {
            str(name): {
                "positives": int((g["asm_label"] == 1).sum()),
                "negatives": int((g["asm_label"] == 0).sum()),
                "breast_pairs": int((g["breast"] == 1).sum()),
            }
            for name, g in matched.groupby("contrast")
        },
        "negative_tier_note": (
            "positive_vs_bimodal_non_asm is the controlled contrast and the one "
            "to lead with: it holds the bimodal/enhancer-like regional character "
            "fixed, so separation cannot come from recognising region class "
            "alone. positive_vs_background is the weaker, less specific test."
        ),
        "breast_pairs": int(len(breast)),
        "breast_positives": int((breast["asm_label"] == 1).sum()),
        "unique_cpgs": int(len(features)),
        "unique_variants": int(matched["Variant_ID"].nunique()),
        "out_of_distribution_caveat": (
            "The model was trained only at HM450 probe positions, which are "
            "enriched at promoters and CpG islands. Every CpG scored here is an "
            "arbitrary genomic CpG, so this is out of distribution with respect "
            "to training POSITIONS -- distinct from being out of distribution in "
            "sequence or in chromatin. State this as a limitation and report the "
            "island-context breakdown alongside the headline."
        ),
        "inputs": {
            p.as_posix(): sha256_file(p)
            for p in (args.asm_table, args.imputation)
        },
        "outputs": {
            pairs_path.as_posix(): sha256_file(pairs_path),
            probes_path.as_posix(): sha256_file(probes_path),
        },
    }
    atomic_json(summary, args.output_dir / "build_summary.json")

    print("=" * 78)
    print("ASM validation scoring input (Task E1)")
    print("=" * 78)
    print(f"  ASM SNP loci, all chromosomes      {len(asm):>10,}")
    print(f"  lifted hg19->hg38                  {len(lifted_all):>10,}")
    print(f"  on {'+'.join(TEST_CHROMS):<31}{len(selected):>10,}")
    print(f"  SNV-CpG pairs built                {len(pairs):>10,}")
    print(f"    positives (CpG in ASM region)    {counters['positive']:>10,}")
    print(f"    negatives, bimodal non-ASM       {counters['negative_bimodal']:>10,}")
    print(f"    negatives, background            {counters['negative_background']:>10,}")
    print(f"  after distance matching            {len(matched):>10,}")
    for name, g in matched.groupby("contrast"):
        print(f"    {str(name):<32}{int((g['asm_label']==1).sum()):>6,} pos / "
              f"{int((g['asm_label']==0).sum()):,} neg")
    print(f"  breast-epithelium pairs            {len(breast):>10,}")
    print(f"  unique CpGs needing context        {len(features):>10,}")
    print(f"  unique variants                    {matched['Variant_ID'].nunique():>10,}")
    print("=" * 78)
    print(f"wrote {pairs_path}\nwrote {probes_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asm-table", type=Path,
                   default=Path("data/external/asm_atlas_natcommun2025/"
                                "41467_2025_57433_MOESM4_ESM.xlsx"))
    p.add_argument("--chain", type=Path,
                   default=Path("data/reference/hg19ToHg38.over.chain.gz"))
    p.add_argument("--genome", type=Path, default=Path("data/hg38.fa"))
    p.add_argument("--reference-dir", type=Path,
                   default=Path("data/reference/BreastEpithelium"))
    p.add_argument("--phylop", type=Path,
                   default=Path("data/reference/hg38.phyloP100way.bw"))
    p.add_argument("--imputation", type=Path,
                   default=Path("data/datafiles_breast_epithelium/feature_imputation.json"))
    p.add_argument("--bimodal-bed", type=Path,
                   default=Path("data/external/asm_atlas/bimodal_chr8_chr9.bed"),
                   help="Merged bimodal methylation regions from Supplementary "
                        "Data 1, used for the stronger negative tier.")
    p.add_argument("--distance-tolerance-bp", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/external/asm_atlas/scoring"))
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    sys.exit(run(parse_args()))
