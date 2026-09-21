#!/usr/bin/env python3
"""Test whether the model's variant signal runs through transcription-factor motifs. Scans
JASPAR motifs on both strands, picks the best position on the wild-type sequence and
evaluates the mutant at that same position, then asks whether disruption magnitude
predicts predicted methylation shift, whether per-factor coupling is signed and
interpretable, and whether mQTL discrimination concentrates among strong disruptors.
Backgrounds are matched on exact variant-CpG distance and local GC quintile, since both
drive the effect on their own; intervals are 1 Mb block bootstraps and per-factor tests
are Benjamini-Hochberg corrected.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/silentmethyl_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger("silentmethyl.motifs")

COLOURS = {"motif": "#B03A2E", "background": "#1F6FB2", "accent": "#B4761A"}
BLOCK_BP = 1_000_000
FLANK = 35
MAX_MOTIF_LENGTH = 30
FULL_TARGET_C_INDEX = 2499
BASES = "ACGT"
BASE_CODE = {b: i for i, b in enumerate(BASES)}

KNOWN_METHYL_SENSITIVE = {
    "CTCF", "NRF1", "CEBPB", "ZBTB33", "BANP", "E2F1", "E2F4", "SP1", "USF1",
    "USF2", "YY1", "EGR1", "ETS1", "ELK1", "MYC", "MAX", "NFYA", "NFYB", "KLF4",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores-dir", type=Path,
                   default=Path("results/journal/genoa_variant_scoring"))
    p.add_argument("--stratum", default="heldout", choices=("heldout", "model_visible"))
    p.add_argument("--model", default="fusion",
                   help="Model name as it appears in the scores directory layout "
                        "(fusion, sequence, kmer_ridge, composition, ...).")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--jaspar", type=Path,
                   default=Path("data/external/jaspar/"
                                "JASPAR_CORE_vertebrates_non-redundant_pfms_jaspar.txt"))
    p.add_argument("--split-template", default="data/datafiles/{split}.csv")
    p.add_argument("--relative-threshold", type=float, default=0.80,
                   help="Relative PWM score defining a motif occurrence.")
    p.add_argument("--contrast-quantile", type=float, default=0.25,
                   help="Q1 and Q3 contrast the top q against the bottom q of "
                        "max motif disruption. A quantile contrast is used instead "
                        "of a fixed threshold because with the full JASPAR library "
                        "essentially every variant falls inside SOME motif "
                        "occurrence, which leaves a binary in/out split with no "
                        "background to compare against.")
    p.add_argument("--significance", type=float, default=5e-8)
    p.add_argument("--min-covered", type=int, default=200,
                   help="Minimum variants inside a factor's motif to test it.")
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--match-tolerance", type=int, default=10)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0,
                   help="Smoke test: random sample of N pairs (not the head -- the "
                        "pair file is position-sorted, so a head slice falls inside "
                        "a handful of 1 Mb blocks and every interval comes back "
                        "empty).")
    p.add_argument("--max-motifs", type=int, default=0,
                   help="0 = all motifs. Positive values are smoke-test only.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/journal/motif_disruption"))
    return p.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def parse_jaspar(path: Path) -> list[dict]:
    """Parse the JASPAR raw PFM format into log-odds PWMs.

    Format:
        >MA0002.2  RUNX1
        A [ 287 234 ... ]
        C [ 496 485 ... ]
        G [ 696 ... ]
        T [ 521 ... ]
    """
    if not path.is_file():
        raise SystemExit(f"JASPAR file not found: {path}")
    motifs, current = [], None
    counts: dict[str, list[float]] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current and len(counts) == 4:
                motifs.append({**current, "counts": counts})
            parts = line[1:].split()
            current = {"matrix_id": parts[0],
                       "name": parts[1] if len(parts) > 1 else parts[0]}
            counts = {}
            continue
        if current is None:
            continue
        base = line[0].upper()
        if base not in BASE_CODE:
            continue
        inside = line[line.find("[") + 1: line.rfind("]")] if "[" in line else line[1:]
        try:
            counts[base] = [float(v) for v in inside.split()]
        except ValueError:
            continue
    if current and len(counts) == 4:
        motifs.append({**current, "counts": counts})

    prepared = []
    for motif in motifs:
        lengths = {len(v) for v in motif["counts"].values()}
        if len(lengths) != 1:
            continue
        length = lengths.pop()
        if not 4 <= length <= MAX_MOTIF_LENGTH:
            continue
        matrix = np.array([motif["counts"][b] for b in BASES], dtype=float).T
        totals = matrix.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            continue
        frequencies = (matrix + 0.25) / (totals + 1.0)
        pwm = np.log2(frequencies / 0.25).astype(np.float32)
        consensus = "".join(BASES[i] for i in frequencies.argmax(axis=1))
        prepared.append({
            "matrix_id": motif["matrix_id"],
            "name": motif["name"].upper(),
            "length": int(length),
            "consensus": consensus,
            "gc_content": float(frequencies[:, [1, 2]].sum(axis=1).mean()),
            "consensus_cpg_count": int(sum(
                consensus[i:i + 2] == "CG" for i in range(len(consensus) - 1))),
            "pwm": pwm,
            "pwm_rc": pwm[::-1, ::-1].copy(),
            "min_score": float(pwm.min(axis=1).sum()),
            "max_score": float(pwm.max(axis=1).sum()),
        })
    LOGGER.info("parsed %d usable motifs from %s", len(prepared), path.name)
    return prepared


def load_pair_scores(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for seed in args.seeds:
        path = (args.scores_dir / args.stratum / args.model / f"seed{seed}"
                / "pair_scores.csv")
        if not path.is_file():
            raise SystemExit(f"missing scores: {path}. Run script 19 first.")
        frames.append(pd.read_csv(path))
    long = pd.concat(frames, ignore_index=True)
    required = {"Pair_UID", "probeID", "Predicted_Delta_M", "distance_bp",
                "abs_distance_bp", "Ref", "Alt", "p_wald", "creates_cpg",
                "destroys_cpg", "cpg_chr", "cpg_pos_hg38"}
    absent = sorted(required - set(long.columns))
    if absent:
        raise SystemExit(f"score files are missing {absent}")

    ensemble = (long.groupby("Pair_UID", sort=False)["Predicted_Delta_M"]
                .mean().rename("Predicted_Delta_M").reset_index())
    metadata = (long[long["Seed"] == long["Seed"].min()]
                if "Seed" in long.columns else long.drop_duplicates("Pair_UID"))
    metadata = metadata.drop(columns=["Predicted_Delta_M"]).drop_duplicates("Pair_UID")
    pairs = metadata.merge(ensemble, on="Pair_UID", how="inner", validate="one_to_one")

    before = len(pairs)
    pairs = pairs[~(pairs["creates_cpg"].astype(bool)
                    | pairs["destroys_cpg"].astype(bool))].reset_index(drop=True)
    LOGGER.info("%d pairs, %d after removing CpG-altering variants", before, len(pairs))

    pairs["_block"] = (pairs["cpg_chr"].astype(str) + ":"
                       + (pairs["cpg_pos_hg38"] // BLOCK_BP).astype(int).astype(str))
    pairs["significant"] = (pairs["p_wald"] < args.significance).astype(int)
    return pairs


def extract_windows(pairs: pd.DataFrame, split_template: str) -> tuple:
    """Integer-encoded +/-FLANK bp of reference sequence centred on each variant."""
    probes = set(pairs["probeID"].astype(str))
    sequences: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = Path(split_template.format(split=split))
        if not path.is_file():
            raise FileNotFoundError(path)
        for chunk in pd.read_csv(path, usecols=["probeID", "Healthy_5000bp_DNA"],
                                 chunksize=20_000):
            chunk = chunk[chunk["probeID"].astype(str).isin(probes)]
            for probe, sequence in zip(chunk["probeID"].astype(str),
                                       chunk["Healthy_5000bp_DNA"]):
                sequences[probe] = str(sequence).upper()
    LOGGER.info("materialised %d reference windows", len(sequences))

    width = 2 * FLANK + 1
    encoded = np.full((len(pairs), width), 4, dtype=np.int8)
    alt_codes = np.full(len(pairs), 4, dtype=np.int8)
    keep = np.zeros(len(pairs), dtype=bool)
    gc = np.full(len(pairs), np.nan)

    for row_index, row in enumerate(pairs.itertuples(index=False)):
        sequence = sequences.get(str(row.probeID))
        if sequence is None or len(sequence) != 5000:
            continue
        centre = FULL_TARGET_C_INDEX + int(row.distance_bp)
        start, stop = centre - FLANK, centre + FLANK + 1
        if start < 0 or stop > len(sequence):
            continue
        window = sequence[start:stop]
        if window[FLANK] != str(row.Ref).upper():
            continue
        codes = np.frombuffer(window.encode("ascii"), dtype=np.uint8)
        mapped = np.full(width, 4, dtype=np.int8)
        for base, code in BASE_CODE.items():
            mapped[codes == ord(base)] = code
        if np.any(mapped == 4):
            continue
        encoded[row_index] = mapped
        alt_codes[row_index] = BASE_CODE.get(str(row.Alt).upper(), 4)
        gc[row_index] = float(np.mean((mapped == 1) | (mapped == 2)))
        keep[row_index] = alt_codes[row_index] != 4

    LOGGER.info("%d/%d pairs have a clean scan window", int(keep.sum()), len(pairs))
    return encoded[keep], alt_codes[keep], gc[keep], pairs[keep].reset_index(drop=True)


def scan_motif(wt: np.ndarray, mut: np.ndarray, motif: dict) -> tuple:
    """Best WT hit covering the variant, and the mutant score at that same site.

    Returns (relative_wt, delta_score) with one entry per pair. The mutant is
    evaluated at the WT-optimal (offset, strand) so that "disruption" cannot be
    an artifact of the motif relocating to a different site.
    """
    length = motif["length"]
    offsets = np.arange(FLANK - length + 1, FLANK + 1)
    offsets = offsets[(offsets >= 0) & (offsets + length <= wt.shape[1])]
    if len(offsets) == 0:
        return None, None

    columns = np.arange(length)
    best = np.full(wt.shape[0], -np.inf, dtype=np.float32)
    best_offset = np.zeros(wt.shape[0], dtype=np.int32)
    best_strand = np.zeros(wt.shape[0], dtype=np.int8)

    for strand, pwm in ((0, motif["pwm"]), (1, motif["pwm_rc"])):
        for offset in offsets:
            score = pwm[columns, wt[:, offset:offset + length]].sum(axis=1)
            better = score > best
            best = np.where(better, score, best)
            best_offset = np.where(better, offset, best_offset)
            best_strand = np.where(better, strand, best_strand)

    mutant = np.empty_like(best)
    for strand, pwm in ((0, motif["pwm"]), (1, motif["pwm_rc"])):
        mask = best_strand == strand
        if not mask.any():
            continue
        rows = np.flatnonzero(mask)
        window = np.take_along_axis(
            mut[rows], best_offset[rows, None] + columns[None, :], axis=1)
        mutant[rows] = pwm[columns, window].sum(axis=1)

    span = motif["max_score"] - motif["min_score"]
    relative = (best - motif["min_score"]) / span
    delta = (mutant - best) / span
    return relative.astype(np.float32), delta.astype(np.float32)


def block_bootstrap(frame: pd.DataFrame, metric, n_boot: int,
                    rng: np.random.Generator) -> tuple[float, float]:
    if frame.empty:
        return (np.nan, np.nan)
    unique, inverse = np.unique(frame["_block"].to_numpy(), return_inverse=True)
    if len(unique) < 5:
        return (np.nan, np.nan)
    positions = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    values = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(unique), len(unique))
        index = np.concatenate([positions[p] for p in picks])
        value = metric(frame.iloc[index])
        if np.isfinite(value):
            values.append(value)
    if len(values) < max(20, n_boot // 10):
        return (np.nan, np.nan)
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def partial_spearman(x, y, controls: pd.DataFrame) -> float:
    """Spearman correlation of x and y after removing linear effects of controls.

    Everything is rank-transformed first, then x and y are residualised on the
    ranked controls by least squares. Without this, a factor whose motif simply
    happens to sit in GC-rich, CpG-island-like sequence would appear to have a
    factor-specific coupling that is really a property of the neighbourhood.
    """
    if len(x) < 30:
        return np.nan
    rank = lambda v: pd.Series(v).rank().to_numpy(dtype=float)  # noqa: E731
    design = np.column_stack([np.ones(len(x))] + [rank(controls[c]) for c in controls])
    try:
        residual = lambda v: rank(v) - design @ np.linalg.lstsq(  # noqa: E731
            design, rank(v), rcond=None)[0]
        rx, ry = residual(x), residual(y)
    except np.linalg.LinAlgError:
        return np.nan
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(p_values)
    out = np.full(len(p_values), np.nan)
    if not finite.any():
        return out
    values = p_values[finite]
    order = np.argsort(values)
    ranked = values[order]
    n = len(ranked)
    adjusted = np.minimum.accumulate((ranked * n / np.arange(n, 0, -1))[::-1])[::-1]
    restored = np.empty(n)
    restored[order] = np.clip(adjusted, 0, 1)
    out[finite] = restored
    return out


def matched_background(frame: pd.DataFrame, positive: np.ndarray,
                       gc_quintile: np.ndarray, tolerance: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Indices of background pairs matched to positives on distance and GC.

    Motif occurrences are not uniformly distributed with respect to either
    variable -- motifs cluster in GC-rich sequence, and GC-rich sequence sits
    closer to CpG islands -- so an unmatched comparison would confound motif
    membership with both.
    """
    distances = frame["abs_distance_bp"].to_numpy(dtype=int)
    pool: dict[tuple[int, int], list[int]] = {}
    for index in np.flatnonzero(~positive):
        pool.setdefault((distances[index], int(gc_quintile[index])), []).append(index)
    for value in pool.values():
        rng.shuffle(value)

    chosen: list[int] = []
    for index in np.flatnonzero(positive):
        quintile = int(gc_quintile[index])
        distance = distances[index]
        for offset in range(0, tolerance + 1):
            candidates = ({distance} if offset == 0
                          else {distance - offset, distance + offset})
            taken = False
            for candidate in candidates:
                bucket = pool.get((candidate, quintile))
                if bucket:
                    chosen.append(bucket.pop())
                    taken = True
                    break
            if taken:
                break
    return np.array(chosen, dtype=int)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if MAX_MOTIF_LENGTH > FLANK + 1:
        raise SystemExit("MAX_MOTIF_LENGTH must not exceed FLANK + 1")
    rng = np.random.default_rng(args.random_seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    motifs = parse_jaspar(args.jaspar)
    if args.max_motifs > 0:
        motifs = motifs[:args.max_motifs]
        LOGGER.warning("SMOKE TEST: limited to %d motifs", len(motifs))

    pairs = load_pair_scores(args)
    if args.limit > 0:
        pairs = (pairs.sample(n=min(args.limit, len(pairs)),
                              random_state=args.random_seed)
                 .sort_index().reset_index(drop=True))
        LOGGER.warning("SMOKE TEST: random sample of %d pairs", len(pairs))
    wt, alt_codes, gc, pairs = extract_windows(pairs, args.split_template)
    if len(pairs) == 0:
        raise SystemExit("no pair produced a clean scan window")

    mut = wt.copy()
    mut[:, FLANK] = alt_codes
    pairs["gc_content"] = gc
    pairs["gc_quintile"] = pd.qcut(pairs["gc_content"], 5, labels=False,
                                   duplicates="drop")
    pairs["abs_delta_m"] = pairs["Predicted_Delta_M"].abs()
    quintiles = pairs["gc_quintile"].to_numpy()

    complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
    reference = pairs["Ref"].astype(str).str.upper()
    alternate = pairs["Alt"].astype(str).str.upper()
    flip = reference.isin(["A", "G"])
    canonical_ref = np.where(flip, reference.map(complement), reference)
    canonical_alt = np.where(flip, alternate.map(complement), alternate)
    substitution = pd.Series([f"{r}>{a}" for r, a in zip(canonical_ref, canonical_alt)],
                             index=pairs.index)
    substitution_dummies = pd.get_dummies(substitution, prefix="sub",
                                          drop_first=True).astype(float)
    pairs = pd.concat([pairs, substitution_dummies], axis=1)
    substitution_columns = list(substitution_dummies.columns)
    LOGGER.info("substitution classes: %s",
                dict(substitution.value_counts().head(6)))

    LOGGER.info("scanning %d motifs over %d pairs", len(motifs), len(pairs))
    hit_any = np.zeros(len(pairs), dtype=bool)
    covered_sets: dict[str, np.ndarray] = {}
    best_disruption = np.zeros(len(pairs), dtype=np.float32)
    top_factor = np.array([""] * len(pairs), dtype=object)
    per_motif_rows = []

    for position, motif in enumerate(motifs, 1):
        relative, delta_score = scan_motif(wt, mut, motif)
        if relative is None:
            continue
        covered = relative >= args.relative_threshold
        n_covered = int(covered.sum())
        hit_any |= covered

        magnitude = np.where(covered, np.abs(delta_score), 0.0)
        improved = magnitude > best_disruption
        best_disruption = np.where(improved, magnitude, best_disruption)
        top_factor[improved] = motif["name"]

        if n_covered < args.min_covered:
            continue

        control_columns = ["gc_content", "abs_distance_bp"] + substitution_columns
        subset = pairs.loc[covered, ["Predicted_Delta_M", "_block"]
                           + control_columns].copy()
        subset["_ds"] = delta_score[covered]
        result = spearmanr(subset["_ds"], subset["Predicted_Delta_M"])
        coupling = float(result.statistic)
        partial = partial_spearman(subset["_ds"], subset["Predicted_Delta_M"],
                                   subset[["gc_content", "abs_distance_bp"]])
        partial_substitution = partial_spearman(
            subset["_ds"], subset["Predicted_Delta_M"], subset[control_columns])
        covered_sets[motif["name"]] = covered.copy()
        low, high = block_bootstrap(
            subset,
            lambda f: float(spearmanr(f["_ds"], f["Predicted_Delta_M"]).statistic),
            args.n_boot, rng)
        per_motif_rows.append({
            "matrix_id": motif["matrix_id"],
            "factor": motif["name"],
            "known_methylation_sensitive": motif["name"] in KNOWN_METHYL_SENSITIVE,
            "motif_length": motif["length"],
            "n_covered": n_covered,
            "n_blocks": int(subset["_block"].nunique()),
            "median_abs_delta_m_covered": float(
                pairs.loc[covered, "abs_delta_m"].median()),
            "motif_gc_content": motif["gc_content"],
            "motif_consensus": motif["consensus"],
            "motif_consensus_cpg_count": motif["consensus_cpg_count"],
            "coupling_spearman": coupling,
            "coupling_partial_gc_distance": partial,
            "coupling_partial_gc_distance_substitution": partial_substitution,
            "coupling_ci_low": low,
            "coupling_ci_high": high,
            "coupling_p_raw": float(result.pvalue),
        })
        if position % 100 == 0:
            LOGGER.info("  %d/%d motifs scanned", position, len(motifs))

    per_motif = pd.DataFrame(per_motif_rows)
    if not per_motif.empty:
        per_motif["coupling_q_bh"] = benjamini_hochberg(
            per_motif["coupling_p_raw"].to_numpy())
        per_motif = per_motif.sort_values(
            "coupling_spearman", key=lambda s: -s.abs()).reset_index(drop=True)
    atomic_csv(per_motif, out / "per_motif_coupling.csv")
    LOGGER.info("%d motifs met the coverage threshold", len(per_motif))

    null_summary = {}
    if not per_motif.empty:
        values = per_motif["coupling_spearman"].to_numpy(dtype=float)
        partials = per_motif["coupling_partial_gc_distance"].to_numpy(dtype=float)
        gc_values = per_motif["motif_gc_content"].to_numpy(dtype=float)
        finite = np.isfinite(values) & np.isfinite(gc_values)
        null_summary = {
            "motifs_tested": int(len(per_motif)),
            "coupling_median": float(np.nanmedian(values)),
            "coupling_iqr": [float(np.nanpercentile(values, 25)),
                             float(np.nanpercentile(values, 75))],
            "coupling_median_partial": float(np.nanmedian(partials)),
            "fraction_negative": float(np.nanmean(values < 0)),
            "significant_negative_q05": int(((per_motif["coupling_q_bh"] < 0.05)
                                             & (values < 0)).sum()),
            "significant_positive_q05": int(((per_motif["coupling_q_bh"] < 0.05)
                                             & (values > 0)).sum()),
            "spearman_motif_gc_vs_coupling": (
                float(spearmanr(gc_values[finite], values[finite]).statistic)
                if finite.sum() > 10 else np.nan),
            "known_methylation_sensitive_tested": int(
                per_motif["known_methylation_sensitive"].sum()),
            "known_methylation_sensitive_median_rank": (
                int(per_motif.index[per_motif["known_methylation_sensitive"]]
                    .to_series().median()) + 1
                if per_motif["known_methylation_sensitive"].any() else None),
        }
        top_names = per_motif.head(15)["factor"].tolist()
        overlaps, rows_overlap = [], []
        for i, first in enumerate(top_names):
            for second in top_names[i + 1:]:
                a, b = covered_sets.get(first), covered_sets.get(second)
                if a is None or b is None:
                    continue
                union = int((a | b).sum())
                jaccard = float((a & b).sum() / union) if union else np.nan
                overlaps.append(jaccard)
                rows_overlap.append({"factor_a": first, "factor_b": second,
                                     "jaccard": jaccard})
        if rows_overlap:
            atomic_csv(pd.DataFrame(rows_overlap), out / "top_motif_overlap.csv")
        null_summary["top15_mean_pairwise_jaccard"] = (
            float(np.nanmean(overlaps)) if overlaps else np.nan)
        null_summary["top15_max_pairwise_jaccard"] = (
            float(np.nanmax(overlaps)) if overlaps else np.nan)
        atomic_json(null_summary, out / "coupling_null_summary.json")

    pairs["in_motif"] = hit_any
    pairs["max_motif_disruption"] = best_disruption
    pairs["top_disrupted_factor"] = top_factor
    LOGGER.info("%d/%d pairs (%.1f%%) fall inside at least one motif occurrence at "
                "relative score >= %.2f", int(hit_any.sum()), len(pairs),
                100 * hit_any.mean(), args.relative_threshold)

    covered_values = best_disruption[hit_any]
    if covered_values.size < 200:
        raise SystemExit(
            f"only {covered_values.size} pairs fall inside any motif occurrence. "
            f"Lower --relative-threshold or check the JASPAR file; the contrast "
            f"cannot be formed.")
    low_cut = float(np.quantile(covered_values, args.contrast_quantile))
    high_cut = float(np.quantile(covered_values, 1 - args.contrast_quantile))
    if high_cut <= low_cut:
        raise SystemExit(
            f"disruption cut points collapsed (low={low_cut:.4f}, high={high_cut:.4f}): "
            f"the score change is too concentrated to form a contrast. Raise "
            f"--relative-threshold so only confident occurrences are scored.")
    strong = hit_any & (best_disruption >= high_cut)
    weak = hit_any & (best_disruption <= low_cut)
    pairs["disruption_group"] = np.where(strong, "strong",
                                         np.where(weak, "weak", "middle"))
    LOGGER.info("disruption contrast over %d covered pairs: strong >= %.4f (n=%d), "
                "weak <= %.4f (n=%d)", int(hit_any.sum()), high_cut,
                int(strong.sum()), low_cut, int(weak.sum()))

    continuous = pairs.loc[hit_any, ["max_motif_disruption", "abs_delta_m", "_block"]]
    coupling_point = float(spearmanr(continuous["max_motif_disruption"],
                                     continuous["abs_delta_m"]).statistic)
    coupling_low, coupling_high = block_bootstrap(
        continuous,
        lambda f: float(spearmanr(f["max_motif_disruption"], f["abs_delta_m"]).statistic),
        args.n_boot, rng)

    weak_pool = pairs.loc[weak]
    background = matched_background(
        pd.concat([pairs.loc[strong], weak_pool]).reset_index(drop=True),
        np.concatenate([np.ones(int(strong.sum()), bool),
                        np.zeros(len(weak_pool), bool)]),
        np.concatenate([quintiles[strong], quintiles[weak]]),
        args.match_tolerance, rng)
    combined = pd.concat([pairs.loc[strong], weak_pool]).reset_index(drop=True)
    if len(background) < 0.8 * int(strong.sum()):
        LOGGER.warning("only %d of %d strong-disruption variants could be matched to "
                       "a weak-disruption counterpart on distance and GC",
                       len(background), int(strong.sum()))
    covered_frame = pairs.loc[strong]
    background_frame = combined.loc[background]
    global_rows = [{
        "group": "continuous_coupling", "n": int(hit_any.sum()),
        "median_abs_delta_m": coupling_point,
        "ci_low": coupling_low, "ci_high": coupling_high,
        "median_abs_distance_bp": np.nan, "mean_gc": np.nan,
    }]
    for label, frame in (("strong_disruption", covered_frame),
                         ("matched_weak_disruption", background_frame)):
        median = lambda f: float(f["abs_delta_m"].median())  # noqa: E731
        low, high = block_bootstrap(frame, median, args.n_boot, rng)
        global_rows.append({
            "group": label, "n": int(len(frame)),
            "median_abs_delta_m": median(frame) if len(frame) else np.nan,
            "ci_low": low, "ci_high": high,
            "median_abs_distance_bp": float(frame["abs_distance_bp"].median())
            if len(frame) else np.nan,
            "mean_gc": float(frame["gc_content"].mean()) if len(frame) else np.nan,
        })
    atomic_csv(pd.DataFrame(global_rows), out / "motif_vs_background.csv")

    discrimination_rows = []

    def discrimination(frame: pd.DataFrame) -> float:
        labels = frame["significant"].to_numpy(dtype=int)
        if len(np.unique(labels)) < 2 or labels.sum() < 20:
            return np.nan
        return float(roc_auc_score(labels, frame["abs_delta_m"]))

    for label, frame in (("strong_disruption", covered_frame),
                         ("weak_disruption", background_frame)):
        low, high = block_bootstrap(frame, discrimination, args.n_boot, rng)
        discrimination_rows.append({
            "group": label, "n": int(len(frame)),
            "n_significant": int(frame["significant"].sum()),
            "auroc": discrimination(frame), "ci_low": low, "ci_high": high,
        })
    atomic_csv(pd.DataFrame(discrimination_rows),
               out / "meqtl_discrimination_by_motif_status.csv")

    make_figure(per_motif, pd.DataFrame(global_rows),
                pd.DataFrame(discrimination_rows), out)

    atomic_json(
        {
            "analysis": "TF motif disruption and SilentMethyl variant effects",
            "purpose": "regulatory enrichment: do variant effects run through TF motifs?",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stratum": args.stratum,
            "model": args.model,
            "seeds": args.seeds,
            "motif_source": str(args.jaspar),
            "motifs_scanned": len(motifs),
            "motifs_tested": int(len(per_motif)),
            "relative_score_threshold": args.relative_threshold,
            "pairs_scanned": int(len(pairs)),
            "pairs_in_any_motif": int(hit_any.sum()),
            "contrast_quantile": args.contrast_quantile,
            "disruption_cut_strong": high_cut,
            "disruption_cut_weak": low_cut,
            "n_strong": int(strong.sum()),
            "n_weak": int(weak.sum()),
            "coupling_null_summary": null_summary,
            "continuous_coupling_spearman": coupling_point,
            "continuous_coupling_ci": [coupling_low, coupling_high],
            "n_boot": args.n_boot,
            "block_size_bp": BLOCK_BP,
            "matching": {"variables": ["exact bp distance", "GC quintile"],
                         "tolerance_bp": args.match_tolerance,
                         "replacement": False},
            "exclusions": ("CpG-creating and CpG-destroying variants are removed: "
                           "their methylation direction is near-deterministic and "
                           "would masquerade as a motif effect"),
            "strand": "both; best WT hit chosen, mutant scored at the same site",
            "interpretation_guide": {
                "Q1": ("does the SIZE of motif disruption predict the size of the "
                       "predicted methylation shift? Reported two ways: a continuous "
                       "rank correlation over all pairs, and a top-vs-bottom quantile "
                       "contrast matched on distance and GC. A binary inside/outside "
                       "split is NOT used: with the full JASPAR library essentially "
                       "every variant falls inside some occurrence, leaving no "
                       "background. Confirmatory either way, not novel."),
                "Q2": ("per-factor coupling between motif score change and predicted "
                       "methylation change; the SIGN is the mechanistic claim"),
                "Q3": ("higher meQTL discrimination inside motifs than outside would "
                       "mean the model's requirement-6 signal is specifically "
                       "TF-binding disruption. A flat result is informative and must "
                       "be reported as such."),
            },
            "caveat": ("PWM occurrence is a sequence-match prediction, not measured "
                       "binding. Without matched ChIP-seq these are candidate sites."),
        },
        out / "run_summary.json",
    )

    print_report(per_motif, pd.DataFrame(global_rows),
                 pd.DataFrame(discrimination_rows), pairs, out, null_summary)
    return 0


def make_figure(per_motif: pd.DataFrame, global_frame: pd.DataFrame,
                discrimination: pd.DataFrame, out: Path) -> None:
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    axis = axes[0]
    if not per_motif.empty:
        top = per_motif.head(18).iloc[::-1]
        y = np.arange(len(top))
        colours = [COLOURS["accent"] if flag else COLOURS["background"]
                   for flag in top["known_methylation_sensitive"]]
        axis.barh(y, top["coupling_spearman"],
                  xerr=[top["coupling_spearman"] - top["coupling_ci_low"],
                        top["coupling_ci_high"] - top["coupling_spearman"]],
                  color=colours, alpha=0.9, capsize=2.5, height=0.72)
        axis.set_yticks(y)
        axis.set_yticklabels(
            [f"{f}{'*' if k else ''}" for f, k in
             zip(top["factor"], top["known_methylation_sensitive"])], fontsize=8)
        axis.axvline(0, color="0.45", linewidth=0.9, linestyle=(0, (1, 2)))
        axis.set_xlabel(r"Spearman($\Delta$ motif score, $\Delta \hat{M}$)", fontsize=9)
        axis.set_title("Per-factor coupling\n(* = known methylation-sensitive)",
                       fontsize=10)
        axis.grid(axis="x", alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)

    axis = axes[1]
    if not discrimination.empty:
        groups = discrimination["group"].tolist()
        x = np.arange(len(groups))
        values = discrimination["auroc"].to_numpy(dtype=float)
        errors = [values - discrimination["ci_low"].to_numpy(dtype=float),
                  discrimination["ci_high"].to_numpy(dtype=float) - values]
        axis.bar(x, values, 0.5, yerr=errors, capsize=4,
                 color=[COLOURS["motif"], COLOURS["background"]],
                 edgecolor="white", linewidth=0.8)
        axis.axhline(0.5, color="0.45", linewidth=0.9, linestyle=(0, (1, 2)))
        axis.set_xticks(x, [g.replace("_", " ") for g in groups], fontsize=9)
        axis.set_ylim(0.45, max(0.70, float(np.nanmax(values)) + 0.05))
        axis.set_ylabel("AUROC, real meQTL vs null", fontsize=9)
        axis.set_title("Is the variant signal concentrated\nin strong motif disruptions?",
                       fontsize=10)
        axis.grid(axis="y", alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(plots / "motif_disruption.png", dpi=400, bbox_inches="tight",
                facecolor="white")
    fig.savefig(plots / "motif_disruption.pdf", bbox_inches="tight")
    plt.close(fig)


def print_report(per_motif: pd.DataFrame, global_frame: pd.DataFrame,
                 discrimination: pd.DataFrame, pairs: pd.DataFrame, out: Path,
                 null_summary: dict) -> None:
    print()
    print("=" * 78)
    print("TF motif disruption and predicted variant effects")
    print("=" * 78)
    print(f"\npairs scanned            : {len(pairs):,}")
    print(f"pairs inside any motif   : {int(pairs['in_motif'].sum()):,} "
          f"({pairs['in_motif'].mean():.1%})")
    print("\nQ1  does the SIZE of motif disruption predict the size of the shift?")
    for _, row in global_frame.iterrows():
        if row["group"] == "continuous_coupling":
            print(f"  continuous  spearman(max disruption, |dM|) over all "
                  f"{row['n']:,} pairs: {row['median_abs_delta_m']:+.4f} "
                  f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]")
            continue
        print(f"  {row['group']:<24} n={row['n']:>7,}  median |dM| "
              f"{row['median_abs_delta_m']:.4f} "
              f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]  "
              f"median dist {row['median_abs_distance_bp']:.0f} bp  "
              f"GC {row['mean_gc']:.3f}")
    print("  (matched medians for distance and GC confirm the matching worked)")

    print("\nQ2  strongest per-factor coupling (* = known methylation-sensitive)")
    if per_motif.empty:
        print("  no factor met the coverage threshold")
    else:
        print(f"  {'factor':<14}{'n':>8}{'rho':>10}{'+GC/dist':>10}{'+subst':>10}"
              f"{'GC':>7}{'q(BH)':>10}")
        for _, row in per_motif.head(15).iterrows():
            star = "*" if row["known_methylation_sensitive"] else " "
            def fmt(value):
                return f"{value:>+10.4f}" if np.isfinite(value) else f"{'n/a':>10}"
            print(f"  {row['factor'][:12]:<12}{star} {row['n_covered']:>7,}"
                  f"{row['coupling_spearman']:>+10.4f}"
                  f"{fmt(row['coupling_partial_gc_distance'])}"
                  f"{fmt(row['coupling_partial_gc_distance_substitution'])}"
                  f"{row['motif_gc_content']:>7.2f}"
                  f"{row['coupling_q_bh']:>10.2e}")
        print("  +GC/dist controls local GC and variant-CpG distance; "
              "+subst adds substitution class")
        known = per_motif[per_motif["known_methylation_sensitive"]]
        if not known.empty:
            print(f"\n  known methylation-sensitive factors tested: {len(known)}; "
                  f"median rank {int(known.index.to_series().median()) + 1} "
                  f"of {len(per_motif)}")
        if null_summary:
            print("\n  IS ANY FACTOR ACTUALLY SPECIAL? (read this before believing "
                  "the table above)")
            print(f"    median coupling across all {null_summary['motifs_tested']} "
                  f"motifs   {null_summary['coupling_median']:+.4f}  "
                  f"IQR [{null_summary['coupling_iqr'][0]:+.4f}, "
                  f"{null_summary['coupling_iqr'][1]:+.4f}]")
            print(f"    median after controlling GC and distance     "
                  f"{null_summary['coupling_median_partial']:+.4f}")
            print(f"    motifs with negative coupling                "
                  f"{null_summary['fraction_negative']:.1%}")
            print(f"    significant at q<0.05: {null_summary['significant_negative_q05']} "
                  f"negative, {null_summary['significant_positive_q05']} positive")
            print(f"    spearman(motif GC, coupling)                 "
                  f"{null_summary['spearman_motif_gc_vs_coupling']:+.4f}")
            print(f"    top-15 mean pairwise Jaccard overlap         "
                  f"{null_summary.get('top15_mean_pairwise_jaccard', float('nan')):.3f}"
                  f"   (high = one motif family, not N factors)")
            print("    -> a median far from zero means a GLOBAL offset, not a")
            print("       factor-specific effect. A strong GC correlation means the")
            print("       result is about sequence composition, not the factor.")

    print("\nQ3  meQTL discrimination, strongly vs weakly motif-disrupting variants")
    for _, row in discrimination.iterrows():
        print(f"  {row['group']:<16} n={row['n']:>7,}  n_sig={row['n_significant']:>6,}"
              f"  AUROC {row['auroc']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
    if len(discrimination) == 2 and discrimination["auroc"].notna().all():
        inside, outside = discrimination["auroc"].iloc[0], discrimination["auroc"].iloc[1]
        overlap = not (discrimination["ci_low"].iloc[0] > discrimination["ci_high"].iloc[1]
                       or discrimination["ci_low"].iloc[1] > discrimination["ci_high"].iloc[0])
        verdict = ("intervals overlap -- report as no detected concentration"
                   if overlap else "intervals separate -- a real concentration")
        print(f"\n  difference {inside - outside:+.4f}: {verdict}")

    print(f"\noutput: {out}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
