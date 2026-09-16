#!/usr/bin/env python3
"""
Evaluate the ASM validation against Do & Tycko 2020 (Tasks E1 + E2). CPU only.

Answers both questions from one catalogue
------------------------------------------
E2, the mentor's three named statistics, all on the SIGNED effect:

    direction concordance   fraction of ASM SNPs where the predicted delta and
                            the observed ALT-minus-REF difference share a sign,
                            with a binomial interval against the 0.5 null
    signed Spearman         rank correlation of predicted vs observed, signed
    AUROC pos vs neg        can the predicted delta separate SNPs with a POSITIVE
                            observed allelic difference from those with a
                            NEGATIVE one -- literally "AUROC for positive versus
                            negative methylation effects"

E1, discrimination, on the distance-matched subset:

    AUROC |delta|           ASM CpGs vs distance-matched CpGs outside every DMR

Unit of analysis for E2
-----------------------
The published effect is per SNP, averaged over the CpGs in its ASM DMR. So the
prediction is aggregated the same way: the mean predicted delta over that SNP's
scored DMR CpGs. Comparing a single CpG's prediction against a DMR-wide
measurement would be a unit mismatch.

Scale
-----
Observed effects are percentage points of methylation; predicted deltas are on
the model's M / beta scale. Magnitudes are NOT comparable and no claim of
magnitude calibration is made -- exactly as for GENOA/eGTEx. Rank, sign and
AUROC are the meaningful comparisons, which is why they are the ones reported.

Usage
-----
    python -u scripts/53e_tycko_evaluate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger("tycko_eval")
PRED = "Predicted_Delta_M"
PRED_ABS = "Absolute_Delta_M"
OBS = "observed_diff_alt_ref"


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


def block_codes(chrom: np.ndarray, position: np.ndarray, size: int):
    keys = pd.Series([f"{c}:{p // size}" for c, p in zip(chrom, position)])
    codes, unique = pd.factorize(keys, sort=True)
    return codes, len(unique)


def block_bootstrap(stat_fn, codes: np.ndarray, n_blocks: int,
                    replicates: int, seed: int) -> tuple[float, float, float]:
    """Percentile interval over 1 Mb blocks, plus P(stat <= null) where given."""
    rng = np.random.default_rng(seed)
    draws = np.full(replicates, np.nan)
    for i in range(replicates):
        multiplicity = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks))
        weight = multiplicity[codes].astype(float)
        keep = weight > 0
        if not keep.any():
            continue
        try:
            draws[i] = stat_fn(keep, weight[keep])
        except (ValueError, ZeroDivisionError):
            continue
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan"), finite
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975)), finite


def evaluate_snp_level(snps: pd.DataFrame, label: str, args, offset: int) -> list[dict]:
    """E2: direction concordance, signed Spearman, AUROC positive vs negative."""
    rows: list[dict] = []
    for arm, block in snps.groupby("Model"):
        block = block[np.isfinite(block[PRED]) & np.isfinite(block[OBS])]
        if len(block) < 10 or block[OBS].abs().min() == block[OBS].abs().max():
            LOGGER.warning("%s / %s: too few usable SNPs (%d)", label, arm, len(block))
            continue
        predicted = block[PRED].to_numpy(dtype=float)
        observed = block[OBS].to_numpy(dtype=float)
        codes, n_blocks = block_codes(block["cpg_chr"].to_numpy(),
                                      block["snp_pos0"].to_numpy(dtype=np.int64),
                                      args.block_size_bp)

        concordance = float(np.mean(np.sign(predicted) == np.sign(observed)))
        rho = float(spearmanr(predicted, observed).statistic)
        positive = (observed > 0).astype(int)
        auroc = (float(roc_auc_score(positive, predicted))
                 if 0 < positive.sum() < len(positive) else float("nan"))

        def c_fn(keep, w):
            return float(np.average(
                (np.sign(predicted[keep]) == np.sign(observed[keep])).astype(float),
                weights=w))

        def r_fn(keep, w):
            return float(spearmanr(predicted[keep], observed[keep]).statistic)

        def a_fn(keep, w):
            y = positive[keep]
            if np.unique(y).size < 2:
                raise ValueError("one class")
            return float(roc_auc_score(y, predicted[keep], sample_weight=w))

        c_lo, c_hi, c_draws = block_bootstrap(c_fn, codes, n_blocks,
                                              args.bootstrap_replicates,
                                              args.random_seed + offset)
        r_lo, r_hi, r_draws = block_bootstrap(r_fn, codes, n_blocks,
                                              args.bootstrap_replicates,
                                              args.random_seed + offset + 1)
        a_lo, a_hi, a_draws = block_bootstrap(a_fn, codes, n_blocks,
                                              args.bootstrap_replicates,
                                              args.random_seed + offset + 2)
        rows.append({
            "Stratum": label, "Arm": arm, "N_SNPs": int(len(block)),
            "N_Observed_Positive": int(positive.sum()),
            "Direction_Concordance": concordance,
            "Direction_CI_Low": c_lo, "Direction_CI_High": c_hi,
            "Direction_P_LE_Half": float(np.mean(c_draws <= 0.5)) if len(c_draws) else float("nan"),
            "Signed_Spearman": rho,
            "Spearman_CI_Low": r_lo, "Spearman_CI_High": r_hi,
            "Spearman_P_LE_Zero": float(np.mean(r_draws <= 0)) if len(r_draws) else float("nan"),
            # NOT case-vs-control: every SNP here is an ASM SNP. The two classes
            # are the SIGN of the measured ALT-REF difference, so this ranks
            # alleles within ASM sites. E1's AUROC_Detection ranks sites.
            "AUROC_Directional": auroc,
            "AUROC_Directional_CI_Low": a_lo, "AUROC_Directional_CI_High": a_hi,
            "AUROC_Directional_P_LE_Half": float(np.mean(a_draws <= 0.5)) if len(a_draws) else float("nan"),
            "N_Genomic_Blocks": int(n_blocks),
        })
    return rows


def evaluate_discrimination(pairs: pd.DataFrame, label: str, args,
                            offset: int) -> list[dict]:
    """E1: |delta| separating ASM CpGs from distance-matched non-DMR CpGs."""
    rows: list[dict] = []
    scores = {"distance_only_baseline": lambda b: -b["abs_distance_bp"].to_numpy(float)}
    for arm in sorted(pairs["Model"].unique()):
        scores[arm] = (lambda b, a=arm: b[PRED_ABS].to_numpy(float))
    for name, getter in scores.items():
        block = pairs[pairs["Model"] == sorted(pairs["Model"].unique())[0]] \
            if name == "distance_only_baseline" else pairs[pairs["Model"] == name]
        y = block["asm_label"].to_numpy(dtype=int)
        if np.unique(y).size < 2:
            continue
        value = getter(block)
        codes, n_blocks = block_codes(block["cpg_chr"].to_numpy(),
                                      block["cpg_pos0"].to_numpy(dtype=np.int64),
                                      args.block_size_bp)

        def fn(keep, w):
            if np.unique(y[keep]).size < 2:
                raise ValueError("one class")
            return float(roc_auc_score(y[keep], value[keep], sample_weight=w))

        lo, hi, draws = block_bootstrap(fn, codes, n_blocks,
                                        args.bootstrap_replicates,
                                        args.random_seed + offset + 5)
        rows.append({
            "Stratum": label, "Score": name, "N_Pairs": int(len(block)),
            "N_Positive": int(y.sum()),
            "AUROC_Detection": float(roc_auc_score(y, value)),
            "CI_Low": lo, "CI_High": hi,
            "P_LE_Half": float(np.mean(draws <= 0.5)) if len(draws) else float("nan"),
            "N_Genomic_Blocks": int(n_blocks),
        })
    return rows


def run(args: argparse.Namespace) -> int:
    scores = pd.read_csv(args.scores_csv)
    LOGGER.info("scored rows: %d, arms %s", len(scores), sorted(scores["Model"].unique()))

    scores["snp_pos0"] = scores["Position_1based"].astype(np.int64) - 1
    positives = scores[scores["asm_label"] == 1].copy()

    # E2 unit: one row per SNP per arm, prediction averaged over its DMR CpGs,
    # matching how the published effect was averaged over the DMR.
    snps = (positives.groupby(["Model", "Variant_ID"], as_index=False)
            .agg(**{
                PRED: (PRED, "mean"),
                OBS: (OBS, "first"),
                "cpg_chr": ("cpg_chr", "first"),
                "snp_pos0": ("snp_pos0", "first"),
                "mammary": ("mammary", "first"),
                "n_cpgs": ("cpg_id", "size"),
                "observed_abs_diff": ("observed_abs_diff", "first"),
            }))
    LOGGER.info("E2 units: %d SNP x arm rows, median %.1f CpGs per SNP",
                len(snps), snps["n_cpgs"].median())

    e2_rows = evaluate_snp_level(snps, "all tissues", args, 0)
    mammary = snps[snps["mammary"] == 1]
    if len(mammary) >= 20:
        e2_rows += evaluate_snp_level(mammary, "mammary", args, 100)
    else:
        LOGGER.warning("mammary stratum too small (%d SNPs); skipped",
                       len(mammary) // max(1, snps["Model"].nunique()))
    strong = snps[snps["observed_abs_diff"] >= args.strong_effect_pp]
    if len(strong) >= 20:
        e2_rows += evaluate_snp_level(
            strong, f"|effect| >= {args.strong_effect_pp}pp", args, 200)
    e2 = pd.DataFrame(e2_rows)

    e1_rows = []
    matched = scores[scores.get("matched_for_e1", 1) == 1] if "matched_for_e1" in scores else scores
    if matched["asm_label"].nunique() > 1:
        e1_rows = evaluate_discrimination(matched, "all tissues", args, 0)
    e1 = pd.DataFrame(e1_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    e2_path = args.output_dir / "tycko_e2_signed_agreement.csv"
    atomic_csv(e2, e2_path)
    if not e1.empty:
        atomic_csv(e1, args.output_dir / "tycko_e1_discrimination.csv")

    baseline = e1[e1["Score"] == "distance_only_baseline"]["AUROC_Detection"] if not e1.empty else []
    baseline_value = float(baseline.iloc[0]) if len(baseline) else float("nan")

    summary = {
        "analysis": "ASM validation against Do & Tycko 2020 (E1 + E2)",
        "analysis_status": "COMPLETE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Do C et al., Genome Biology 21:153 (2020), Table S2",
        "e2_unit": "one ASM index SNP; prediction = mean predicted delta over the "
                   "CpGs scored inside that SNP's ASM DMR, matching how the "
                   "published effect was averaged over the DMR",
        "auroc_note": (
            "TWO DIFFERENT AUROCs, and they answer different questions. "
            "E2's AUROC_Directional contains NO non-ASM entities: every SNP in "
            "it is an ASM SNP and the two classes are the SIGN of the measured "
            "ALT-REF difference, so it asks which ALLELE carries the "
            "methylation. It is a continuous-margin restatement of direction "
            "concordance and is NOT independent of it. E1's AUROC_Detection "
            "separates ASM CpGs from distance-matched non-DMR CpGs and asks "
            "which SITE is allele-specifically methylated. Do not compare the "
            "two numbers to each other, and do not read either column name as "
            "case-versus-control."),
        "scale_note": (
            "Observed effects are percentage points of methylation; predicted "
            "deltas are on the model's M scale. Magnitudes are NOT comparable and "
            "no magnitude calibration is claimed. Rank, sign and AUROC are the "
            "meaningful comparisons."
        ),
        "block_size_bp": int(args.block_size_bp),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "strong_effect_threshold_pp": float(args.strong_effect_pp),
        "e2_results": e2.to_dict(orient="records"),
        "e1_results": e1.to_dict(orient="records") if not e1.empty else [],
        "e1_distance_only_baseline": baseline_value,
        "out_of_distribution_caveat": (
            "Trained only at HM450 positions, enriched at promoters and islands; "
            "every CpG scored here is an arbitrary genomic CpG."
        ),
        "inputs": {args.scores_csv.as_posix(): sha256_file(args.scores_csv)},
    }
    (args.output_dir / "tycko_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("=" * 96)
    print("E2 -- signed agreement with observed allelic methylation (Do & Tycko 2020)")
    print("=" * 96)
    print(f"  {'stratum':<22}{'arm':<10}{'n':>6}{'direction':>11}{'95% CI':>18}"
          f"{'sSpearman':>11}{'AUROCdir':>10}")
    for _, r in e2.iterrows():
        print(f"  {r['Stratum']:<22}{r['Arm']:<10}{int(r['N_SNPs']):>6}"
              f"{r['Direction_Concordance']:>11.4f}"
              f"   [{r['Direction_CI_Low']:.3f}, {r['Direction_CI_High']:.3f}]"
              f"{r['Signed_Spearman']:>11.4f}{r['AUROC_Directional']:>10.4f}")
    if not e1.empty:
        print("\n" + "=" * 96)
        print("E1 -- discrimination, ASM CpGs vs distance-matched non-DMR CpGs")
        print("=" * 96)
        for _, r in e1.iterrows():
            print(f"  {r['Stratum']:<22}{r['Score']:<26}{int(r['N_Pairs']):>7,}"
                  f"{r['AUROC_Detection']:>9.4f}   [{r['CI_Low']:.3f}, {r['CI_High']:.3f}]")
        print(f"\n  distance-only baseline = {baseline_value:.4f}")
    print("=" * 96)
    print(f"wrote {e2_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores-csv", type=Path,
                   default=Path("results/journal/asm_validation_tycko/tycko_pair_scores.csv"))
    p.add_argument("--block-size-bp", type=int, default=1_000_000)
    p.add_argument("--bootstrap-replicates", type=int, default=2_000)
    p.add_argument("--random-seed", type=int, default=20260915)
    p.add_argument("--strong-effect-pp", type=float, default=20.0)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/journal/asm_validation_tycko"))
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    sys.exit(run(parse_args()))
