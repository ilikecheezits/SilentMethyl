#!/usr/bin/env python3
"""
Evaluate the ASM validation (Task E1). CPU only, seconds to run, rerunnable.

The question
------------
Does SilentMethyl assign larger predicted variant effects to SNV-CpG pairs that
actually show allele-specific methylation than to distance-matched pairs that do
not? The model has never seen an ASM measurement, and every CpG scored here is an
arbitrary genomic CpG rather than an array probe, so this is a genuinely external
test of whether the learned sequence->methylation response transfers.

What is and is not being measured
----------------------------------
The published atlas tables give significance and sample membership but **no
signed allelic methylation difference and no per-allele beta**. So:

    AUROC vs distance-matched negatives      measured here
    direction concordance                    NOT possible from these tables
    signed Spearman                          NOT possible from these tables

That is a property of the source data, not a choice. Getting the signed
statistics needs per-allele methylation -- either CanASM (server down since at
least 15 Sep 2026) or a build on GSE186458's read-level data. Report the
limitation; do not quietly substitute |delta| agreement for direction agreement.

Two contrasts, and which one to believe
----------------------------------------
positive_vs_bimodal_non_asm   the controlled test -- LEAD WITH THIS
positive_vs_background        the weaker, less specific test

ASM regions are a subset of the atlas's bimodal methylation regions, which are
CpG-dense, intermediate-methylation and enhancer-like. Against plain background
CpGs the model could separate them by recognising that regional character alone,
with nothing allele-specific involved. The bimodal-but-not-ASM contrast holds
that character fixed. If the background contrast separates and the bimodal one
does not, the honest conclusion is that the model recognises the region class,
not allele-specific methylation.

Protocol, matched to the GENOA/eGTEx work
------------------------------------------
- positives and negatives matched on |variant-to-CpG distance|, 10 bp tolerance
- 1 Mb block bootstrap over genomic blocks, 2,000 replicates
- distance-only baseline reported alongside; matching should pin it near 0.5,
  and if it does not, the matching failed and the headline AUROC is not
  interpretable
- fusion and sequence arms both reported; the sequence arm is the tissue-agnostic
  comparator

Usage
-----
    python -u scripts/53c_asm_evaluate.py
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
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger("silentmethyl.asm_eval")
SCORE_COLUMN = "Absolute_Delta_M"


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


def block_auc(labels: np.ndarray, scores: np.ndarray, chrom: np.ndarray,
              position: np.ndarray, block_size: int, replicates: int,
              seed: int) -> dict[str, float]:
    """AUROC with a 1 Mb block bootstrap over the CpG's genomic block."""
    if np.unique(labels).size < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_le_half": float("nan"), "n_blocks": 0}
    observed = float(roc_auc_score(labels, scores))
    keys = pd.Series([f"{c}:{p // block_size}" for c, p in zip(chrom, position)])
    codes, unique = pd.factorize(keys, sort=True)
    n_blocks = len(unique)
    if n_blocks < 2 or replicates <= 0:
        return {"auroc": observed, "ci_low": float("nan"), "ci_high": float("nan"),
                "p_le_half": float("nan"), "n_blocks": int(n_blocks)}

    rng = np.random.default_rng(seed)
    draws = np.full(replicates, np.nan)
    for i in range(replicates):
        multiplicity = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks))
        weight = multiplicity[codes].astype(float)
        keep = weight > 0
        sub_labels = labels[keep]
        if np.unique(sub_labels).size < 2:
            continue
        draws[i] = roc_auc_score(sub_labels, scores[keep], sample_weight=weight[keep])
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return {"auroc": observed, "ci_low": float("nan"), "ci_high": float("nan"),
                "p_le_half": float("nan"), "n_blocks": int(n_blocks)}
    return {
        "auroc": observed,
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "p_le_half": float(np.mean(finite <= 0.5)),
        "n_blocks": int(n_blocks),
    }


def evaluate(frame: pd.DataFrame, label: str, args: argparse.Namespace,
             offset: int) -> list[dict]:
    rows = []
    labels = frame["asm_label"].to_numpy(dtype=int)
    chrom = frame["cpg_chr"].to_numpy()
    position = frame["cpg_pos0"].to_numpy(dtype=np.int64)

    for arm in sorted(frame["Model"].unique()):
        sub = frame[frame["Model"] == arm]
        stats = block_auc(
            sub["asm_label"].to_numpy(dtype=int),
            sub[SCORE_COLUMN].to_numpy(dtype=float),
            sub["cpg_chr"].to_numpy(),
            sub["cpg_pos0"].to_numpy(dtype=np.int64),
            args.block_size_bp, args.bootstrap_replicates,
            args.random_seed + offset,
        )
        rows.append({"Stratum": label, "Score": arm, "N_Pairs": int(len(sub)),
                     "N_Positive": int((sub["asm_label"] == 1).sum()), **stats})

    one_arm = frame[frame["Model"] == sorted(frame["Model"].unique())[0]]
    stats = block_auc(
        one_arm["asm_label"].to_numpy(dtype=int),
        -one_arm["abs_distance_bp"].to_numpy(dtype=float),
        one_arm["cpg_chr"].to_numpy(),
        one_arm["cpg_pos0"].to_numpy(dtype=np.int64),
        args.block_size_bp, args.bootstrap_replicates, args.random_seed + offset + 7,
    )
    rows.append({"Stratum": label, "Score": "distance_only_baseline",
                 "N_Pairs": int(len(one_arm)),
                 "N_Positive": int((one_arm["asm_label"] == 1).sum()), **stats})
    return rows


def balance_table(frame: pd.DataFrame) -> pd.DataFrame:
    one = frame[frame["Model"] == sorted(frame["Model"].unique())[0]]
    rows = []
    for value, group in one.groupby("asm_label"):
        rows.append({
            "asm_label": int(value),
            "n": int(len(group)),
            "abs_distance_mean": float(group["abs_distance_bp"].mean()),
            "abs_distance_median": float(group["abs_distance_bp"].median()),
            "abs_distance_p10": float(group["abs_distance_bp"].quantile(0.10)),
            "abs_distance_p90": float(group["abs_distance_bp"].quantile(0.90)),
        })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> int:
    scores = pd.read_csv(args.scores_csv)
    required = ["Model", "asm_label", "cpg_chr", "cpg_pos0", "abs_distance_bp",
                SCORE_COLUMN, "breast"]
    missing = [c for c in required if c not in scores.columns]
    if missing:
        raise ValueError(f"{args.scores_csv} is missing {missing}")
    LOGGER.info("loaded %d scored rows, arms=%s",
                len(scores), sorted(scores["Model"].unique()))

    if "contrast" not in scores.columns:
        raise ValueError(f"{args.scores_csv} lacks the `contrast` column")

    rows: list[dict] = []
    offset = 0
    for contrast, block in scores.groupby("contrast"):
        strata = [(f"{contrast} | all tissues", block)]
        breast = block[block["breast"] == 1]
        if len(breast) and breast["asm_label"].nunique() > 1:
            strata.append((f"{contrast} | breast", breast))
        else:
            LOGGER.warning("%s: breast stratum unusable (one class or empty)", contrast)
        for label, frame in strata:
            offset += 100
            LOGGER.info("evaluating %s (%d rows)", label, len(frame))
            rows.extend(evaluate(frame, label, args, offset))
    table = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "asm_discrimination.csv"
    balance_path = args.output_dir / "asm_matching_balance.csv"
    atomic_csv(table, table_path)
    atomic_csv(balance_table(scores), balance_path)

    baselines = table[table["Score"] == "distance_only_baseline"]
    worst = float(np.nanmax(np.abs(baselines["auroc"].to_numpy(dtype=float) - 0.5))) \
        if len(baselines) else float("nan")
    matching_ok = bool(np.isfinite(worst) and worst <= 0.05)
    baseline_value = {str(r["Stratum"]): float(r["auroc"]) for _, r in baselines.iterrows()}

    summary = {
        "analysis": "ASM validation (Task E1) -- discrimination against "
                    "distance-matched negatives",
        "analysis_status": "COMPLETE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "score_column": SCORE_COLUMN,
        "block_size_bp": int(args.block_size_bp),
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "random_seed": int(args.random_seed),
        "distance_only_baseline_auroc": baseline_value,
        "worst_baseline_deviation_from_half": worst,
        "matching_sane": matching_ok,
        "which_contrast_to_lead_with": (
            "positive_vs_bimodal_non_asm -- it holds the bimodal/enhancer-like "
            "region class fixed. positive_vs_background cannot distinguish "
            "detecting ASM from detecting the region class."
        ),
        "matching_sanity_note": (
            "The distance-only baseline must sit near 0.5 after matching. If it "
            "does not, the matching failed and the headline AUROC is confounded "
            "by distance rather than measuring the model."
        ),
        "results": table.to_dict(orient="records"),
        "not_measured": {
            "direction_concordance": "requires a signed allelic methylation "
                                     "difference; absent from the atlas tables",
            "signed_spearman": "same reason",
        },
        "out_of_distribution_caveat": (
            "Trained only at HM450 probe positions, which are enriched at "
            "promoters and CpG islands. Every CpG here is an arbitrary genomic "
            "CpG, so this is out of distribution with respect to training "
            "POSITIONS. A limitation to state, and also a generalisation test."
        ),
        "inputs": {args.scores_csv.as_posix(): sha256_file(args.scores_csv)},
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("=" * 78)
    print(f"ASM validation (Task E1) -- AUROC on {SCORE_COLUMN}")
    print("=" * 78)
    print(f"  {'stratum':<44}{'score':<24}{'n':>8}{'AUROC':>8}  95% CI")
    for _, r in table.iterrows():
        ci = (f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
              if np.isfinite(r["ci_low"]) else "n/a")
        print(f"  {str(r['Stratum']):<44}{str(r['Score']):<24}{int(r['N_Pairs']):>8,}"
              f"{r['auroc']:>8.4f}  {ci}")
    print("-" * 78)
    print(f"  worst distance-baseline deviation from 0.5 = {worst:.4f} -> matching "
          f"{'OK' if matching_ok else 'FAILED, do not interpret the AUROCs'}")
    print("=" * 78)
    print(f"wrote {table_path}\nwrote {balance_path}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scores-csv", type=Path,
                   default=Path("results/journal/asm_validation/asm_pair_scores.csv"))
    p.add_argument("--block-size-bp", type=int, default=1_000_000)
    p.add_argument("--bootstrap-replicates", type=int, default=2_000)
    p.add_argument("--random-seed", type=int, default=20260915)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/journal/asm_validation"))
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    sys.exit(run(parse_args()))
