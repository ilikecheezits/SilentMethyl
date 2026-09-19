#!/usr/bin/env python3
"""Join per-tissue mQTL labels onto the union predictions.

Why this exists
---------------
The nine tissues were scored once, as a union of 80,919 variant-CpG pairs, because
the model's prediction for a pair does not depend on which tissue's mQTL table the
pair came from -- only the LABELS differ. That saved 54 GPU tasks. The cost is that
`pair_scores.csv` carries label columns (pvalue, beta_ref_to_alt, se, maf,
ma_count, ma_samples) inherited from whichever tissue sorted first, which is
BreastMammaryTissue.

Those columns are placeholders and must never be used. Running the downstream
analyses on them would silently produce a breast-only result wearing a nine-tissue
label -- the numbers would look plausible and be wrong.

This script replaces them, per tissue, and writes into the directory layout
scripts/31_transfer_discrimination.py and scripts/40_meqtl_tissue_specificity.py
already expect:

    results/journal/egtex_multitissue_scoring/by_tissue/<Tissue>/heldout/<model>/seed<N>/pair_scores.csv

The join is an INNER join on (probeID, Variant_ID), so each tissue keeps only the
pairs eGTEx actually tested there -- 70,218 for Prostate up to 77,421 for Lung,
not all 80,919. Output row counts are asserted against the cohort sizes recorded
when the cohorts were built, so a silent join failure cannot pass.

Usage (run from the repository root)
------------------------------------
    python -u data/split_predictions_by_tissue.py
    python -u data/split_predictions_by_tissue.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCORING = Path("results/journal/egtex_multitissue_scoring/heldout")
LABELS = Path("data/external/egtex_multitissue/scoring")
OUT_ROOT = Path("results/journal/egtex_multitissue_scoring/by_tissue")

KEY = ["probeID", "Variant_ID"]
LABEL_COLS = ["beta_ref_to_alt", "se", "pvalue", "maf", "ma_count", "ma_samples"]

EXPECTED = {
    "BreastMammaryTissue": 76893,
    "ColonTransverse": 76453,
    "KidneyCortex": 75873,
    "Lung": 77421,
    "MuscleSkeletal": 74864,
    "Ovary": 76985,
    "Prostate": 70218,
    "Testis": 73524,
    "WholeBlood": 75517,
}


def load_labels() -> dict:
    out = {}
    for tissue in sorted(EXPECTED):
        path = LABELS / tissue / "egtex_scoring_input_heldout.csv"
        if not path.is_file():
            raise SystemExit(f"STOP: missing {path}")
        frame = pd.read_csv(path, usecols=KEY + LABEL_COLS)
        if len(frame) != EXPECTED[tissue]:
            raise SystemExit(
                f"STOP: {tissue} has {len(frame):,} rows, expected "
                f"{EXPECTED[tissue]:,}. The cohort changed since it was built; "
                f"update EXPECTED rather than ignoring this.")
        dupes = int(frame.duplicated(subset=KEY).sum())
        if dupes:
            raise SystemExit(f"STOP: {tissue} has {dupes:,} duplicate keys; "
                             f"an inner join would multiply rows")
        out[tissue] = frame
        logging.info("labels %-22s %7d pairs", tissue, len(frame))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="check inputs and report planned output, write nothing")
    ap.add_argument("--models", nargs="+", default=["fusion", "sequence"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    if not SCORING.is_dir():
        raise SystemExit(f"STOP: {SCORING} not found -- run from the repo root")

    labels = load_labels()
    written, report = 0, []

    for model in args.models:
        for seed in args.seeds:
            src = SCORING / model / f"seed{seed}" / "pair_scores.csv"
            if not src.is_file():
                raise SystemExit(f"STOP: missing {src}")
            pred = pd.read_csv(src)
            missing = [c for c in KEY if c not in pred.columns]
            if missing:
                raise SystemExit(f"STOP: {src} lacks {missing}")
            logging.info("read %s (%d rows)", src, len(pred))

            base = pred.drop(columns=[c for c in LABEL_COLS + ["source_tissue"]
                                      if c in pred.columns])

            for tissue, lab in labels.items():
                joined = base.merge(lab, on=KEY, how="inner")
                if len(joined) != EXPECTED[tissue]:
                    raise SystemExit(
                        f"STOP: {tissue} {model} seed{seed} joined to "
                        f"{len(joined):,} rows, expected {EXPECTED[tissue]:,}. "
                        f"The union does not cover this tissue -- do not proceed.")
                if joined["pvalue"].isna().any():
                    raise SystemExit(f"STOP: {tissue} {model} seed{seed} has "
                                     f"NaN p-values after the join")
                joined["source_tissue"] = tissue

                dst = OUT_ROOT / tissue / "heldout" / model / f"seed{seed}"
                report.append({"tissue": tissue, "model": model, "seed": seed,
                               "rows": int(len(joined)),
                               "significant_5e8": int((joined["pvalue"] < 5e-8).sum()),
                               "path": str(dst / "pair_scores.csv")})
                if not args.dry_run:
                    dst.mkdir(parents=True, exist_ok=True)
                    joined.to_csv(dst / "pair_scores.csv", index=False)
                    written += 1
            del pred, base

    frame = pd.DataFrame(report)
    print()
    print("=" * 70)
    print("PER-TISSUE COHORTS  (rows and significant pairs are per model-seed,")
    print("so they are identical across the six -- only predictions differ)")
    print("=" * 70)
    summary = (frame.groupby("tissue")
               .agg(rows=("rows", "first"),
                    significant=("significant_5e8", "first"),
                    files=("path", "count")).reset_index())
    print(f"{'tissue':<24}{'pairs':>9}{'sig p<5e-8':>12}{'files':>7}")
    for _, r in summary.iterrows():
        print(f"{r['tissue']:<24}{r['rows']:>9,}{r['significant']:>12,}{r['files']:>7}")
    print(f"{'TOTAL':<24}{'':>9}{summary['significant'].sum():>12,}"
          f"{summary['files'].sum():>7}")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        return 0

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "split_summary.json").open("w") as fh:
        json.dump({
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": str(SCORING),
            "join_key": KEY,
            "replaced_columns": LABEL_COLS,
            "why": ("the union was scored once because model predictions do not "
                    "depend on the tissue label; these columns carried "
                    "BreastMammaryTissue values as placeholders and are replaced "
                    "here with each tissue's own measurements"),
            "files_written": written,
            "per_tissue": summary.to_dict("records"),
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"\nwrote {written} files under {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
