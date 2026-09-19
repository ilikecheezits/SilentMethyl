#!/usr/bin/env python3
"""Turn Melody-ST scores into per-tissue cohorts that scripts/31 can read.

Why a separate script from 34
-----------------------------
`34_melody_by_tissue.py` consumes the Melody-MT union file, which carries one
`delta_<track>` column per output channel and needs them averaged. The ST runs
are one CSV per (checkpoint, tissue) with `Predicted_Delta_M` already computed,
so the join is simpler and the track-averaging logic does not apply. Same
guarantee though: SilentMethyl's per-tissue `pair_scores.csv` is the TEMPLATE,
and only `Predicted_Delta_M` / `Absolute_Delta_M` are replaced, so both models
are evaluated on byte-identical rows.

What the two arms are for
-------------------------
BREAST -> all nine tissues. Melody-ST is genuinely single-tissue ("trained on
one cell type at a time using a single bigWig track as supervision", Methods),
which makes it the architecture-matched comparison for SilentMethyl. Melody's
own Fig 3H is Melody-MT cross-TRACK validation -- one model trained on all 39
cell types, reading a different output channel -- so it is head selection, not
transfer. This arm is the transfer experiment their paper does not contain.

LUNG -> Lung only. The tissue-matched arm. ST-Breast vs ST-Lung on the SAME Lung
rows isolates what tissue-matching buys with architecture, training-data volume
and scoring procedure all held constant. That is the cleanest measurement of
tissue-matching available in either paper, and it is two numbers.

Usage (run from the repository root)
------------------------------------
    python -u scripts/35_melody_st_by_tissue.py --dry-run
    python -u scripts/35_melody_st_by_tissue.py

Then, per tissue:

    python -u scripts/31_transfer_discrimination.py \
        --reference results/journal/egtex_multitissue_scoring/by_tissue/Lung::fusion::42,43,44 \
        --compare   results/journal/melody_st/by_tissue/Lung::melody_st_breast::42 \
        --output-dir results/journal/melody_st_head_to_head/Lung
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("melody.st_by_tissue")

KEY = ["probeID", "Variant_ID"]

DROP_PREFIXES = ("Predicted_Delta_Beta", "Absolute_Delta_Beta", "Delta_Beta_",
                 "WT_M_RC", "MUT_M_RC", "WT_Beta_RC", "MUT_Beta_RC",
                 "Weights_Path", "Weights_SHA256")

ARMS = [
    ("melody_st_breast", "st_breast_{tissue}",
     ["BreastMammaryTissue", "ColonTransverse", "KidneyCortex", "Lung",
      "MuscleSkeletal", "Ovary", "Prostate", "Testis", "WholeBlood"]),
    ("melody_st_lung", "st_lung_{tissue}", ["Lung"]),
]


def load_template(root: Path, tissue: str, model: str, seed: int) -> Path:
    return root / tissue / "heldout" / model / f"seed{seed}" / "pair_scores.csv"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--st-root", type=Path,
                    default=Path("results/journal/melody_st"))
    ap.add_argument("--template-root", type=Path,
                    default=Path("results/journal/egtex_multitissue_scoring/by_tissue"))
    ap.add_argument("--template-model", default="fusion")
    ap.add_argument("--template-seed", type=int, default=42)
    ap.add_argument("--out-root", type=Path,
                    default=Path("results/journal/melody_st/by_tissue"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    report, written, missing = [], 0, []
    for arm, stem, tissues in ARMS:
        for tissue in tissues:
            src = args.st_root / (stem.format(tissue=tissue) + ".csv")
            if not src.is_file():
                missing.append(str(src))
                LOGGER.warning("%-18s %-22s MISSING %s", arm, tissue, src)
                continue

            template = load_template(args.template_root, tissue,
                                     args.template_model, args.template_seed)
            if not template.is_file():
                raise SystemExit(f"STOP: template not found: {template}")

            st = pd.read_csv(src)
            for col in KEY + ["Predicted_Delta_M"]:
                if col not in st.columns:
                    raise SystemExit(
                        f"STOP: {src} has no {col!r} column. It must be a "
                        f"`--format ours` run of scripts/33_melody_scoring.py, "
                        f"not a `--format melody` benchmark run.")
            if st.duplicated(subset=KEY).any():
                raise SystemExit(f"STOP: {src} has duplicate {KEY} keys; an "
                                 f"inner join would multiply rows")

            base = pd.read_csv(template)
            joined = base.merge(st[KEY + ["Predicted_Delta_M"]], on=KEY,
                                how="left", suffixes=("_silentmethyl", ""))
            if len(joined) != len(base):
                raise SystemExit(f"STOP: {arm}/{tissue} join changed row count "
                                 f"{len(base)} -> {len(joined)}")

            unmatched = int(joined["Predicted_Delta_M"].isna().sum())
            if unmatched:
                raise SystemExit(
                    f"STOP: {arm}/{tissue} has {unmatched:,} pairs with no "
                    f"Melody-ST score. A partial comparison would silently "
                    f"evaluate the two models on different rows.")

            joined["Absolute_Delta_M"] = joined["Predicted_Delta_M"].abs()
            drop = [c for c in joined.columns
                    if c.startswith(DROP_PREFIXES) or "Gate" in c
                    or c.endswith("_silentmethyl")]
            joined = joined.drop(columns=drop)
            joined["Model"] = arm

            dst = args.out_root / tissue / "heldout" / arm / "seed42"
            report.append({"arm": arm, "tissue": tissue,
                           "rows": int(len(joined)),
                           "significant_5e8": int((joined["pvalue"] < 5e-8).sum()),
                           "source": str(src),
                           "path": str(dst / "pair_scores.csv")})
            if not args.dry_run:
                dst.mkdir(parents=True, exist_ok=True)
                joined.to_csv(dst / "pair_scores.csv", index=False)
                written += 1
            LOGGER.info("%-18s %-22s %7d pairs", arm, tissue, len(joined))

    if not report:
        raise SystemExit(
            "STOP: no Melody-ST score files found under "
            f"{args.st_root}. Has job run_melody_st.sbatch finished?")

    frame = pd.DataFrame(report)
    print()
    print("=" * 76)
    print("MELODY-ST PER-TISSUE COHORTS  (same rows as SilentMethyl by construction)")
    print("=" * 76)
    print(f"{'arm':<20}{'tissue':<24}{'pairs':>9}{'sig p<5e-8':>12}")
    for _, r in frame.iterrows():
        print(f"{r['arm']:<20}{r['tissue']:<24}{r['rows']:>9,}"
              f"{r['significant_5e8']:>12,}")
    if missing:
        print()
        for m in missing:
            print(f"MISSING  {m}")
        print("(a missing file means that tissue was skipped by the scoring job; "
              "check logs/melody_st/*.err before interpreting anything)")

    if {"melody_st_breast", "melody_st_lung"}.issubset(set(frame["arm"])):
        print()
        print("Both arms present for Lung. The matched-vs-unmatched comparison:")
        print("  python -u scripts/31_transfer_discrimination.py \\")
        print("    --reference results/journal/melody_st/by_tissue/Lung::melody_st_lung::42 \\")
        print("    --compare   results/journal/melody_st/by_tissue/Lung::melody_st_breast::42 \\")
        print("    --output-dir results/journal/melody_st_matched_vs_unmatched/Lung")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "melody_st_by_tissue_summary.json").open("w") as fh:
        json.dump({
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "template": (f"{args.template_root}/<Tissue>/heldout/"
                         f"{args.template_model}/seed{args.template_seed}"),
            "method": ("SilentMethyl's per-tissue pair_scores.csv is the template; "
                       "only Predicted_Delta_M and Absolute_Delta_M are replaced, "
                       "so both models are evaluated on identical rows"),
            "arms": {
                "melody_st_breast": ("single-tissue breast model applied to every "
                                     "tissue; the architecture-matched analogue of "
                                     "SilentMethyl's zero-shot transfer"),
                "melody_st_lung": ("tissue-matched single-tissue model, Lung only; "
                                   "the comparison arm that isolates what "
                                   "tissue-matching buys"),
            },
            "missing_inputs": missing,
            "per_cohort": frame.to_dict("records"),
            "files_written": written,
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {written} files under {args.out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
