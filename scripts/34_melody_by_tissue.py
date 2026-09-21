#!/usr/bin/env python3
"""Split Melody-MT union scores into per-tissue cohorts that 31_transfer_discrimination.py
can read. The cohorts are taken from the SilentMethyl scoring input rather than rebuilt
from Melody's output, so both models are compared on identical rows.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("melody.by_tissue")

KEY = ["probeID", "Variant_ID"]

TISSUE_TRACKS = {
    "BreastMammaryTissue": ["GSM5652347_Breast-Luminal-Epithelial-Z000000V2",
                            "GSM5652350_Breast-Basal-Epithelial-Z000000V6"],
    "ColonTransverse":     ["GSM5652370_Colon-Right-Epithelial-Z000000V0",
                            "GSM5652198_Colon-Fibroblasts-Z0000042A"],
    "KidneyCortex":        ["GSM5652189_Kidney-Tubular-Endothel-Z0000042R"],
    "Lung":                ["GSM5652354_Lung-Alveolar-Epithelial-Z000000T1",
                            "GSM5652335_Lung-Bronchus-Epithelial-Z000000QD"],
    "MuscleSkeletal":      ["GSM5652205_Skeletal-Muscle-Z00000427"],
    "Ovary":               ["GSM5652270_Ovary-Epithelial-Z000000QT"],
    "Prostate":            ["GSM5652338_Prostate-Epithelial-Z000000RV"],
    "WholeBlood":          ["GSM5652317_Blood-B-Z000000UB",
                            "GSM5652277_Blood-T-CD3-Z000000TV",
                            "GSM5652299_Blood-NK-Z000000TM",
                            "GSM5652302_Blood-Monocytes-Z000000TP",
                            "GSM5652313_Blood-Granulocytes-Z000000TZ"],
}
EXCLUDED = {"Testis": "no testis cell type exists among Melody's 39 tracks"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--melody", type=Path,
                    default=Path("results/journal/melody/union_pair_scores_15track.csv"))
    ap.add_argument("--template-root", type=Path,
                    default=Path("results/journal/egtex_multitissue_scoring/by_tissue"))
    ap.add_argument("--template-model", default="fusion")
    ap.add_argument("--template-seed", type=int, default=42)
    ap.add_argument("--out-root", type=Path,
                    default=Path("results/journal/melody/by_tissue"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if not args.melody.is_file():
        raise SystemExit(f"STOP: {args.melody} not found")

    mel = pd.read_csv(args.melody)
    missing = [c for c in KEY if c not in mel.columns]
    if missing:
        raise SystemExit(f"STOP: {args.melody} lacks {missing}; it must be the "
                         f"--format ours run, not --format melody")
    if mel.duplicated(subset=KEY).any():
        raise SystemExit("STOP: Melody scores contain duplicate (probeID, "
                         "Variant_ID) keys; an inner join would multiply rows")
    LOGGER.info("Melody scores: %d rows", len(mel))

    all_tracks = sorted({t for v in TISSUE_TRACKS.values() for t in v})
    absent = [t for t in all_tracks if f"delta_{t}" not in mel.columns]
    if absent:
        raise SystemExit(
            f"STOP: the scoring run did not include {len(absent)} needed "
            f"track(s):\n  " + "\n  ".join(absent) +
            "\n(a --export comma-splitting bug once truncated this list to one "
            "track; check summary.json['tracks'])")

    report, written = [], 0
    for tissue, tracks in TISSUE_TRACKS.items():
        template = (args.template_root / tissue / "heldout" /
                    args.template_model / f"seed{args.template_seed}" /
                    "pair_scores.csv")
        if not template.is_file():
            raise SystemExit(f"STOP: template not found: {template}")
        base = pd.read_csv(template)
        cols = [f"delta_{t}" for t in tracks]
        joined = base.merge(mel[KEY + cols], on=KEY, how="left")
        if len(joined) != len(base):
            raise SystemExit(f"STOP: {tissue} join changed row count "
                             f"{len(base)} -> {len(joined)}")

        unmatched = int(joined[cols[0]].isna().sum())
        if unmatched:
            raise SystemExit(
                f"STOP: {tissue} has {unmatched:,} pairs with no Melody score. "
                f"The union scoring did not cover this tissue's cohort; do not "
                f"proceed with a partial comparison.")

        joined["Predicted_Delta_M"] = joined[cols].mean(axis=1)
        joined["Absolute_Delta_M"] = joined["Predicted_Delta_M"].abs()
        for drop in ("Predicted_Delta_Beta", "Absolute_Delta_Beta",
                     "Delta_Beta_FWD", "Delta_Beta_RC",
                     "Delta_Beta_RC_Absolute_Difference",
                     "Delta_Beta_RC_Sign_Agree",
                     "WT_M_RC_Avg", "MUT_M_RC_Avg",
                     "WT_Beta_RC_Avg", "MUT_Beta_RC_Avg",
                     "Weights_Path", "Weights_SHA256"):
            if drop in joined.columns:
                joined = joined.drop(columns=drop)
        for gate in [c for c in joined.columns if "Gate" in c]:
            joined = joined.drop(columns=gate)
        joined["Model"] = "melody"
        joined["Melody_Tracks"] = ";".join(tracks)

        dst = args.out_root / tissue / "heldout" / "melody" / "seed42"
        report.append({"tissue": tissue, "rows": int(len(joined)),
                       "tracks": len(tracks),
                       "significant_5e8": int((joined["pvalue"] < 5e-8).sum()),
                       "path": str(dst / "pair_scores.csv")})
        if not args.dry_run:
            dst.mkdir(parents=True, exist_ok=True)
            joined.to_csv(dst / "pair_scores.csv", index=False)
            written += 1
        LOGGER.info("%-22s %7d pairs, %d track(s)", tissue, len(joined), len(tracks))

    frame = pd.DataFrame(report)
    print()
    print("=" * 72)
    print("MELODY PER-TISSUE COHORTS  (same rows as SilentMethyl by construction)")
    print("=" * 72)
    print(f"{'tissue':<24}{'pairs':>9}{'sig p<5e-8':>12}{'tracks':>8}")
    for _, r in frame.iterrows():
        print(f"{r['tissue']:<24}{r['rows']:>9,}{r['significant_5e8']:>12,}"
              f"{r['tracks']:>8}")
    print()
    for tissue, why in EXCLUDED.items():
        print(f"EXCLUDED  {tissue}: {why}")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)
    with (args.out_root / "melody_by_tissue_summary.json").open("w") as fh:
        json.dump({
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "melody_scores": str(args.melody),
            "template": (f"{args.template_root}/<Tissue>/heldout/"
                         f"{args.template_model}/seed{args.template_seed}"),
            "method": ("SilentMethyl's per-tissue pair_scores.csv is the template; "
                       "only Predicted_Delta_M and Absolute_Delta_M are replaced, "
                       "so both models are evaluated on identical rows"),
            "tissue_tracks": TISSUE_TRACKS,
            "excluded": EXCLUDED,
            "per_tissue": frame.to_dict("records"),
            "files_written": written,
        }, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {written} files under {args.out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
