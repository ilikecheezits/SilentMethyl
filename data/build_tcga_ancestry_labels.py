#!/usr/bin/env python3
"""
Attach genetic-ancestry labels to the TCGA-BRCA samples SilentMethyl uses.

Why this exists
---------------
The mentor asked for **ancestry analyses**. Two things deliver that:

  1. Cross-cohort replication -- GENOA (African American) against eGTEx
     (European-dominant). Handled by the GENOA harmonizer.
  2. Ancestry-stratified prediction error on your OWN data. That needs a label
     per TCGA participant, which is what this script produces.

The labels come from the GDC's openly published ancestry calls
(gdc.cancer.gov/about-data/publications/CCG-AIM-2020) -- derived from genotype,
but released as open supplemental files. No controlled access.

What it writes
--------------
  tcga_ancestry_labels.csv   participant, ancestry_call, source, plus admixture
                             proportions where available
  ancestry_summary.json      group counts for the training normals and, if
                             present, the tumour cohort

Honest caveat this script prints for you
-----------------------------------------
Genetic ancestry is a continuous quantity summarised here as a discrete label.
Small groups give unstable per-group error estimates. The summary reports group
sizes precisely so you can say which strata are actually powered -- and so a
reviewer does not have to ask.

Usage (run from the repository root)
------------------------------------
    python -u data/build_tcga_ancestry_labels.py --inspect
    python -u data/build_tcga_ancestry_labels.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_ANCESTRY_DIR = Path("data/external/tcga_ancestry")
DEFAULT_NORMAL_IDS = Path("data/datafiles/tcga_normal_sample_ids.json")
DEFAULT_TUMOR_SUMMARY = Path("data/external/tcga_tumor/tcga_tumor_summary.json")
DEFAULT_OUT = Path("data/external/tcga_ancestry")

PARTICIPANT_RE = re.compile(r"(TCGA[-.][A-Z0-9]{2}[-.][A-Z0-9]{4})", re.IGNORECASE)


def participant_of(value) -> str | None:
    m = PARTICIPANT_RE.search(str(value))
    return m.group(1).replace(".", "-").upper() if m else None


def sniff_table(path: Path) -> pd.DataFrame | None:
    """Read a GDC ancestry file without assuming its delimiter."""
    if not path.exists():
        return None
    for sep in ("\t", ","):
        try:
            df = pd.read_csv(path, sep=sep, low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception:  # noqa: BLE001
            continue
    return None


def find_columns(df: pd.DataFrame) -> dict:
    """Locate the participant-ID column and the ancestry-call column."""
    pid_col = None
    best_hits = 0
    for c in df.columns:
        hits = df[c].astype(str).head(200).map(lambda v: participant_of(v) is not None).sum()
        if hits > best_hits:
            best_hits, pid_col = hits, c

    call_col = None
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ("ancestry", "ethnicity", "population",
                                 "consensus", "call", "assign")):
            if c != pid_col:
                call_col = c
                break

    admix = [c for c in df.columns
             if any(k in str(c).lower() for k in ("afr", "eur", "eas", "sas", "amr", "admix"))]
    return {"participant": pid_col, "call": call_col, "admixture": admix,
            "participant_hit_rate": best_hits / min(200, len(df)) if len(df) else 0.0}


def load_sample_participants(path: Path) -> set:
    if not path.exists():
        return set()
    ids = json.loads(path.read_text())
    if isinstance(ids, dict):
        ids = ids.get("normal_columns") or ids.get("samples") or list(ids.values())
    return {p for p in (participant_of(s) for s in ids) if p}


def cmd_inspect(args) -> int:
    files = sorted(args.ancestry_dir.glob("*"))
    files = [f for f in files if f.is_file() and f.suffix.lower() in
             {".csv", ".txt", ".tsv"}]
    if not files:
        print(f"no ancestry tables found under {args.ancestry_dir}")
        return 1

    for f in files:
        print(f"\n=== {f.name} ({f.stat().st_size/1e6:.2f} MB) ===")
        df = sniff_table(f)
        if df is None:
            print("  could not parse")
            continue
        cols = find_columns(df)
        print(f"  rows x cols      : {df.shape[0]:,} x {df.shape[1]}")
        print(f"  participant col  : {cols['participant']} "
              f"(match rate {cols['participant_hit_rate']:.0%})")
        print(f"  ancestry call col: {cols['call']}")
        if cols["admixture"]:
            print(f"  admixture cols   : {cols['admixture'][:8]}")
        if cols["call"]:
            vc = df[cols["call"]].astype(str).value_counts().head(10)
            print("  call values:")
            for k, v in vc.items():
                print(f"      {k:<28} {v:>7,}")
        print(f"  columns: {list(df.columns)[:14]}")

    normals = load_sample_participants(args.normal_ids)
    print(f"\ntraining-normal participants to label: {len(normals)}")
    return 0


def cmd_run(args) -> int:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    candidates = sorted(p for p in args.ancestry_dir.glob("*")
                        if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".tsv"}
                        and not p.name.startswith("tcga_ancestry_labels"))
    if not candidates:
        raise SystemExit(f"no ancestry tables under {args.ancestry_dir}")

    merged = None
    used = []
    for f in candidates:
        df = sniff_table(f)
        if df is None:
            continue
        cols = find_columns(df)
        if not cols["participant"] or cols["participant_hit_rate"] < 0.5:
            logging.info("skipping %s (no reliable participant column)", f.name)
            continue
        sub = pd.DataFrame({"participant": df[cols["participant"]].map(participant_of)})
        if cols["call"]:
            sub[f"call_{f.stem}"] = df[cols["call"]].astype(str)
        for a in cols["admixture"][:8]:
            sub[f"{f.stem}__{a}"] = pd.to_numeric(df[a], errors="coerce")
        sub = sub.dropna(subset=["participant"]).drop_duplicates("participant")
        merged = sub if merged is None else merged.merge(sub, on="participant", how="outer")
        used.append({"file": f.name, "participants": int(len(sub)),
                     "call_column": cols["call"], "admixture_columns": cols["admixture"][:8]})
        logging.info("%s -> %d participants", f.name, len(sub))

    if merged is None or merged.empty:
        raise SystemExit("no usable ancestry labels parsed. Run --inspect.")

    # Consensus call: first non-null across sources, in file order.
    call_cols = [c for c in merged.columns if c.startswith("call_")]
    if call_cols:
        merged["ancestry_call"] = merged[call_cols].bfill(axis=1).iloc[:, 0]
    else:
        merged["ancestry_call"] = pd.NA

    target = out / "tcga_ancestry_labels.csv"
    merged.to_csv(target, index=False)

    # ---- coverage against the cohorts we actually score
    normals = load_sample_participants(args.normal_ids)
    labelled = set(merged.dropna(subset=["ancestry_call"])["participant"])
    normal_counts = Counter(
        merged.set_index("participant")["ancestry_call"].get(p, "UNLABELLED")
        for p in normals
    ) if normals else Counter()

    summary = {
        "analysis": "TCGA ancestry labels joined to SilentMethyl cohorts",
        "purpose": "mentor requirement: ancestry analyses",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": ("GDC open supplemental ancestry calls, "
                   "gdc.cancer.gov/about-data/publications/CCG-AIM-2020"),
        "access": "open; derived from genotype but released as open supplemental files",
        "files_used": used,
        "participants_labelled": int(len(labelled)),
        "output_file": str(target),
        "training_normal_participants": len(normals),
        "training_normal_group_counts": dict(normal_counts),
        "caveat": (
            "Genetic ancestry is continuous and is summarised here as a discrete "
            "label. Groups with few participants give unstable per-group error "
            "estimates -- report group sizes alongside any stratified metric and "
            "state which strata are powered."
        ),
    }
    with (out / "ancestry_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print()
    print("=" * 62)
    print(f"participants labelled : {len(labelled):,}")
    print(f"output                : {target}")
    if normals:
        print(f"\ntraining-normal cohort ({len(normals)} participants):")
        for group, n in sorted(normal_counts.items(), key=lambda kv: -kv[1]):
            flag = "  <- too small to stratify on" if n < 10 and group != "UNLABELLED" else ""
            print(f"  {str(group):<28} {n:>4}{flag}")
        print("\nReport these group sizes with any ancestry-stratified metric.")
        print("Strata under ~10 samples will not support a stable error estimate.")
    print("=" * 62)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ancestry-dir", type=Path, default=DEFAULT_ANCESTRY_DIR)
    ap.add_argument("--normal-ids", type=Path, default=DEFAULT_NORMAL_IDS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--inspect", action="store_true",
                    help="show detected columns and call values; write nothing")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    return cmd_inspect(args) if args.inspect else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
