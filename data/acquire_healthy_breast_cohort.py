#!/usr/bin/env python3
"""
Build a truly-healthy breast methylation cohort from GEO GSE69914.

Why this dataset specifically
-----------------------------
SilentMethyl's targets are medians of 97 TCGA solid-tissue-normal samples --
tissue taken adjacent to a tumour. The manuscript lists field effects and
cell-composition heterogeneity as a limitation, citing Teschendorff et al.
Nat Commun 2016. GSE69914 is the dataset from that paper, and it is the right
instrument for the question for three reasons:

  1. **Illumina HumanMethylation450.** Probe IDs are platform-stable, so this
     joins directly on probeID. No liftover, no chain file -- the same property
     that let the GENOA harmonizer avoid lifting a single CpG. The fact that GEO
     reports coordinates in hg19 is irrelevant to an array join.

  2. **It contains both arms in one study.** 50 normals from healthy women AND 84
     tumour-adjacent normals, same platform, same processing. Comparing healthy
     against adjacent *within* this series removes the batch and platform
     confound that would dominate a comparison against TCGA. Only after that
     comparison is established does it make sense to bring the result to our own
     targets.

  3. **The study that raised the concern can settle it.**

What this is FOR, and what it is not for
-----------------------------------------
**Validation, not training.** A 50-sample median is noisier than the existing
97-sample median; substituting it would trade a known bias for added variance.
The value here is an independent target to score frozen checkpoints against:

  comparable performance -> field effects are not materially contaminating the
                            targets, stated with evidence rather than hedged
  better                 -> interesting, and worth explaining
  worse                  -> the contamination is real and now quantified

All three outcomes are reportable, and the cohort also serves multi-cohort testing.

Output schema deliberately matches data/build_tcga_tumor_cohort.py, so
scripts/21_tcga_tumor_domain_shift.py consumes these files with no changes:
  probeID, Median_Beta, M_Value_Target, Binary_State_Target, n_samples_observed

Two confounds that must be stated with any result
--------------------------------------------------
**Cell composition.** Healthy-donor and tumour-adjacent breast tissue differ in
epithelial/stromal/adipose fractions. A beta difference is not automatically
evidence of a field effect.

**Age.** GEO reports incomplete metadata for this series and supplies no age
field. TCGA normals come from cancer patients and skew older, and methylation is
strongly age-dependent. Either state this as a limitation or estimate age from
methylation directly (--emit-clock-probes writes the CpG subset needed for that).

Usage (run from the repository root)
------------------------------------
    python -u data/acquire_healthy_breast_cohort.py --inspect     # DO THIS FIRST
    python -u data/acquire_healthy_breast_cohort.py --classify-only
    python -u data/acquire_healthy_breast_cohort.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("silentmethyl.healthy")

ACCESSION = "GSE69914"
SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE69nnn/GSE69914/matrix/"
    "GSE69914_series_matrix.txt.gz"
)
EPSILON = 1e-4

# Sample-group patterns, applied to the concatenated title + characteristics text.
# Order matters: the first pattern that matches wins, so the more specific
# BRCA1-carrier and cancer patterns are tested before the general normal ones.
#
# These are HYPOTHESES about the series' own wording, not verified fact. Run
# --inspect first and confirm the counts land near 50 healthy / 84 adjacent /
# 263 cancer before trusting them. The script refuses to write cohorts whose
# counts disagree with the published composition unless --force is given.
# ORDER IS LOAD-BEARING. adjacent_normal must be tested BEFORE cancer: a label
# like "normal adjacent to tumour" contains the word "tumour", so a cancer-first
# ordering silently files every adjacent sample as a tumour. That exact mistake
# was caught in testing, and the composition check below is what caught it.
GROUP_PATTERNS: list[tuple[str, str]] = [
    ("brca1_carrier", r"brca1"),
    ("adjacent_normal", r"normal[\s_-]*adjacent|adjacent[\s_-]*normal|\bnadj\b|"
                        r"tumou?r[\s_-]*adjacent|adjacent[\s_-]*to|\badjacent\b"),
    ("cancer", r"\bcancer\b|\btumou?r\b|\bcarcinoma\b|\bmalignant\b"),
    ("healthy_normal", r"\bnormal\b|\bhealthy\b|\bcontrol\b"),
]

# Published composition, used only as a sanity check on the classifier.
EXPECTED = {"healthy_normal": 50, "adjacent_normal": 84, "cancer": 263}
TOLERANCE = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=SERIES_MATRIX_URL)
    p.add_argument("--raw-dir", type=Path,
                   default=Path("data/external/healthy_breast"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("data/external/healthy_breast"))
    p.add_argument("--probe-source", type=Path, default=Path("data/datafiles/test.csv"),
                   help="CSV whose probeID column defines the probe subset. Default "
                        "is the held-out split, which is all the evaluation needs.")
    p.add_argument("--all-probes", action="store_true",
                   help="Keep every probe instead of the held-out subset. Produces a "
                        "much larger file for no benefit to the planned analysis.")
    p.add_argument("--inspect", action="store_true",
                   help="Download only the metadata header, print sample titles and "
                        "characteristics, and show how the classifier would group "
                        "them. Writes nothing. RUN THIS FIRST.")
    p.add_argument("--classify-only", action="store_true",
                   help="Classify samples and write the audit, but do not parse the "
                        "expression matrix.")
    p.add_argument("--keep-raw", action="store_true",
                   help="Keep the downloaded series matrix. Default is to delete it "
                        "after extraction, per the project's filter-on-arrival policy; "
                        "the SHA-256 and URL are recorded in the summary either way.")
    p.add_argument("--force", action="store_true",
                   help="Write cohorts even if group counts disagree with the "
                        "published composition. Only use after --inspect explains why.")
    p.add_argument("--emit-clock-probes", action="store_true",
                   help="Also write per-sample betas for probes needed by epigenetic "
                        "clocks, so age can be estimated when metadata is missing.")
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> Path:
    """Resume-capable download, matching data/acquire_external_cohorts.py."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        LOGGER.info("already present: %s (%.1f MB)", target,
                    target.stat().st_size / 1e6)
        return target
    LOGGER.info("downloading %s", url)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        seen = 0
        while chunk := response.read(4 * 1024 * 1024):
            handle.write(chunk)
            seen += len(chunk)
            if total:
                LOGGER.info("  %.0f%% (%.1f/%.1f MB)", 100 * seen / total,
                            seen / 1e6, total / 1e6)
    partial.replace(target)
    LOGGER.info("wrote %s (%.1f MB)", target, target.stat().st_size / 1e6)
    return target


def read_metadata(path: Path) -> tuple[dict, int]:
    """Header lines of a GEO series matrix, plus the line index where data starts."""
    metadata: dict[str, list[str]] = {}
    data_line = None
    with gzip.open(path, "rt", errors="replace") as handle:
        for index, raw in enumerate(handle):
            line = raw.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                data_line = index
                break
            if line.startswith("!"):
                parts = line.split("\t")
                key = parts[0].lstrip("!")
                values = [v.strip().strip('"') for v in parts[1:]]
                metadata.setdefault(key, []).extend(values)
    if data_line is None:
        raise SystemExit(f"{path} has no !series_matrix_table_begin marker")
    return metadata, data_line


def classify(metadata: dict) -> pd.DataFrame:
    """One row per sample: accession, the text used, and the assigned group."""
    accessions = metadata.get("Sample_geo_accession", [])
    if not accessions:
        raise SystemExit("no !Sample_geo_accession lines found")

    fields = [key for key in metadata
              if key.startswith("Sample_title")
              or key.startswith("Sample_characteristics")
              or key.startswith("Sample_source_name")]
    blobs = [""] * len(accessions)
    for key in fields:
        values = metadata[key]
        for i in range(min(len(values), len(blobs))):
            blobs[i] = f"{blobs[i]} | {values[i]}".strip(" |")

    groups = []
    for blob in blobs:
        lowered = blob.lower()
        assigned = "unclassified"
        for name, pattern in GROUP_PATTERNS:
            if re.search(pattern, lowered):
                assigned = name
                break
        groups.append(assigned)

    return pd.DataFrame({"geo_accession": accessions,
                         "descriptor": blobs,
                         "group": groups})


def check_composition(counts: Counter) -> list[str]:
    problems = []
    for group, expected in EXPECTED.items():
        found = counts.get(group, 0)
        if abs(found - expected) > TOLERANCE * expected:
            problems.append(
                f"{group}: found {found}, published composition says ~{expected}")
    return problems


def summarise_group(path: Path, data_line: int, columns: list[str],
                    sample_order: list[str], probes: set[str] | None) -> pd.DataFrame:
    """Median beta across the named samples, streaming the matrix once."""
    wanted = [sample_order.index(c) for c in columns]
    rows: list[tuple] = []
    with gzip.open(path, "rt", errors="replace") as handle:
        for index, raw in enumerate(handle):
            if index <= data_line + 1:          # skip header + column-name row
                continue
            if raw.startswith("!series_matrix_table_end"):
                break
            parts = raw.rstrip("\n").split("\t")
            probe = parts[0].strip().strip('"')
            if probes is not None and probe not in probes:
                continue
            values = np.array(
                [_to_float(parts[1 + i]) if 1 + i < len(parts) else np.nan
                 for i in wanted], dtype=float)
            observed = int(np.isfinite(values).sum())
            if observed == 0:
                continue
            rows.append((probe, float(np.nanmedian(values)), observed))

    frame = pd.DataFrame(rows, columns=["probeID", "Median_Beta", "n_samples_observed"])
    clipped = np.clip(frame["Median_Beta"].to_numpy(dtype=float), EPSILON, 1 - EPSILON)
    frame["M_Value_Target"] = np.log2(clipped / (1.0 - clipped))
    frame["Binary_State_Target"] = (frame["Median_Beta"] > 0.5).astype(np.int8)
    return frame[["probeID", "Median_Beta", "M_Value_Target",
                  "Binary_State_Target", "n_samples_observed"]]


def _to_float(token: str) -> float:
    token = token.strip().strip('"')
    if not token or token.upper() in {"NA", "NAN", "NULL", "NONE"}:
        return np.nan
    try:
        return float(token)
    except ValueError:
        return np.nan


def load_probe_subset(path: Path) -> set[str] | None:
    if not path.is_file():
        LOGGER.warning("%s not found -- keeping all probes", path)
        return None
    head = pd.read_csv(path, nrows=0)
    column = "probeID" if "probeID" in head.columns else head.columns[0]
    probes = set(pd.read_csv(path, usecols=[column])[column].astype(str))
    LOGGER.info("restricting to %d probes from %s", len(probes), path.name)
    return probes


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    raw = args.raw_dir / f"{ACCESSION}_series_matrix.txt.gz"
    download(args.url, raw)
    metadata, data_line = read_metadata(raw)
    samples = classify(metadata)
    counts = Counter(samples["group"])

    if args.inspect:
        print(f"\n{ACCESSION}: {len(samples)} samples\n")
        print("group assignment")
        for group, n in counts.most_common():
            print(f"  {group:<18} {n:>4}")
        problems = check_composition(counts)
        if problems:
            print("\n  DISAGREES WITH PUBLISHED COMPOSITION:")
            for problem in problems:
                print(f"    {problem}")
            print("  -> the patterns in GROUP_PATTERNS do not match this series'")
            print("     wording. Read the descriptors below and edit them.")
        else:
            print("\n  counts are consistent with the published composition")
        print("\nfirst descriptor per group (this is what the patterns matched on)")
        for group in counts:
            example = samples[samples["group"] == group]["descriptor"].iloc[0]
            print(f"\n  [{group}]\n    {example[:400]}")
        unclassified = samples[samples["group"] == "unclassified"]
        if not unclassified.empty:
            print(f"\n{len(unclassified)} UNCLASSIFIED samples, first five descriptors:")
            for text in unclassified["descriptor"].head(5):
                print(f"    {text[:300]}")
        age_keys = [k for k in metadata if "characteristics" in k.lower()]
        print(f"\ncharacteristics fields present: {age_keys if age_keys else 'none'}")
        print("\nWrites nothing. Re-run without --inspect once the groups look right.")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(samples, args.output_dir / "sample_classification_audit.csv")

    problems = check_composition(counts)
    if problems and not args.force:
        raise SystemExit(
            "group counts disagree with the published composition:\n  "
            + "\n  ".join(problems)
            + "\n\nThe classifier's patterns probably do not match this series' "
              "wording. Run --inspect, read the descriptors, edit GROUP_PATTERNS. "
              "Pass --force only if you have confirmed the counts are right and the "
              "published numbers are what is stale.")
    if args.classify_only:
        print("classification written; matrix not parsed")
        return 0

    sample_order = metadata["Sample_geo_accession"]
    probes = None if args.all_probes else load_probe_subset(args.probe_source)

    written = {}
    for group in ("healthy_normal", "adjacent_normal"):
        columns = samples.loc[samples["group"] == group, "geo_accession"].tolist()
        if not columns:
            LOGGER.warning("no samples in group %s", group)
            continue
        LOGGER.info("summarising %s (%d samples)", group, len(columns))
        frame = summarise_group(raw, data_line, columns, sample_order, probes)
        if frame.empty:
            LOGGER.warning("%s produced no probes", group)
            continue
        target = args.output_dir / f"gse69914_{group}.csv"
        atomic_csv(frame, target)
        written[group] = {"file": str(target), "samples": len(columns),
                          "probes": int(len(frame)),
                          "bytes": int(target.stat().st_size)}
        LOGGER.info("  -> %s (%d probes)", target, len(frame))

    digest = sha256_file(raw)
    raw_bytes = raw.stat().st_size
    if not args.keep_raw:
        raw.unlink()
        LOGGER.info("removed the raw series matrix; SHA-256 recorded in the summary")

    atomic_json(
        {
            "analysis": "GSE69914 healthy-donor breast methylation cohort",
            "purpose": ("independent validation target; tests whether the "
                        "tumour-adjacent origin of the training targets materially "
                        "contaminates them"),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "accession": ACCESSION,
            "source_url": args.url,
            "source_sha256": digest,
            "source_bytes": raw_bytes,
            "raw_retained": bool(args.keep_raw),
            "platform": "Illumina HumanMethylation450",
            "join_key": ("probeID -- platform-stable, so no liftover is required "
                         "despite GEO reporting hg19 coordinates"),
            "group_counts": dict(counts),
            "published_composition": EXPECTED,
            "cohorts": written,
            "probe_subset_source": (None if args.all_probes
                                    else str(args.probe_source)),
            "target_convention": ("median beta across observed samples; beta clipped "
                                  "to [1e-4, 1-1e-4]; M = log2(b/(1-b)); binary = "
                                  "beta > 0.5 -- identical to "
                                  "data/build_training_data.py and "
                                  "data/build_tcga_tumor_cohort.py"),
            "intended_use": ("VALIDATION ONLY. A 50-sample median is noisier than the "
                             "existing 97-sample median; substituting it into training "
                             "would trade a known bias for added variance."),
            "confounds_to_state": {
                "cell_composition": ("healthy-donor and tumour-adjacent breast differ "
                                     "in epithelial/stromal/adipose fractions; a beta "
                                     "difference is not automatically a field effect"),
                "age": ("GEO reports incomplete metadata for this series and supplies "
                        "no age field. TCGA normals come from cancer patients and skew "
                        "older, and methylation is strongly age-dependent."),
            },
            "why_both_arms_matter": ("this series contains healthy and tumour-adjacent "
                                     "normals on the same platform in the same study, "
                                     "so the healthy-vs-adjacent contrast is free of "
                                     "the batch confound that would dominate a "
                                     "comparison against TCGA"),
        },
        args.output_dir / "healthy_cohort_summary.json",
    )

    print()
    print("=" * 70)
    for group, info in written.items():
        print(f"{group:<18} {info['samples']:>4} samples  {info['probes']:>7,} probes  "
              f"{info['bytes']/1e6:>6.1f} MB")
    print(f"\noutput: {args.output_dir}")
    print("\nBoth files use the same schema as data/build_tcga_tumor_cohort.py, so")
    print("scripts/21_tcga_tumor_domain_shift.py scores them with --tumor-dir pointed")
    print("here. Report the healthy-vs-adjacent contrast within this series first;")
    print("it is the only comparison free of a platform confound.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
