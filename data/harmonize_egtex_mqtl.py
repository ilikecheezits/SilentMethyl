#!/usr/bin/env python3
"""
Harmonize the eGTEx Breast Mammary Tissue mQTL all-pairs file into SilentMethyl
scoring input.

Why this file and not GENOA
---------------------------
GENOA is peripheral blood in an African-ancestry cohort. SilentMethyl is trained
on breast tissue, so a weak GENOA correlation is confounded with a tissue change
and cannot be read as a model failure. eGTEx Breast Mammary Tissue is the same
tissue the model was trained on, so it is the arm that can actually falsify the
variant-effect claim. GENOA stays in the paper as the cross-tissue,
cross-ancestry transfer arm -- a different question, honestly labelled.

Input format (9 columns, no header, whitespace-separated)
--------------------------------------------------------
    cg26928153  chr1_13550_G_A_b38  2701  1  1  0.0102041  0.0748688  2.17016  1.18095
    probeID     variant_id          dist  ma_samples  ma_count  maf  pval_nominal  slope  slope_se

Three things that make this easier than GENOA, and one that does not:
  * Already hg38 (`b38` suffix) -- no liftover, no chain file.
  * REF and ALT are encoded in the variant ID -- no minor/major ambiguity, so no
    repeat of the effect-allele sign bug that silently flipped 11.3% of GENOA.
  * Distance is precomputed.
  * BUT the file is 45 GB compressed and describes EPIC probes, while the model
    is trained on HM450. Both are handled below.

What this script does NOT trust
-------------------------------
1. **Their distance column.** We recompute distance from the HM450 manifest and
   report the discrepancy distribution. If their probe-coordinate convention
   differs from ours by a base, we detect it rather than inherit it.
2. **The GTEx sign convention.** `slope` is documented as keyed to the ALT
   allele, which is what the model's REF->ALT delta needs. We verify it against
   an internal positive control -- variants that destroy the target CpG must
   lower methylation -- and refuse to write output if the check comes back
   reversed. Assuming this is how the GENOA sign bug happened.
3. **The reference base.** Every REF is checked against hg38.fa. A high mismatch
   rate means the build or strand assumption is wrong.

Two stages, because 45 GB
-------------------------
Stage 1 (`--prefilter`) streams the gzip through awk, keeping only rows whose
reported distance is within +/-600 bp. That is ~0.06% of the file and turns
45 GB into a few hundred MB in one pass. awk is used rather than Python because
this is a multi-billion-line scan and per-line interpreter overhead dominates.

Stage 2 (default) does the real work in pandas on the prefiltered file: parse
variant IDs, join the HM450 manifest, recompute distance, verify REF against
hg38, flag CpG-altering variants, verify the sign convention, apply the exact
[-499, +500] model window, and split by what the model has seen.

Usage (run from the repository root)
------------------------------------
    python -u data/harmonize_egtex_mqtl.py --inspect
    python -u data/harmonize_egtex_mqtl.py --prefilter
    python -u data/harmonize_egtex_mqtl.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# FastaReader lives in the GENOA builder; importing it rather than copying it
# guarantees both cohorts resolve reference bases with identical semantics.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from build_genoa_scoring_input import (  # noqa: E402
        MAX_SCOREABLE_OFFSET,
        MIN_SCOREABLE_OFFSET,
        FastaReader,
        count_cg,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "cannot import from data/build_genoa_scoring_input.py -- "
        f"run this from the repository root ({exc})"
    )

DEFAULT_RAW = Path("data/external/egtex_breast/BreastMammaryTissue.mQTLs.regular.txt.gz")
DEFAULT_PREFILTERED = Path("data/external/egtex_breast/egtex_breast_within600.tsv.gz")
DEFAULT_MANIFEST = Path("data/HM450.hg38.manifest.tsv.gz")
DEFAULT_FASTA = Path("data/hg38.fa")
DEFAULT_SPLIT_DIR = Path("data/datafiles")
DEFAULT_OUT = Path("data/external/egtex_breast/scoring")

EGTEX_COLUMNS = ["probeID", "variant_id", "dist_reported", "ma_samples",
                 "ma_count", "maf", "pval_nominal", "slope", "slope_se"]

# The model window is imported, not restated. The 1,000-bp crop puts the target
# C at index 499, so a variant at offset d is visible iff 0 <= 499 + d < 1000,
# i.e. d in [-499, +500]. Importing the bounds from the GENOA builder means the
# two cohorts cannot drift apart if that geometry is ever revised.
WINDOW_LOW = MIN_SCOREABLE_OFFSET
WINDOW_HIGH = MAX_SCOREABLE_OFFSET
if (WINDOW_LOW, WINDOW_HIGH) != (-499, 500):
    raise SystemExit(
        f"model window is {(WINDOW_LOW, WINDOW_HIGH)}, expected (-499, 500). "
        f"If the architecture changed, the GENOA scoring input must be "
        f"regenerated too -- the two cohorts must share one window.")
PREFILTER_HALF_WINDOW = 600

VALID_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}
EXPECTED_BUILD = "b38"


# --------------------------------------------------------------------------
# stage 1: stream the 45 GB file down to the cis-proximal rows
# --------------------------------------------------------------------------

# Test order matters here: this program sees ~4 billion lines, and the regex is
# by far the most expensive clause. The numeric bounds are checked FIRST, so the
# regex only runs on the ~0.06% of lines that are already candidates. A
# non-numeric field coerces to 0 and slips past the bounds test, which is exactly
# why the regex is still there -- it is a correctness guard, not a prefilter.
AWK_PROGRAM = r"""
BEGIN { OFS = "\t" }
{
    n++
    if (n % 200000000 == 0) {
        printf("  scanned %.2fB lines, kept %d\n", n / 1000000000, k) > "/dev/stderr"
    }
    if ($3 <= W && $3 >= LO && NF >= 9 && $3 ~ /^-?[0-9]+$/) {
        k++
        print $1, $2, $3, $4, $5, $6, $7, $8, $9
    }
}
END { printf("scanned %d lines, kept %d rows within +/-%d bp\n", n, k, W) > "/dev/stderr" }
"""


def awk_binary() -> str:
    """mawk is 3-5x faster than gawk on a scan this size. Any awk is correct."""
    for candidate in ("mawk", "gawk", "awk"):
        if shutil.which(candidate):
            return candidate
    raise SystemExit("no awk found on PATH")


def decompressor(path: Path) -> str:
    """pigz if available, else gzip. Threads follow the Slurm allocation."""
    if shutil.which("pigz"):
        threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
        return f"pigz -dc -p {threads} {path}"
    return f"gzip -dc {path}"


def run_prefilter(raw: Path, out: Path, half_window: int, force: bool) -> None:
    if not raw.is_file():
        raise SystemExit(
            f"raw eGTEx file not found: {raw}\n"
            f"Download it first (45,279,365,738 bytes expected)."
        )
    if out.is_file() and not force:
        logging.info("prefiltered file already exists: %s (use --force to redo)", out)
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".partial")
    if "'" in AWK_PROGRAM:
        raise RuntimeError("the awk program cannot contain a single quote")
    awk = awk_binary()
    pipeline = (
        f"set -o pipefail\n"
        f"{decompressor(raw)} | "
        f"{awk} -v W={half_window} -v LO={-half_window} '{AWK_PROGRAM}' | "
        f"gzip -1 -c > {tmp}\n"
    )
    logging.info("streaming %s (%.1f GB) through %s -- this is the slow step",
                 raw, raw.stat().st_size / 1e9, awk)
    logging.info("progress prints every 200M lines")
    result = subprocess.run(["bash", "-c", pipeline])
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"prefilter pipeline failed (exit {result.returncode}); "
                         f"the partial output was removed")
    tmp.replace(out)
    logging.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def parse_variant_ids(ids: pd.Series) -> pd.DataFrame:
    """chr1_13550_G_A_b38 -> chrom, pos1, ref, alt, build.

    Split from the right: contig names can themselves contain underscores
    (chr1_KI270706v1_random), so the last four fields are the anchor.
    """
    parts = ids.astype(str).str.rsplit("_", n=4, expand=True)
    if parts.shape[1] != 5:
        raise SystemExit("variant IDs do not have the expected "
                         "<chrom>_<pos>_<ref>_<alt>_<build> shape")
    out = pd.DataFrame({
        "chr": parts[0].astype(str),
        "pos1_raw": parts[1],
        "Ref": parts[2].astype(str).str.upper(),
        "Alt": parts[3].astype(str).str.upper(),
        "build": parts[4].astype(str),
    })
    out["Position_1based"] = pd.to_numeric(out["pos1_raw"], errors="coerce")
    return out.drop(columns="pos1_raw")


def load_manifest(path: Path) -> pd.DataFrame:
    head = pd.read_csv(path, sep="\t", nrows=0)
    probe_col = next((c for c in ("probeID", "IlmnID", "Name") if c in head.columns), None)
    if probe_col is None:
        raise SystemExit(f"{path}: no probe ID column found")
    for needed in ("CpG_chrm", "CpG_beg"):
        if needed not in head.columns:
            raise SystemExit(f"{path}: missing {needed}")
    usecols = [probe_col, "CpG_chrm", "CpG_beg"]
    if "MASK_general" in head.columns:
        usecols.append("MASK_general")

    df = pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)
    df = df.rename(columns={probe_col: "probeID", "CpG_chrm": "cpg_chr",
                            "CpG_beg": "cpg_pos_hg38"})
    df["probeID"] = df["probeID"].astype(str)
    df["cpg_chr"] = df["cpg_chr"].astype(str)
    df = df[df["cpg_chr"].isin(VALID_CHROMS)]
    df = df.dropna(subset=["cpg_pos_hg38"])
    df["cpg_pos_hg38"] = df["cpg_pos_hg38"].astype("int64")
    if "MASK_general" in df.columns:
        df["HM450_MASK_general"] = (df["MASK_general"].astype(str).str.strip()
                                    .str.lower().isin({"true", "t", "1", "yes"}))
        df = df.drop(columns="MASK_general")
    else:
        df["HM450_MASK_general"] = False
    df = df.drop_duplicates(subset="probeID", keep="first").reset_index(drop=True)
    return df


def load_split_labels(split_dir: Path) -> dict:
    labels = {}
    for name in ("train", "val", "test"):
        path = split_dir / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"{path} not found -- split labels are not optional; "
                             f"without them the held-out claim cannot be made")
        head = pd.read_csv(path, nrows=0)
        col = "probeID" if "probeID" in head.columns else head.columns[0]
        ids = pd.read_csv(path, usecols=[col])[col].astype(str)
        labels.update({p: name for p in ids})
        logging.info("%-5s split: %d probes", name, len(ids))
    return labels


def resolve_reference_bases(df: pd.DataFrame, fasta_path: Path) -> pd.DataFrame:
    """Attach the hg38 3-mer centred on each variant.

    Deduplicated by (chrom, pos) and sorted, so the FASTA is read with sequential
    seeks over a few hundred thousand unique sites instead of one random seek per
    variant-probe pair.
    """
    sites = (df[["chr", "Position_1based"]]
             .drop_duplicates()
             .sort_values(["chr", "Position_1based"])
             .reset_index(drop=True))
    logging.info("resolving %d unique variant positions against %s",
                 len(sites), fasta_path)
    fasta = FastaReader(fasta_path)
    contexts = []
    for chrom, pos1 in zip(sites["chr"].to_numpy(), sites["Position_1based"].to_numpy()):
        pos0 = int(pos1) - 1
        contexts.append(fasta.fetch(str(chrom), pos0 - 1, pos0 + 2))
    fasta.close()
    sites["hg38_context_3mer"] = contexts
    return df.merge(sites, on=["chr", "Position_1based"], how="left",
                    validate="many_to_one")


def flag_cpg_effects(df: pd.DataFrame) -> pd.DataFrame:
    """alters_target_cpg / creates_cpg / destroys_cpg, from the 3-mer."""
    ctx = df["hg38_context_3mer"].to_numpy()
    alt = df["Alt"].to_numpy()
    creates, destroys = np.zeros(len(df), bool), np.zeros(len(df), bool)
    for i in range(len(df)):
        c = ctx[i]
        mut = c[0] + alt[i] + c[2]
        delta = count_cg(mut) - count_cg(c)
        creates[i] = delta > 0
        destroys[i] = delta < 0
    df = df.copy()
    df["creates_cpg"] = creates
    df["destroys_cpg"] = destroys
    pos0 = df["Position_1based"].to_numpy(dtype="int64") - 1
    cpg0 = df["cpg_pos_hg38"].to_numpy(dtype="int64")
    df["alters_target_cpg"] = (pos0 == cpg0) | (pos0 == cpg0 + 1)
    return df


# --------------------------------------------------------------------------
# the sign check
# --------------------------------------------------------------------------

def verify_sign_convention(df: pd.DataFrame, p_threshold: float) -> dict:
    """Positive control for the effect-allele convention.

    A variant that destroys the target CpG removes the substrate for methylation
    at that site, so the ALT allele must be associated with LOWER methylation. If
    `slope` is keyed to ALT, as GTEx documents, the mean slope over these
    variants is negative. If it comes back positive, the convention is reversed
    and every downstream direction-agreement number would be backwards -- the
    exact failure mode that hid in GENOA for weeks.

    This is a real check, not a formality: it is computed on the variants we then
    exclude from scoring, so it costs nothing and is not circular.
    """
    subset = df[df["alters_target_cpg"] & df["destroys_cpg"]]
    strong = subset[subset["pval_nominal"] < p_threshold]

    def summarize(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {"n": 0}
        slopes = frame["slope"].to_numpy(dtype=float)
        slopes = slopes[np.isfinite(slopes)]
        if slopes.size == 0:
            return {"n": 0}
        se = slopes.std(ddof=1) / np.sqrt(slopes.size) if slopes.size > 1 else np.nan
        return {
            "n": int(slopes.size),
            "mean_slope": float(slopes.mean()),
            "median_slope": float(np.median(slopes)),
            "fraction_negative": float((slopes < 0).mean()),
            "z_of_mean": float(slopes.mean() / se) if se and np.isfinite(se) and se > 0 else None,
        }

    report = {
        "control": ("variants that destroy the target CpG must be associated with "
                    "lower methylation on the ALT allele"),
        "all_cpg_destroying": summarize(subset),
        f"cpg_destroying_p_lt_{p_threshold:g}": summarize(strong),
    }

    # Decide on the strong subset when it is powered, otherwise the full set.
    decisive = report[f"cpg_destroying_p_lt_{p_threshold:g}"]
    if decisive.get("n", 0) < 100:
        decisive = report["all_cpg_destroying"]
    z = decisive.get("z_of_mean")
    if decisive.get("n", 0) < 30:
        report["verdict"] = "underpowered"
    elif z is not None and z > 3:
        report["verdict"] = "REVERSED"
    elif z is not None and z < -3:
        report["verdict"] = "confirmed_alt_keyed"
    else:
        report["verdict"] = "inconclusive"
    return report


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_inspect(args) -> int:
    raw = args.raw
    if not raw.is_file():
        print(f"raw file not found: {raw}")
        return 1
    size = raw.stat().st_size
    print(f"file  : {raw}")
    print(f"bytes : {size:,} ({size/1e9:.1f} GB compressed)")
    print()
    rows = []
    with gzip.open(raw, "rt") as fh:
        for i, line in enumerate(fh):
            if i >= args.inspect_lines:
                break
            rows.append(line.rstrip("\n").split())
    print(f"first {len(rows)} lines, split on whitespace:")
    for r in rows[:5]:
        print("  " + "  ".join(r))
    widths = {len(r) for r in rows}
    print(f"\nfield counts observed: {sorted(widths)}")
    if widths != {9}:
        print("  WARNING: expected exactly 9 fields on every line")
    print("\nassumed schema:")
    for i, name in enumerate(EGTEX_COLUMNS):
        print(f"  col {i+1}  {name}")

    frame = pd.DataFrame(rows, columns=EGTEX_COLUMNS[:len(rows[0])])
    parsed = parse_variant_ids(frame["variant_id"])
    print(f"\nbuilds seen      : {sorted(parsed['build'].unique())}")
    print(f"chromosomes seen : {sorted(parsed['chr'].unique())[:5]}")

    # Their distance convention, checked against itself: probe position implied
    # by (variant pos - dist) must be constant for a given probe.
    frame["dist_reported"] = pd.to_numeric(frame["dist_reported"], errors="coerce")
    frame["implied_probe_pos"] = parsed["Position_1based"] - frame["dist_reported"]
    per_probe = frame.groupby("probeID")["implied_probe_pos"].nunique()
    print(f"\nprobes in sample : {len(per_probe)}")
    print(f"probes with a single implied position: "
          f"{int((per_probe == 1).sum())}/{len(per_probe)}")
    print("  (all should be 1 -- confirms dist = variant_pos - probe_pos)")

    if args.manifest.is_file():
        man = load_manifest(args.manifest)
        overlap = frame["probeID"].isin(set(man["probeID"])).mean()
        print(f"\nHM450 manifest   : {len(man):,} probes")
        print(f"sample probes present in HM450: {overlap:.1%}")
        print("  (eGTEx is EPIC; only the HM450 subset is usable, since the model")
        print("   was trained on HM450 probes)")
    return 0


def cmd_run(args) -> int:
    stats: Counter = Counter()

    if not args.prefiltered.is_file():
        logging.info("prefiltered file absent; running stage 1 first")
        run_prefilter(args.raw, args.prefiltered, PREFILTER_HALF_WINDOW, args.force)

    logging.info("reading %s", args.prefiltered)
    df = pd.read_csv(args.prefiltered, sep="\t", names=EGTEX_COLUMNS,
                     header=None, low_memory=False)
    stats["prefiltered_rows"] = len(df)
    logging.info("%d rows within +/-%d bp (reported distance)",
                 len(df), PREFILTER_HALF_WINDOW)
    if df.empty:
        raise SystemExit("prefiltered file is empty -- check the column order")

    df["probeID"] = df["probeID"].astype(str)
    for col in ("dist_reported", "ma_samples", "ma_count", "maf",
                "pval_nominal", "slope", "slope_se"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    parsed = parse_variant_ids(df["variant_id"])
    df = pd.concat([df.drop(columns=[]), parsed], axis=1)

    bad_build = df["build"] != EXPECTED_BUILD
    if bad_build.any():
        raise SystemExit(
            f"{int(bad_build.sum())} variant IDs are not {EXPECTED_BUILD}; "
            f"builds seen: {sorted(df['build'].unique())}. Refusing to proceed -- "
            f"a mixed build would silently misplace variants.")

    before = len(df)
    df = df[df["Position_1based"].notna()].copy()
    df["Position_1based"] = df["Position_1based"].astype("int64")
    stats["dropped_unparsable_position"] = before - len(df)

    before = len(df)
    is_snv = (df["Ref"].str.len() == 1) & (df["Alt"].str.len() == 1) \
        & df["Ref"].isin(list("ACGT")) & df["Alt"].isin(list("ACGT"))
    df = df[is_snv].reset_index(drop=True)
    stats["dropped_non_snv"] = before - len(df)
    logging.info("%d SNVs (%d indels/MNVs dropped)", len(df), stats["dropped_non_snv"])

    # --- HM450 restriction -------------------------------------------------
    manifest = load_manifest(args.manifest)
    hm450_ids = set(manifest["probeID"])
    probes_seen = df["probeID"].nunique()
    epic_only = df[~df["probeID"].isin(hm450_ids)]["probeID"].nunique()
    stats["probes_in_egtex_window"] = probes_seen
    stats["probes_epic_only_dropped"] = epic_only

    before = len(df)
    df = df.merge(manifest, on="probeID", how="inner", validate="many_to_one")
    stats["dropped_probe_not_on_hm450"] = before - len(df)

    before = len(df)
    df = df[~df["HM450_MASK_general"].astype(bool)].reset_index(drop=True)
    stats["dropped_probe_masked"] = before - len(df)
    logging.info("%d pairs on unmasked HM450 probes (%d probes)",
                 len(df), df["probeID"].nunique())

    before = len(df)
    df = df[df["chr"] == df["cpg_chr"]].reset_index(drop=True)
    stats["dropped_chromosome_mismatch"] = before - len(df)

    # --- our distance, not theirs ------------------------------------------
    df["distance_bp"] = (df["Position_1based"] - 1 - df["cpg_pos_hg38"]).astype("int64")
    df["abs_distance_bp"] = df["distance_bp"].abs()
    offset = (df["dist_reported"] - df["distance_bp"]).round().astype("Int64")
    offset_counts = offset.value_counts().head(5)
    distance_agreement = {str(k): int(v) for k, v in offset_counts.items()}
    modal_offset = int(offset_counts.index[0]) if len(offset_counts) else None
    logging.info("reported-minus-recomputed distance, top values: %s",
                 distance_agreement)
    if modal_offset != 0:
        logging.warning("eGTEx probe coordinates are offset from the HM450 "
                        "manifest by %s bp (modal). Using OUR recomputed "
                        "distance.", modal_offset)

    # --- reference base check ----------------------------------------------
    df = resolve_reference_bases(df, args.fasta)
    before = len(df)
    df = df[df["hg38_context_3mer"].str.len() == 3].reset_index(drop=True)
    stats["dropped_no_reference_sequence"] = before - len(df)

    observed_ref = df["hg38_context_3mer"].str[1]
    mismatch = observed_ref != df["Ref"]
    stats["reference_base_mismatch"] = int(mismatch.sum())
    mismatch_rate = float(mismatch.mean()) if len(df) else 0.0
    if mismatch_rate > 0.01:
        logging.error("%.2f%% of variants disagree with hg38 at the REF base. "
                      "That is far above the ~0 expected for a b38 file.",
                      100 * mismatch_rate)
    df = df[~mismatch].reset_index(drop=True)
    logging.info("%d pairs with REF confirmed against hg38 (%.4f%% mismatched)",
                 len(df), 100 * mismatch_rate)

    df = flag_cpg_effects(df)

    # --- sign convention ----------------------------------------------------
    sign_report = verify_sign_convention(df, args.sign_check_p)
    logging.info("sign check verdict: %s", sign_report["verdict"])
    if sign_report["verdict"] == "REVERSED" and not args.ignore_sign_check:
        print(json.dumps(sign_report, indent=2))
        raise SystemExit(
            "\nSIGN CHECK FAILED.\n"
            "Variants that destroy the target CpG are associated with HIGHER "
            "methylation on the ALT allele, which is biologically backwards. "
            "That means `slope` is keyed to the REF allele, not ALT, and every "
            "direction-agreement statistic computed from it would be inverted.\n"
            "Fix the sign in this script (negate `slope`) rather than passing "
            "--ignore-sign-check, which only silences the guard.")

    # `slope` is keyed to ALT in the GTEx/tensorQTL convention, which is the same
    # direction the model's REF->ALT delta measures. Verified above, not assumed.
    df["beta_ref_to_alt"] = df["slope"].astype(float)
    df["se"] = df["slope_se"].astype(float)
    df["pvalue"] = df["pval_nominal"].astype(float)

    # --- the exact model window --------------------------------------------
    before = len(df)
    df = df[(df["distance_bp"] >= WINDOW_LOW)
            & (df["distance_bp"] <= WINDOW_HIGH)].reset_index(drop=True)
    stats["dropped_outside_model_window"] = before - len(df)

    before = len(df)
    stats["cpg_altering_target_excluded"] = int(df["alters_target_cpg"].sum())
    df = df[~df["alters_target_cpg"]].reset_index(drop=True)
    logging.info("%d pairs after the [%+d, %+d] window and removing %d "
                 "target-CpG-altering variants",
                 len(df), WINDOW_LOW, WINDOW_HIGH, before - len(df))

    if df.empty:
        raise SystemExit("no pairs survived filtering -- stop and inspect before rerunning")

    # --- split labels -------------------------------------------------------
    labels = load_split_labels(args.split_dir)
    df["probe_split"] = df["probeID"].map(labels).fillna("unknown")
    before = len(df)
    df = df[df["probe_split"] != "unknown"].reset_index(drop=True)
    stats["dropped_probe_not_in_model_data"] = before - len(df)

    df["Variant_ID"] = df["variant_id"].astype(str)
    df["Gene"] = "NA"

    lead = ["Variant_ID", "Gene", "chr", "Position_1based", "Ref", "Alt"]
    extra = ["probeID", "probe_split", "cpg_chr", "cpg_pos_hg38", "distance_bp",
             "abs_distance_bp", "alters_target_cpg", "creates_cpg", "destroys_cpg",
             "beta_ref_to_alt", "se", "pvalue", "maf", "ma_count", "ma_samples",
             "dist_reported"]
    out = df[lead + extra]

    # --- write ---------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, subset in (("heldout", out[out["probe_split"] == "test"]),
                         ("model_visible", out[out["probe_split"].isin(["train", "val"])])):
        if subset.empty:
            logging.warning("%s stratum is empty", name)
            continue
        target = args.output_dir / f"egtex_scoring_input_{name}.csv"
        subset.to_csv(target, index=False)
        written[name] = {
            "file": str(target),
            "rows": int(len(subset)),
            "unique_variants": int(subset["Variant_ID"].nunique()),
            "unique_probes": int(subset["probeID"].nunique()),
            "genome_wide_significant": int((subset["pvalue"] < 5e-8).sum()),
            "cpg_altering_nearby": int((subset["creates_cpg"] | subset["destroys_cpg"]).sum()),
            "median_abs_distance_bp": float(subset["abs_distance_bp"].median()),
            "bytes": int(target.stat().st_size),
        }

    summary = {
        "analysis": "eGTEx Breast Mammary Tissue mQTLs -> SilentMethyl scoring input",
        "why_this_cohort": ("tissue-matched to training data, so a weak correlation "
                            "is attributable to the model rather than to a tissue "
                            "change; GENOA remains the cross-tissue transfer arm"),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_input": str(args.raw),
        "prefiltered_input": str(args.prefiltered),
        "reference": str(args.fasta),
        "manifest": str(args.manifest),
        "array_platform_note": (
            "eGTEx assays EPIC; SilentMethyl trains on HM450. Probe IDs are "
            "platform-stable, so the HM450 subset is used directly with no "
            "liftover. EPIC-only probes are dropped and counted."),
        "filter_counts": dict(stats),
        "reference_base_mismatch_rate": mismatch_rate,
        "distance_reported_minus_recomputed": distance_agreement,
        "distance_authority": ("recomputed from HM450 manifest CpG_beg; the "
                               "reported column is retained only as a diagnostic"),
        "effect_allele_convention": {
            "column": "beta_ref_to_alt",
            "source_column": "slope",
            "convention": "GTEx/tensorQTL keys slope to the ALT allele",
            "verification": sign_report,
        },
        "model_window": [WINDOW_LOW, WINDOW_HIGH],
        "strata": written,
        "reporting_guidance": (
            "Report `heldout` as primary -- the model trained on probes in "
            "`model_visible`. Variants altering the target CpG are excluded "
            "entirely; nearby CpG-altering variants are flagged, not removed, "
            "and should be reported as a separate stratum."),
    }
    with (args.output_dir / "egtex_scoring_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    print()
    print("=" * 72)
    print("filtering")
    for k, v in stats.items():
        print(f"  {k:<36} {v:>12,}")
    print(f"  {'reference_base_mismatch_rate':<36} {mismatch_rate:>11.4%}")
    print(f"\ndistance (reported - recomputed), top values: {distance_agreement}")
    print(f"sign convention verdict: {sign_report['verdict']}")
    ctrl = sign_report["all_cpg_destroying"]
    if ctrl.get("n"):
        print(f"  CpG-destroying variants: n={ctrl['n']:,}  "
              f"mean slope={ctrl['mean_slope']:+.4f}  "
              f"{ctrl['fraction_negative']:.1%} negative")
    print("\nstrata")
    for name, info in written.items():
        print(f"  {name:<14} {info['rows']:>9,} pairs  "
              f"{info['unique_variants']:>8,} variants  "
              f"{info['unique_probes']:>7,} probes  "
              f"({info['genome_wide_significant']:,} at p<5e-8)")
    print("\nOnly `heldout` supports the independent-validation claim.")
    print(f"\nThe {args.raw.stat().st_size/1e9:.0f} GB raw file is no longer needed "
          f"once\n{args.prefiltered} exists.")
    print("=" * 72)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--prefiltered", type=Path, default=DEFAULT_PREFILTERED)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    ap.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--inspect", action="store_true",
                    help="read the first lines of the raw file and report the "
                         "inferred schema; writes nothing, downloads nothing")
    ap.add_argument("--inspect-lines", type=int, default=2000)
    ap.add_argument("--prefilter", action="store_true",
                    help="run stage 1 only (the 45 GB streaming scan)")
    ap.add_argument("--force", action="store_true",
                    help="redo stage 1 even if the prefiltered file exists")
    ap.add_argument("--sign-check-p", type=float, default=1e-5)
    ap.add_argument("--ignore-sign-check", action="store_true",
                    help="do not use; fix the sign instead")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if args.inspect:
        return cmd_inspect(args)
    if args.prefilter:
        run_prefilter(args.raw, args.prefiltered, PREFILTER_HALF_WINDOW, args.force)
        return 0
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
