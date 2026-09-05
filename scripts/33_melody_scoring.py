#!/usr/bin/env python3
"""Score variant effects with Melody, mirroring the authors' own procedure.

Why this exists
---------------
A head-to-head against Melody is only meaningful if Melody is run the way its
authors run it. This reimplements their `SNPEffectDataset` and
`check_meqtl_batched_*` logic from `eqtl_pure_util.py` rather than inventing a
scoring scheme, so any difference in the result is a difference in the models
and not in how we drove them.

Their procedure, copied deliberately
------------------------------------
1. The 10-kb window is centred on the MIDPOINT of the CpG and the SNP, not on
   the CpG:            center = (CpG_start + SNP_start) // 2
2. The variant index is `SNP_start - fetch_start - 1` (their coordinates are
   1-based; the -1 converts).
3. The reference base is OVERWRITTEN with the stated REF allele rather than
   checked against the genome. We reproduce that, but we also count how often
   the genome disagrees and refuse to continue past a set rate -- a silent
   coordinate error would otherwise corrupt every prediction.
4. The effect is the SUM of per-position (ALT - REF) predictions over
   [CpG_start - margin, CpG_end + margin), divided by `cpg_number`.
   Their CSVs have CpG_start == CpG_end, so margin must be >= 1.
5. Pairs are skipped when the ALT allele is not a single base or when the SNP
   start and end differ. Both are their filters.

Two input conventions
---------------------
--format melody   their benchmark CSVs (meqtl/dataset/processed/GTEX/*.csv):
                  chrom, SNP_region_start/end, SNP_ref, SNP_alt,
                  CPG_region_start/end, effect_size
                  Use this to VALIDATE: GTEX_WholeBlood should reproduce their
                  published Pearson r of about 0.4158 over n = 1334.

--format ours     our scoring inputs (egtex_scoring_input_heldout.csv):
                  chr, Position_1based, Ref, Alt, probeID, cpg_pos_hg38, ...
                  Use this to COMPARE: identical pairs to SilentMethyl, so the
                  output drops straight into scripts/31.

Output carries `Predicted_Delta_M` so that scripts/31_transfer_discrimination.py
can consume it as another model without modification.

Usage
-----
    # validate against their published number first
    python -u scripts/33_melody_scoring.py --format melody \\
        --input-csv data/external/melody/Melody_repo/meqtl/dataset/processed/GTEX/GTEX_WholeBlood.csv \\
        --tracks GSM5652317_Blood-B-Z000000UB \\
        --report-correlation

    # then score our cohort
    python -u scripts/33_melody_scoring.py --format ours \\
        --input-csv data/external/egtex_multitissue/scoring/Lung/egtex_scoring_input_heldout.csv \\
        --tracks GSM5652354_Lung-Alveolar-Epithelial-Z000000T1,GSM5652335_Lung-Bronchus-Epithelial-Z000000QD
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

LOGGER = logging.getLogger("melody_scoring")

BASES = {"A": 0, "C": 1, "G": 2, "T": 3,
         "a": 0, "c": 1, "g": 2, "t": 3}
HALF_WINDOW = 5000          # their half_model_input_len default
REF_MISMATCH_ABORT = 0.05   # refuse to proceed past this rate


def one_hot(seq: str) -> np.ndarray:
    """(L, 4) float32, zeros for N -- their BASE_TO_ONEHOT convention."""
    out = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        j = BASES.get(base)
        if j is not None:
            out[i, j] = 1.0
    return out


def normalise(frame: pd.DataFrame, fmt: str) -> pd.DataFrame:
    """Bring either input convention to their column names."""
    if fmt == "melody":
        need = ["chrom", "SNP_region_start", "SNP_region_end", "SNP_ref",
                "SNP_alt", "CPG_region_start", "CPG_region_end"]
        missing = [c for c in need if c not in frame.columns]
        if missing:
            raise SystemExit(f"STOP: --format melody needs {missing}")
        out = frame.copy()
        if "effect_size" not in out.columns:
            out["effect_size"] = np.nan
        if "cpg_number" not in out.columns:
            out["cpg_number"] = 1
        return out

    need = ["chr", "Position_1based", "Ref", "Alt", "probeID", "cpg_pos_hg38"]
    missing = [c for c in need if c not in frame.columns]
    if missing:
        raise SystemExit(f"STOP: --format ours needs {missing}")
    out = pd.DataFrame({
        "chrom": frame["chr"].astype(str),
        "SNP_region_start": frame["Position_1based"].astype(int),
        "SNP_region_end": frame["Position_1based"].astype(int),
        "SNP_ref": frame["Ref"].astype(str),
        "SNP_alt": frame["Alt"].astype(str),
        "CPG_region_start": frame["cpg_pos_hg38"].astype(int),
        "CPG_region_end": frame["cpg_pos_hg38"].astype(int),
        "cpg_number": 1,
    })
    # Carry identifiers through so the output joins back to our cohort.
    for col in ("probeID", "Variant_ID", "pvalue", "beta_ref_to_alt", "se",
                "abs_distance_bp", "distance_bp", "probe_split"):
        if col in frame.columns:
            out[col] = frame[col].to_numpy()
    out["effect_size"] = (frame["beta_ref_to_alt"].to_numpy()
                          if "beta_ref_to_alt" in frame.columns else np.nan)
    return out


def build_pairs(frame: pd.DataFrame, genome, counters: dict):
    """Yield (index, ref_onehot, alt_onehot, cpg_rel_start, cpg_rel_end)."""
    for idx, row in frame.iterrows():
        alt, ref = str(row["SNP_alt"]), str(row["SNP_ref"])
        if len(alt) != 1 or len(ref) != 1:
            counters["not_single_base"] += 1
            continue
        snp_start = int(row["SNP_region_start"])
        if snp_start != int(row["SNP_region_end"]):
            counters["snp_start_ne_end"] += 1
            continue
        cpg_start, cpg_end = int(row["CPG_region_start"]), int(row["CPG_region_end"])

        center = (cpg_start + snp_start) // 2
        fetch_start = max(0, center - HALF_WINDOW)
        fetch_end = center + HALF_WINDOW
        try:
            # pyfaidx returns a plain str under as_raw=True and a Sequence
            # object otherwise; str() is correct for both.
            piece = genome[str(row["chrom"])][fetch_start:fetch_end]
            seq = piece if isinstance(piece, str) else str(piece)
        except (KeyError, ValueError):
            counters["chrom_not_in_genome"] += 1
            continue
        if len(seq) != 2 * HALF_WINDOW:
            counters["short_window"] += 1
            continue

        mut_idx = snp_start - fetch_start - 1
        if not (0 <= mut_idx < len(seq)):
            counters["variant_outside_window"] += 1
            continue

        # They overwrite the reference base without checking. We do the same,
        # but we count disagreements -- silently building a REF sequence that
        # is not the genome is how a coordinate bug reaches a figure.
        if seq[mut_idx].upper() != ref.upper():
            counters["reference_base_mismatch"] += 1

        ref_oh = one_hot(seq)
        ref_oh[mut_idx] = 0.0
        ref_oh[mut_idx, BASES[ref.upper()]] = 1.0
        alt_oh = ref_oh.copy()
        alt_oh[mut_idx] = 0.0
        alt_oh[mut_idx, BASES[alt.upper()]] = 1.0

        counters["scoreable"] += 1
        yield (idx, ref_oh, alt_oh,
               cpg_start - fetch_start, cpg_end - fetch_start)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-csv", type=Path, required=True)
    ap.add_argument("--format", choices=("ours", "melody"), required=True)
    ap.add_argument("--repo", type=Path,
                    default=Path("data/external/melody/Melody_repo"))
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="default: <repo>/drive/Melody-MT-39.pth")
    ap.add_argument("--genome", type=Path, default=Path("data/hg38.fa"))
    ap.add_argument("--tracks", required=True,
                    help="comma-separated track names; predictions are AVERAGED "
                         "across them, which is their 'related tracks' setting")
    ap.add_argument("--n-track", type=int, default=39)
    ap.add_argument("--margin", type=int, default=1,
                    help="positions either side of the CpG summed over; must be "
                         ">=1 because CpG_start == CpG_end in these cohorts")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="smoke test only")
    ap.add_argument("--report-correlation", action="store_true",
                    help="print Pearson r against effect_size, for validating "
                         "against their published numbers")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if args.margin < 1:
        raise SystemExit("STOP: --margin must be >= 1 or the summation window "
                         "is empty (CpG_start == CpG_end)")

    import torch
    from pyfaidx import Fasta
    sys.path.insert(0, str(args.repo.resolve()))
    from stateless import load_ckpt          # noqa: E402
    from models import Melody                # noqa: E402
    from global_constants import track_39_names  # noqa: E402

    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    unknown = [t for t in tracks if t not in track_39_names]
    if unknown:
        raise SystemExit(
            f"STOP: unknown track(s) {unknown}.\nAvailable:\n  " +
            "\n  ".join(track_39_names))
    track_idx = [track_39_names.index(t) for t in tracks]
    LOGGER.info("averaging over %d track(s): %s", len(tracks), tracks)

    frame = normalise(pd.read_csv(args.input_csv, dtype=str), args.format)
    for col in ("SNP_region_start", "SNP_region_end", "CPG_region_start",
                "CPG_region_end", "cpg_number"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["SNP_region_start", "CPG_region_start"])
    frame["effect_size"] = pd.to_numeric(frame["effect_size"], errors="coerce")
    if args.limit:
        frame = frame.head(args.limit).copy()
        LOGGER.warning("SMOKE TEST: limited to %d rows", len(frame))
    LOGGER.info("%d rows from %s", len(frame), args.input_csv)

    genome = Fasta(str(args.genome), as_raw=True, sequence_always_upper=True)
    ckpt = args.checkpoint or (args.repo / "drive" / "Melody-MT-39.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Melody(n_track=args.n_track)
    load_ckpt(model, str(ckpt))
    model = model.to(device).eval()
    LOGGER.info("device=%s checkpoint=%s", device, ckpt)

    counters = {k: 0 for k in (
        "scoreable", "not_single_base", "snp_start_ne_end", "short_window",
        "variant_outside_window", "chrom_not_in_genome",
        "reference_base_mismatch", "cpg_window_out_of_bounds")}

    rows, batch = [], []

    def flush():
        if not batch:
            return
        idxs = [b[0] for b in batch]
        ref = torch.from_numpy(np.stack([b[1] for b in batch])).to(device).transpose(1, 2)
        alt = torch.from_numpy(np.stack([b[2] for b in batch])).to(device).transpose(1, 2)
        with torch.no_grad():
            pr = model(ref)
            pa = model(alt)
        pr = pr[0] if isinstance(pr, (list, tuple)) else pr
        pa = pa[0] if isinstance(pa, (list, tuple)) else pa
        pred_len = pr.shape[-1]
        for j, (idx, _, _, cs, ce) in enumerate(batch):
            lo, hi = cs - args.margin, ce + args.margin
            if not (0 <= lo < pred_len and 0 < hi <= pred_len and lo < hi):
                counters["cpg_window_out_of_bounds"] += 1
                continue
            n_cpg = float(frame.at[idx, "cpg_number"]) or 1.0
            per_track = [float((pa[j, t, lo:hi] - pr[j, t, lo:hi]).sum().item()) / n_cpg
                         for t in track_idx]
            rows.append({"row": idx, "Predicted_Delta_M": float(np.mean(per_track)),
                         **{f"delta_{tracks[k]}": v for k, v in enumerate(per_track)}})
        batch.clear()

    from tqdm import tqdm
    for item in tqdm(build_pairs(frame, genome, counters), total=len(frame),
                     desc="scoring"):
        batch.append(item)
        if len(batch) >= args.batch_size:
            flush()
    flush()

    LOGGER.info("counters: %s", counters)
    if counters["scoreable"] == 0:
        raise SystemExit("STOP: nothing scoreable; see counters above")
    mismatch_rate = counters["reference_base_mismatch"] / counters["scoreable"]
    if mismatch_rate > REF_MISMATCH_ABORT:
        raise SystemExit(
            f"STOP: {mismatch_rate:.1%} of variants disagree with the genome at "
            f"the stated reference base. Melody's code overwrites the base "
            f"without checking, so this would pass silently -- but a rate this "
            f"high means a coordinate convention mismatch, not bad data.")
    LOGGER.info("reference-base mismatch rate: %.3f%%", 100 * mismatch_rate)

    scored = pd.DataFrame(rows).set_index("row")
    out = frame.join(scored, how="inner")
    out["Absolute_Delta_M"] = out["Predicted_Delta_M"].abs()
    out["Model"] = "melody"
    out["Melody_Tracks"] = ";".join(tracks)
    out["Melody_Margin"] = args.margin

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_csv": str(args.input_csv), "format": args.format,
        "checkpoint": str(ckpt), "tracks": tracks, "margin": args.margin,
        "counters": counters,
        "reference_base_mismatch_rate": mismatch_rate,
        "rows_scored": int(len(out)),
        "mirrors": ("eqtl_pure_util.SNPEffectDataset and "
                    "check_meqtl_batched_multi_margin_multi_distance_multi_track_model"),
    }
    if args.report_correlation and out["effect_size"].notna().any():
        keep = out["effect_size"].notna() & out["Predicted_Delta_M"].notna()
        if keep.sum() >= 2:
            r = float(np.corrcoef(out.loc[keep, "effect_size"].astype(float),
                                  out.loc[keep, "Predicted_Delta_M"])[0, 1])
            summary["pearson_r_vs_effect_size"] = r
            summary["n_for_correlation"] = int(keep.sum())
            print(f"\nPearson r vs observed effect: {r:.4f}  (n = {int(keep.sum())})")
            print("Compare against Melody's published value for this dataset. "
                  "A close match validates the track choice and the whole setup.")

    with args.output.with_suffix(".summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    print(f"\nwrote {args.output} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
