#!/usr/bin/env python
"""Build a synthetic matched-background pool for the ClinVar test.

Why synthetic. The comparators for a matched-background test must be variants
that look like the target -- same substitution class, same CpG effect, similar
distance to the CpG -- but are not themselves the thing being tested. The only
real held-out variant table we have (`testing_data_test_only.csv`) yields 472
model-visible variants, which is far too thin: 17 of 35 ClinVar variants fell
through to matching tiers where substitution class is no longer held fixed, and
12 ended up drawing over 10% of their background from other ClinVar variants.

The sequences and epigenomic features in this project are properties of the
PROBE, not of the variant. So a comparator is just a different single-base
substitution inside a held-out probe window, and we can make as many as the
matching needs. That gives:

  * a pool of arbitrary size, so matching lands at strict tiers
  * zero overlap with the ClinVar set, by construction
  * no allele-frequency confound (using common meQTL variants would introduce
    one, since ClinVar-pathogenic variants are rare)
  * provenance that is trivially auditable: same genome, same probe, one base

The claim the test then supports is: ClinVar-pathogenic variants produce larger
predicted methylation effects than substitution- and distance-matched SNVs at
the same class of locus. That is the standard in-silico-mutagenesis background.

What this deliberately does NOT do: it does not choose substitutions to match
particular ClinVar variants. Positions and alternate alleles are drawn at
random, and the matcher in scripts/05 does the matching afterwards, exactly as
it does for any other cohort. Constructing the background around the targets
would risk tailoring it to them.

    python -u data/build_synthetic_background.py --n 15000
    python -u data/preflight_clinvar_background.py --pool <emitted csv>

Scale --n up if the pre-flight still reports weak tiers; each variant costs
about 0.25 s of V100 time across three seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

WINDOW_SIZE = 1000
CENTER_C, CENTER_G = 499, 500          # protected CpG inside the 1,000 bp crop
BASES = ("A", "C", "G", "T")

TABULAR_FEATURES = [
    "Ref_ATAC_Signal", "Ref_H3K4me3_Signal", "Ref_H3K27ac_Signal",
    "Ref_H3K27me3_Signal", "Ref_H3K9me3_Signal", "Ref_H3K36me3_Signal",
    "Ref_H3K4me1_Signal", "Target_Base_PhyloP_100way_1",
    "Target_Base_PhyloP_100way_2",
]
MISSING_FEATURES = [f"{n}_Missing" for n in TABULAR_FEATURES]
CARRY = ["probeID", "Healthy_5000bp_DNA", *TABULAR_FEATURES, *MISSING_FEATURES]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probes", default="data/datafiles/test.csv",
                   help="Held-out probe table with Healthy_5000bp_DNA and features.")
    p.add_argument("--n", type=int, default=15000,
                   help="Synthetic variants to emit.")
    p.add_argument("--max-per-probe", type=int, default=3,
                   help="Cap per probe so the pool is not dominated by a few loci.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/journal/clinvar_matched_background/"
                                    "synthetic_background_pool.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    probes_path = Path(args.probes)
    if not probes_path.exists():
        print(f"MISSING: {probes_path}")
        return 1

    try:
        from training_common import centered_crop
    except Exception as exc:
        print(f"import failed: {type(exc).__name__}: {exc}")
        print("run from the repository root with the silentmethyl env active")
        return 1

    rng = np.random.default_rng(args.seed)
    probes = pd.read_csv(probes_path, low_memory=False)
    missing_cols = [c for c in CARRY if c not in probes.columns]
    if missing_cols:
        print(f"STOP: {probes_path} lacks {missing_cols}")
        return 1
    print(f"held-out probes available: {len(probes)}")

    # Every emitted row must survive scripts/05's own validation, so apply the
    # same conditions here rather than discovering the losses later:
    #   - the crop must be a real 1,000 bp window centred on CG
    #   - exactly one base differs between WT and mutant
    #   - that base is not one of the two protected CpG positions
    rows: list[dict] = []
    per_probe: dict[str, int] = {}
    skipped = {"short_sequence": 0, "not_cpg_centred": 0, "ambiguous_base": 0}
    order = rng.permutation(len(probes))
    attempts = 0

    for pos in order:
        if len(rows) >= args.n:
            break
        row = probes.iloc[int(pos)]
        probe = str(row["probeID"])
        if per_probe.get(probe, 0) >= args.max_per_probe:
            continue
        try:
            wt = centered_crop(str(row["Healthy_5000bp_DNA"]), WINDOW_SIZE)
        except ValueError:
            skipped["short_sequence"] += 1
            continue
        if wt[CENTER_C:CENTER_G + 1] != "CG":
            skipped["not_cpg_centred"] += 1
            continue

        full = str(row["Healthy_5000bp_DNA"]).upper()
        offset = (len(full) // 2) - (WINDOW_SIZE // 2)   # crop index -> full index

        made = 0
        for _ in range(args.max_per_probe * 4):          # bounded retries
            if len(rows) >= args.n or made >= args.max_per_probe:
                break
            attempts += 1
            i = int(rng.integers(0, WINDOW_SIZE))
            if i in (CENTER_C, CENTER_G):
                continue
            ref = wt[i]
            if ref not in BASES:
                skipped["ambiguous_base"] += 1
                continue
            alt = str(rng.choice([b for b in BASES if b != ref]))
            mutated = full[:offset + i] + alt + full[offset + i + 1:]

            rec = {c: row[c] for c in CARRY}
            rec["Mutated_5000bp_DNA"] = mutated
            rec["Model_Split"] = "test"
            # Candidate_ID is what make_variant_uid prefers, so this fixes the
            # UID and guarantees it cannot collide with a ClinVar rsID-based one.
            rec["Candidate_ID"] = f"SYN|{probe}|{i}|{ref}>{alt}"
            rec["Selected_Gene_Name"] = "SYNTHETIC"
            rows.append(rec)
            per_probe[probe] = per_probe.get(probe, 0) + 1
            made += 1

    pool = pd.DataFrame(rows)
    if pool.empty:
        print("STOP: no synthetic variants could be generated.")
        return 1
    if pool["Candidate_ID"].duplicated().any():
        n_dup = int(pool["Candidate_ID"].duplicated().sum())
        print(f"dropping {n_dup} duplicate Candidate_ID(s)")
        pool = pool.drop_duplicates("Candidate_ID")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(out, index=False)

    print(f"emitted {len(pool)} synthetic variants across "
          f"{pool['probeID'].nunique()} probes ({attempts} draws)")
    if any(skipped.values()):
        print(f"probes skipped: {skipped}")
    print(f"wrote {out}")

    summary = {
        "probe_table": str(probes_path),
        "probes_available": int(len(probes)),
        "requested": int(args.n),
        "emitted": int(len(pool)),
        "probes_used": int(pool["probeID"].nunique()),
        "max_per_probe": int(args.max_per_probe),
        "random_seed": int(args.seed),
        "probes_skipped": skipped,
        "design_note": (
            "Positions and alternate alleles drawn uniformly at random within "
            "the 1,000 bp model window, excluding the protected centred CpG. "
            "No attempt is made to match particular ClinVar variants; matching "
            "is performed afterwards by scripts/05 exactly as for any cohort."
        ),
    }
    (out.parent / "synthetic_background_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"wrote {out.parent / 'synthetic_background_summary.json'}")
    print("\nnext: python -u data/preflight_clinvar_background.py "
          f"--pool {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
