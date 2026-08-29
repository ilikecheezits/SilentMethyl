#!/usr/bin/env python
"""Build the breast-cancer GWAS risk-variant cohort for the matched-background test.

Pre-registered in results/journal/gwas_matched_background/preregistration.json.
Read that first; this script only implements it.

Pipeline, with every drop counted and reported:

  GWAS Catalog breast associations
    -> chr8/9 only (the held-out chromosomes)
    -> within 500 bp of a held-out HM450 probe
    -> alleles resolved: REF from Ensembl GRCh38, ALT = the catalog's risk allele
    -> reference base verified against Healthy_5000bp_DNA at the computed offset
    -> emitted with the columns scripts/05 requires

The reference-base check is the load-bearing one. The offset of a variant inside
the 5,000 bp probe window is computed from genomic coordinates, and if the
coordinate convention were wrong by even one base every variant would be scored
at the wrong position while looking perfectly valid. So the base read at the
computed offset must equal the reference allele Ensembl reports. The convention
is determined empirically across candidate offsets and must reach the agreement
threshold below, or the build halts.

    python -u data/build_gwas_cohort.py --gwas /tmp/gwas/<associations>.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("results/journal/gwas_matched_background")
ENSEMBL = "https://rest.ensembl.org/variation/human/{rsid}?content-type=application/json"

WINDOW_FULL = 5000
CENTER_C_FULL = 2499          # index of the CpG C inside Healthy_5000bp_DNA
CROP_LOW, CROP_HIGH = 2000, 3000   # the 1,000 bp the model actually sees
PROTECTED = {2499, 2500}
MAX_DISTANCE = 500
AGREEMENT_REQUIRED = 0.90     # reference-base agreement needed to accept an offset

TABULAR_FEATURES = [
    "Ref_ATAC_Signal", "Ref_H3K4me3_Signal", "Ref_H3K27ac_Signal",
    "Ref_H3K27me3_Signal", "Ref_H3K9me3_Signal", "Ref_H3K36me3_Signal",
    "Ref_H3K4me1_Signal", "Target_Base_PhyloP_100way_1",
    "Target_Base_PhyloP_100way_2",
]
MISSING_FEATURES = [f"{n}_Missing" for n in TABULAR_FEATURES]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gwas", required=True, help="GWAS Catalog associations TSV")
    p.add_argument("--probes", default="data/datafiles/test.csv")
    p.add_argument("--trait", default="breast")
    p.add_argument("--chroms", nargs="+", default=["8", "9"])
    p.add_argument("--cache", default=str(OUT_DIR / "ensembl_cache.json"))
    p.add_argument("--sleep", type=float, default=0.15,
                   help="Pause between Ensembl calls; be polite to a public API.")
    p.add_argument("--out", default=str(OUT_DIR / "gwas_cohort.csv"))
    return p.parse_args()


def ensembl_lookup(rsids: list[str], cache_path: Path, sleep: float) -> dict:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [r for r in rsids if r not in cache]
    print(f"Ensembl: {len(rsids)} rsIDs, {len(cache)} cached, {len(todo)} to fetch")
    for i, rsid in enumerate(todo, 1):
        try:
            with urllib.request.urlopen(ENSEMBL.format(rsid=rsid), timeout=30) as fh:
                cache[rsid] = json.load(fh)
        except Exception as exc:                       # noqa: BLE001
            cache[rsid] = {"_error": f"{type(exc).__name__}: {exc}"}
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")
        time.sleep(sleep)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache))
    return cache


def pick_mapping(record: dict, chrom: str, pos: int) -> dict | None:
    """Choose the GRCh38 primary-assembly mapping that agrees with the catalog.

    A variant can map to patches and alt contigs; taking mappings[0] blindly
    would sometimes silently pick one.
    """
    for m in record.get("mappings", []):
        if (str(m.get("seq_region_name")) == str(chrom)
                and m.get("assembly_name") == "GRCh38"
                and int(m.get("start", -1)) == int(pos)):
            return m
    return None


def main() -> int:
    args = parse_args()
    gwas_path, probes_path = Path(args.gwas), Path(args.probes)
    for p in (gwas_path, probes_path):
        if not p.exists():
            print(f"MISSING: {p}")
            return 1

    drops: dict[str, int] = {}

    def drop(reason: str, n: int = 1) -> None:
        drops[reason] = drops.get(reason, 0) + n

    print("=" * 70)
    print("1. GWAS CATALOG")
    print("=" * 70)
    g = pd.read_csv(gwas_path, sep="\t", low_memory=False)
    bc = g[g["DISEASE/TRAIT"].astype(str)
            .str.contains(args.trait, case=False, na=False)].copy()
    print(f"associations matching '{args.trait}': {len(bc)}")
    bc = bc[bc["CHR_ID"].astype(str).isin(args.chroms)]
    bc["CHR_POS"] = pd.to_numeric(bc["CHR_POS"], errors="coerce")
    bc = bc.dropna(subset=["CHR_POS"])
    bc["CHR_POS"] = bc["CHR_POS"].astype(int)
    print(f"on chr{'/'.join(args.chroms)} with a numeric position: {len(bc)}")

    risk = bc["STRONGEST SNP-RISK ALLELE"].astype(str).str.rsplit("-", n=1)
    bc["RISK_ALLELE"] = risk.str[-1].str.strip().str.upper()
    bc = bc[bc["SNPS"].astype(str).str.startswith("rs")]
    # One row per rsID: keep the most significant association for its risk allele.
    bc["_p"] = pd.to_numeric(bc.get("P-VALUE"), errors="coerce").fillna(1.0)
    bc = bc.sort_values("_p").drop_duplicates("SNPS", keep="first")
    print(f"unique rsIDs: {len(bc)}")
    bad_allele = ~bc["RISK_ALLELE"].isin(list("ACGT"))
    drop("risk allele not a single base (e.g. '?')", int(bad_allele.sum()))
    bc = bc[~bad_allele]
    print(f"with a usable risk allele: {len(bc)}")

    print("\n" + "=" * 70)
    print("2. PROXIMITY TO HELD-OUT PROBES")
    print("=" * 70)
    probes = pd.read_csv(probes_path, low_memory=False)
    need = ["chr", "pos", "probeID", "Healthy_5000bp_DNA",
            *TABULAR_FEATURES, *MISSING_FEATURES]
    missing = [c for c in need if c not in probes.columns]
    if missing:
        print(f"STOP: {probes_path} lacks {missing}")
        return 1
    probes["_c"] = probes["chr"].astype(str).str.replace("chr", "", regex=False)

    pairs = []
    for c in args.chroms:
        p = probes[probes["_c"] == c].sort_values("pos")
        v = bc[bc["CHR_ID"].astype(str) == c]
        if p.empty or v.empty:
            continue
        pp = p["pos"].to_numpy()
        rows = p.index.to_numpy()
        idx = np.searchsorted(pp, v["CHR_POS"].to_numpy())
        for k, (_, var) in enumerate(v.iterrows()):
            for j in (idx[k] - 1, idx[k]):
                if 0 <= j < len(pp) and abs(int(pp[j]) - int(var["CHR_POS"])) <= MAX_DISTANCE:
                    pairs.append({"rsid": str(var["SNPS"]), "chrom": c,
                                  "snp_pos": int(var["CHR_POS"]),
                                  "risk_allele": var["RISK_ALLELE"],
                                  "gene": str(var.get("MAPPED_GENE", "") or "GWAS"),
                                  "probe_row": int(rows[j])})
    pf = pd.DataFrame(pairs).drop_duplicates(["rsid", "probe_row"])
    print(f"variant-probe pairs within {MAX_DISTANCE} bp: {len(pf)}")
    if pf.empty:
        print("STOP: no pairs.")
        return 1
    print(f"  distinct rsIDs {pf['rsid'].nunique()}, "
          f"distinct probes {pf['probe_row'].nunique()}")

    print("\n" + "=" * 70)
    print("3. ALLELE RESOLUTION (Ensembl GRCh38)")
    print("=" * 70)
    cache = ensembl_lookup(sorted(pf["rsid"].unique()), Path(args.cache), args.sleep)

    refs: dict[str, str] = {}
    for rsid in pf["rsid"].unique():
        rec = cache.get(rsid, {})
        if "_error" in rec:
            drop("Ensembl lookup failed")
            continue
        sub = pf[pf["rsid"] == rsid].iloc[0]
        m = pick_mapping(rec, sub["chrom"], sub["snp_pos"])
        if m is None:
            drop("no GRCh38 mapping agreeing with the catalog position")
            continue
        alleles = str(m.get("allele_string", "")).split("/")
        if len(alleles) < 2 or alleles[0] not in "ACGT" or len(alleles[0]) != 1:
            drop("allele string not a simple SNV")
            continue
        refs[rsid] = alleles[0]
    print(f"reference alleles resolved: {len(refs)} / {pf['rsid'].nunique()}")

    pf = pf[pf["rsid"].isin(refs)].copy()
    pf["ref"] = pf["rsid"].map(refs)
    same = pf["ref"] == pf["risk_allele"]
    drop("risk allele equals reference", int(same.sum()))
    pf = pf[~same]
    print(f"pairs with ALT != REF: {len(pf)}")

    print("\n" + "=" * 70)
    print("4. OFFSET CONVENTION AND REFERENCE-BASE CHECK")
    print("=" * 70)
    # Determine the coordinate convention empirically rather than assuming it.
    # Fitting here is to the reference genome, not to any outcome.
    best, scores = None, {}
    for delta in (-1, 0, 1):
        ok = 0
        for _, r in pf.iterrows():
            probe = probes.loc[r["probe_row"]]
            i = CENTER_C_FULL + (r["snp_pos"] - int(probe["pos"])) + delta
            seq = str(probe["Healthy_5000bp_DNA"]).upper()
            if 0 <= i < len(seq) and seq[i] == r["ref"]:
                ok += 1
        scores[delta] = ok / len(pf)
        print(f"  offset {delta:+d}: reference base agrees for {ok}/{len(pf)} "
              f"({100 * scores[delta]:.1f}%)")
    best = max(scores, key=scores.get)
    if scores[best] < AGREEMENT_REQUIRED:
        print(f"\nSTOP: best agreement {100 * scores[best]:.1f}% is below the "
              f"{100 * AGREEMENT_REQUIRED:.0f}% required. The coordinate "
              f"conventions do not line up and every variant would be scored at "
              f"the wrong position. Not proceeding.")
        return 1
    print(f"  adopted offset {best:+d}")

    print("\n" + "=" * 70)
    print("5. EMIT COHORT")
    print("=" * 70)
    rows = []
    for _, r in pf.iterrows():
        probe = probes.loc[r["probe_row"]]
        seq = str(probe["Healthy_5000bp_DNA"]).upper()
        i = CENTER_C_FULL + (r["snp_pos"] - int(probe["pos"])) + best
        if not (0 <= i < len(seq)):
            drop("offset outside the 5,000 bp window")
            continue
        if seq[i] != r["ref"]:
            drop("reference base mismatch at the adopted offset")
            continue
        if not (CROP_LOW <= i < CROP_HIGH):
            drop("outside the 1,000 bp the model sees")
            continue
        if i in PROTECTED:
            drop("alters the target CpG (unscoreable)")
            continue
        mutated = seq[:i] + r["risk_allele"] + seq[i + 1:]
        rec = {c: probe[c] for c in ["probeID", "Healthy_5000bp_DNA",
                                     *TABULAR_FEATURES, *MISSING_FEATURES]}
        rec.update({
            "Mutated_5000bp_DNA": mutated,
            "Model_Split": "test",
            "Candidate_ID": f"GWAS|{r['rsid']}|{r['ref']}>{r['risk_allele']}|{probe['probeID']}",
            "Selected_Gene_Name": r["gene"],
            "GWAS_rsID": r["rsid"],
            "GWAS_Risk_Allele": r["risk_allele"],
            "GWAS_Ref_Allele": r["ref"],
            "GWAS_Distance_bp": abs(int(probe["pos"]) - r["snp_pos"]),
        })
        rows.append(rec)

    cohort = pd.DataFrame(rows)
    if cohort.empty:
        print("STOP: nothing survived.")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out, index=False)

    n_probe = cohort["probeID"].nunique()
    per = cohort.groupby("probeID").size()
    print(f"emitted {len(cohort)} variants across {n_probe} probes")
    print(f"  variants per probe: max {per.max()}, median {per.median():.0f}")
    print(f"  distance: median {cohort['GWAS_Distance_bp'].median():.0f} bp")
    print(f"\nexclusion cascade: {drops}")
    print(f"wrote {out}")

    (out.parent / "gwas_cohort_summary.json").write_text(json.dumps({
        "gwas_file": str(gwas_path), "trait": args.trait, "chroms": args.chroms,
        "max_distance_bp": MAX_DISTANCE,
        "adopted_offset": best, "offset_agreement": scores,
        "emitted_variants": int(len(cohort)), "distinct_probes": int(n_probe),
        "max_variants_per_probe": int(per.max()),
        "exclusions": drops,
    }, indent=2) + "\n")
    print(f"wrote {out.parent / 'gwas_cohort_summary.json'}")
    print("\nnext: preflight against the existing background pool")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
