#!/usr/bin/env python3
import gzip
import pysam
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import motifs
from pyjaspar import jaspardb
import warnings

warnings.filterwarnings("ignore")

VARIANT_CHR = "chr8"
VARIANT_POS = 70148394
REF_ALLELE = "G"
ALT_ALLELE = "A"
WINDOW_SIZE = 25

TEST_CSV_PATH = Path("data/datafiles/test.csv")
CPG_ISLAND_PATH = Path("data/HM450.hg38.manifest.CpGIsland.tsv.gz")
GTF_PATH = Path("data/reference/gencode.v44.annotation.gtf.gz")
FASTA_PATH = "data/hg38.fa"

def check_genomic_region(chrom, pos_1based):
    promoter_intervals, utr_intervals, gene_intervals = [], [], []

    with gzip.open(GTF_PATH, "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[0] != chrom: continue

            if "NCOA2" not in fields[8]: continue

            start, end = int(fields[3]), int(fields[4])
            strand, feature = fields[6], fields[2]

            if feature == "transcript":
                tss = start if strand == "+" else end
                if strand == "+":
                    promoter_intervals.append((max(1, tss - 1500), tss + 200))
                else:
                    promoter_intervals.append((max(1, tss - 200), tss + 1500))
            elif feature == "UTR":
                utr_intervals.append((start, end))
            elif feature == "gene":
                gene_intervals.append((start, end))

    def is_in(intervals, pos):
        return any(s <= pos <= e for s, e in intervals)

    if is_in(promoter_intervals, pos_1based): return "Promoter/TSS"
    if is_in(utr_intervals, pos_1based): return "UTR"
    if is_in(gene_intervals, pos_1based): return "Gene body"
    return "Intergenic"

def get_epigenetic_quartiles_and_cpg(chrom, pos):
    test_df = pd.read_csv(TEST_CSV_PATH, usecols=["probeID", "chr", "pos", "Ref_ATAC_Signal", "Ref_H3K27ac_Signal"])

    observed_atac = pd.to_numeric(test_df["Ref_ATAC_Signal"], errors="coerce").dropna()
    atac_pct = observed_atac.rank(method="average", pct=True)
    test_df.loc[observed_atac.index, "ATAC_Stratum"] = pd.cut(
        atac_pct, bins=[0.0, 0.25, 0.50, 0.75, 1.0], labels=["Q1 low", "Q2", "Q3", "Q4 high"], include_lowest=True
    ).astype(str)

    observed_h3k = pd.to_numeric(test_df["Ref_H3K27ac_Signal"], errors="coerce").dropna()
    h3k_pct = observed_h3k.rank(method="average", pct=True)
    test_df.loc[observed_h3k.index, "H3K27ac_Stratum"] = pd.cut(
        h3k_pct, bins=[0.0, 0.25, 0.50, 0.75, 1.0], labels=["Q1 low", "Q2", "Q3", "Q4 high"], include_lowest=True
    ).astype(str)

    test_df = test_df[test_df["chr"] == chrom].copy()
    test_df["dist"] = (test_df["pos"] - pos).abs()
    closest_probe = test_df.sort_values("dist").iloc[0]

    header = pd.read_csv(CPG_ISLAND_PATH, sep="\t", nrows=0)
    probe_col = next((name for name in ("probeID", "Probe_ID", "IlmnID", "Name") if name in header.columns), None)
    rel_col = next((name for name in ("CGIposition", "Relation_to_UCSC_CpG_Island", "Relation_to_Island") if name in header.columns), None)

    cpg_df = pd.read_csv(CPG_ISLAND_PATH, sep="\t", usecols=[probe_col, rel_col])
    cpg_df.rename(columns={probe_col: "probeID", rel_col: "CGIposition"}, inplace=True)

    probe_cpg = cpg_df[cpg_df["probeID"] == closest_probe["probeID"]]["CGIposition"].fillna("Open sea").values
    raw_cpg = str(probe_cpg[0]).lower() if len(probe_cpg) > 0 else "open sea"

    if "shore" in raw_cpg: cpg_context = "Shore"
    elif "shelf" in raw_cpg: cpg_context = "Shelf"
    elif "island" in raw_cpg or "cgi" in raw_cpg: cpg_context = "Island"
    else: cpg_context = "Open sea"

    return closest_probe["probeID"], closest_probe["ATAC_Stratum"], closest_probe["H3K27ac_Stratum"], cpg_context

def get_sequences():
    fasta = pysam.FastaFile(FASTA_PATH)

    start_0based = (VARIANT_POS - 1) - WINDOW_SIZE
    end_0based = (VARIANT_POS - 1) + WINDOW_SIZE + 1

    ref_seq = fasta.fetch(VARIANT_CHR, start_0based, end_0based).upper()
    actual_ref_allele = ref_seq[WINDOW_SIZE]

    if actual_ref_allele != REF_ALLELE:
        print(f"Warning: Expected reference allele {REF_ALLELE}, but found {actual_ref_allele} in FASTA.")

    alt_seq = ref_seq[:WINDOW_SIZE] + ALT_ALLELE + ref_seq[WINDOW_SIZE+1:]
    return ref_seq, alt_seq

def scan_motifs(ref_seq, alt_seq):
    print("Loading JASPAR database (this takes a moment on first run)...")
    jdb = jaspardb(release='JASPAR2024')
    tf_motifs = jdb.fetch_motifs(collection='CORE', tax_group='vertebrates')

    results = []

    print(f"Scanning {len(tf_motifs)} vertebrate TF motifs...")
    for tf in tf_motifs:
        try:
            m = motifs.create(tf.counts.values())
            m.pseudocounts = 0.5
            pssm = m.counts.normalize(pseudocounts=0.5).log_odds()

            ref_scores = [score for pos, score in pssm.calculate(ref_seq)]
            max_ref = max(ref_scores) if ref_scores else None

            alt_scores = [score for pos, score in pssm.calculate(alt_seq)]
            max_alt = max(alt_scores) if alt_scores else None

            if max_ref is not None and max_alt is not None:
                delta = max_alt - max_ref
                if abs(delta) > 1.0:
                    results.append({
                        "TF_Matrix_ID": tf.matrix_id,
                        "TF_Name": tf.name,
                        "Ref_Score": round(max_ref, 3),
                        "Alt_Score": round(max_alt, 3),
                        "Delta_Score": round(delta, 3)
                    })
        except Exception:
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        df["Abs_Delta"] = df["Delta_Score"].abs()
        df = df.sort_values(by="Abs_Delta", ascending=False).drop(columns=["Abs_Delta"])
    return df

if __name__ == "__main__":
    print("="*60)
    print(f"Analyzing Variant: {VARIANT_CHR}:g.{VARIANT_POS}{REF_ALLELE}>{ALT_ALLELE}")
    print("="*60)

    print("\n[1/3] Extracting Biological Context...")
    region = check_genomic_region(VARIANT_CHR, VARIANT_POS)
    probe, atac, h3k, cpg = get_epigenetic_quartiles_and_cpg(VARIANT_CHR, VARIANT_POS)

    print(f"  > Genomic Region: {region}")
    print(f"  > Closest Target CpG Probe: {probe}")
    print(f"  > CpG Island Context: {cpg}")
    print(f"  > ATAC-seq Stratum: {atac}")
    print(f"  > H3K27ac Stratum: {h3k}")

    print(f"\n[2/3] Extracting Genomic Sequence from {FASTA_PATH}...")
    ref, alt = get_sequences()
    print(f"  > Reference Window ({WINDOW_SIZE*2 + 1}bp): {ref}")
    print(f"  > Alternate Window ({WINDOW_SIZE*2 + 1}bp): {alt}")

    print("\n[3/3] Calculating JASPAR TF Motif Overlaps...")
    overlap_df = scan_motifs(ref, alt)

    print("\n--- Transcription Factors with Altered Binding Scores ---")
    if overlap_df.empty:
        print("No significant motif binding changes (>1.0 log-odds) detected spanning the mutated position.")
    else:
        print(overlap_df.head(20).to_string(index=False))
