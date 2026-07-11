import concurrent.futures
import gzip
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
import requests
from pyfaidx import Fasta
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
DATAFILES_DIR = DATA_DIR / "datafiles"
DATAFILES_DIR.mkdir(parents=True, exist_ok=True)

# --- SWITCH TO LUNG ADENOCARCINOMA ---
GDC_URL = "https://api.gdc.cancer.gov/ssms"
CBIO_URL = "https://www.cbioportal.org/api"
STUDY_ID = "luad_tcga" # Changed from brca_tcga
MAX_THREADS = 16

FASTA_PATH = DATA_DIR / "hg38.fa"
MANIFEST_PATH = DATA_DIR / "HM450.hg38.manifest.tsv.gz"
# Ensure you download the LUAD 450k array data from TCGA!
METH_PATH = DATA_DIR / "TCGA-LUAD.methylation450.tsv.gz" 

# --- CRITICAL: LUNG EPIGENETIC TRACKS ---
# For a true zero-shot test, these MUST point to Normal Lung (or LUAD) ENCODE tracks, 
# not the breast tracks used in training.
BASE_REF = DATA_DIR / "reference_lung" 
BW_PATHS = {
    "Ref_ATAC_Signal": BASE_REF / "Lung_ATAC_seq.bw",
    "Ref_H3K4me3_Signal": BASE_REF / "Lung_H3K4me3.bw",
    "Ref_H3K27ac_Signal": BASE_REF / "Lung_H3K27ac.bw",
    "Ref_H3K27me3_Signal": BASE_REF / "Lung_H3K27me3.bw",
    "Ref_H3K9me3_Signal": BASE_REF / "Lung_H3K9me3.bw",
    "Ref_H3K36me3_Signal": BASE_REF / "Lung_H3K36me3.bw",
    "Ref_H3K4me1_Signal": BASE_REF / "Lung_H3K4me1.bw",
    "Target_Base_PhyloP_100way": DATA_DIR / "hg38.phyloP100way.bw"
}

def get_synonymous_mutations_gdc():
    print(f"[*] Fetching Synonymous Mutations for TCGA-LUAD from GDC...")
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-LUAD"]}},
            {"op": "in", "content": {"field": "consequence.transcript.consequence_type", "value": ["synonymous_variant"]}}
        ]
    }
    
    params = {
        "filters": json.dumps(filters),
        "format": "JSON",
        "size": "5000",
        "fields": "ssm_id,genomic_dna_change,consequence.transcript.gene.symbol"
    }
    
    response = requests.post(GDC_URL, json=params)
    data = response.json()
    
    mutations = []
    if "data" in data and "hits" in data["data"]:
        for hit in data["data"]["hits"]:
            genes = set()
            if "consequence" in hit:
                for cons in hit["consequence"]:
                    if "transcript" in cons and "gene" in cons["transcript"]:
                        genes.add(cons["transcript"]["gene"].get("symbol", "UNKNOWN"))
            
            mutations.append({
                "ssm_id": hit["ssm_id"],
                "GDC_Genomic_DNA_Change": hit["genomic_dna_change"],
                "Gene": ",".join(genes)
            })
            
    df = pd.DataFrame(mutations).drop_duplicates(subset=["GDC_Genomic_DNA_Change"])
    print(f"[✓] Found {len(df)} unique synonymous mutations.")
    return df

def map_mutations_to_tcga_patients(df_muts):
    print("[*] Mapping mutations to LUAD patient barcodes via cBioPortal...")
    ssm_ids = df_muts["ssm_id"].tolist()
    
    patient_mapping = []
    chunk_size = 500
    
    for i in tqdm(range(0, len(ssm_ids), chunk_size)):
        chunk = ssm_ids[i:i+chunk_size]
        url = f"{CBIO_URL}/molecular-profiles/{STUDY_ID}_mutations/mutations/fetch"
        try:
            res = requests.post(url, json={"entrezGeneIds": [], "sampleListId": f"{STUDY_ID}_all"})
            data = res.json()
            for mut in data:
                patient_barcode = mut.get("sampleId", "")[:12] 
                genomic_change = f"chr{mut.get('chr')}:g.{mut.get('startPosition')}{mut.get('referenceAllele')}>{mut.get('variantAllele')}"
                patient_mapping.append({
                    "GDC_Genomic_DNA_Change": genomic_change,
                    "TCGA_Patient_Barcode": patient_barcode
                })
        except Exception as e:
            continue
            
    df_map = pd.DataFrame(patient_mapping).drop_duplicates()
    df_merged = pd.merge(df_muts, df_map, on="GDC_Genomic_DNA_Change", how="inner")
    print(f"[✓] Mapped {len(df_merged)} mutation-patient pairs.")
    return df_merged

def parse_genomic_change(change_str):
    match = re.match(r"(chr[0-9XY]+):g\.(\d+)([A-Z]+)>([A-Z]+)", change_str)
    if not match: return None
    return match.groups()

def get_epigenetic_signals(chrom, pos):
    signals = {}
    for mark, bw_path in BW_PATHS.items():
        if not bw_path.exists():
            signals[mark] = np.nan
            continue
        try:
            bw = pyBigWig.open(str(bw_path))
            if mark == "Target_Base_PhyloP_100way":
                val = bw.stats(chrom, pos-1, pos, type="mean")[0]
                signals[mark + "_1"] = val if val is not None else 0.0
                signals[mark + "_2"] = val if val is not None else 0.0
            else:
                val = bw.stats(chrom, pos-1, pos, type="mean")[0]
                signals[mark] = val if val is not None else 0.0
            bw.close()
        except:
            signals[mark] = np.nan
    return signals

def process_mutation(row, fasta, df_manifest):
    parsed = parse_genomic_change(row["GDC_Genomic_DNA_Change"])
    if not parsed: return None
    chrom, pos, ref, alt = parsed
    pos = int(pos)
    
    target_cpgs = df_manifest[(df_manifest["chr"] == chrom) & (abs(df_manifest["pos"] - pos) <= 2499)]
    if target_cpgs.empty: return None
    
    cpg_row = target_cpgs.iloc[0]
    probe_id = cpg_row["probeID"]
    
    half_window = 2500
    start = pos - half_window - 1
    end = pos + half_window - 1 
    
    try:
        wt_seq = str(fasta[chrom][start:end]).upper()
        if wt_seq[half_window:half_window+len(ref)] != ref: return None
        mut_seq = wt_seq[:half_window] + alt + wt_seq[half_window+len(ref):]
        if len(mut_seq) > 5000: mut_seq = mut_seq[:5000]
        elif len(mut_seq) < 5000: mut_seq = mut_seq.ljust(5000, "N")
    except:
        return None

    epi_signals = get_epigenetic_signals(chrom, pos)
    
    return {
        "chr": chrom, "pos": pos, "probeID": probe_id,
        "Gene": row["Gene"], "GDC_Genomic_DNA_Change": row["GDC_Genomic_DNA_Change"],
        "TCGA_Patient_Barcode": row["TCGA_Patient_Barcode"],
        "Healthy_5000bp_DNA": wt_seq, "Mutated_5000bp_DNA": mut_seq,
        **epi_signals
    }

def main():
    print("=== Building TCGA-LUAD Zero-Shot Validation Dataset ===")
    df_muts = get_synonymous_mutations_gdc()
    df_mapped = map_mutations_to_tcga_patients(df_muts)
    
    print("[*] Loading HM450k Manifest...")
    df_manifest = pd.read_csv(MANIFEST_PATH, sep="\t", usecols=["probeID", "CpG_chrm", "CpG_beg"], compression="gzip")
    df_manifest.columns = ["probeID", "chr", "pos"]
    
    print("[*] Loading Reference Genome...")
    fasta = Fasta(str(FASTA_PATH))
    
    print(f"[*] Extracting Sequences and LUAD Epigenetic Contexts...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(process_mutation, row, fasta, df_manifest): row for _, row in df_mapped.iterrows()}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            res = future.result()
            if res: results.append(res)
            
    df_final = pd.DataFrame(results)
    
    print("[*] Filtering for valid LUAD Methylation Targets...")
    df_meth = pd.read_csv(METH_PATH, sep="\t", index_col=0, compression="gzip")
    
    def get_patient_meth(probe_id, patient_barcode):
        patient_cols = [c for c in df_meth.columns if patient_barcode in c]
        if not patient_cols or probe_id not in df_meth.index: return np.nan
        val = df_meth.loc[probe_id, patient_cols].mean()
        return val

    df_final["True_Mutated_Beta"] = df_final.apply(lambda r: get_patient_meth(r["probeID"], r["TCGA_Patient_Barcode"]), axis=1)
    df_final = df_final.dropna(subset=["True_Mutated_Beta"])
    
    # Calculate M-values securely
    def beta_to_m(beta):
        b_safe = max(min(beta, 0.999), 0.001)
        return np.log2(b_safe / (1 - b_safe))

    df_final["True_Mutated_M_Value"] = df_final["True_Mutated_Beta"].apply(beta_to_m)
    
    output_path = DATAFILES_DIR / "luad_zero_shot_validation.csv"
    df_final.to_csv(output_path, index=False)
    print(f"[✓] Zero-Shot Cohort Saved! Extracted {len(df_final)} fully validated LUAD mutations to {output_path}")

if __name__ == "__main__":
    main()