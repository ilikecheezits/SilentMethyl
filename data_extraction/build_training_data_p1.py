import os
import pandas as pd
import pyBigWig
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
INPUT_CSV = "actual_data/held_out_test_data.csv"
OUTPUT_CSV = "actual_data/held_out_test_data_full.csv"
FASTA_OUT = "reference/held_out_test_wt_101bp.fasta"

BW_PATHS = {
    "Ref_ATAC_Signal": "reference/ATAC_seq.bw",
    "Ref_H3K4me3_Signal": "reference/H3K4me3.bw",
    "Ref_H3K27ac_Signal": "reference/H3K27ac.bw",
    "Ref_H3K27me3_Signal": "reference/H3K27me3.bw",
    "Ref_H3K9me3_Signal": "reference/H3K9me3.bw",
    "Target_Base_PhyloP_100way": "reference/hg38.phyloP100way.bw"
}
# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_bw_signal(bw_obj, chrom, start, end):
    try:
        if chrom not in bw_obj.chroms():
            if chrom.replace('chr', '') in bw_obj.chroms():
                chrom = chrom.replace('chr', '')
            else:
                return 0.0
                
        stat = bw_obj.stats(chrom, start, end, type="mean")
        if stat is None or stat[0] is None:
            return 0.0
        return float(stat[0])
    except Exception as e:
        return 0.0

# ==============================================================================
# MAIN EXECUTION (CHUNKED STREAMING)
# ==============================================================================
print("[*] Opening Epigenetic Reference Tracks...")
bw_files = {name: pyBigWig.open(path) for name, path in BW_PATHS.items()}

# Wipe the FASTA output clean if it exists so we can safely append to it
open(FASTA_OUT, 'w').close()

chunk_size = 5000  # Process 5,000 rows at a time to keep RAM usage incredibly low
first_chunk = True

print(f"[*] Processing massive CSV in chunks of {chunk_size}...")

# Stream the dataframe instead of loading it all at once
for chunk_idx, chunk in enumerate(pd.read_csv(INPUT_CSV, chunksize=chunk_size)):
    print(f"    -> Crunching chunk {chunk_idx + 1}...")
    
    new_features = {name: [] for name in BW_PATHS.keys()}
    fasta_lines = []

    for idx, row in chunk.iterrows():
        chrom = str(row['chr'])
        pos = int(row['pos'])
        seq_5000 = str(row['Healthy_5000bp_DNA'])
        
        start = max(0, pos - 50)
        end = pos + 51
        
        # 1. Epigenetic extraction
        for feature_name, bw_obj in bw_files.items():
            if "PhyloP" in feature_name:
                val = get_bw_signal(bw_obj, chrom, pos, pos + 1)
            else:
                val = get_bw_signal(bw_obj, chrom, start, end)
            new_features[feature_name].append(val)
            
        # 2. 101bp sequence extraction
        if pd.isna(seq_5000) or len(seq_5000) < 5000:
            seq_101 = "N" * 101
        else:
            seq_101 = seq_5000[2450:2551]
            
        fasta_lines.append(f">seq_{idx}\n{seq_101}\n")

    # Append features to this specific chunk
    for feature_name, values in new_features.items():
        chunk[feature_name] = values

    # Append to FASTA
    with open(FASTA_OUT, "a") as f:
        f.writelines(fasta_lines)

    # Append to output CSV
    if first_chunk:
        chunk.to_csv(OUTPUT_CSV, index=False, mode='w')  # Write headers on first pass
        first_chunk = False
    else:
        chunk.to_csv(OUTPUT_CSV, index=False, mode='a', header=False) # Append without headers

# Cleanup
for bw in bw_files.values():
    bw.close()

print("[+] Phase 2 Feature pipeline complete! No RAM crashed today.")
