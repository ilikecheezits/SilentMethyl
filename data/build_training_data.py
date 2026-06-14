import os
import gc
import gzip
import pandas as pd
import numpy as np
import pyBigWig
from tqdm import tqdm
from pyfaidx import Fasta
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
BASE_DIR = "/ocean/projects/med250012p/szhang37/SilentMethyl/data/"
os.makedirs(BASE_DIR, exist_ok=True)

# Output paths
TRAIN_OUT = os.path.join(BASE_DIR, "train_data.csv")
VAL_OUT = os.path.join(BASE_DIR, "val_data.csv")
TEST_OUT = os.path.join(BASE_DIR, "test_data.csv")

# Input/Reference paths
FASTA_PATH = os.path.join(BASE_DIR, "hg38.fa")
MANIFEST_PATH = os.path.join(BASE_DIR, "HM450.hg38.manifest.tsv.gz")
METH_PATH = os.path.join(BASE_DIR, "TCGA-BRCA.methylation450.tsv.gz")

BASE_REF = os.path.join(BASE_DIR, "reference/")
BW_PATHS = {
    "Ref_ATAC_Signal": BASE_REF + "ATAC_seq.bw",
    "Ref_H3K4me3_Signal": BASE_REF + "H3K4me3.bw",
    "Ref_H3K27ac_Signal": BASE_REF + "H3K27ac.bw",
    "Ref_H3K27me3_Signal": BASE_REF + "H3K27me3.bw",
    "Ref_H3K9me3_Signal": BASE_REF + "H3K9me3.bw",
    "Target_Base_PhyloP_100way": BASE_REF + "hg38.phyloP100way.bw"
}

WINDOW_SIZE = 5000
HALF_WINDOW = WINDOW_SIZE // 2

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def get_bw_signal(bw_obj, chrom, start, end):
    """Safely extracts mean signal from a BigWig file."""
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
    except Exception:
        return 0.0

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("==========================================")
    print("--- STEP 1: DNA SEQUENCE EXTRACTION ---")
    print("==========================================")
    
    print("[*] Loading Genome...")
    genome = Fasta(FASTA_PATH)
    
    print("[*] Loading HM450 hg38 Manifest...")
    df_manifest = pd.read_csv(MANIFEST_PATH, sep='\t', usecols=['probeID', 'CpG_chrm', 'CpG_beg'])
    df_manifest.rename(columns={'CpG_chrm': 'chr', 'CpG_beg': 'pos'}, inplace=True)
    
    valid_chrs = set([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'])
    df_manifest = df_manifest[df_manifest['chr'].isin(valid_chrs)].dropna().reset_index(drop=True)
    df_manifest['pos'] = df_manifest['pos'].astype(int)
    
    seqs = []
    valid_mask = []
    
    num_faulty = 0
    for idx, row in tqdm(df_manifest.iterrows(), total=len(df_manifest), desc="Extracting DNA"):
        chrom, pos = row['chr'], row['pos']
        start_idx = (pos + 1) - HALF_WINDOW
        end_idx = start_idx + WINDOW_SIZE
    
        if start_idx < 0 or end_idx > len(genome[chrom]):
            valid_mask.append(False)
            seqs.append(np.nan)
            continue
        if str(genome[chrom][2499:2501]).upper() != "CG":
            num_faulty += 1
        print(str(genome[chrom][2490:2510]).upper())     
        seqs.append(str(genome[chrom][start_idx:end_idx]).upper())
        valid_mask.append(True)
    
    df_manifest['Healthy_5000bp_DNA'] = seqs
    df_seq = df_manifest[valid_mask].copy().reset_index(drop=True)
    relevant_probes = set(df_seq['probeID'])
    
    del df_manifest, seqs
    gc.collect()
    print(f"num faulty:{num_faulty}")
    print("\n==========================================")
    print("--- STEP 2: EPIGENETIC TARGET CALCULATION ---")
    print("==========================================")
    
    with gzip.open(METH_PATH, 'rt') as f:
        meth_header = f.readline().strip().split('\t')
    
    probe_col_name = meth_header[0]
    healthy_cols = [col for col in meth_header if '-11' in col]  # Only Solid Tissue Normal 
    
    chunk_list = []
    print("[*] Processing methylation data in chunks to preserve RAM...")
    for chunk in tqdm(pd.read_csv(METH_PATH, sep='\t', usecols=[probe_col_name] + healthy_cols,
                             chunksize=50000, dtype={col: 'float32' for col in healthy_cols})):
        
        chunk.rename(columns={probe_col_name: 'probeID'}, inplace=True)
        chunk = chunk[chunk['probeID'].isin(relevant_probes)].copy()
    
        if not chunk.empty:
            chunk = chunk[~chunk[healthy_cols].isna().all(axis=1)].copy()
            if not chunk.empty:
                beta_vals = chunk[healthy_cols].values
                chunk['Median_Beta'] = np.nanmedian(beta_vals, axis=1)
                
                # Clip values to prevent log(0) errors 
                beta_clipped = np.clip(chunk['Median_Beta'].values, 0.0001, 0.9999)
                chunk['M_Value_Target'] = np.log2(beta_clipped / (1.0 - beta_clipped))
                chunk['Binary_State_Target'] = (chunk['Median_Beta'] > 0.5).astype(int)
    
                chunk_list.append(chunk[['probeID', 'Median_Beta', 'M_Value_Target', 'Binary_State_Target']])
        
        del chunk
        gc.collect()
    
    target_map = pd.concat(chunk_list).drop_duplicates(subset='probeID')
    del chunk_list
    gc.collect()
    
    df_master = pd.merge(df_seq, target_map, on='probeID', how='inner')
    print(f"[✓] Merged DNA and Targets. Shape: {df_master.shape}")

    print("\n==========================================")
    print("--- STEP 3: BIGWIG FEATURE EXTRACTION ---")
    print("==========================================")
    
    print("[*] Opening Reference Epigenetic Tracks...")
    bw_files = {name: pyBigWig.open(path) for name, path in BW_PATHS.items()}
    
    new_features = {name: [] for name in BW_PATHS.keys()}
    
    for idx, row in tqdm(df_master.iterrows(), total=len(df_master), desc="Mining BigWigs"):
        chrom = str(row['chr'])
        pos = int(row['pos'])
        
        start = max(0, pos - 50)
        end = pos + 51
        
        for feature_name, bw_obj in bw_files.items():
            if "PhyloP" in feature_name:
                val = get_bw_signal(bw_obj, chrom, pos, pos + 1) # Single base for PhyloP 
            else:
                val = get_bw_signal(bw_obj, chrom, start, end) # 101bp window for Epigenetics 
            new_features[feature_name].append(val)
            
    for feature_name, values in new_features.items():
        df_master[feature_name] = values
        
    for bw in bw_files.values():
        bw.close()

    print("\n==========================================")
    print("--- STEP 4: TRAIN/VAL/TEST SPLIT & SAVE ---")
    print("==========================================")
    
    # Shuffle and split: 80% Train, 10% Validation, 10% Test
    # Using stratify on Binary_State_Target to ensure balanced classes across splits
    print("[*] Splitting data (80/10/10) with stratification...")
    train_df, temp_df = train_test_split(
        df_master, test_size=0.20, random_state=42, stratify=df_master['Binary_State_Target']
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['Binary_State_Target']
    )
    
    print(f"    -> Train Set: {len(train_df)} rows")
    print(f"    -> Validation Set: {len(val_df)} rows")
    print(f"    -> Test Set: {len(test_df)} rows")
    
    print("[*] Writing CSV files...")
    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)
    
    print("[✓] ALL STEPS COMPLETE! Training data successfully built and split.")

if __name__ == "__main__":
    main()
