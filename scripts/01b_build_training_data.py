import pandas as pd
import numpy as np
from pyfaidx import Fasta
import os
import gzip
from tqdm import tqdm

def generate_mock_patient_data(manifest_df, num_patients=10):
    """Generates fake beta values if you don't have patient data yet."""
    print(f"[*] Generating mock beta values for {num_patients} patients...")
    probes = manifest_df['probeID'].unique()
    
    # Generate random beta values between 0.01 and 0.99
    mock_betas = np.random.uniform(0.01, 0.99, size=(len(probes), num_patients))
    df_betas = pd.DataFrame(mock_betas, index=probes, columns=[f"Patient_{i}" for i in range(num_patients)])
    df_betas.index.name = "probeID"
    return df_betas.reset_index()

def main():
    genome_path = "data/raw/hg38.fa"
    manifest_path = "data/raw/EPIC.hg38.manifest.tsv.gz"
    patient_data_path = "data/raw/patient_beta_values.csv" # Your future patient data
    output_path = "data/processed/perfect_baseline_data.csv"
    
    # 1. Load the hg38 Genome
    print("[*] Loading hg38 reference genome into RAM...")
    genome = Fasta(genome_path, sequence_always_upper=True)
    
    # 2. Load the Zhou Lab Manifest
    print("[*] Loading Zhou Lab hg38 Manifest...")
    manifest = pd.read_csv(manifest_path, sep='\t', usecols=['probeID', 'chrm_A', 'MAPINFO_A'])
    manifest = manifest.rename(columns={'chrm_A': 'chr', 'MAPINFO_A': 'pos'})
    manifest = manifest.dropna(subset=['chr', 'pos'])
    manifest['pos'] = manifest['pos'].astype(int)
    
    # 3. Load or Generate Patient Data
    if os.path.exists(patient_data_path):
        print(f"[*] Loading actual patient data from {patient_data_path}...")
        patient_df = pd.read_csv(patient_data_path)
    else:
        print("[!] No patient data found. Creating a mock dataset to test the pipeline...")
        patient_df = generate_mock_patient_data(manifest)
        patient_df.to_csv("data/raw/mock_patient_betas.csv", index=False)

    # 4. The "Consensus Aggregation" (Solves the One-to-Many problem)
    print("[*] Aggregating Patient Beta Values into 'Consensus Healthy State'...")
    # Assume patient columns are all columns except 'probeID'
    patient_cols = [c for c in patient_df.columns if c != 'probeID']
    
    # Calculate median beta across all patients for each probe
    consensus_df = pd.DataFrame()
    consensus_df['probeID'] = patient_df['probeID']
    consensus_df['Median_Beta'] = patient_df[patient_cols].median(axis=1)
    
    # Calculate M-Value: log2(Beta / (1 - Beta))
    # Clip betas to avoid log(0)
    clipped_betas = np.clip(consensus_df['Median_Beta'], 0.001, 0.999)
    consensus_df['M_Value_Target'] = np.log2(clipped_betas / (1 - clipped_betas))
    
    # Calculate Binary State
    consensus_df['Binary_State_Target'] = (consensus_df['Median_Beta'] > 0.5).astype(int)
    
    # Merge with coordinates
    final_df = pd.merge(consensus_df, manifest, on='probeID', how='inner')
    
    # 5. The Mathematical Sequence Extractor
    print("[*] Extracting perfectly centered 5,000bp sequences...")
    valid_rows = []
    
    for _, row in tqdm(final_df.iterrows(), total=len(final_df)):
        chrom = str(row['chr'])
        # Add 'chr' prefix if missing (Zhou manifest usually has it, but just in case)
        if not chrom.startswith('chr'):
            chrom = f"chr{chrom}"
            
        if chrom not in genome:
            continue
            
        pos = row['pos'] # This is 1-based coordinate of the 'C'
        
        # We fetch a slightly larger window to safely hunt for the exact CG
        # Pyfaidx uses 1-based indexing for slicing, just like Illumina maps!
        search_start = pos - 2510
        search_end = pos + 2510
        
        if search_start < 1 or search_end > len(genome[chrom]):
            continue # Skip sequences too close to the end of a chromosome
            
        raw_seq = str(genome[chrom][search_start - 1 : search_end])
        
        # Hunt for the CG near the mathematical center
        approx_center = 2510
        search_window = raw_seq[approx_center - 10 : approx_center + 10]
        
        if "CG" in search_window:
            # Find exact local offset
            local_offset = search_window.index("CG")
            true_c_idx = (approx_center - 10) + local_offset
            
            # Crop exactly 2500 left and 2500 right
            final_seq = raw_seq[true_c_idx - 2500 : true_c_idx + 2500]
            
            # Double check our math
            if final_seq[2500:2502] == "CG":
                row['Healthy_5000bp_DNA'] = final_seq
                valid_rows.append(row)
            else:
                # This should mathematically never trigger, but good for safety
                pass
                
    # 6. Save the final flawless dataset
    perfect_df = pd.DataFrame(valid_rows)
    perfect_df.to_csv(output_path, index=False)
    print(f"\n[✓] SUCCESS! Saved {len(perfect_df)} perfectly centered sequences to {output_path}")

if __name__ == "__main__":
    main()
