import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyfaidx import Fasta
from pyjaspar import jaspardb
from Bio.Seq import Seq
import multiprocessing as mp
import logging
import warnings

# Suppress Biopython warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
TEST_CSV_PATH = 'data/actual_data/actual_testing_data.csv'
HG38_PATH = 'data/hg38.fa' # Update to your hg38.fa path
OUTPUT_CSV = 'jaspar_motif_disruptions.csv'

# Motif configuration
WINDOW_SIZE = 41  # 20bp left + 1bp mutation + 20bp right
SCORE_THRESHOLD = 5.0 # Minimum log-odds score to be considered a "real" binding site

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
def parse_mutation_id(mut_id):
    """Extracts position, ref, and alt from standard mutation IDs."""
    mut_str = str(mut_id).upper()
    if mut_str == 'NAN': return None, None, None
    match = re.search(r'(\d+)\s*([ACGT])\s*>\s*([ACGT])', mut_str)
    if match: return int(match.group(1)), match.group(2), match.group(3)
    return None, None, None
# =============================================================================
# 3. JASPAR INITIALIZATION
# =============================================================================
logging.info("[*] Downloading/Loading JASPAR Database...")
jdb = jaspardb(release='JASPAR2022') # Using 2022 for stability

logging.info("[*] Fetching and filtering for Vertebrate Motifs...")
all_motifs = jdb.fetch_motifs(collection='CORE')

# Robust manual filtering to bypass pyjaspar version API differences
motifs = []
for m in all_motifs:
    if hasattr(m, 'tax_group') and m.tax_group:
        # tax_group is often a list like ['vertebrates'], so we check the string representation safely
        if 'vertebrate' in str(m.tax_group).lower():
            motifs.append(m)

# Fallback just in case the metadata is missing
if not motifs:
    logging.warning("[!] Taxonomy filter returned 0. Falling back to scoring ALL CORE motifs.")
    motifs = all_motifs

logging.info(f"[*] Precomputing PSSMs for {len(motifs)} Transcription Factors...")
pssm_dict = {}
for motif in motifs:
    # Add a pseudocount to prevent log(0) -inf math errors
    pwm = motif.counts.normalize(pseudocounts=0.5)
    pssm = pwm.log_odds()
    pssm_dict[motif.name] = pssm

# =============================================================================
# 4. PARSING THE DATASET
# =============================================================================
logging.info("[*] Loading Genome and Test CSV...")
genome = Fasta(HG38_PATH)
df = pd.read_csv(TEST_CSV_PATH)

valid_pairs = []
for idx, row in df.iterrows():
    chrom = str(row['chr']).strip()
    mut_id = row['Mutation_ID']
    gene = str(row['Gene']) if pd.notna(row['Gene']) else f"Intergenic"
    
    if chrom not in genome: continue
    
    mut_pos, ref_base, alt_base = parse_mutation_id(mut_id)
    if not mut_pos: continue
        
    # Extract 41bp window exactly centered on the mutation
    start_idx = (mut_pos - 1) - 20
    end_idx = start_idx + WINDOW_SIZE
    
    # Boundary safety
    if start_idx < 0 or end_idx > len(genome[chrom]): continue
        
    wt_seq = str(genome[chrom][start_idx:end_idx]).upper()
    
    # Verify the reference base matches hg38 before mutating
    if wt_seq[20] != ref_base: continue 
        
    mut_seq = wt_seq[:20] + alt_base + wt_seq[21:]
    
    valid_pairs.append({
        'Gene': gene,
        'Mutation_ID': mut_id,
        'WT_Seq': wt_seq,
        'MUT_Seq': mut_seq
    })

logging.info(f"[*] Successfully extracted {len(valid_pairs)} sequence pairs for scanning.")

# =============================================================================
# 5. WORKER FUNCTION FOR MULTIPROCESSING
# =============================================================================
def scan_sequence_pair(item):
    """
    Scans a single WT and MUT sequence pair against all ~800 JASPAR motifs.
    Returns the TF with the most violent thermodynamic disruption.
    """
    wt_seq = Seq(item['WT_Seq'])
    mut_seq = Seq(item['MUT_Seq'])
    
    wt_seq_rc = wt_seq.reverse_complement()
    mut_seq_rc = mut_seq.reverse_complement()
    
    max_disruption = 0
    top_tf = None
    wt_max_score = 0
    mut_max_score = 0
    
    for tf_name, pssm in pssm_dict.items():
        # Score Wild-Type (Forward and Reverse)
        try:
            wt_fwd = max(pssm.calculate(wt_seq))
            wt_rev = max(pssm.calculate(wt_seq_rc))
            best_wt = max(wt_fwd, wt_rev)
            
            # Score Mutated (Forward and Reverse)
            mut_fwd = max(pssm.calculate(mut_seq))
            mut_rev = max(pssm.calculate(mut_seq_rc))
            best_mut = max(mut_fwd, mut_rev)
            
            # Did it destroy or create a motif?
            disruption = abs(best_wt - best_mut)
            
            # Only care if the sequence actually contained a strong motif to begin with
            # (or if it strongly created one)
            if max(best_wt, best_mut) > SCORE_THRESHOLD:
                if disruption > max_disruption:
                    max_disruption = disruption
                    top_tf = tf_name
                    wt_max_score = best_wt
                    mut_max_score = best_mut
        except Exception:
            # Catch lengths issues (if PSSM is longer than 41bp)
            continue
            
    # Calculate the Delta
    delta_score = mut_max_score - wt_max_score
    
    return {
        'Gene': item['Gene'],
        'Mutation_ID': item['Mutation_ID'],
        'Top_Disrupted_TF': top_tf,
        'WT_Motif_Score': round(wt_max_score, 2),
        'MUT_Motif_Score': round(mut_max_score, 2),
        'Motif_Delta_Score': round(delta_score, 2),
        'Absolute_Disruption': round(max_disruption, 2)
    }

# =============================================================================
# 6. PARALLEL EXECUTION
# =============================================================================
if __name__ == '__main__':
    logging.info(f"[*] Starting Motif Scanning on {mp.cpu_count()} CPU Cores...")
    
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use imap to get a clean tqdm progress bar for multiprocessing
        for res in tqdm(pool.imap_unordered(scan_sequence_pair, valid_pairs), total=len(valid_pairs)):
            if res['Top_Disrupted_TF'] is not None:
                results.append(res)
                
    # =============================================================================
    # 7. EXPORT RESULTS
    # =============================================================================
    df_results = pd.DataFrame(results)
    
    # Sort by the most massive thermodynamic disruptions
    df_results = df_results.sort_values(by='Absolute_Disruption', ascending=False)
    
    df_results.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"[✓] Complete! Disruption analysis saved to {OUTPUT_CSV}")
