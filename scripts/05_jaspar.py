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
TEST_CSV_PATH = 'data/datafiles/testing_data.csv'
HG38_PATH = 'data/hg38.fa' 
OUTPUT_CSV = 'results/baseline/jaspar_motif_disruptions.csv'

# Motif configuration
WINDOW_SIZE = 41  # 20bp left + 1bp mutation + 20bp right
SCORE_THRESHOLD = 5.0 # Minimum log-odds score to be considered a "real" binding site

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

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
jdb = jaspardb(release='JASPAR2022') 

logging.info("[*] Fetching and filtering for Vertebrate Motifs...")
all_motifs = jdb.fetch_motifs(collection='CORE')

motifs = []
for m in all_motifs:
    if hasattr(m, 'tax_group') and m.tax_group:
        if 'vertebrate' in str(m.tax_group).lower():
            motifs.append(m)

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
        
    start_idx = (mut_pos - 1) - 20
    end_idx = start_idx + WINDOW_SIZE
    
    if start_idx < 0 or end_idx > len(genome[chrom]): continue
        
    wt_seq = str(genome[chrom][start_idx:end_idx]).upper()
    
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
    Scans a single WT and MUT sequence pair against all JASPAR motifs.
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
        try:
            wt_fwd = max(pssm.calculate(wt_seq))
            wt_rev = max(pssm.calculate(wt_seq_rc))
            best_wt = max(wt_fwd, wt_rev)
            
            mut_fwd = max(pssm.calculate(mut_seq))
            mut_rev = max(pssm.calculate(mut_seq_rc))
            best_mut = max(mut_fwd, mut_rev)
            
            disruption = abs(best_wt - best_mut)
            
            if max(best_wt, best_mut) > SCORE_THRESHOLD:
                if disruption > max_disruption:
                    max_disruption = disruption
                    top_tf = tf_name
                    wt_max_score = best_wt
                    mut_max_score = best_mut
        except Exception:
            continue
            
    delta_score = mut_max_score - wt_max_score
    
    # NEW: Calculate Percentage Loss/Gain (relative to the max possible score found)
    max_possible = max(wt_max_score, mut_max_score)
    percent_change = (delta_score / max_possible) * 100 if max_possible > 0 else 0
    
    return {
        'Gene': item['Gene'],
        'Mutation_ID': item['Mutation_ID'],
        'Top_Disrupted_TF': top_tf,
        'WT_Motif_Score': round(wt_max_score, 2),
        'MUT_Motif_Score': round(mut_max_score, 2),
        'Motif_Delta_Score': round(delta_score, 2),
        'Absolute_Disruption': round(max_disruption, 2),
        'Percentage_Change': round(percent_change, 1)
    }

# =============================================================================
# 6. PARALLEL EXECUTION
# =============================================================================
if __name__ == '__main__':
    logging.info(f"[*] Starting Motif Scanning on {mp.cpu_count()} CPU Cores...")
    
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(scan_sequence_pair, valid_pairs), total=len(valid_pairs)):
            if res['Top_Disrupted_TF'] is not None:
                results.append(res)
                
    # =============================================================================
    # 7. EXPORT RESULTS
    # =============================================================================
    df_results = pd.DataFrame(results)
    
    # Sort by the most massive Percentage disruptions (destroying a motif entirely)
    df_results['Abs_Percentage'] = df_results['Percentage_Change'].abs()
    df_results = df_results.sort_values(by='Abs_Percentage', ascending=False).drop(columns=['Abs_Percentage'])
    
    df_results.to_csv(OUTPUT_CSV, index=False)
    logging.info(f"[✓] Complete! Disruption analysis saved to {OUTPUT_CSV}")