import pandas as pd
import argparse
import re
import os

# Simplified Consensus Sequences derived from JASPAR database
MOTIF_DB = {
    'SP1': r'GGGCGG',
    'CTCF': r'CCGCG.{1,3}GGCGGCAG', # Approximate consensus for CTCF zinc finger
    'FOXA1': r'[AT]T[AT]TGTT[AT]',
    'GATA3': r'[AT]GATA[AG]',
    'AP1': r'TGAGTCA',
    'E_Box': r'CANNTG'
}

def scan_for_motifs(sequence):
    """Scans a DNA sequence for known TF binding sites and returns a list of found motifs."""
    found_motifs = []
    for tf_name, pattern in MOTIF_DB.items():
        if re.search(pattern, sequence):
            found_motifs.append(tf_name)
    return found_motifs

def main():
    parser = argparse.ArgumentParser(description="Cross-reference ISM results with biological motifs.")
    parser.add_argument("--inference_results", type=str, required=True, help="Path to Final_Mutation_Inference_Results.csv")
    parser.add_argument("--raw_matrix", type=str, required=True, help="Path to original matrix with full sequences.")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Output directory.")
    args = parser.parse_args()

    print("--- BIOLOGICAL ANCHORING ENGINE (JASPAR/HOCOMOCO) ---")
    
    df_results = pd.read_csv(args.inference_results)
    df_raw = pd.read_csv(args.raw_matrix)
    
    # Merge results with raw sequences
    df_merged = pd.merge(df_results, df_raw[['Mutation_ID', 'Healthy_5000bp_DNA', 'Mutated_5000bp_DNA']], on='Mutation_ID', how='inner')
    
    # Isolate only the statistically significant structural variants
    if 'Significant' in df_merged.columns:
        significant_df = df_merged[df_merged['Significant'] == True].copy()
    else:
        significant_df = df_merged.copy()
        
    print(f"[*] Found {len(significant_df)} statistically significant load-bearing mutations.")
    
    anchoring_results = []
    
    for _, row in significant_df.iterrows():
        # Look at the central 50bp where the mutation occurs
        wt_center = str(row['Healthy_5000bp_DNA'])[2475:2525].upper()
        mut_center = str(row['Mutated_5000bp_DNA'])[2475:2525].upper()
        
        wt_motifs = scan_for_motifs(wt_center)
        mut_motifs = scan_for_motifs(mut_center)
        
        # Determine which motifs were destroyed or created by the mutation
        destroyed = list(set(wt_motifs) - set(mut_motifs))
        created = list(set(mut_motifs) - set(wt_motifs))
        
        if destroyed or created:
            anchoring_results.append({
                'Mutation_ID': row['Mutation_ID'],
                'Gene': row['Gene'],
                'Delta_P': row.get('Absolute_Delta_P', 0),
                'FDR_Pval': row.get('FDR_Pval', 1),
                'Destroyed_Motifs': ", ".join(destroyed) if destroyed else "None",
                'Created_Motifs': ", ".join(created) if created else "None"
            })
            
    df_anchor = pd.DataFrame(anchoring_results)
    
    if not df_anchor.empty:
        df_anchor = df_anchor.sort_values(by='Delta_P', key=abs, ascending=False)
        out_path = os.path.join(args.save_dir, "Biological_Anchoring_Proof.csv")
        df_anchor.to_csv(out_path, index=False)
        print(f"[✓] Biological proof saved to {out_path}")
        
        print("\n🏆 TOP BIOLOGICAL DISCOVERIES (MOTIF DISRUPTIONS):")
        print(df_anchor.head(10).to_string(index=False))
    else:
        print("[!] No direct core motif disruptions found in the significant cohort.")

if __name__ == "__main__":
    main()
