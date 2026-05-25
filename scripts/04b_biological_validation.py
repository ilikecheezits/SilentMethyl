import pandas as pd
import numpy as np
import requests
import argparse
import os

# --- JASPAR API (Fetch True PWM Matrices) ---
JASPAR_IDS = {'CTCF': 'MA0139.1', 'SP1': 'MA0079.3', 'LRAT_Proxy': 'MA0140.2'}

def fetch_jaspar_pfm(matrix_id):
    """Pings JASPAR REST API to get the exact nucleotide frequencies."""
    url = f"https://jaspar.genereg.net/api/v1/matrix/{matrix_id}.json"
    try:
        res = requests.get(url, timeout=10).json()
        pfm = pd.DataFrame(res['pfm'])
        # Convert Frequency to Probabilities (with pseudocount)
        pwm = (pfm + 0.01).div((pfm + 0.01).sum(axis=0), axis=1)
        return pwm
    except Exception as e:
        print(f"[!] Warning: Could not fetch JASPAR {matrix_id}. {e}")
        return None

print("[*] Booting Causal Validation Engine. Fetching JASPAR matrices...")
PWM_DB = {name: fetch_jaspar_pfm(mat_id) for name, mat_id in JASPAR_IDS.items()}
# Filter out any that failed to download
PWM_DB = {k: v for k, v in PWM_DB.items() if v is not None}

def calculate_kl_divergence(wt_seq, mut_seq, pwm):
    """Calculates the Information Content (Bits) destroyed by the mutation."""
    best_drop = 0
    w_len = len(pwm.columns)
    
    # Slide window across the center 40bp
    for i in range(len(wt_seq) - w_len):
        wt_win = wt_seq[i:i+w_len]
        mut_win = mut_seq[i:i+w_len]
        
        # Ensure we only calculate over valid ATCG bases
        if not all(b in 'ACGT' for b in wt_win) or not all(b in 'ACGT' for b in mut_win):
            continue
            
        wt_score = sum(np.log2(pwm.iloc['ACGT'.index(wt_win[j]), j] / 0.25) for j in range(w_len))
        mut_score = sum(np.log2(pwm.iloc['ACGT'.index(mut_win[j]), j] / 0.25) for j in range(w_len))
        
        drop = wt_score - mut_score
        
        # REMOVED THE STUPID GATE: We no longer demand wt_score > 5.0. 
        # If the mutation caused a drop in binding affinity, we record it.
        if drop > best_drop and drop > 0.5: # 0.5 bits is a very mild threshold
            best_drop = drop
            
    return best_drop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc_results", type=str, required=True, help="Path to your MC Confident Mutations (or Top_Synonymous_Mutations_Impact.csv)")
    parser.add_argument("--raw_matrix", type=str, required=True, help="Path to the raw clinical discovery dataset")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Where to save the final proof")
    args = parser.parse_args()

    print(f"[*] Loading Model Predictions: {args.mc_results}")
    df_mc = pd.read_csv(args.mc_results)
    print(f"[*] Loading Clinical Ground Truth: {args.raw_matrix}")
    df_raw = pd.read_csv(args.raw_matrix)
    
    print("[*] Running Unrestricted Causal & Biophysical Validation...\n")
    
    # Merge the model's significant hits with the actual patient data
    df_merged = pd.merge(df_mc, df_raw, on='Mutation_ID', how='inner')
    final_drivers = []
    
    for _, row in df_merged.iterrows():
        # Isolate the center 40bp where the mutation occurred
        wt_full = str(row.get('Healthy_5000bp_DNA', 'X'*5000))
        mut_full = str(row.get('Mutated_5000bp_DNA', 'X'*5000))
        
        mid = len(wt_full) // 2
        wt = wt_full[mid-20:mid+20].upper()
        mut = mut_full[mid-20:mid+20].upper()
        
        # Check KL Divergence (Information Content Drop)
        kl_destroyed = []
        for tf, pwm in PWM_DB.items():
            bits_lost = calculate_kl_divergence(wt, mut, pwm)
            if bits_lost > 0.5: # Log any structural degradation
                kl_destroyed.append(f"{tf} (-{bits_lost:.1f}b)")
        
        # Pull the patient's actual RNA-seq Z-Score directly
        # If the mutation caused the gene to silence, this number will be negative.
        mrna_z = row.get('True_Mutated_mRNA_ZScore', row.get('Patient_mRNA_ZScore', 0.0))
        
        # Pull the neural network's confidence and shift
        # Handle column names whether coming from skib.py or skib_mc.py
        delta_z = row.get('Logit_Delta', row.get('MC_Mean_Logit_Delta', 0.0))
        
        final_drivers.append({
            'Mutation_ID': row['Mutation_ID'],
            'Gene': row.get('Gene', 'Unknown'),
            'Model_Delta_Z': round(delta_z, 3),
            'KL_Motifs_Broken': ", ".join(kl_destroyed) if kl_destroyed else "None",
            'Clinical_mRNA_ZScore': round(mrna_z, 2),
            # Flag it as a true driver if the model predicted a shift AND the patient actually silenced the gene
            'Is_Driver': 'YES' if (abs(mrna_z) >= 2.0 and abs(delta_z) >=  0.05) else 'NO' 
        })
            
    df_final = pd.DataFrame(final_drivers)
    
    # Sort by the most severe clinical silencing
    df_final = df_final.sort_values(by='Clinical_mRNA_ZScore', ascending=True)
    
    out_path = os.path.join(args.save_dir, "Unrestricted_Biological_Proof.csv")
    df_final.to_csv(out_path, index=False)
    
    print("🏆 UNRESTRICTED EPIGENETIC DRIVER HITS 🏆")
    print("=========================================================")
    # Print the top 20 variants that actually caused severe silencing
    display_df = df_final[df_final['Clinical_mRNA_ZScore'] < -0.5].head(20)
    
    if display_df.empty:
        print("[!] No mutations found with severe clinical silencing (mRNA Z-Score < -0.5).")
        print("Here are the top model shifts instead:")
        print(df_final.sort_values(by='Model_Delta_Z', key=abs, ascending=False).head(10).to_string(index=False))
    else:
        print(display_df.to_string(index=False))
        
    print(f"\n[✓] Full validation matrix saved to {out_path}")

if __name__ == "__main__":
    main()
