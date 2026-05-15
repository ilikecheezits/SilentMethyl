import os
import sys
import argparse
import torch
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

# Hook up the src/ directory so we can import your architecture locally
sys.path.append(os.path.join(os.getcwd(), 'src'))
from model.architecture import SilentMethylModel, perform_triton_surgery
from model.lora_utils import inject_lora_adapters

def main():
    parser = argparse.ArgumentParser(description="Run In-Silico Mutagenesis on Synonymous Variants")
    parser.add_argument("--mutation_matrix", type=str, required=True, help="Path to your specific mutation testing data (CSV).")
    parser.add_argument("--weights_path", type=str, default="SilentMethyl_Best_Weights.pth", help="Path to trained weights.")
    parser.add_argument("--region_vocab", type=str, default="data/processed/SilentMethyl_RegionVocab.pkl", help="Path to region vocab.")
    parser.add_argument("--island_vocab", type=str, default="data/processed/SilentMethyl_IslandVocab.pkl", help="Path to island vocab.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Where to save the ISM results.")
    # Notice we removed dict_path because your CSV already has the DNA!
    args = parser.parse_args()

    # 1. HARDWARE SETUP
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps') # Apple Silicon
    else:
        device = torch.device('cpu')
    print(f"[*] ISM Engine Online. Running on: {device}")

    # 2. LOAD DATA
    print(f"[*] Loading Fully Assembled Mutation Data from: {args.mutation_matrix}")
    ism_df = pd.read_csv(args.mutation_matrix)
    
    # Ensure no missing DNA sequences
    ism_df = ism_df.dropna(subset=['Healthy_5000bp_DNA', 'Mutated_5000bp_DNA']).reset_index(drop=True)
    print(f"[✓] Secured {len(ism_df)} valid variants for scanning.")

    # 3. LOAD VOCABULARIES & TOKENIZER
    print("[*] Loading Vocabularies and DNABERT-2 Tokenizer...")
    REGION_VOCAB = joblib.load(args.region_vocab)
    ISLAND_VOCAB = joblib.load(args.island_vocab)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    # 4. LOAD ARCHITECTURE
    print("[*] Initializing AI Architecture and Supercomputer Weights...")
    print("[*] Initializing AI Architecture and Supercomputer Weights...")
    
    # Run the exact same surgery and config loading as the training script
    local_model_dir = perform_triton_surgery()
    config = AutoConfig.from_pretrained(local_model_dir, trust_remote_code=True)
    
    # Initialize the model with the config and the lengths of your loaded vocabs
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    
    # Inject LoRA and load your trained weights
    model = inject_lora_adapters(model) 
    model.load_state_dict(torch.load(args.weights_path, map_location=device), strict=False)
    model.eval()
    print("[✓] Brain Loaded Successfully.")

    # 5. IN-SILICO MUTAGENESIS LOOP
    results = []
    print("\n[*] Commencing Variant Effect Prediction (VEP)...")
    
    with torch.no_grad():
        for idx, row in tqdm(ism_df.iterrows(), total=len(ism_df), desc="Scanning Variants"):
            
            # --- EXTRACT FULL METADATA VECTORS (No Zero-Padding!) ---
            # Create the Wild-Type Metadata Vector
            wt_num_feats = torch.tensor([[
                row.get('Age', 0), row.get('True_Mutated_mRNA_ZScore', 0),
                row.get('WT_GC_Content', 0), row.get('WT_CpG_Count', 0),
                row.get('WT_CpG_OE_Ratio', 0), row.get('WT_GC_Skew', 0),
                row.get('WT_Shore_Asymmetry', 0), row.get('WT_FOXA1_Motifs', 0),
                row.get('WT_GATA3_Motifs', 0), row.get('WT_AP1_Motifs', 0),
                row.get('WT_CTCF_Motifs', 0), row.get('WT_SP1_Motifs', 0), 
                row.get('WT_TpG_CpA_Clock', 0), row.get('WT_Poly_A_Tracts', 0), 
                row.get('WT_Alu_Proxy', 0), row.get('WT_G4_Quadruplex_Proxy', 0),
                row.get('WT_ERE_Motifs', 0), row.get('WT_E_Box_Motifs', 0), 
                row.get('WT_YY1_Motifs', 0), row.get('WT_HRE_Motifs', 0)
            ]], dtype=torch.float32).to(device)

            # Create the Mutated Metadata Vector
            mut_num_feats = torch.tensor([[
                row.get('Age', 0), row.get('True_Mutated_mRNA_ZScore', 0),
                row.get('Mut_GC_Content', 0), row.get('Mut_CpG_Count', 0),
                row.get('Mut_CpG_OE_Ratio', 0), row.get('Mut_GC_Skew', 0),
                row.get('Mut_Shore_Asymmetry', 0), row.get('Mut_FOXA1_Motifs', 0),
                row.get('Mut_GATA3_Motifs', 0), row.get('Mut_AP1_Motifs', 0),
                row.get('Mut_CTCF_Motifs', 0), row.get('Mut_SP1_Motifs', 0), 
                row.get('Mut_TpG_CpA_Clock', 0), row.get('Mut_Poly_A_Tracts', 0), 
                row.get('Mut_Alu_Proxy', 0), row.get('Mut_G4_Quadruplex_Proxy', 0),
                row.get('Mut_ERE_Motifs', 0), row.get('Mut_E_Box_Motifs', 0), 
                row.get('Mut_YY1_Motifs', 0), row.get('Mut_HRE_Motifs', 0)
            ]], dtype=torch.float32).to(device)

            # Vocab Indices
            reg_idx = torch.tensor([REGION_VOCAB.get(row.get('Gene_Region', 'Unknown'), 0)], dtype=torch.long).to(device)
            isl_idx = torch.tensor([ISLAND_VOCAB.get(row.get('CpG_Island_Status', 'Unknown'), 0)], dtype=torch.long).to(device)
            wt_tata_idx = torch.tensor([row.get('WT_TATA_Box_Present', 0)], dtype=torch.long).to(device)
            mut_tata_idx = torch.tensor([row.get('Mut_TATA_Box_Present', 0)], dtype=torch.long).to(device)

            # --- RUN WILD-TYPE (With strict 1000bp Center Cropping) ---
            wt_seq = str(row['Healthy_5000bp_DNA']).upper()
            wt_center = wt_seq[2000:3000] if len(wt_seq) >= 3000 else wt_seq
            wt_tokens = tokenizer(wt_center, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
            
            wt_logits, _ = model(
                wt_tokens['input_ids'].to(device), wt_tokens['attention_mask'].to(device), 
                wt_num_feats, reg_idx, isl_idx, wt_tata_idx
            )
            wt_prob = torch.sigmoid(wt_logits).item()

            # --- RUN MUTATED (With strict 1000bp Center Cropping) ---
            mut_seq = str(row['Mutated_5000bp_DNA']).upper()
            mut_center = mut_seq[2000:3000] if len(mut_seq) >= 3000 else mut_seq
            mut_tokens = tokenizer(mut_center, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
            
            mut_logits, _ = model(
                mut_tokens['input_ids'].to(device), mut_tokens['attention_mask'].to(device), 
                mut_num_feats, reg_idx, isl_idx, mut_tata_idx
            )
            mut_prob = torch.sigmoid(mut_logits).item()
            
            # --- RECORD IMPACT ---
            delta_p = mut_prob - wt_prob
            rel_shift = (delta_p / wt_prob * 100) if wt_prob > 0 else 0.0

            results.append({
                'Mutation_ID': row.get('Mutation_ID', f"Var_{idx}"),
                'Gene': row.get('Gene', 'Unknown'),
                'WT_Prob': wt_prob,
                'Mut_Prob': mut_prob,
                'Raw_Delta': delta_p,
                'Relative_Shift_%': rel_shift,
                'Abs_Delta': abs(delta_p)
            })

    # 6. SAVE & DISPLAY RESULTS
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='Abs_Delta', ascending=False)
    
    out_file = os.path.join(args.save_dir, "Top_Synonymous_Mutations_Impact.csv")
    results_df.to_csv(out_file, index=False)
    print(f"\n[✓] Full scan complete! Results saved to: {out_file}")

    print("\n" + "="*80)
    print("🚨 TOP 15 SYNONYMOUS VARIANTS (MASSIVE EPIGENETIC DISRUPTIONS) 🚨")
    print("="*80)
    
    display_cols = ['Mutation_ID', 'Gene', 'WT_Prob', 'Mut_Prob', 'Raw_Delta', 'Relative_Shift_%']
    print(results_df.head(15)[display_cols].to_string(index=False))

if __name__ == "__main__":
    main()