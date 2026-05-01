import random
import numpy as np
import torch
from tqdm import tqdm
import argparse
import os
import sys
import joblib
from transformers import AutoConfig, AutoTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model.architecture import SilentMethylModel
from model.lora_utils import inject_lora_adapters, load_lora_weights
from inference.engine import predict_epigenetics

# Standard Genetic Code Map
CODON_MAP = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

def get_synonymous_mutation(sequence, position):
    """Finds a random mutation at 'position' that does not change the amino acid."""
    codon_start = position - (position % 3)
    base_idx_in_codon = position % 3
    
    codon = sequence[codon_start : codon_start + 3]
    if len(codon) < 3: return None

    original_aa = CODON_MAP.get(codon)
    if not original_aa: return None

    possible_bases = ['A', 'C', 'G', 'T']
    valid_variants = []

    for b in possible_bases:
        if b == sequence[position]: continue
        new_codon = list(codon)
        new_codon[base_idx_in_codon] = b
        new_codon = "".join(new_codon)

        if CODON_MAP.get(new_codon) == original_aa:
            valid_variants.append(b)

    return random.choice(valid_variants) if valid_variants else None

def main():
    parser = argparse.ArgumentParser(description="Run Rigorous Silent Permutation Test.")
    parser.add_argument("--wt_sequence", type=str, required=True, help="Wild-type DNA sequence.")
    parser.add_argument("--real_delta", type=float, required=True, help="The real delta_p from the actual mutation.")
    parser.add_argument("--n_permutations", type=int, default=100, help="Number of silent permutations to generate.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory containing artifacts.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the trained model weights.")
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M", help="Hugging Face model path.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and artifacts
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    REGION_VOCAB = joblib.load(os.path.join(args.base_dir, "SilentMethyl_RegionVocab.pkl"))
    ISLAND_VOCAB = joblib.load(os.path.join(args.base_dir, "SilentMethyl_IslandVocab.pkl"))
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    model = inject_lora_adapters(model)
    model = load_lora_weights(model, args.weights_path, device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    scaler_age = joblib.load(os.path.join(args.base_dir, "Scaler_Age.pkl"))
    scaler_seq = joblib.load(os.path.join(args.base_dir, "Scaler_Seq.pkl"))

    print(f"[*] Generating {args.n_permutations} Random SILENT Mutations...")
    silent_deltas = []
    attempts = 0
    
    dummy_age = 50 
    dummy_gene_region = 'Gene-body'
    dummy_island_status = 'Open-sea'
    
    with torch.no_grad():
        wt_prob = predict_epigenetics(args.wt_sequence, dummy_age, dummy_gene_region, dummy_island_status, model, tokenizer, scaler_age, scaler_seq, REGION_VOCAB, ISLAND_VOCAB, device)
    
        pbar = tqdm(total=args.n_permutations)
        while len(silent_deltas) < args.n_permutations and attempts < 10000:
            attempts += 1
            pos = random.randint(2000, 3000)
            new_base = get_synonymous_mutation(args.wt_sequence, pos)

            if new_base:
                seq_list = list(args.wt_sequence)
                seq_list[pos] = new_base
                test_seq = "".join(seq_list)

                mut_prob = predict_epigenetics(test_seq, dummy_age, dummy_gene_region, dummy_island_status, model, tokenizer, scaler_age, scaler_seq, REGION_VOCAB, ISLAND_VOCAB, device)
                delta = mut_prob - wt_prob
                silent_deltas.append(delta)
                pbar.update(1)
        pbar.close()

    z_score_silent = (args.real_delta - np.mean(silent_deltas)) / (np.std(silent_deltas) + 1e-9)

    print(f"🔬 FINAL RIGOROUS SILENT PROOF:")
    print(f"-> Z-SCORE (vs Silent Background): {z_score_silent:.2f}")

if __name__ == "__main__":
    main()
