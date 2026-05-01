import torch
import joblib
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def extract_spatial_features(seq):
    seq = str(seq).upper()
    length = len(seq)
    if length == 0: return [0] * 19

    g, c = seq.count('G'), seq.count('C')
    gc_content = (g + c) / length

    cpg_total = seq.count('CG')
    oe_ratio = (cpg_total * length) / (c * g) if (c * g) > 0 else 0
    gc_skew = (g - c) / (g + c) if (g + c) > 0 else 0

    left_seq, right_seq = seq[:2500], seq[2501:]
    cpg_left, cpg_right = left_seq.count('CG'), right_seq.count('CG')
    shore_asymmetry = abs(cpg_left - cpg_right)

    tata = 1 if "TATAAA" in seq else 0

    foxa1 = seq.count('TGTTTAC') + seq.count('GTAAACA')
    gata3 = seq.count('AGATAA') + seq.count('AGATAG') + seq.count('TGATAA') + seq.count('TGATAG')
    ap1_motifs = seq.count('TGACTCA') + seq.count('TGAGTCA')

    ctcf = seq.count('CCGCG') + seq.count('GGCAG')
    sp1 = seq.count('GGGCGG') + seq.count('CCGCCC')
    tpg_cpa_clock = (seq.count('TG') + seq.count('CA')) / length
    poly_a = sum(1 for _ in seq.split('A'*6)[:-1]) + sum(1 for _ in seq.split('T'*6)[:-1])
    alu_proxy = seq.count('AGCT')
    g4_proxy = seq.count('GGGG') + seq.count('CCCC')

    ere_motifs = sum(1 for i in range(length - 13) if seq[i:i+5] == 'GGTCA' and seq[i+8:i+13] == 'TGACC')
    e_box = seq.count('CACGTG')
    yy1_motifs = seq.count('GCCAT') + seq.count('ATGGC')
    hre_motifs = seq.count('ACGTG') + seq.count('GCGTG') + seq.count('CACGT') + seq.count('CACGC')

    return gc_content, cpg_total, oe_ratio, tata, gc_skew, shore_asymmetry, foxa1, gata3, ap1_motifs, ctcf, sp1, tpg_cpa_clock, poly_a, alu_proxy, g4_proxy, ere_motifs, e_box, yy1_motifs, hre_motifs

def predict_epigenetics(dna_seq_5000bp, age, gene_region, island_status, model, tokenizer, scaler_age, scaler_seq, region_vocab, island_vocab, device):
    # 1. Math computation
    gc_content, cpg_total, oe_ratio, tata, gc_skew, shore_asymmetry, foxa1, gata3, ap1_motifs, ctcf, sp1, tpg_cpa_clock, poly_a, alu_proxy, g4_proxy, ere_motifs, e_box, yy1_motifs, hre_motifs = extract_spatial_features(dna_seq_5000bp)

    age_scaled = scaler_age.transform([[age]])[0][0]

    seq_features = pd.DataFrame([[
        gc_content, cpg_total, oe_ratio, gc_skew, shore_asymmetry,
        foxa1, gata3, ap1_motifs, ctcf, sp1, tpg_cpa_clock, poly_a,
        alu_proxy, g4_proxy, ere_motifs, e_box, yy1_motifs, hre_motifs
    ]], columns=[
        'GC_Content', 'CpG_Count', 'CpG_OE_Ratio', 'GC_Skew', 'Shore_Asymmetry',
        'FOXA1_Motifs', 'GATA3_Motifs', 'AP1_Motifs', 'CTCF_Motifs', 'SP1_Motifs',
        'TpG_CpA_Clock', 'Poly_A_Tracts', 'Alu_Proxy', 'G4_Quadruplex_Proxy',
        'ERE_Motifs', 'E_Box_Motifs', 'YY1_Motifs', 'HRE_Motifs'
    ])

    seq_scaled = scaler_seq.transform(seq_features)[0]

    num_tensor = torch.tensor([[age_scaled] + list(seq_scaled)], dtype=torch.float32).to(device)
    tata_idx = torch.tensor([tata], dtype=torch.long).to(device)

    reg_idx = region_vocab.get(gene_region, region_vocab.get('Unknown', 0))
    isl_idx = island_vocab.get(island_status, island_vocab.get('Unknown', 0))
    reg_tensor = torch.tensor([reg_idx], dtype=torch.long).to(device)
    isl_tensor = torch.tensor([isl_idx], dtype=torch.long).to(device)
    tata_tensor = torch.tensor([tata], dtype=torch.long).to(device)

    # 2. Extract strictly 1000bp core for the Language Model
    dna_center = str(dna_seq_5000bp)[2000:3000].upper()
    encoding = tokenizer(dna_center, truncation=True, max_length=512, padding='max_length', return_tensors='pt')

    with torch.no_grad():
        class_logits, reg_logits = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device), num_tensor, reg_tensor, isl_tensor, tata_tensor)
        beta_prob = torch.sigmoid(reg_logits).item()

    return beta_prob


def run_mass_inference(model, dataloader, device):
    """Batched torch.no_grad() pipeline for genome-wide Mutagenesis Sweeps."""
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Mass Inference Progress"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            num_feats = batch['numerical_features'].to(device)
            reg_idx = batch['region_idx'].to(device)
            isl_idx = batch['island_idx'].to(device)
            tata_idx = batch['tata_idx'].to(device)

            _, pred_reg_logits = model(input_ids, attention_mask, num_feats, reg_idx, isl_idx, tata_idx)

            pred_beta = torch.sigmoid(pred_reg_logits)
            clamped_beta = torch.clamp(pred_beta, 0.0, 1.0)
            all_predictions.extend(clamped_beta.cpu().numpy().flatten())

    return all_predictions
