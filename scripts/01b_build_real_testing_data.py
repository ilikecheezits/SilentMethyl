import argparse
import requests
import json
import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
from pyfaidx import Fasta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# CONSTANTS & CONFIGURATION
# ==========================================
GDC_URL = "https://api.gdc.cancer.gov/ssms"
CBIO_URL = "https://www.cbioportal.org/api"
STUDY_ID = "brca_tcga"

def fetch_cbio_data(item, df_clinical, meth_profile, mrna_profile, age_col):
    """Worker function for Multi-Threaded cBioPortal API fetching."""
    gene_symbol = item['gene']
    if gene_symbol == "Unknown": 
        return None
    
    try:
        # Fetch Entrez ID
        res = requests.get(f"{CBIO_URL}/genes?keyword={gene_symbol}").json()
        if not res: return None
        entrez_id = res[0].get('entrezGeneId')

        # Fetch Methylation and mRNA
        meth_data = requests.post(f"{CBIO_URL}/molecular-profiles/{meth_profile}/molecular-data/fetch",
                                  json={"entrezGeneIds": [entrez_id], "sampleIds": item['patients']}).json()
        mrna_data = requests.post(f"{CBIO_URL}/molecular-profiles/{mrna_profile}/molecular-data/fetch",
                                  json={"entrezGeneIds": [entrez_id], "sampleIds": item['patients']}).json()

        if meth_data and mrna_data:
            meth_vals = [d.get('value', 0) for d in meth_data if 'value' in d]
            mrna_vals = [d.get('value', 0) for d in mrna_data if 'value' in d]

            if meth_vals and mrna_vals:
                patient_barcodes = [p[:12] for p in item['patients']]
                ages = df_clinical[df_clinical.index.isin(patient_barcodes)][age_col].dropna()
                real_age = ages.mean() if not ages.empty else 58.0 

                return {
                    "Gene": gene_symbol,
                    "HGVSp_Protein_Notation": item['protein'],
                    "GDC_Genomic_DNA_Change": item['dna_change'],
                    "TCGA_Patient_Count": len(item['patients']),
                    "Age": real_age, 
                    "True_Mutated_Beta": sum(meth_vals) / len(meth_vals),
                    "True_Mutated_mRNA_ZScore": sum(mrna_vals) / len(mrna_vals)
                }
    except Exception:
        pass
    return None

def calculate_spatial_features(seq):
    """Vectorized calculation of structural DNA grammar features."""
    seq = str(seq).upper()
    length = len(seq)
    if length == 0: return [0] * 19

    g, c = seq.count('G'), seq.count('C')
    gc_content = (g + c) / length

    cpg_total = seq.count('CG')
    oe_ratio = (cpg_total * length) / (c * g) if (c * g) > 0 else 0
    gc_skew = (g - c) / (g + c) if (g + c) > 0 else 0

    left_seq, right_seq = seq[:2500], seq[2501:]
    shore_asymmetry = abs(left_seq.count('CG') - right_seq.count('CG'))

    tata = 1 if "TATAAA" in seq else 0
    foxa1 = seq.count('TGTTTAC') + seq.count('GTAAACA')
    gata3 = seq.count('AGATAA') + seq.count('AGATAG') + seq.count('TGATAA') + seq.count('TGATAG')
    ap1 = seq.count('TGACTCA') + seq.count('TGAGTCA')
    ctcf = seq.count('CCGCG') + seq.count('GGCAG')
    sp1 = seq.count('GGGCGG') + seq.count('CCGCCC')
    tpg_cpa = (seq.count('TG') + seq.count('CA')) / length
    poly_a = sum(1 for _ in seq.split('A'*6)[:-1]) + sum(1 for _ in seq.split('T'*6)[:-1])
    alu = seq.count('AGCT')
    g4 = seq.count('GGGG') + seq.count('CCCC')
    ere = sum(1 for i in range(length - 13) if seq[i:i+5] == 'GGTCA' and seq[i+8:i+13] == 'TGACC')
    e_box = seq.count('CACGTG')
    yy1 = seq.count('GCCAT') + seq.count('ATGGC')
    hre = seq.count('ACGTG') + seq.count('GCGTG') + seq.count('CACGT') + seq.count('CACGC')

    return gc_content, cpg_total, oe_ratio, tata, gc_skew, shore_asymmetry, foxa1, gata3, ap1, ctcf, sp1, tpg_cpa, poly_a, alu, g4, ere, e_box, yy1, hre

def main(args):
    print("--- STEP 1: LOAD GENOMES & CLINICAL METADATA ---")
    genome = Fasta(args.fasta_path)
    df_clinical = pd.read_csv(args.clinical_path, sep='\t', index_col=0, low_memory=False)
    df_clinical.index = df_clinical.index.astype(str).str[:12]
    age_col = 'age_at_initial_pathologic_diagnosis' if 'age_at_initial_pathologic_diagnosis' in df_clinical.columns else 'age'

    print("--- STEP 2: FAST ILLUMINA MANIFEST PARSING ---")
    df_official = pd.read_csv(args.geo_path, compression='gzip', skiprows=7, low_memory=False)
    df_official['CHR'] = 'chr' + df_official['CHR'].astype(str)
    df_official['MAPINFO'] = pd.to_numeric(df_official['MAPINFO'], errors='coerce')
    df_official = df_official.dropna(subset=['MAPINFO']).sort_values(['CHR', 'MAPINFO'])
    
    df_official['Gene_Region'] = df_official['UCSC_RefGene_Group'].astype(str).str.split(';').str[0].replace('nan', 'Intergenic')
    df_official['CpG_Island'] = df_official['Relation_to_UCSC_CpG_Island'].astype(str).str.split(';').str[0].replace('nan', 'OpenSea')

    # Pre-compute dictionary for O(log N) binary searches
    chrom_probes = {}
    for chrom, group in df_official.groupby('CHR'):
        chrom_probes[chrom] = {
            'pos': group['MAPINFO'].values,
            'region': group['Gene_Region'].values,
            'island': group['CpG_Island'].values
        }

    def get_fast_annotations(chrom, pos):
        """Binary Search implementation -> Reduces lookup from O(N) to O(log N)"""
        if chrom not in chrom_probes: return 'Intergenic', 'OpenSea'
        positions = chrom_probes[chrom]['pos']
        idx = np.searchsorted(positions, pos)
        
        # Edge cases & closest match
        if idx == 0: best_idx = 0
        elif idx == len(positions): best_idx = len(positions) - 1
        else:
            best_idx = idx - 1 if (pos - positions[idx - 1]) < (positions[idx] - pos) else idx
            
        return chrom_probes[chrom]['region'][best_idx], chrom_probes[chrom]['island'][best_idx]

    print("--- STEP 3: API FUSION & MULTI-THREADING ---")
    profiles = requests.get(f"{CBIO_URL}/studies/{STUDY_ID}/molecular-profiles").json()
    METH_PROFILE = next((p['molecularProfileId'] for p in profiles if p.get('molecularAlterationType') == "METHYLATION"), None)
    MRNA_PROFILE = next((p['molecularProfileId'] for p in profiles if p.get('molecularAlterationType') == "MRNA_EXPRESSION"), None)
    cbio_samples = requests.get(f"{CBIO_URL}/sample-lists/{STUDY_ID}_all/sample-ids").json()

    filt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "consequence.transcript.consequence_type", "value": ["synonymous_variant"]}}
    ]}
    
    params = {"filters": json.dumps(filt), "expand": "occurrence.case,consequence.transcript.gene", 
              "fields": "genomic_dna_change,occurrence.case.submitter_id,consequence.transcript.gene.symbol,consequence.transcript.hgvsp", 
              "format": "JSON", "size": str(args.gdc_size)}
              
    mutations = requests.get(GDC_URL, params=params).json().get('data', {}).get('hits', [])

    mutation_groups = {}
    for m in mutations:
        dna_change = m.get('genomic_dna_change', 'Unknown')
        gene_symbol = m.get('consequence', [{}])[0].get('transcript', {}).get('gene', {}).get('symbol', 'Unknown')
        protein = m.get('consequence', [{}])[0].get('transcript', {}).get('hgvsp', 'Unknown')
        if dna_change not in mutation_groups:
            mutation_groups[dna_change] = {'gene': gene_symbol, 'protein': protein, 'patients': set()}
        for occ in m.get('occurrence', []):
            barcode = occ.get('case', {}).get('submitter_id')
            if barcode:
                mutation_groups[dna_change]['patients'].update([s for s in cbio_samples if s.startswith(barcode)])

    processing_list = [{"dna_change": k, "gene": v['gene'], "protein": v['protein'], "patients": list(v['patients'])}
                       for k, v in mutation_groups.items() if v['patients']]

    real_world_hits = []
    # FIX: ThreadPoolExecutor fires requests simultaneously, ending sequential blocking
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(fetch_cbio_data, item, df_clinical, METH_PROFILE, MRNA_PROFILE, age_col) for item in processing_list]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Multi-Threaded API Fetch"):
            res = future.result()
            if res: real_world_hits.append(res)

    df = pd.DataFrame(real_world_hits).sort_values("TCGA_Patient_Count", ascending=False)
    
    print("--- STEP 4: SEQUENCE EXTRACTION & GOLDEN CONTROLS ---")
    mut_regex = re.compile(r'(chr[0-9XY]+):g\.(\d+)([A-Z]+)>([A-Z]+)')
    
    def extract_row(change):
        match = mut_regex.match(change)
        if not match: return None
        chrom, pos, ref, alt = match.groups()
        pos = int(pos)
        region, island = get_fast_annotations(chrom, pos)
        
        try:
            wt_seq = str(genome[chrom][pos-2501:pos+2499]).upper()
            if len(wt_seq) == 5000 and wt_seq[2500] == ref:
                mut_seq = wt_seq[:2500] + alt + wt_seq[2501:]
                return wt_seq, mut_seq, chrom, region, island
        except Exception: pass
        return None

    extracted = [extract_row(c) for c in df['GDC_Genomic_DNA_Change']]
    df['Healthy_5000bp_DNA'] = [x[0] if x else None for x in extracted]
    df['Mutated_5000bp_DNA'] = [x[1] if x else None for x in extracted]
    df['CpG_chrm'] = [x[2] if x else None for x in extracted]
    df['Gene_Region'] = [x[3] if x else 'Intergenic' for x in extracted]
    df['CpG_Island_Status'] = [x[4] if x else 'OpenSea' for x in extracted]
    df = df.dropna(subset=['Mutated_5000bp_DNA']).copy()
    df['Mutation_ID'] = df['GDC_Genomic_DNA_Change']

    # --- INJECT GOLDEN CONTROLS ---
    controls = [
        {"chrom": "chr17", "pos": 43125270, "gene": "BRCA1", "id": "GOLDEN_CTRL_1_CpG_DESTROY", "type": "DESTROY_CPG", "beta": 0.05},
        {"chrom": "chr6", "pos": 151206123, "gene": "ESR1", "id": "GOLDEN_CTRL_2_CpG_CREATE", "type": "CREATE_CPG", "beta": 0.88},
        {"chrom": "chr19", "pos": 55115768, "gene": "PPP1R12C", "id": "GOLDEN_CTRL_3_NEUTRAL", "type": "NEUTRAL", "beta": df['True_Mutated_Beta'].mean() if not df.empty else 0.5}
    ]
    
    golden_rows = []
    for c in controls:
        reg, isl = get_fast_annotations(c['chrom'], c['pos'])
        wt_seq = list(str(genome[c['chrom']][c['pos']-2501:c['pos']+2499]).upper())
        
        idx = -1
        if c['type'] == "DESTROY_CPG": idx = "".join(wt_seq).find('CG', 2450)
        elif c['type'] == "CREATE_CPG": idx = "".join(wt_seq).find('TG', 2450)
        elif c['type'] == "NEUTRAL": idx = "".join(wt_seq).find('A', 2450)

        mut_seq = wt_seq.copy()
        if idx != -1:
            mut_seq[idx] = 'T' if c['type'] != "CREATE_CPG" else 'C'
            
        golden_rows.append({
            'Mutation_ID': c['id'], 'Gene': f"CTRL_{c['gene']}", 'Age': 58.0, 
            'True_Mutated_Beta': c['beta'], 'True_Mutated_mRNA_ZScore': 0.0,
            'CpG_chrm': c['chrom'], 'Gene_Region': reg, 'CpG_Island_Status': isl,
            'Healthy_5000bp_DNA': "".join(wt_seq), 'Mutated_5000bp_DNA': "".join(mut_seq),
            'HGVSp_Protein_Notation': "p.Control", 'GDC_Genomic_DNA_Change': f"{c['chrom']}:g.CTRL",
            'TCGA_Patient_Count': 999
        })
    
    df = pd.concat([pd.DataFrame(golden_rows), df], ignore_index=True)

    print("--- STEP 5: CALCULATING & SCALING SPATIAL FEATURES ---")
    feat_cols = ['GC_Content', 'CpG_Count', 'CpG_OE_Ratio', 'GC_Skew', 'Shore_Asymmetry', 'FOXA1_Motifs', 'GATA3_Motifs', 'AP1_Motifs', 'CTCF_Motifs', 'SP1_Motifs', 'TpG_CpA_Clock', 'Poly_A_Tracts', 'Alu_Proxy', 'G4_Quadruplex_Proxy', 'ERE_Motifs', 'E_Box_Motifs', 'YY1_Motifs', 'HRE_Motifs']
    
    # Calculate raw features
    wt_feats = pd.DataFrame(df['Healthy_5000bp_DNA'].apply(calculate_spatial_features).tolist(), columns=[f"WT_{c}" for c in feat_cols])
    mut_feats = pd.DataFrame(df['Mutated_5000bp_DNA'].apply(calculate_spatial_features).tolist(), columns=[f"Mut_{c}" for c in feat_cols])
    
    final_df = pd.concat([df.drop(columns=['Healthy_5000bp_DNA', 'Mutated_5000bp_DNA']), 
                          df[['Healthy_5000bp_DNA', 'Mutated_5000bp_DNA']], wt_feats, mut_feats], axis=1)

    # --- CRITICAL FIX: APPLY TRAINING SCALERS TO INFERENCE DATA ---
    print("[*] Applying Training Scalers to Inference Data...")
    import joblib
    try:
        scaler_age = joblib.load("checkpoints/Scaler_Age.pkl") # Update path to where 01_build_data.py saves it
        scaler_seq = joblib.load("checkpoints/Scaler_Seq.pkl")
        
        final_df['Age'] = scaler_age.transform(final_df[['Age']])
        
        wt_feat_names = [f"WT_{c}" for c in feat_cols]
        final_df[wt_feat_names] = scaler_seq.transform(final_df[wt_feat_names])
        print("[✓] Inference features successfully scaled to match training distribution.")
    except Exception as e:
        print(f"[!] WARNING: Could not find/apply scalers. Ensure models are trained first. Error: {e}")

    final_df.to_csv(args.output_csv, index=False)
    print(f"[✓] Pipeline complete! High-performance dataset ({len(final_df)} rows) saved to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdc_size", type=int, default=250)
    parser.add_argument("--fasta_path", required=True, help="Local path to hg38.fa")
    parser.add_argument("--clinical_path", required=True, help="Local path to BRCA_clinicalMatrix")
    parser.add_argument("--geo_path", required=True, help="Local path to GPL13534.csv.gz")
    parser.add_argument("--output_csv", required=True)
    main(parser.parse_args())