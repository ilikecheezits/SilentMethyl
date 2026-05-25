import os
import sys
import requests
import json
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from pyfaidx import Fasta
import concurrent.futures

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = ""
os.makedirs(BASE_DIR, exist_ok=True)

GDC_URL = "https://api.gdc.cancer.gov/ssms"
CBIO_URL = "https://www.cbioportal.org/api"
STUDY_ID = "brca_tcga"
MAX_THREADS = 16  # Safe limit for the cBioPortal API

print("==========================================")
print("--- STEP 1: ENVIRONMENT & GENOME SETUP ---")
print("==========================================")

# Download hg38 if not present
if not os.path.exists('hg38.fa'):
    print("[*] Downloading Human Genome hg38.fa... (This might take a while)")
    os.system('wget -q -O hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz')
    os.system('gunzip hg38.fa.gz')

print("[*] Loading Human Genome into Memory... (This takes ~30s)")
genome = Fasta('hg38.fa')

print("\n==========================================")
print("--- STEP 2: GENOME-WIDE SYNONYMOUS DISCOVERY ---")
print("==========================================")

# 1. Syncing cBioPortal Profiles
try:
    profiles = requests.get(f"{CBIO_URL}/studies/{STUDY_ID}/molecular-profiles").json()
    METH_PROFILE = next((p['molecularProfileId'] for p in profiles if p.get('molecularAlterationType') == "METHYLATION"), None)
    MRNA_PROFILE = next((p['molecularProfileId'] for p in profiles if p.get('molecularAlterationType') == "MRNA_EXPRESSION"), None)
    cbio_samples = requests.get(f"{CBIO_URL}/sample-lists/{STUDY_ID}_all/sample-ids").json()
    print(f"[✓] Synced cBioPortal databases. Ready to cross-reference.")
except Exception as e:
    print(f"[!] Setup Error: {e}")
    raise SystemExit

# Pull the actual TCGA Clinical Matrix
CLINICAL_PATH = "BRCA_clinicalMatrix"
if not os.path.exists(CLINICAL_PATH):
    print("[*] Downloading TCGA Clinical Matrix for True Ages...")
    os.system(f'wget -q -O {CLINICAL_PATH} "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FBRCA_clinicalMatrix"')

df_clinical = pd.read_csv(CLINICAL_PATH, sep='\t', index_col=0, low_memory=False)
df_clinical.index = df_clinical.index.astype(str).str[:12]
age_col = 'age_at_initial_pathologic_diagnosis' if 'age_at_initial_pathologic_diagnosis' in df_clinical.columns else 'age'

# 2. Querying the NIH GDC
print("[*] Pinging NIH GDC Supercomputers for Synonymous Variants...")
filt = {
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "consequence.transcript.consequence_type", "value": ["synonymous_variant"]}}
    ]
}

params = {
    "filters": json.dumps(filt),
    "expand": "occurrence.case,consequence.transcript.gene",
    "fields": "genomic_dna_change,occurrence.case.submitter_id,consequence.transcript.gene.symbol,consequence.transcript.hgvsp",
    "format": "JSON",
    "size": "500000"  # Note: increasing this pulls more raw data, but takes longer overall. 
}

mutations = requests.get(GDC_URL, params=params).json().get('data', {}).get('hits', [])
print(f"[✓] Downloaded {len(mutations)} raw synonymous mutations from the NIH.")

# 3. Data Fusion: Mapping GDC -> cBioPortal
mutation_groups = {}
for m in mutations:
    dna_change = m.get('genomic_dna_change', 'Unknown')
    gene_symbol = "Unknown"
    protein_notation = "Unknown"
    
    consequences = m.get('consequence', [])
    if consequences:
        transcript = consequences[0].get('transcript', {})
        if isinstance(transcript, dict):
            gene = transcript.get('gene', {})
            if isinstance(gene, dict): gene_symbol = gene.get('symbol', 'Unknown')
            protein_notation = transcript.get('hgvsp', 'Unknown')

    if dna_change not in mutation_groups:
        mutation_groups[dna_change] = {'gene': gene_symbol, 'protein': protein_notation, 'patients': set()}

    for occurrence in m.get('occurrence', []):
        barcode = occurrence.get('case', {}).get('submitter_id')
        if barcode:
            for match in [s for s in cbio_samples if s.startswith(barcode)]:
                mutation_groups[dna_change]['patients'].add(match)

processing_list = [{"dna_change": k, "gene": v['gene'], "protein": v['protein'], "patients": list(v['patients'])}
                   for k, v in mutation_groups.items() if v['patients']]

# 4. Multithreaded Unrolled Data Fetching
print(f"[*] Blasting cBioPortal with {MAX_THREADS} Parallel Threads...")

def fetch_patient_data(item):
    local_hits = []
    gene_symbol = item['gene']
    if gene_symbol == "Unknown": return []

    try:
        # Fetch Entrez ID
        entrez_response = requests.get(f"{CBIO_URL}/genes?keyword={gene_symbol}")
        if not entrez_response.ok or not entrez_response.json(): return []
        entrez_id = entrez_response.json()[0].get('entrezGeneId')

        # Fetch Methylation and mRNA
        meth_data = requests.post(f"{CBIO_URL}/molecular-profiles/{METH_PROFILE}/molecular-data/fetch",
                                  json={"entrezGeneIds": [entrez_id], "sampleIds": item['patients']}).json()
        mrna_data = requests.post(f"{CBIO_URL}/molecular-profiles/{MRNA_PROFILE}/molecular-data/fetch",
                                  json={"entrezGeneIds": [entrez_id], "sampleIds": item['patients']}).json()

        if meth_data and mrna_data:
            meth_dict = {d['sampleId']: d.get('value', np.nan) for d in meth_data if 'sampleId' in d and 'value' in d}
            mrna_dict = {d['sampleId']: d.get('value', np.nan) for d in mrna_data if 'sampleId' in d and 'value' in d}

            # UNROLL MODIFICATION: Treat each patient individually
            for patient in item['patients']:
                p_barcode = patient[:12]
                
                age_val = 58.0 # Fallback
                if p_barcode in df_clinical.index:
                    a = df_clinical.loc[p_barcode, age_col]
                    age_val = a.iloc[0] if isinstance(a, pd.Series) else a
                
                meth_val = meth_dict.get(patient, np.nan)
                mrna_val = mrna_dict.get(patient, np.nan)
                
                if pd.notna(meth_val) and pd.notna(mrna_val):
                    local_hits.append({
                        "Gene": gene_symbol,
                        "HGVSp_Protein_Notation": item['protein'],
                        "GDC_Genomic_DNA_Change": item['dna_change'],
                        "TCGA_Patient_Barcode": patient,
                        "Age": age_val, 
                        "True_Mutated_Beta": meth_val,
                        "True_Mutated_mRNA_ZScore": mrna_val
                    })
    except Exception:
        pass
    return local_hits

real_world_hits = []
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = [executor.submit(fetch_patient_data, item) for item in processing_list]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Mining Clinical Arrays"):
        result = future.result()
        if result:
            real_world_hits.extend(result)

df = pd.DataFrame(real_world_hits)
if not df.empty:
    print(f"\n[✓] SUCCESS! Unrolled processing generated {len(df)} distinct mutation/patient pairings.")
else:
    print("\n[!] 0 hits. No patients had complete data.")
    raise SystemExit

print("\n==========================================")
print("--- STEP 3: FAST 5000bp SEQUENCE EXTRACTION ---")
print("==========================================")

GEO_PATH = "GPL13534_HumanMethylation450.csv.gz"
if not os.path.exists(GEO_PATH):
    print("[*] Downloading Official Illumina Map for Structural Annotations...")
    GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GPL13534&format=file&file=GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz"
    os.system(f'wget -q -O {GEO_PATH} "{GEO_URL}"')

print("[*] Parsing Illumina Manifest...")
df_official = pd.read_csv(GEO_PATH, compression='gzip', skiprows=7, low_memory=False)
df_annotations = df_official[['CHR', 'MAPINFO', 'UCSC_RefGene_Group', 'Relation_to_UCSC_CpG_Island']].copy()
df_annotations['CHR'] = 'chr' + df_annotations['CHR'].astype(str)
df_annotations['MAPINFO'] = pd.to_numeric(df_annotations['MAPINFO'], errors='coerce')
df_annotations = df_annotations.dropna(subset=['MAPINFO'])

df_annotations['Gene_Region'] = df_annotations['UCSC_RefGene_Group'].astype(str).str.split(';').str[0].replace('nan', 'Intergenic')
df_annotations['CpG_Island_Status'] = df_annotations['Relation_to_UCSC_CpG_Island'].astype(str).str.split(';').str[0].replace('nan', 'OpenSea')
df_annotations = df_annotations.sort_values(['CHR', 'MAPINFO']).reset_index(drop=True)

def get_real_annotations(chrom, pos):
    chrm_data = df_annotations[df_annotations['CHR'] == chrom]
    if chrm_data.empty: return 'Intergenic', 'OpenSea'
    idx = (np.abs(chrm_data['MAPINFO'] - pos)).argmin()
    closest_probe = chrm_data.iloc[idx]
    return closest_probe['Gene_Region'], closest_probe['CpG_Island_Status']

mutation_regex = re.compile(r'(chr[0-9XY]+):g\.(\d+)([A-Z]+)>([A-Z]+)')

def extract_data(genomic_change):
    match = mutation_regex.match(genomic_change)
    if not match: return None, None, None, 'Intergenic', 'OpenSea'

    chrom, pos, ref_allele, mut_allele = match.groups()
    pos = int(pos)
    start_pos = pos - 2500
    end_pos = pos + 2499

    region, island = get_real_annotations(chrom, pos)

    try:
        if chrom in genome:
            healthy_seq = str(genome[chrom][start_pos-1:end_pos]).upper()
            if len(healthy_seq) == 5000 and healthy_seq[2500] == ref_allele:
                mutated_seq = healthy_seq[:2500] + mut_allele + healthy_seq[2501:]
                return healthy_seq, mutated_seq, chrom, region, island
    except Exception:
        pass
    return None, None, None, 'Intergenic', 'OpenSea'

print(f"[*] Slicing DNA for dataset...")
extraction_cache = {}
extracted = []
for change in df['GDC_Genomic_DNA_Change']:
    if change not in extraction_cache:
        extraction_cache[change] = extract_data(change)
    extracted.append(extraction_cache[change])

df['Healthy_5000bp_DNA'] = [x[0] for x in extracted]
df['Mutated_5000bp_DNA'] = [x[1] for x in extracted]
df['CpG_chrm'] = [x[2] for x in extracted]
df['Gene_Region'] = [x[3] for x in extracted]
df['CpG_Island_Status'] = [x[4] for x in extracted]
df.dropna(subset=['Mutated_5000bp_DNA'], inplace=True)
df['Mutation_ID'] = df['GDC_Genomic_DNA_Change'] + "_" + df['TCGA_Patient_Barcode']

def calculate_spatial_features(seq):
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

feature_cols = [
    'GC_Content', 'CpG_Count', 'CpG_OE_Ratio', 'TATA_Box_Present',
    'GC_Skew', 'Shore_Asymmetry', 'FOXA1_Motifs', 'GATA3_Motifs', 'AP1_Motifs',
    'CTCF_Motifs', 'SP1_Motifs', 'TpG_CpA_Clock', 'Poly_A_Tracts', 'Alu_Proxy', 'G4_Quadruplex_Proxy',
    'ERE_Motifs', 'E_Box_Motifs', 'YY1_Motifs', 'HRE_Motifs'
]

print("[*] Calculating spatial features for Wild-Type DNA...")
wt_features = df['Healthy_5000bp_DNA'].apply(calculate_spatial_features).tolist()
df_wt = pd.DataFrame(wt_features, columns=[f"WT_{c}" for c in feature_cols], index=df.index)

print("[*] Calculating spatial features for Mutated DNA...")
mut_features = df['Mutated_5000bp_DNA'].apply(calculate_spatial_features).tolist()
df_mut = pd.DataFrame(mut_features, columns=[f"Mut_{c}" for c in feature_cols], index=df.index)

df = pd.concat([df, df_wt, df_mut], axis=1)

print("\n==========================================")
print("--- STEP 4: ENGINEERING GOLDEN BIOLOGICAL CONTROLS ---")
print("==========================================")

control_anchors = {
    "BRCA1_Promoter": {"chrom": "chr17", "pos": 43125270, "gene": "BRCA1"},
    "ESR1_Enhancer": {"chrom": "chr6", "pos": 151206123, "gene": "ESR1"},
    "AAVS1_Safe_Harbor": {"chrom": "chr19", "pos": 55115768, "gene": "PPP1R12C"}
}

def build_real_control(anchor_key, mut_type, mut_id, hgvsp, true_beta):
    chrom = control_anchors[anchor_key]["chrom"]
    pos = control_anchors[anchor_key]["pos"]
    gene = control_anchors[anchor_key]["gene"]

    start_pos = pos - 2500
    end_pos = pos + 2499
    healthy_seq = str(genome[chrom][start_pos-1:end_pos]).upper()

    real_region, real_island = get_real_annotations(chrom, pos)
    real_age = 58.0  # Controls can safely use the baseline age

    seq_mut = list(healthy_seq)
    dna_change_str = ""

    if mut_type == "DESTROY_CPG":
        idx = healthy_seq.find('CG', 2450)
        if idx != -1:
            seq_mut[idx] = 'T'
            dna_change_str = f"{chrom}:g.{pos+(idx-2500)}C>T"
    elif mut_type == "CREATE_CPG":
        idx = healthy_seq.find('TG', 2450)
        if idx != -1:
            seq_mut[idx] = 'C'
            dna_change_str = f"{chrom}:g.{pos+(idx-2500)}T>C"
    elif mut_type == "NEUTRAL_SILENT":
        idx = healthy_seq.find('A', 2450)
        if idx != -1:
            seq_mut[idx] = 'T'
            dna_change_str = f"{chrom}:g.{pos+(idx-2500)}A>T"

    mutated_seq = "".join(seq_mut)

    return {
        'Mutation_ID': mut_id,
        'Gene': f"CTRL_{gene}",
        'TCGA_Patient_Barcode': "CTRL_PATIENT",
        'Age': real_age,
        'True_Mutated_Beta': true_beta,
        'True_Mutated_mRNA_ZScore': 0.0,
        'CpG_chrm': chrom,
        'Gene_Region': real_region,
        'CpG_Island_Status': real_island,
        'HGVSp_Protein_Notation': hgvsp,
        'GDC_Genomic_DNA_Change': dna_change_str,
        'TCGA_Patient_Count': 999,
        'Healthy_5000bp_DNA': healthy_seq,
        'Mutated_5000bp_DNA': mutated_seq
    }

cohort_baseline_beta = df['True_Mutated_Beta'].mean() if not df.empty else 0.5
golden_controls = [
    build_real_control("BRCA1_Promoter", "DESTROY_CPG", "GOLDEN_CTRL_1_CpG_DESTROY", "p.CpG_Loss", 0.05),
    build_real_control("ESR1_Enhancer", "CREATE_CPG", "GOLDEN_CTRL_2_CpG_CREATE", "p.CpG_Gain", 0.88),
    build_real_control("AAVS1_Safe_Harbor", "NEUTRAL_SILENT", "GOLDEN_CTRL_3_NEUTRAL", "p.Neutral", cohort_baseline_beta)
]
df_golden = pd.DataFrame(golden_controls)

print("[*] Calculating Real Spatial Features for Golden Controls...")
wt_features = df_golden['Healthy_5000bp_DNA'].apply(calculate_spatial_features).tolist()
df_wt_golden = pd.DataFrame(wt_features, columns=[f"WT_{c}" for c in feature_cols], index=df_golden.index)
mut_features = df_golden['Mutated_5000bp_DNA'].apply(calculate_spatial_features).tolist()
df_mut_golden = pd.DataFrame(mut_features, columns=[f"Mut_{c}" for c in feature_cols], index=df_golden.index)

for col in df_wt_golden.columns: df_golden[col] = df_wt_golden[col]
for col in df_mut_golden.columns: df_golden[col] = df_mut_golden[col]

df = pd.concat([df_golden, df], ignore_index=True)

# 5. Final Organize and Save
metadata_cols = [
    'Mutation_ID', 'Gene', 'TCGA_Patient_Barcode', 'Age', 'True_Mutated_Beta', 'True_Mutated_mRNA_ZScore',
    'CpG_chrm', 'Gene_Region', 'CpG_Island_Status',
    'HGVSp_Protein_Notation', 'GDC_Genomic_DNA_Change'
]
seq_cols = ['Healthy_5000bp_DNA', 'Mutated_5000bp_DNA']
wt_feat_cols = [f"WT_{c}" for c in feature_cols]
mut_feat_cols = [f"Mut_{c}" for c in feature_cols]

final_cols = metadata_cols + seq_cols + wt_feat_cols + mut_feat_cols
df = df[final_cols]

OUTPUT_PATH = os.path.join(BASE_DIR, "Final_Discovery_Dataset_MultiOmics.csv")
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n[✓] EXTRACTION COMPLETE!")
print(f"[✓] Saved {len(df)} unrolled variant pairings to: {OUTPUT_PATH}")
