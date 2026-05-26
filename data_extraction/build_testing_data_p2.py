import os
import requests
import json
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from pyfaidx import Fasta
import concurrent.futures
import pyBigWig

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = "actual_data"
os.makedirs(BASE_DIR, exist_ok=True)

GDC_URL = "https://api.gdc.cancer.gov/ssms"
CBIO_URL = "https://www.cbioportal.org/api"
STUDY_ID = "brca_tcga"
MAX_THREADS = 16  # The old reliable thread count

# Absolute paths for Bridges-2
BASE_REF = "/ocean/projects/med250012p/szhang37/SilentMethyl/data/reference/"
EPIGENETIC_PATHS = {
    'Ref_ATAC_Signal': BASE_REF + "ATAC_seq.bw",
    'Ref_H3K4me3_Signal': BASE_REF + "H3K4me3.bw",
    'Ref_H3K27ac_Signal': BASE_REF + "H3K27ac.bw",
    'Ref_H3K27me3_Signal': BASE_REF + "H3K27me3.bw",
    'Ref_H3K9me3_Signal': BASE_REF + "H3K9me3.bw",
    'Target_Base_PhyloP_100way': BASE_REF + "hg38.phyloP100way.bw"
}

print("==========================================")
print("--- STEP 1: ENVIRONMENT & GENOME SETUP ---")
print("==========================================")

if not os.path.exists('hg38.fa'):
    print("[*] Downloading Human Genome hg38.fa...")
    os.system('wget -q -O hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz')
    os.system('gunzip hg38.fa.gz')

print("[*] Loading Human Genome into Memory...")
genome = Fasta('hg38.fa')

bw_handles = {}
for name, path in EPIGENETIC_PATHS.items():
    if os.path.exists(path):
        bw_handles[name] = pyBigWig.open(path)
    else:
        print(f"[!] WARNING: Missing BigWig file: {path}")

print("\n==========================================")
print("--- STEP 2: GENOME-WIDE MUTATION DISCOVERY ---")
print("==========================================")

try:
    profiles = requests.get(f"{CBIO_URL}/studies/{STUDY_ID}/molecular-profiles").json()
    METH_PROFILE = next((p['molecularProfileId'] for p in profiles if p.get('molecularAlterationType') == "METHYLATION"), None)
    cbio_samples = requests.get(f"{CBIO_URL}/sample-lists/{STUDY_ID}_all/sample-ids").json()
except Exception as e:
    print(f"[!] Setup Error: {e}")
    raise SystemExit

print("[*] Pinging NIH GDC Supercomputers...")
filt = {
    "op": "and",
    "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
        {"op": "in", "content": {"field": "consequence.transcript.consequence_type", "value": ["synonymous_variant"]}},
        {"op": "in", "content": {"field": "occurrence.case.samples.sample_type", "value": ["Solid Tissue Normal"]}}
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
print(f"[✓] Downloaded {len(mutations)} raw mutations from the NIH.")

mutation_groups = {}
for m in mutations:
    dna_change = m.get('genomic_dna_change', 'Unknown')
    gene_symbol = "Unknown"
    
    consequences = m.get('consequence', [])
    if consequences:
        transcript = consequences[0].get('transcript', {})
        if isinstance(transcript, dict):
            gene = transcript.get('gene', {})
            if isinstance(gene, dict): gene_symbol = gene.get('symbol', 'Unknown')

    if dna_change not in mutation_groups:
        mutation_groups[dna_change] = {'gene': gene_symbol, 'patients': set()}

    for occurrence in m.get('occurrence', []):
        barcode = occurrence.get('case', {}).get('submitter_id')
        if barcode:
            for match in [s for s in cbio_samples if s.startswith(barcode)]:
                mutation_groups[dna_change]['patients'].add(match)

# Clean up list and prep for mapping
unique_genes = list(set([v['gene'] for v in mutation_groups.values() if v['gene'] != 'Unknown']))

# INSTANT HGNC MAPPING
# INSTANT HGNC MAPPING (With Compute Node Fallback)
print(f"[*] Attempting to download HGNC database to map {len(unique_genes)} genes instantly...")
gene_to_entrez = {}

try:
    # The updated HGNC server path
    hgnc_url = "https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&col=md_eg_id&status=Approved&hgnc_dbt=dic&order_by=gd_app_sym_sort&format=text&submit=submit"
    hgnc_df = pd.read_csv(hgnc_url, sep='\t', low_memory=False)
    # The columns in this custom export are named slightly differently:
    valid_hgnc = hgnc_df.dropna(subset=['Approved symbol', 'NCBI Gene ID'])
    gene_to_entrez = dict(zip(valid_hgnc['Approved symbol'], valid_hgnc['NCBI Gene ID'].astype(int)))
    print(f"[+] HGNC instant mapping successful!")
    
except Exception as e:
    print(f"[!] Direct download failed ({e}). Falling back to cBioPortal API...")
    
    def get_entrez(gene):
        try:
            r = requests.get(f"{CBIO_URL}/genes?keyword={gene}", timeout=10)
            if r.ok and r.json(): return gene, r.json()[0].get('entrezGeneId')
        except: pass
        return gene, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(get_entrez, g): g for g in unique_genes}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Mapping Genes via API"):
            g, eid = future.result()
            if eid: gene_to_entrez[g] = eid

processing_list = [{"dna_change": k, "gene": v['gene'], "patients": list(v['patients'])}
                   for k, v in mutation_groups.items() if v['patients'] and v['gene'] in gene_to_entrez]
print(f"[*] Blasting cBioPortal with {MAX_THREADS} Parallel Threads (Using the reliable method)...")

def fetch_patient_data(item):
    local_hits = []
    gene_symbol = item['gene']
    entrez_id = gene_to_entrez.get(gene_symbol)
    
    if not entrez_id: return []

    try:
        # Generate the normal barcodes (-11) to ask for them simultaneously
        normal_barcodes = [p[:-2] + "11" for p in item['patients']]
        combined_search = item['patients'] + normal_barcodes

        meth_data = requests.post(f"{CBIO_URL}/molecular-profiles/{METH_PROFILE}/molecular-data/fetch",
                                  json={"entrezGeneIds": [int(entrez_id)], "sampleIds": combined_search}).json()

        if meth_data:
            meth_dict = {d['sampleId']: d.get('value', np.nan) for d in meth_data if 'sampleId' in d and 'value' in d}

            for patient in item['patients']:
                tumor_beta = meth_dict.get(patient, np.nan)
                normal_barcode = patient[:-2] + "11"
                normal_beta = meth_dict.get(normal_barcode, np.nan) # Try to get the matched normal
                
                if pd.notna(tumor_beta):
                    # M-Value conversion for Tumor
                    beta_safe = max(0.0001, min(0.9999, tumor_beta))
                    m_val = np.log2(beta_safe / (1 - beta_safe))
                    
                    # M-Value conversion for Normal (if it exists)
                    wt_m_val = np.nan
                    if pd.notna(normal_beta):
                        wt_beta_safe = max(0.0001, min(0.9999, normal_beta))
                        wt_m_val = np.log2(wt_beta_safe / (1 - wt_beta_safe))
                    
                    local_hits.append({
                        "Gene": gene_symbol,
                        "GDC_Genomic_DNA_Change": item['dna_change'],
                        "TCGA_Patient_Barcode": patient,
                        "True_Mutated_Beta": tumor_beta,
                        "True_Mutated_M_Value": m_val,
                        "True_Wild_Type_Beta": normal_beta,
                        "True_Wild_Type_M_Value": wt_m_val
                    })
    except Exception:
        pass
    return local_hits

real_world_hits = []
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = [executor.submit(fetch_patient_data, item) for item in processing_list]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Mining Methylation Arrays"):
        result = future.result()
        if result:
            real_world_hits.extend(result)

df = pd.DataFrame(real_world_hits)
if df.empty:
    print("\n[!] 0 hits. No patients had complete data.")
    raise SystemExit

print("\n==========================================")
print("--- STEP 3: SEQUENCE & EPIGENETIC INJECTION ---")
print("==========================================")

GEO_PATH = "GPL13534_HumanMethylation450.csv.gz"
if not os.path.exists(GEO_PATH):
    print("[*] Downloading Official Illumina Map for Structural Annotations...")
    GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GPL13534&format=file&file=GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz"
    os.system(f'wget -q -O {GEO_PATH} "{GEO_URL}"')

print("[*] Parsing Illumina Manifest...")
df_official = pd.read_csv(GEO_PATH, compression='gzip', skiprows=7, low_memory=False)
id_col = 'ID' if 'ID' in df_official.columns else 'IlmnID' if 'IlmnID' in df_official.columns else 'Name'
df_annotations = df_official[['CHR', 'MAPINFO', 'UCSC_RefGene_Group', 'Relation_to_UCSC_CpG_Island', id_col]].copy()
df_annotations = df_annotations.rename(columns={id_col: 'TargetID'})

df_annotations['CHR'] = 'chr' + df_annotations['CHR'].astype(str)
df_annotations['MAPINFO'] = pd.to_numeric(df_annotations['MAPINFO'], errors='coerce')
df_annotations = df_annotations.dropna(subset=['MAPINFO'])

df_annotations['Gene_Region'] = df_annotations['UCSC_RefGene_Group'].astype(str).str.split(';').str[0].replace('nan', 'Intergenic')
df_annotations['CpG_Island_Status'] = df_annotations['Relation_to_UCSC_CpG_Island'].astype(str).str.split(';').str[0].replace('nan', 'OpenSea')
df_annotations = df_annotations.sort_values(['CHR', 'MAPINFO']).reset_index(drop=True)

mutation_regex = re.compile(r'(chr[0-9XY]+):g\.(\d+)([A-Z]+)>([A-Z]+)')

def extract_and_build(row):
    match = mutation_regex.match(row['GDC_Genomic_DNA_Change'])
    if not match: return None

    chrom, mut_pos, ref_allele, mut_allele = match.groups()
    mut_pos = int(mut_pos)

    # 1. Find the target CpG site closest to the mutation
    chrm_data = df_annotations[df_annotations['CHR'] == chrom]
    if chrm_data.empty: return None
    idx = (np.abs(chrm_data['MAPINFO'] - mut_pos)).argmin()
    closest_probe = chrm_data.iloc[idx]
    
    cpg_pos = int(closest_probe['MAPINFO'])
    probe_id = closest_probe['TargetID']
    region = closest_probe['Gene_Region']
    island = closest_probe['CpG_Island_Status']

    # 2. Strict Boundary Check
    offset = mut_pos - cpg_pos
    if abs(offset) > 499:
        return None

    # 3. Build Centered Sequences
    start_pos = cpg_pos - 2500
    end_pos = cpg_pos + 2499
    
    if chrom not in genome: return None
    healthy_seq = str(genome[chrom][start_pos-1:end_pos]).upper()
    
    if len(healthy_seq) != 5000: return None
    
    mut_idx = 2500 + offset
    if healthy_seq[mut_idx:mut_idx+len(ref_allele)] != ref_allele:
        return None 
        
    mutated_seq = healthy_seq[:mut_idx] + mut_allele + healthy_seq[mut_idx+len(ref_allele):]

    # 4. Extract Tabular Epigenetics at the CpG
    bw_features = {}
    for name, bw in bw_handles.items():
        try:
            val = bw.stats(chrom, cpg_pos, cpg_pos + 1, type="mean")[0]
            bw_features[name] = float(val) if val is not None else 0.0
        except:
            bw_features[name] = 0.0

    return {
        'probeID': probe_id,
        'chr': chrom,
        'pos': cpg_pos,
        'Gene_Region': region,
        'CpG_Island_Status': island,
        'Healthy_5000bp_DNA': healthy_seq,
        'Mutated_5000bp_DNA': mutated_seq,
        **bw_features
    }

print(f"[*] Constructing multi-modal features and strict centering...")
results = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    feat = extract_and_build(row)
    if feat:
        merged = {**row.to_dict(), **feat}
        results.append(merged)

df_final = pd.DataFrame(results)

# ==========================================
# STEP 4: FINAL METADATA FORMATTING
# ==========================================
# Generate required missing metadata placeholders
df_final['Mutation_ID'] = df_final['GDC_Genomic_DNA_Change'] + "_" + df_final['TCGA_Patient_Barcode']
df_final['Mutation_Type_SNV_Indel'] = np.where(df_final['GDC_Genomic_DNA_Change'].str.contains('del|ins'), 'Indel', 'SNV')
df_final['CADD_Phred_Score'] = 0.0 # Placeholder for downstream processing
df_final['Is_TAD_Boundary'] = 0    # Placeholder
df_final['Distance_To_Nearest_TSS'] = 0 # Placeholder
df_final['True_Wild_Type_M_Value'] = np.nan # TCGA rarely provides matched normal; using NaN fallback

# Reorder columns strictly to your specification
final_columns = [
    # Metadata
    'probeID', 'chr', 'pos', 'Mutation_ID', 'Gene', 'Mutation_Type_SNV_Indel', 'CADD_Phred_Score',
    'Gene_Region', 'CpG_Island_Status', 'Is_TAD_Boundary', 'Distance_To_Nearest_TSS',
    'True_Wild_Type_Beta', 'True_Wild_Type_M_Value', 'True_Mutated_Beta', 'True_Mutated_M_Value', 
    # Sequence
    'Healthy_5000bp_DNA', 'Mutated_5000bp_DNA',
    # Tabular
    'Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 'Ref_H3K27me3_Signal', 
    'Ref_H3K9me3_Signal', 'Target_Base_PhyloP_100way'
]

# Drop anything not explicitly requested to keep it perfectly clean
df_final = df_final[[col for col in final_columns if col in df_final.columns]]

OUTPUT_PATH = os.path.join(BASE_DIR, "testing_data_phase2.csv")
df_final.to_csv(OUTPUT_PATH, index=False)

print(f"\n[✓] EXTRACTION COMPLETE!")
print(f"[✓] Saved {len(df_final)} strictly aligned pairings to: {OUTPUT_PATH}")
