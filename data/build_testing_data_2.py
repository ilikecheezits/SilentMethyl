import os
import requests
import json
import pandas as pd
import numpy as np
import re
import gzip
from tqdm import tqdm
from pyfaidx import Fasta
import concurrent.futures
import pyBigWig

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = "/ocean/projects/med250012p/szhang37/SilentMethyl/data/"
os.makedirs(BASE_DIR, exist_ok=True)

GDC_URL = "https://api.gdc.cancer.gov/ssms"
CBIO_URL = "https://www.cbioportal.org/api"
STUDY_ID = "brca_tcga"
MAX_THREADS = 16

FASTA_PATH = os.path.join(BASE_DIR, "hg38.fa")
MANIFEST_PATH = os.path.join(BASE_DIR, "HM450.hg38.manifest.tsv.gz")

BASE_REF = os.path.join(BASE_DIR, "reference/")
BW_PATHS = {
    "Ref_ATAC_Signal": BASE_REF + "ATAC_seq.bw",
    "Ref_H3K4me3_Signal": BASE_REF + "H3K4me3.bw",
    "Ref_H3K27ac_Signal": BASE_REF + "H3K27ac.bw",
    "Ref_H3K27me3_Signal": BASE_REF + "H3K27me3.bw",
    "Ref_H3K9me3_Signal": BASE_REF + "H3K9me3.bw",
    "Target_Base_PhyloP_100way_1": BASE_REF + "hg38.phyloP100way.bw",
    "Target_Base_PhyloP_100way_2": BASE_REF + "hg38.phyloP100way.bw"    
}

def get_bw_signal(bw_obj, chrom, start, end):
    """Safely extracts mean signal from a BigWig file."""
    try:
        if chrom not in bw_obj.chroms():
            if chrom.replace('chr', '') in bw_obj.chroms():
                chrom = chrom.replace('chr', '')
            else:
                return 0.0
        stat = bw_obj.stats(chrom, start, end, type="mean")
        return float(stat[0]) if stat and stat[0] is not None else 0.0
    except Exception:
        return 0.0

print("==========================================")
print("--- STEP 1: ENVIRONMENT & GENOME SETUP ---")
print("==========================================")

print("[*] Loading Human Genome into Memory...")
genome = Fasta(FASTA_PATH)

bw_handles = {}
for name, path in BW_PATHS.items():
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

print("[*] Pinging NIH GDC Supercomputers for sSNVs...")
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
    "size": "500000"
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

unique_genes = list(set([v['gene'] for v in mutation_groups.values() if v['gene'] != 'Unknown']))

print(f"[*] Attempting to map {len(unique_genes)} genes instantly...")
gene_to_entrez = {}

try:
    hgnc_url = "https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&col=md_eg_id&status=Approved&hgnc_dbt=dic&order_by=gd_app_sym_sort&format=text&submit=submit"
    hgnc_df = pd.read_csv(hgnc_url, sep='\t', low_memory=False)
    valid_hgnc = hgnc_df.dropna(subset=['Approved symbol', 'NCBI Gene ID'])
    gene_to_entrez = dict(zip(valid_hgnc['Approved symbol'], valid_hgnc['NCBI Gene ID'].astype(int)))
    print(f"[+] HGNC instant mapping successful!")
except Exception as e:
    print(f"[!] HGNC mapping failed. Using cBioPortal API fallback...")
    
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
def fetch_patient_data(item):
    local_hits = []
    gene_symbol = item['gene']
    entrez_id = gene_to_entrez.get(gene_symbol)
    
    if not entrez_id: return []

    try:
        normal_barcodes = [p[:-2] + "11" for p in item['patients']]
        combined_search = item['patients'] + normal_barcodes

        meth_data = requests.post(f"{CBIO_URL}/molecular-profiles/{METH_PROFILE}/molecular-data/fetch",
                                  json={"entrezGeneIds": [int(entrez_id)], "sampleIds": combined_search}).json()

        if meth_data:
            meth_dict = {d['sampleId']: d.get('value', np.nan) for d in meth_data if 'sampleId' in d and 'value' in d}

            for patient in item['patients']:
                tumor_beta = meth_dict.get(patient, np.nan)
                normal_barcode = patient[:-2] + "11"
                normal_beta = meth_dict.get(normal_barcode, np.nan) 
                
                if pd.notna(tumor_beta):
                    beta_safe = max(0.0001, min(0.9999, tumor_beta))
                    m_val = np.log2(beta_safe / (1 - beta_safe))
                    
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
                        "Matched_Normal_Beta": normal_beta, # Might be NaN, that's okay
                        "Matched_Normal_M_Value": wt_m_val
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
    print("\n[!] 0 hits. No patients had complete paired data.")
    raise SystemExit

print("\n==========================================")
print("--- STEP 3: SEQUENCE & EPIGENETIC INJECTION ---")
print("==========================================")

print("[*] Loading Wanding Zhou hg38 Manifest...")
df_manifest = pd.read_csv(MANIFEST_PATH, sep='\t', usecols=['probeID', 'CpG_chrm', 'CpG_beg'])
df_manifest.rename(columns={'CpG_chrm': 'chr', 'CpG_beg': 'pos'}, inplace=True)
valid_chrs = set([f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'])
df_manifest = df_manifest[df_manifest['chr'].isin(valid_chrs)].dropna().reset_index(drop=True)
df_manifest['pos'] = df_manifest['pos'].astype(int)

mutation_regex = re.compile(r'(chr[0-9XY]+):g\.(\d+)([A-Z]+)>([A-Z]+)')

def extract_and_build(row):
    match = mutation_regex.match(row['GDC_Genomic_DNA_Change'])
    if not match: return None

    chrom, mut_pos, ref_allele, mut_allele = match.groups()
    mut_pos = int(mut_pos)
    mut_pos_0based = mut_pos - 1  # VCF format is 1-based, convert to 0-based
    
    # 1. Find the closest target CpG site using 0-based math
    chrm_data = df_manifest[df_manifest['chr'] == chrom]
    if chrm_data.empty: return None
    idx = (np.abs(chrm_data['pos'] - mut_pos_0based)).argmin()
    closest_probe = chrm_data.iloc[idx]
    
    cpg_pos = int(closest_probe['pos'])
    probe_id = closest_probe['probeID']

    # 2. Strict Boundary Check (Mutation must be within our window)
    offset = mut_pos_0based - cpg_pos
    if abs(offset) > 2499:
        return None

    # 3. Build Centered WT Sequence (Same 0-based logic as training)
    start_idx = cpg_pos - 2499
    end_idx = start_idx + 5000 
    
    if start_idx < 0 or end_idx > len(genome[chrom]): return None
    
    healthy_seq = str(genome[chrom][start_idx:end_idx]).upper()
    
    # Strict filter to match training data
    if healthy_seq[2499:2501] != "CG":
        return None
    
    # 4. Inject Mutation
    mut_idx = 2499 + offset
    if healthy_seq[mut_idx : mut_idx + len(ref_allele)] != ref_allele:
        return None # Sequence doesn't match reference allele, VCF mismatch
        
    mutated_seq = healthy_seq[:mut_idx] + mut_allele + healthy_seq[mut_idx + len(ref_allele):]

    # 5. Extract Tabular Epigenetics (100bp / 1bp for PhyloP)
    start_100 = max(0, cpg_pos - 49)
    end_100 = cpg_pos + 51
    
    bw_features = {}
    for name, bw in bw_handles.items():
        if name == "Target_Base_PhyloP_100way_1":
            val = get_bw_signal(bw, chrom, cpg_pos, cpg_pos + 1)
        elif name == "Target_Base_PhyloP_100way_2":
            val = get_bw_signal(bw, chrom, cpg_pos + 1, cpg_pos + 2)
        else:
            val = get_bw_signal(bw, chrom, start_100, end_100)
        bw_features[name] = val

    return {
        'probeID': probe_id,
        'chr': chrom,
        'pos': cpg_pos,
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

METH_PATH = os.path.join(BASE_DIR, "TCGA-BRCA.methylation450.tsv.gz")

print("\n==========================================")
print("--- STEP 3.5: FETCHING COHORT NORMAL BASELINES ---")
print("==========================================")
relevant_probes = set(df_final['probeID'])
print(f"[*] Mining local TCGA file for {len(relevant_probes)} cohort-average normal baselines...")

with gzip.open(METH_PATH, 'rt') as f:
    meth_header = f.readline().strip().split('\t')

probe_col_name = meth_header[0]
healthy_cols = [col for col in meth_header if '-11' in col]

probe_to_normal = {}
for chunk in pd.read_csv(METH_PATH, sep='\t', usecols=[probe_col_name] + healthy_cols, chunksize=50000):
    chunk.rename(columns={probe_col_name: 'probeID'}, inplace=True)
    chunk = chunk[chunk['probeID'].isin(relevant_probes)]
    if not chunk.empty:
        for _, row in chunk.iterrows():
            vals = pd.to_numeric(row[healthy_cols], errors='coerce').dropna()
            if not vals.empty:
                probe_to_normal[row['probeID']] = np.median(vals)

def get_cohort_m_val(beta):
    if pd.isna(beta): return np.nan
    b_safe = max(0.0001, min(0.9999, beta))
    return np.log2(b_safe / (1 - b_safe))

# If a patient had a matched normal, keep it. Otherwise, use the cohort median!
df_final['Cohort_Normal_Beta'] = df_final['probeID'].map(probe_to_normal)
df_final['True_Wild_Type_Beta'] = df_final['Matched_Normal_Beta'].combine_first(df_final['Cohort_Normal_Beta'])
df_final['True_Wild_Type_M_Value'] = df_final['True_Wild_Type_Beta'].apply(get_cohort_m_val)

# Drop rows where we STILL couldn't find a normal baseline (rare, but possible if probe is dropped)
df_final = df_final.dropna(subset=['True_Wild_Type_Beta'])
print(f"[✓] Baselines secured. {len(df_final)} fully matched pairs remain.")

# ==========================================
# STEP 4: FINAL METADATA FORMATTING
# ==========================================
# Clean column order to mirror training constraints perfectly
final_columns = [
    'chr', 'pos', 'probeID', 'Gene', 'GDC_Genomic_DNA_Change', 'TCGA_Patient_Barcode',
    'Healthy_5000bp_DNA', 'Mutated_5000bp_DNA',
    'True_Wild_Type_Beta', 'True_Wild_Type_M_Value',
    'True_Mutated_Beta', 'True_Mutated_M_Value',
    'Ref_ATAC_Signal', 'Ref_H3K4me3_Signal', 'Ref_H3K27ac_Signal', 
    'Ref_H3K27me3_Signal', 'Ref_H3K9me3_Signal', 
    'Target_Base_PhyloP_100way_1', 'Target_Base_PhyloP_100way_2'
]

df_final = df_final[[col for col in final_columns if col in df_final.columns]]

OUTPUT_PATH = os.path.join(BASE_DIR, "testing_data.csv")
df_final.to_csv(OUTPUT_PATH, index=False)

for bw in bw_handles.values():
    bw.close()

print(f"\n[✓] EXTRACTION COMPLETE!")
print(f"[✓] Saved {len(df_final)} precisely aligned, matched test pairings to: {OUTPUT_PATH}")
