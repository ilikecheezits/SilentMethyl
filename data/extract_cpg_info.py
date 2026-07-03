import pandas as pd
import requests
import re
import os

# --- Configuration ---
MANIFEST_PATH = "HM450.hg38.manifest.tsv.gz" # Ensure this path is correct
CANDIDATES = [
    {"gene": "MSRA", "mut_id": "chr8:g.10428264C>T", "probe_id": "cg14264678"},
    {"gene": "DDC", "mut_id": "chr7:g.50543912G>A", "probe_id": "cg05346287"},
    {"gene": "CHD5", "mut_id": "chr1:g.6128067C>T", "probe_id": "cg12135344"}
]

print("[*] Loading Illumina hg38 Manifest for CpG Context...")
# Load only the columns we need to save RAM
cols_to_use = ["probeID", "CpG_chrm", "CpG_beg", "UCSC_RefGene_Group", "Relation_to_Island"]
manifest = pd.read_csv(MANIFEST_PATH, sep="\t", usecols=cols_to_use, low_memory=False)
manifest.set_index("probeID", inplace=True)

def check_ucsc_ccre(chrom, pos):
    """Queries UCSC API for ENCODE cCRE overlaps at the exact mutation coordinate."""
    url = f"https://api.genome.ucsc.edu/getData/track?genome=hg38&track=encodeCcreCombined&chrom={chrom}&start={pos-1}&end={pos}"
    try:
        res = requests.get(url, timeout=5).json()
        if "encodeCcreCombined" in res and res["encodeCcreCombined"]:
            elements = [item.get("name", "Unknown_cCRE") for item in res["encodeCcreCombined"]]
            return ", ".join(elements)
        return "None"
    except Exception as e:
        return f"API_Error: {e}"

def parse_mutation(mut_id):
    """Extracts chromosome and 1-based position from GDC mutation string."""
    match = re.match(r"(chr[0-9XY]+):g\.(\d+)[A-Z]+>[A-Z]+", mut_id)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

results = []

print("[*] Annotating Candidates...")
for item in CANDIDATES:
    probe = item["probe_id"]
    mut_id = item["mut_id"]
    gene = item["gene"]
    
    chrom, mut_pos = parse_mutation(mut_id)
    
    if probe not in manifest.index:
        print(f"[!] Probe {probe} not found in manifest.")
        continue
        
    probe_data = manifest.loc[probe]
    cpg_pos = int(probe_data["CpG_beg"])
    
    # 1. Distance Calculation (Mutation Position - CpG Position)
    distance = mut_pos - cpg_pos
    
    # 2. CpG Context (Island, Shore, Shelf, Open Sea)
    cpg_context = probe_data["Relation_to_Island"]
    if pd.isna(cpg_context):
        cpg_context = "Open Sea"
        
    # 3. Gene Region (Promoter/TSS, Body, UTR)
    gene_region = probe_data["UCSC_RefGene_Group"]
    if pd.isna(gene_region):
        gene_region = "Intergenic"
        
    # 4. Regulatory Overlap (ENCODE cCREs via UCSC REST API)
    ccre_overlap = check_ucsc_ccre(chrom, mut_pos)
    
    results.append({
        "Gene": gene,
        "Mutation": mut_id,
        "Target_CpG": probe,
        "Distance_to_CpG": distance,
        "CpG_Context": cpg_context,
        "Gene_Region": gene_region,
        "ENCODE_cCRE_Overlap": ccre_overlap
    })

df_results = pd.DataFrame(results)
print("\n--- Final Annotations ---")
print(df_results.to_string(index=False))

OUTPUT_FILE = "candidate_annotations.csv"
df_results.to_csv(OUTPUT_FILE, index=False)
print(f"\n[✓] Saved to {OUTPUT_FILE}")