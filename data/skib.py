import pandas as pd
import numpy as np
import gzip
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_CSV = "actual_data/actual_testing_data.csv"
OUTPUT_CSV = "actual_data/actual_testing_data_filled.csv"

# [!] IMPORTANT: Point this to the exact same file you used in your training notebook!
METH_PATH = "TCGA-BRCA.methylation450.tsv.gz"

print(f"[*] Loading testing dataset: {INPUT_CSV}")
df_test = pd.read_csv(INPUT_CSV)
existing_probes = set(df_test['probeID'].unique())

# ==========================================
# 1. SCAN AND LOAD HEALTHY DATA
# ==========================================
print('[*] Scanning Methylation Data for Healthy (-11) Columns...')
with gzip.open(METH_PATH, 'rt') as f:
    meth_header = f.readline().strip().split('\t')

probe_col_name = meth_header[0]
healthy_cols = [col for col in meth_header if '-11' in col]

print(f'[*] Found {len(healthy_cols)} healthy (-11) sample columns.')
if len(healthy_cols) == 0:
    raise ValueError('No healthy columns found! Check that METH_PATH points to your training matrix.')

print('[*] Loading strictly necessary data into RAM (using float32)...')
df_meth = pd.read_csv(
    METH_PATH,
    sep='\t',
    usecols=[probe_col_name] + healthy_cols,
    dtype={col: 'float32' for col in healthy_cols}  # explicit float to prevent object NaNs
)
df_meth.rename(columns={probe_col_name: 'probeID'}, inplace=True)

# Filter down ONLY to the probes present in our test dataset to save RAM
df_meth = df_meth[df_meth['probeID'].isin(existing_probes)].copy()

# ==========================================
# 2. VECTORIZED MATH
# ==========================================
print('[*] Cleaning and calculating baselines vectorially...')
all_nan_mask = df_meth[healthy_cols].isna().all(axis=1)
print(f'[!] Dropping {all_nan_mask.sum()} test probes that have ALL healthy values NaN.')
df_meth = df_meth[~all_nan_mask].copy()

# Vectorized median (ignoring individual NaNs)
beta_vals = df_meth[healthy_cols].values
df_meth['True_Wild_Type_Beta'] = np.nanmedian(beta_vals, axis=1)

# Vectorized M-Value with safe clip
beta_clipped = np.clip(df_meth['True_Wild_Type_Beta'].values, 0.0001, 0.9999)
df_meth['True_Wild_Type_M_Value'] = np.log2(beta_clipped / (1.0 - beta_clipped))

target_map = df_meth[['probeID', 'True_Wild_Type_Beta', 'True_Wild_Type_M_Value']]

# ==========================================
# 3. MERGE AND SAVE
# ==========================================
print('[*] Merging newly calculated baselines into testing data...')

# Drop the empty placeholder columns from the test set if they exist
cols_to_drop = [c for c in ['True_Wild_Type_Beta', 'True_Wild_Type_M_Value'] if c in df_test.columns]
df_test.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# Merge on probeID
df_test_filled = pd.merge(df_test, target_map, on='probeID', how='inner')

print(f"\n[✓] UPDATE COMPLETE! Saved to: {OUTPUT_CSV}")
print(f"Final testing dataset size: {df_test_filled.shape[0]} rows.")
df_test_filled.to_csv(OUTPUT_CSV, index=False)
