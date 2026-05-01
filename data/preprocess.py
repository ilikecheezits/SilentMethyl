import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler

def clean_training_data(raw_matrix_path, output_path):
    """Filters out confounders (e.g., ensuring 100% Female cohort for Breast Cancer)."""
    df = pd.read_csv(raw_matrix_path)
    
    # Strictly enforce Healthy Normal baseline and Female cohort
    if 'Gender' in df.columns:
        df = df[df['Gender'].str.upper() == 'FEMALE']
    if 'Sample_Type' in df.columns:
        df = df[df['Sample_Type'] == 'Solid Tissue Normal']
        
    df.to_csv(output_path, index=False)
    print(f"[✓] Cleaned training data saved to {output_path} (N={len(df)})")
    return df

def preprocess_data(matrix_df, dict_df, base_dir):
    """
    Performs imputation and normalization, and saves scalers.
    """
    print("[*] Imputing missing data to prevent crashes...")
    mean_age = matrix_df['Age'].mean()
    matrix_df['Age'] = matrix_df['Age'].fillna(mean_age)
    dict_df['Gene_Region'] = dict_df['Gene_Region'].fillna('Unknown')
    dict_df['CpG_Island_Status'] = dict_df['CpG_Island_Status'].fillna('Unknown')

    print("[*] Normalizing Biological Features...")
    scaler_age = StandardScaler()
    matrix_df[['Age']] = scaler_age.fit_transform(matrix_df[['Age']])

    scaler_seq = StandardScaler()
    seq_cols = [
        'GC_Content', 'CpG_Count', 'CpG_OE_Ratio', 'GC_Skew', 'Shore_Asymmetry',
        'FOXA1_Motifs', 'GATA3_Motifs', 'AP1_Motifs', 'CTCF_Motifs', 'SP1_Motifs',
        'TpG_CpA_Clock', 'Poly_A_Tracts', 'Alu_Proxy', 'G4_Quadruplex_Proxy',
        'ERE_Motifs', 'E_Box_Motifs', 'YY1_Motifs', 'HRE_Motifs'
    ]
    existing_cols = [c for c in seq_cols if c in dict_df.columns]
    dict_df[existing_cols] = scaler_seq.fit_transform(dict_df[existing_cols])

    print("[✓] Features Normalized (Mean=0, StdDev=1).")

    SCALER_AGE_PATH = os.path.join(base_dir, "Scaler_Age.pkl")
    SCALER_SEQ_PATH = os.path.join(base_dir, "Scaler_Seq.pkl")
    joblib.dump(scaler_age, SCALER_AGE_PATH)
    joblib.dump(scaler_seq, SCALER_SEQ_PATH)
    print(f"[✓] Scalers preserved and exported to {base_dir}.")
    
    return matrix_df, dict_df

def build_vocabularies(dict_df, base_dir):
    """Dynamically builds embedding vocabularies from raw data."""
    regions = set(dict_df['Gene_Region'].unique())
    islands = set(dict_df['CpG_Island_Status'].unique())
    
    region_vocab = {name: idx for idx, name in enumerate(regions)}
    region_vocab['Unknown'] = region_vocab.get('Unknown', len(region_vocab))
    
    island_vocab = {name: idx for idx, name in enumerate(islands)}
    island_vocab['Unknown'] = island_vocab.get('Unknown', len(island_vocab))
    
    joblib.dump(dict_df.set_index('probeID').to_dict('index'), os.path.join(base_dir, "SilentMethyl_SeqDict.pkl"))
    joblib.dump(region_vocab, os.path.join(base_dir, "SilentMethyl_RegionVocab.pkl"))
    joblib.dump(island_vocab, os.path.join(base_dir, "SilentMethyl_IslandVocab.pkl"))
    print(f"[✓] Vocabularies preserved and exported to {base_dir}.")

def preprocess_inference_data(df, scaler_age, scaler_seq):
    """Preprocesses the inference data using loaded scalers."""
    df['Age_scaled'] = scaler_age.transform(df[['Age']])[0]

    feature_cols = [
        'GC_Content', 'CpG_Count', 'CpG_OE_Ratio', 'GC_Skew', 'Shore_Asymmetry',
        'FOXA1_Motifs', 'GATA3_Motifs', 'AP1_Motifs', 'CTCF_Motifs', 'SP1_Motifs',
        'TpG_CpA_Clock', 'Poly_A_Tracts', 'Alu_Proxy', 'G4_Quadruplex_Proxy',
        'ERE_Motifs', 'E_Box_Motifs', 'YY1_Motifs', 'HRE_Motifs'
    ]
    wt_feat_cols = [f"WT_{c}" for c in feature_cols]
    mut_feat_cols = [f"Mut_{c}" for c in feature_cols]

    wt_scaled = scaler_seq.transform(df[wt_feat_cols])
    mut_scaled = scaler_seq.transform(df[mut_feat_cols])

    df['wt_scaled_features'] = list(wt_scaled)
    df['mut_scaled_features'] = list(mut_scaled)

    return df
