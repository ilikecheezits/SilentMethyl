import torch
from torch.utils.data import Dataset

class MultiOmicsDataset(Dataset):
    def __init__(self, matrix_df, seq_dict, region_vocab, island_vocab, tokenizer, max_length=512):
        self.matrix_df = matrix_df.reset_index(drop=True)
        self.seq_dict = seq_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.region_vocab = region_vocab
        self.island_vocab = island_vocab

    def __len__(self):
        return len(self.matrix_df)

    def __getitem__(self, idx):
        row = self.matrix_df.iloc[idx]
        dna_5000 = str(row['Mutated_5000bp_DNA']).upper()
        dna_center = dna_5000[2000:3000] if len(dna_5000) >= 3000 else dna_5000

        encoding = self.tokenizer(
            dna_center, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )

        numerical_features = torch.tensor([
            row.get('Age', 0),
            row.get('Mut_GC_Content', 0), row.get('Mut_CpG_Count', 0),
            row.get('Mut_CpG_OE_Ratio', 0), row.get('Mut_GC_Skew', 0),
            row.get('Mut_Shore_Asymmetry', 0), row.get('Mut_FOXA1_Motifs', 0),
            row.get('Mut_GATA3_Motifs', 0), row.get('Mut_AP1_Motifs', 0),
            row.get('Mut_CTCF_Motifs', 0), row.get('Mut_SP1_Motifs', 0),
            row.get('Mut_TpG_CpA_Clock', 0), row.get('Mut_Poly_A_Tracts', 0),
            row.get('Mut_Alu_Proxy', 0), row.get('Mut_G4_Quadruplex_Proxy', 0),
            row.get('Mut_ERE_Motifs', 0), row.get('Mut_E_Box_Motifs', 0),
            row.get('Mut_YY1_Motifs', 0), row.get('Mut_HRE_Motifs', 0)
        ], dtype=torch.float32)

        region_idx = torch.tensor(self.region_vocab.get(row.get('Gene_Region', 'Unknown'), 0), dtype=torch.long)
        island_idx = torch.tensor(self.island_vocab.get(row.get('CpG_Island_Status', 'Unknown'), 0), dtype=torch.long)
        tata_idx = torch.tensor(row.get('Mut_TATA_Box_Present', 0), dtype=torch.long)
        target_beta = torch.tensor([row.get('True_Mutated_Beta', 0.0)], dtype=torch.float32)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'numerical_features': numerical_features,
            'region_idx': region_idx,
            'island_idx': island_idx,
            'tata_idx': tata_idx,
            'targets': target_beta
        }

class GenomicVariantDataset(Dataset):
    def __init__(self, df, tokenizer, region_vocab, island_vocab, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.region_vocab = region_vocab
        self.island_vocab = island_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Text Processing (Wild-Type vs Mutated)
        wt_seq = str(row['Healthy_5000bp_DNA']).upper()
        mut_seq = str(row['Mutated_5000bp_DNA']).upper()
        wt_center = wt_seq[2000:3000] if len(wt_seq) >= 3000 else wt_seq
        mut_center = mut_seq[2000:3000] if len(mut_seq) >= 3000 else mut_seq

        wt_inputs = self.tokenizer(wt_center, return_tensors="pt", truncation=True, max_length=self.max_length, padding="max_length")
        mut_inputs = self.tokenizer(mut_center, return_tensors="pt", truncation=True, max_length=self.max_length, padding="max_length")

        # Vocab Processing
        region_idx = torch.tensor(self.region_vocab.get(row.get('Gene_Region', 'Unknown'), 0), dtype=torch.long)
        island_idx = torch.tensor(self.island_vocab.get(row.get('CpG_Island_Status', 'Unknown'), 0), dtype=torch.long)

        # Numerical Vectors
        wt_num_tensor = torch.tensor([
            row.get('Age', 0), row.get('WT_GC_Content', 0), row.get('WT_CpG_Count', 0), 
            row.get('WT_CpG_OE_Ratio', 0), row.get('WT_GC_Skew', 0), row.get('WT_Shore_Asymmetry', 0), 
            row.get('WT_FOXA1_Motifs', 0), row.get('WT_GATA3_Motifs', 0), row.get('WT_AP1_Motifs', 0), 
            row.get('WT_CTCF_Motifs', 0), row.get('WT_SP1_Motifs', 0), row.get('WT_TpG_CpA_Clock', 0), 
            row.get('WT_Poly_A_Tracts', 0), row.get('WT_Alu_Proxy', 0), row.get('WT_G4_Quadruplex_Proxy', 0), 
            row.get('WT_ERE_Motifs', 0), row.get('WT_E_Box_Motifs', 0), row.get('WT_YY1_Motifs', 0), 
            row.get('WT_HRE_Motifs', 0)
        ], dtype=torch.float32)

        mut_num_tensor = torch.tensor([
            row.get('Age', 0), row.get('Mut_GC_Content', 0), row.get('Mut_CpG_Count', 0), 
            row.get('Mut_CpG_OE_Ratio', 0), row.get('Mut_GC_Skew', 0), row.get('Mut_Shore_Asymmetry', 0), 
            row.get('Mut_FOXA1_Motifs', 0), row.get('Mut_GATA3_Motifs', 0), row.get('Mut_AP1_Motifs', 0), 
            row.get('Mut_CTCF_Motifs', 0), row.get('Mut_SP1_Motifs', 0), row.get('Mut_TpG_CpA_Clock', 0), 
            row.get('Mut_Poly_A_Tracts', 0), row.get('Mut_Alu_Proxy', 0), row.get('Mut_G4_Quadruplex_Proxy', 0), 
            row.get('Mut_ERE_Motifs', 0), row.get('Mut_E_Box_Motifs', 0), row.get('Mut_YY1_Motifs', 0), 
            row.get('Mut_HRE_Motifs', 0)
        ], dtype=torch.float32)

        return {
            'mutation_id': row.get('Mutation_ID', 'Unknown'),
            'gene': row.get('Gene', 'Unknown'),
            'hgvsp': row.get('HGVSp_Protein_Notation', 'Unknown'),
            'true_beta': row.get('True_Mutated_Beta', 0.0),
            'region_idx': region_idx,
            'island_idx': island_idx,
            'wt_input_ids': wt_inputs['input_ids'].squeeze(0),
            'wt_attention_mask': wt_inputs['attention_mask'].squeeze(0),
            'wt_num_tensor': wt_num_tensor,
            'wt_tata_idx': torch.tensor(row.get('WT_TATA_Box_Present', 0), dtype=torch.long),
            'mut_input_ids': mut_inputs['input_ids'].squeeze(0),
            'mut_attention_mask': mut_inputs['attention_mask'].squeeze(0),
            'mut_num_tensor': mut_num_tensor,
            'mut_tata_idx': torch.tensor(row.get('Mut_TATA_Box_Present', 0), dtype=torch.long)
        }