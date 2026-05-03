import torch
from torch.utils.data import Dataset

class MultiOmicsDataset(Dataset):
    def __init__(self, matrix_df, seq_dict, region_vocab, island_vocab, tokenizer, max_length=512):
        self.matrix_df = matrix_df.reset_index(drop=True)
        # We keep seq_dict in the init signature so we don't have to change 02_train_model.py, 
        # but we won't actually use it since the matrix has everything natively.
        self.seq_dict = seq_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.region_vocab = region_vocab
        self.island_vocab = island_vocab

    def __len__(self):
        return len(self.matrix_df)

    def __getitem__(self, idx):
        # FIX 1: Use self.matrix_df instead of the undefined self.df
        row = self.matrix_df.iloc[idx]
        
        # FIX 2: Pull the mutated sequence directly from the matrix row
        dna_5000 = str(row['Mutated_5000bp_DNA']).upper()
        
        # DIVIDE AND CONQUER: Give Language Model the center 1000bp only
        dna_center = dna_5000[2000:3000] if len(dna_5000) >= 3000 else dna_5000

        encoding = self.tokenizer(
            dna_center, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )

        # FIX 3: Pull all numerical features directly from the 'Mut_' columns in the matrix
        numerical_features = torch.tensor([
            row['Age'],
            row.get('Mut_GC_Content', 0), 
            row.get('Mut_CpG_Count', 0),
            row.get('Mut_CpG_OE_Ratio', 0), 
            row.get('Mut_GC_Skew', 0),
            row.get('Mut_Shore_Asymmetry', 0), 
            row.get('Mut_FOXA1_Motifs', 0),
            row.get('Mut_GATA3_Motifs', 0), 
            row.get('Mut_AP1_Motifs', 0),
            row.get('Mut_CTCF_Motifs', 0), 
            row.get('Mut_SP1_Motifs', 0),
            row.get('Mut_TpG_CpA_Clock', 0), 
            row.get('Mut_Poly_A_Tracts', 0),
            row.get('Mut_Alu_Proxy', 0), 
            row.get('Mut_G4_Quadruplex_Proxy', 0),
            row.get('Mut_ERE_Motifs', 0), 
            row.get('Mut_E_Box_Motifs', 0),
            row.get('Mut_YY1_Motifs', 0), 
            row.get('Mut_HRE_Motifs', 0)
        ], dtype=torch.float32)

        # FIX 4: Pull vocabulary columns directly from the row
        region_idx = torch.tensor(self.region_vocab.get(row.get('Gene_Region', 'Unknown'), 0), dtype=torch.long)
        island_idx = torch.tensor(self.island_vocab.get(row.get('CpG_Island_Status', 'Unknown'), 0), dtype=torch.long)
        tata_idx = torch.tensor(row.get('Mut_TATA_Box_Present', 0), dtype=torch.long)
        
        # FIX 5: Target the correct True_Mutated_Beta column
        target_beta = torch.tensor([row['True_Mutated_Beta']], dtype=torch.float32)

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
        self.df = df
        self.tokenizer = tokenizer
        self.region_vocab = region_vocab
        self.island_vocab = island_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Shared data
        region_idx = torch.tensor([self.region_vocab.get(row['Gene_Region'], self.region_vocab.get('Unknown', 0))], dtype=torch.long)
        island_idx = torch.tensor([self.island_vocab.get(row['CpG_Island_Status'], self.island_vocab.get('Unknown', 0))], dtype=torch.long)
        age_scaled = row['Age_scaled']

        # Wild-Type data
        wt_seq = str(row['Healthy_5000bp_DNA'])[2000:3000]
        wt_inputs = self.tokenizer(wt_seq, return_tensors="pt", truncation=True, max_length=self.max_length, padding="max_length")
        wt_num_tensor = torch.tensor([[age_scaled] + row['wt_scaled_features']], dtype=torch.float32)
        wt_tata_idx = torch.tensor([int(row['WT_TATA_Box_Present'])], dtype=torch.long)

        # Mutated data
        mut_seq = str(row['Mutated_5000bp_DNA'])[2000:3000]
        mut_inputs = self.tokenizer(mut_seq, return_tensors="pt", truncation=True, max_length=self.max_length, padding="max_length")
        mut_num_tensor = torch.tensor([[age_scaled] + row['mut_scaled_features']], dtype=torch.float32)
        mut_tata_idx = torch.tensor([int(row['Mut_TATA_Box_Present'])], dtype=torch.long)
        
        return {
            'mutation_id': row['Mutation_ID'],
            'gene': row['Gene'],
            'hgvsp': row['HGVSp_Protein_Notation'],
            'true_beta': row['True_Mutated_Beta'],
            'region_idx': region_idx.squeeze(0),
            'island_idx': island_idx.squeeze(0),
            'wt_input_ids': wt_inputs['input_ids'].squeeze(0),
            'wt_attention_mask': wt_inputs['attention_mask'].squeeze(0),
            'wt_num_tensor': wt_num_tensor.squeeze(0),
            'wt_tata_idx': wt_tata_idx.squeeze(0),
            'mut_input_ids': mut_inputs['input_ids'].squeeze(0),
            'mut_attention_mask': mut_inputs['attention_mask'].squeeze(0),
            'mut_num_tensor': mut_num_tensor.squeeze(0),
            'mut_tata_idx': mut_tata_idx.squeeze(0)
        }