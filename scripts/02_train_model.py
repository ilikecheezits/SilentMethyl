import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
import argparse
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import MultiOmicsDataset
from model.architecture import SilentMethylModel, perform_triton_surgery
from training.loss import FocalLossWithLogits
from model.lora_utils import inject_lora_adapters, load_lora_weights

def main():
    parser = argparse.ArgumentParser(description="Train SilentMethyl Model")
    parser.add_argument("--matrix_path", type=str, required=True, help="Path to the cleaned training data matrix CSV.")
    parser.add_argument("--dict_path", type=str, required=True, help="Path to the sequence dictionary pickle or CSV.")
    parser.add_argument("--region_vocab_path", type=str, required=True, help="Path to the region vocabulary pickle.")
    parser.add_argument("--island_vocab_path", type=str, required=True, help="Path to the island vocabulary pickle.")
    parser.add_argument("--save_dir", default="./checkpoints", help="Directory to save model checkpoints.")
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M", help="Hugging Face model path.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="Number of steps per epoch.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers.")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # --- RIGOROUS LOGGING SETUP ---
    log_file = os.path.join(args.save_dir, "training_debug.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    logger.info("--- EPIGENETICS MASTER LOOP (TRANSPARENT DEBUG MODE) ---")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # =========================================
    # Data Loading
    # =========================================
    logger.info("[*] Loading preprocessed data and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    matrix_df = pd.read_csv(args.matrix_path)
    dict_df = pd.read_csv(args.dict_path)
    
    dict_df = dict_df.loc[:, ~dict_df.columns.str.contains('^Unnamed')]
    id_col = 'probeID' if 'probeID' in dict_df.columns else dict_df.columns[0]
    
    if 'Healthy_5000bp_DNA' in dict_df.columns:
        logger.info("  -> [✓] Mapping 'Healthy_5000bp_DNA' to 'Mutated_5000bp_DNA'")
        dict_df['Mutated_5000bp_DNA'] = dict_df['Healthy_5000bp_DNA']
        
    meta_features = [
        'GC_Content', 'CpG_Count', 'CpG_OE_Ratio', 'GC_Skew', 
        'Shore_Asymmetry', 'FOXA1_Motifs', 'GATA3_Motifs', 'AP1_Motifs',
        'CTCF_Motifs', 'SP1_Motifs', 'TpG_CpA_Clock', 'Poly_A_Tracts',
        'Alu_Proxy', 'G4_Quadruplex_Proxy', 'ERE_Motifs', 'E_Box_Motifs',
        'YY1_Motifs', 'HRE_Motifs', 'TATA_Box_Present'
    ]
    
    for feat in meta_features:
        if feat in dict_df.columns:
            dict_df[f'Mut_{feat}'] = dict_df[feat]
            
    logger.info(f"  -> [✓] Mapped Biological Metadata features to 'Mut_' prefix.")
    
    GLOBAL_SEQ_DICT = {}
    for _, row in tqdm(dict_df.iterrows(), total=len(dict_df), desc="Building Dict"):
        row_dict = row.to_dict()
        pid = str(row_dict[id_col]).strip()
        GLOBAL_SEQ_DICT[pid] = row_dict

    REGION_VOCAB = joblib.load(args.region_vocab_path)
    ISLAND_VOCAB = joblib.load(args.island_vocab_path)

    # =========================================
    # Model Initialization
    # =========================================
    logger.info("--- INITIALIZING ARCHITECTURE ---")
    local_model_dir = perform_triton_surgery()
    config = AutoConfig.from_pretrained(local_model_dir, trust_remote_code=True)
    
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    
    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download(repo_id=args.model_path, filename="pytorch_model.bin")
    pretrained_state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.bert.load_state_dict(pretrained_state_dict, strict=False)
    
    model = inject_lora_adapters(model)
    logger.info("[✓] Model Architecture Built Successfully.")

    # =========================================
    # Data Merging & Splitting
    # =========================================
    logger.info("[*] Injecting DNA and Metadata from Sequence Dictionary...")
    
    mut_cols = [c for c in dict_df.columns if c.startswith('Mut_')]
    cols_to_extract = [id_col, 'CpG_chrm', 'Mutated_5000bp_DNA'] + mut_cols
    
    dna_map = dict_df[cols_to_extract].copy()
    dna_map.rename(columns={id_col: 'CpG_Target', 'CpG_chrm': 'Chromosome'}, inplace=True)

    matrix_df = matrix_df.merge(dna_map, on='CpG_Target', how='inner')
    matrix_df['True_Pos'] = np.nan 
    
    logger.info(f"  -> [✓] Coordinates, DNA & Metadata merged. Samples: {len(matrix_df)}")
    
    unique_genes = matrix_df['Gene'].dropna().unique() 
    train_val_genes, test_genes = train_test_split(unique_genes, test_size=0.10, random_state=42)
    train_genes, val_genes = train_test_split(train_val_genes, test_size=0.15, random_state=42)

    train_matrix = matrix_df[matrix_df['Gene'].isin(train_genes)].reset_index(drop=True)
    val_matrix = matrix_df[matrix_df['Gene'].isin(val_genes)].reset_index(drop=True)
    
    train_dataset = MultiOmicsDataset(train_matrix, GLOBAL_SEQ_DICT, REGION_VOCAB, ISLAND_VOCAB, tokenizer)
    val_dataset = MultiOmicsDataset(val_matrix, GLOBAL_SEQ_DICT, REGION_VOCAB, ISLAND_VOCAB, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # =========================================
    # Training Setup
    # =========================================
    dna_params = [p for n, p in model.named_parameters() if p.requires_grad and ('lora' in n or 'dna_head' in n or 'dna_weight' in n)]
    meta_params = [p for n, p in model.named_parameters() if p.requires_grad and ('meta_head' in n or 'emb' in n or 'meta_weight' in n)]

    optimizer = optim.AdamW([
        {'params': dna_params, 'lr': args.lr, 'weight_decay': 0.01},
        {'params': meta_params, 'lr': args.lr * 0.1, 'weight_decay': 0.01}
    ])

    criterion_class = FocalLossWithLogits(alpha=0.5, gamma=2.0)
    criterion_reg = nn.HuberLoss(delta=0.1)

    SAVE_PATH = os.path.join(args.save_dir, "SilentMethyl_Best_Weights.pth")
    best_val_loss = float('inf')

    TOTAL_STEPS = args.steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS // args.grad_accum_steps)
    
    # =========================================
    # Training Loop
    # =========================================
    logger.info("[*] LAUNCHING TRAINING WITH LIVE DIAGNOSTICS...")
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_auroc': [], 'batch_loss': []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_total_loss = 0.0
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs} [TRAIN]")
        train_iter = iter(train_loader)

        for step in pbar:
            try: batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            num_feats = batch['numerical_features'].to(device)
            reg_idx, isl_idx = batch['region_idx'].to(device), batch['island_idx'].to(device)
            tata_idx = batch['tata_idx'].to(device)
            true_beta = batch['targets'].to(device).view(-1, 1)

            global_step = (epoch - 1) * args.steps_per_epoch + step
            is_warmup = global_step < 1000
            
            if not is_warmup:
                dropout_prob = 0.75 
                batch_size_current = num_feats.size(0)
                modality_mask = (torch.rand(batch_size_current, 1, device=device) > dropout_prob).float()
                num_feats = num_feats * modality_mask
                mask_long = modality_mask.squeeze(-1).long()
                reg_idx = reg_idx * mask_long
                isl_idx = isl_idx * mask_long
                tata_idx = tata_idx * mask_long
            else:
                num_feats = num_feats * 0

            # --- MODEL FORWARD PASS ---
            combined_logits, debug_dict = model(
                input_ids, attention_mask, num_feats, reg_idx, isl_idx, tata_idx, dna_only=is_warmup
            )
            
            # --- TELEMETRY CHECK (Runs exactly once at the start of each Epoch) ---
            if step == 0:
                logger.info(f"\n[🔬 EPOCH {epoch} MODEL TELEMETRY 🔬]")
                logger.info(f"Is Warmup Active? {is_warmup}")
                logger.info(f"Learnable DNA Weight  : {debug_dict['dna_weight_val']:.4f}")
                logger.info(f"Learnable Meta Weight : {debug_dict['meta_weight_val']:.4f}")
                if not is_warmup:
                    logger.info(f"Tabular Min/Max/Mean  : [{num_feats.min():.2f}, {num_feats.max():.2f}, {num_feats.float().mean():.2f}] (Proves data isn't starved!)")
                    logger.info(f"Sample DNA Logit      : {debug_dict['dna_logits'][0].item():.4f}")
                    logger.info(f"Sample Meta Logit     : {debug_dict['meta_logits'][0].item():.4f}")
                logger.info(f"Target True Beta      : {true_beta[0].item():.4f}")
                logger.info("-" * 40)

            pred_beta = torch.sigmoid(combined_logits)
            target_class = (true_beta > 0.5).float()

            loss_class = criterion_class(combined_logits, target_class)
            loss_reg = criterion_reg(pred_beta, true_beta)

            loss = (0.25 * loss_class + 0.75 * loss_reg) / args.grad_accum_steps

            loss.backward()

            real_loss = loss.item() * args.grad_accum_steps
            train_total_loss += real_loss
            history['batch_loss'].append(real_loss)

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
            pbar.set_postfix({'Loss': f"{real_loss:.4f}", 'VRAM': f"{vram_mb:.0f}MB"})

        avg_train_loss = train_total_loss / args.steps_per_epoch

        # --- VALIDATION ---
        model.eval()
        val_total_loss = 0
        VAL_STEPS = min(500, len(val_loader))
        val_iter = iter(val_loader)
        all_true, all_pred = [], []

        with torch.no_grad():
            for _ in range(VAL_STEPS):
                try: batch = next(val_iter)
                except StopIteration: break

                input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                num_feats = batch['numerical_features'].to(device)
                reg_idx, isl_idx = batch['region_idx'].to(device), batch['island_idx'].to(device)
                tata_idx = batch['tata_idx'].to(device)
                true_beta = batch['targets'].to(device).view(-1, 1)

                combined_logits, _ = model(input_ids, attention_mask, num_feats, reg_idx, isl_idx, tata_idx, dna_only=False)
                pred_beta = torch.sigmoid(combined_logits)

                loss_class = criterion_class(combined_logits, (true_beta > 0.5).float())
                loss_reg = criterion_reg(pred_beta, true_beta)
                val_total_loss += (0.25 * loss_class + 0.75 * loss_reg).item()

                all_true.extend(true_beta.cpu().numpy().flatten())
                all_pred.extend(pred_beta.cpu().numpy().flatten())

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        avg_val_loss = val_total_loss / VAL_STEPS

        val_mae = mean_absolute_error(all_true, all_pred)
        val_rmse = np.sqrt(mean_squared_error(all_true, all_pred))

        logger.info(f"\n{'='*50}")
        logger.info(f"--- SUB-EPOCH {epoch} SUMMARY ---")
        logger.info(f"-> Train Loss:     {avg_train_loss:.4f}")
        logger.info(f"-> Val Loss:       {avg_val_loss:.4f}")
        logger.info(f"-> Val RMSE:       {val_rmse:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            logger.info(f"[✓] NEW HIGH SCORE! Weights secured.")
        logger.info(f"{'='*50}\n")

if __name__ == "__main__":
    main()