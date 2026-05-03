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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import MultiOmicsDataset
from model.architecture import SilentMethylModel, perform_triton_surgery
from training.loss import FocalLossWithLogits
from model.lora_utils import inject_lora_adapters, load_lora_weights

def main():
    parser = argparse.ArgumentParser(description="Train SilentMethyl Model")
    parser.add_argument("--matrix_path", type=str, required=True, help="Path to the cleaned training data matrix CSV.")
    parser.add_argument("--dict_path", type=str, required=True, help="Path to the sequence dictionary pickle.")
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

    print("--- EPIGENETICS MASTER LOOP (COMPETITION MODE) ---")

    # =========================================
    # Environment Setup
    # =========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # =========================================
    # Data Loading
    # =========================================
    print("[*] Loading preprocessed data...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    matrix_df = pd.read_csv(args.matrix_path)
    print("[*] Parsing Sequence Dictionary from CSV...")
    
    # 1. Read CSV normally (without forcing an index yet)
    dict_df = pd.read_csv(args.dict_path)
    
    # 2. Destroy the invisible row-number column if Excel/Pandas added it
    dict_df = dict_df.loc[:, ~dict_df.columns.str.contains('^Unnamed')]
    
    # 3. Now that the garbage is gone, the FIRST remaining column is the true ID column
    id_column = dict_df.columns[0]
    
    # 4. Set that specific column as the index and convert to dictionary
    GLOBAL_SEQ_DICT = dict_df.set_index(id_column).to_dict('index')
    REGION_VOCAB = joblib.load(args.region_vocab_path)
    ISLAND_VOCAB = joblib.load(args.island_vocab_path)

    # =========================================
    # Model Initialization
    # =========================================
    print("--- INITIALIZING ARCHITECTURE (WITH LOCAL SURGERY) ---")
    local_model_dir = perform_triton_surgery()
    config = AutoConfig.from_pretrained(local_model_dir, trust_remote_code=True)
    
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    
    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download(repo_id=args.model_path, filename="pytorch_model.bin")
    pretrained_state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.bert.load_state_dict(pretrained_state_dict, strict=False)
    
    model = inject_lora_adapters(model)
    print("[✓] Surgery Complete! Model Architecture Built Successfully.")


    # =========================================
    # Data Splitting
    # =========================================
    print("[*] Implementing Competition-Grade Splitting...")

    # Pull directly from the matrix column instead of the dictionary
    matrix_df['Chromosome'] = matrix_df['CpG_chrm']
    # Force the fallback to Sequential Binning (since exact position isn't in the matrix)
    matrix_df['True_Pos'] = None

    test_matrix = matrix_df[matrix_df['Chromosome'] == 'chr1'].copy()
    discovery_matrix = matrix_df[matrix_df['Chromosome'] != 'chr1'].copy()

    if discovery_matrix['True_Pos'].isnull().any():
        print("  -> [!] Exact positions not found. Engaging Sequential Index Binning.")
        discovery_matrix = discovery_matrix.sort_values(by='Chromosome').reset_index(drop=True)
        discovery_matrix['Block_ID'] = discovery_matrix['Chromosome'] + "_Block_" + (discovery_matrix.index // 50).astype(str)
    else:
        print("  -> [✓] Exact positions found. Engaging 1-Megabase Binning.")
        BLOCK_SIZE = 1_000_000
        discovery_matrix['Block_ID'] = discovery_matrix['Chromosome'] + "_MB_" + (discovery_matrix['True_Pos'] // BLOCK_SIZE).astype(str)

    unique_blocks = discovery_matrix['Block_ID'].unique()
    if len(unique_blocks) < 2:
        print("  -> [!] Smoke Test Detected: Too few blocks. Engaging probe-level fallback split.")
        train_matrix, val_matrix = train_test_split(discovery_matrix, test_size=0.15, random_state=args.seed)
    else:
        train_blocks, val_blocks = train_test_split(unique_blocks, test_size=0.15, random_state=args.seed)
        train_matrix = discovery_matrix[discovery_matrix['Block_ID'].isin(train_blocks)].copy()
        val_matrix = discovery_matrix[discovery_matrix['Block_ID'].isin(val_blocks)].copy()

    print(f"  -> Training Probes: {len(train_matrix)}")
    print(f"  -> Validation Probes: {len(val_matrix)}")
    print(f"  -> BLIND TEST Probes (Chr1): {len(test_matrix)}")

    train_dataset = MultiOmicsDataset(train_matrix, GLOBAL_SEQ_DICT, REGION_VOCAB, ISLAND_VOCAB, tokenizer)
    val_dataset = MultiOmicsDataset(val_matrix, GLOBAL_SEQ_DICT, REGION_VOCAB, ISLAND_VOCAB, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # =========================================
    # Training Setup
    # =========================================
    print("[*] Configuring Differential Optimizers (LoRA Boost Active)...")
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if 'lora' not in n and p.requires_grad]

    optimizer = optim.AdamW([
        {'params': lora_params, 'lr': args.lr, 'weight_decay': 0.01},
        {'params': head_params, 'lr': args.lr, 'weight_decay': 0.00}
    ])

    criterion_class = FocalLossWithLogits(alpha=0.5, gamma=2.0)
    criterion_reg = nn.HuberLoss(delta=0.1)

    SAVE_PATH = os.path.join(args.save_dir, "SilentMethyl_Best_Weights.pth")
    best_val_loss = float('inf')

    TOTAL_STEPS = args.steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS)
    
    # =========================================
    # Training Loop
    # =========================================
    print("[*] LAUNCHING TRAINING WITH LIVE DIAGNOSTICS...")
    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'val_auroc': [], 'val_pearson': [], 'batch_loss': []}

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

            pred_class_logits, pred_reg_logits = model(input_ids, attention_mask, num_feats, reg_idx, isl_idx, tata_idx)
            pred_beta = torch.sigmoid(pred_reg_logits)
            target_class = (true_beta > 0.5).float()

            loss_class = criterion_class(pred_class_logits, target_class)
            loss_reg = criterion_reg(pred_beta, true_beta)

            loss = (0.25 * loss_class + 0.75 * loss_reg) / args.grad_accum_steps

            if torch.isnan(loss):
                raise ValueError("Training halted to prevent weight corruption.")

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
        val_pbar = tqdm(range(VAL_STEPS), desc=f"Epoch {epoch} [VALIDATION]", leave=False)
        val_iter = iter(val_loader)

        all_true, all_pred = [], []

        with torch.no_grad():
            for step in val_pbar:
                try: batch = next(val_iter)
                except StopIteration: break

                input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                num_feats = batch['numerical_features'].to(device)
                reg_idx, isl_idx = batch['region_idx'].to(device), batch['island_idx'].to(device)
                tata_idx = batch['tata_idx'].to(device)
                true_beta = batch['targets'].to(device).view(-1, 1)

                pred_class_logits, pred_reg_logits = model(input_ids, attention_mask, num_feats, reg_idx, isl_idx, tata_idx)
                pred_beta = torch.sigmoid(pred_reg_logits)

                loss_class = criterion_class(pred_class_logits, (true_beta > 0.5).float())
                loss_reg = criterion_reg(pred_beta, true_beta)
                val_total_loss += (0.25 * loss_class + 0.75 * loss_reg).item()

                all_true.extend(true_beta.cpu().numpy().flatten())
                all_pred.extend(pred_beta.cpu().numpy().flatten())

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        avg_val_loss = val_total_loss / VAL_STEPS

        val_mae = mean_absolute_error(all_true, all_pred)
        val_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
        val_pearson, _ = pearsonr(all_true, all_pred) if np.std(all_pred) > 0 else (0.0, 0.0)

        try:
            val_auroc = roc_auc_score((all_true > 0.5).astype(int), all_pred)
        except ValueError:
            val_auroc = 0.5

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"{'='*50}")
        print(f"--- SUB-EPOCH {epoch} COMPETITION TELEMETRY ---")
        print(f"-> Train Loss:     {avg_train_loss:.4f}")
        print(f"-> Val Loss:       {avg_val_loss:.4f}")
        print(f"-> Val RMSE:       {val_rmse:.4f}")
        print(f"-> Val MAE:        {val_mae:.4f}")
        print(f"-> Val Pearson R:  {val_pearson:.4f}")
        print(f"-> Val AUROC:      {val_auroc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"[✓] NEW HIGH SCORE! Weights secured.")

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        sns.kdeplot(all_true, color='blue', label='True Biology', fill=True, alpha=0.3, ax=axes[0])
        sns.kdeplot(all_pred, color='red', label='AI Prediction', fill=True, alpha=0.3, ax=axes[0])
        axes[0].set_title(f"Distribution Alignment")
        axes[0].set_xlim(0, 1)
        axes[0].legend()

        epochs_range = range(1, epoch + 1)
        axes[1].plot(epochs_range, history['train_loss'], color='blue', marker='o', label='Train Loss')
        axes[1].plot(epochs_range, history['val_loss'], color='red', marker='s', label='Val Loss')
        axes[1].set_title(f"Macro Learning Curves")
        axes[1].set_xlabel("Sub-Epoch")
        axes[1].legend()

        axes[2].plot(history['batch_loss'], color='gray', alpha=0.3, label='Raw Batch Loss')
        if len(history['batch_loss']) > 50:
            smoothed = np.convolve(history['batch_loss'], np.ones(50)/50, mode='valid')
            axes[2].plot(np.arange(49, len(history['batch_loss'])), smoothed, color='blue', label='Smoothed (n=50)')
        axes[2].set_title(f"Micro Batch Telemetry")
        axes[2].set_xlabel("Global Batch Step")
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, f"training_history_epoch_{epoch}.png"))
        #plt.show()
        print(f"{'='*50}")

if __name__ == "__main__":
    main()