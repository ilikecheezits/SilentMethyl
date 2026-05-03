import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from scipy.stats import pearsonr
from tqdm import tqdm
from torch.utils.data import DataLoader
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoConfig
import argparse
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import MultiOmicsDataset
from model.architecture import SilentMethylModel, perform_triton_surgery
from model.lora_utils import inject_lora_adapters

def main():
    parser = argparse.ArgumentParser(description="Evaluate the SilentMethyl model on the blind test set.")
    parser.add_argument("--matrix_path", type=str, required=True, help="Path to the cleaned training data matrix CSV.")
    parser.add_argument("--dict_path", type=str, required=True, help="Path to the sequence dictionary CSV (Ignored in V2).")
    parser.add_argument("--region_vocab_path", type=str, required=True, help="Path to the region vocabulary pickle.")
    parser.add_argument("--island_vocab_path", type=str, required=True, help="Path to the island vocabulary pickle.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the trained model weights.")
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M", help="Hugging Face model path.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    args = parser.parse_args()

    print("--- FINAL EVALUATION: BLIND TEST ON CHROMOSOME 1 ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    print("[*] Loading data and artifacts...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    matrix_df = pd.read_csv(args.matrix_path)
    
    # V2 FIX: We no longer need the dictionary! We pass an empty dict 
    # to satisfy the Dataset class signature, but sequences are pulled natively.
    GLOBAL_SEQ_DICT = {}
    
    REGION_VOCAB = joblib.load(args.region_vocab_path)
    ISLAND_VOCAB = joblib.load(args.island_vocab_path)

    # V2 FIX: Pull the Chromosome dynamically from the matrix
    matrix_df['Chromosome'] = matrix_df.get('CpG_chrm', 'chr1')
    test_matrix = matrix_df[matrix_df['Chromosome'] == 'chr1'].copy()
    
    # Fallback just in case testing data doesn't have chr1
    if len(test_matrix) == 0:
        print("[!] WARNING: No Chromosome 1 probes found. Falling back to 15% random sample.")
        test_matrix = matrix_df.sample(frac=0.15, random_state=42)

    print(f"[*] Found {len(test_matrix)} test probes for blind evaluation.")

    test_dataset = MultiOmicsDataset(test_matrix, GLOBAL_SEQ_DICT, REGION_VOCAB, ISLAND_VOCAB, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("[*] Rebuilding Architecture...")
    local_model_dir = perform_triton_surgery()
    config = AutoConfig.from_pretrained(local_model_dir, trust_remote_code=True)
    
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    model = inject_lora_adapters(model)
    model.load_state_dict(torch.load(args.weights_path, map_location=device), strict=False)
    model.eval()

    print("[*] Running Inference on Blind Holdout...")
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            num_feats = batch['numerical_features'].to(device)
            reg_idx = batch['region_idx'].to(device)
            isl_idx = batch['island_idx'].to(device)
            tata_idx = batch['tata_idx'].to(device)
            true_beta = batch['targets'].to(device).view(-1, 1)

            _, pred_reg_logits = model(input_ids, attention_mask, num_feats, reg_idx, isl_idx, tata_idx)
            pred_beta = torch.sigmoid(pred_reg_logits)

            all_true.extend(true_beta.cpu().numpy().flatten())
            all_pred.extend(pred_beta.cpu().numpy().flatten())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # =========================================
    # Calculate & Display Metrics
    # =========================================
    test_mae = mean_absolute_error(all_true, all_pred)
    test_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
    test_pearson, _ = pearsonr(all_true, all_pred) if np.std(all_pred) > 0 else (0.0, 0.0)

    try:
        test_auroc = roc_auc_score((all_true > 0.5).astype(int), all_pred)
    except ValueError:
        test_auroc = 0.5

    print(f"\n{'='*50}")
    print(f"--- BLIND TEST METRICS (CHROMOSOME 1 - {len(all_true)} Probes) ---")
    print(f"-> Test RMSE:       {test_rmse:.4f}")
    print(f"-> Test MAE:        {test_mae:.4f}")
    print(f"-> Test Pearson R:  {test_pearson:.4f}")
    print(f"-> Test AUROC:      {test_auroc:.4f}")
    print(f"{'='*50}")

    # =========================================
    # Visualization
    # =========================================
    plt.figure(figsize=(10, 5))
    sns.kdeplot(all_true, color='blue', label='True Biology (Chr1)', fill=True, alpha=0.3)
    sns.kdeplot(all_pred, color='red', label='AI Prediction (Chr1)', fill=True, alpha=0.3)
    plt.title('Chromosome 1 Holdout: Distribution Alignment')
    plt.xlabel('Methylation Beta Value')
    plt.ylabel('Density')
    plt.xlim(0, 1)
    plt.legend()
    plt.style.use('dark_background')
    
    save_dir = os.path.dirname(args.weights_path)
    plot_path = os.path.join(save_dir, "blind_test_alignment.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[✓] Evaluation plot saved to {plot_path}")

if __name__ == "__main__":
    main()
