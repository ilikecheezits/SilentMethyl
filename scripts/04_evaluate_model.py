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
from model.lora_utils import inject_lora_adapters, load_lora_weights

def main():
    parser = argparse.ArgumentParser(description="Evaluate the SilentMethyl model on the blind test set.")
    parser.add_argument("--matrix_path", type=str, required=True, help="Path to the cleaned training data matrix CSV.")
    parser.add_argument("--dict_path", type=str, required=True, help="Path to the sequence dictionary pickle.")
    parser.add_argument("--region_vocab_path", type=str, required=True, help="Path to the region vocabulary pickle.")
    parser.add_argument("--island_vocab_path", type=str, required=True, help="Path to the island vocabulary pickle.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the trained model weights.")
    parser.add_argument("--model_path", default="zhihan1996/DNABERT-2-117M", help="Hugging Face model path.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers.")
    parser.add_argument("--save_dir", default=".", help="Directory to save evaluation plots.")
    args = parser.parse_args()

    print("--- FINAL EVALUATION: BLIND TEST ON CHROMOSOME 1 ---")

    # =========================================
    # Environment Setup
    # =========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # =========================================
    # Load Data and Artifacts
    # =========================================
    print("[*] Loading data and artifacts...")
    matrix_df = pd.read_csv(args.matrix_path)
    GLOBAL_SEQ_DICT = joblib.load(args.dict_path)
    REGION_VOCAB = joblib.load(args.region_vocab_path)
    ISLAND_VOCAB = joblib.load(args.island_vocab_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    matrix_df['Chromosome'] = matrix_df['CpG_Target'].apply(lambda x: GLOBAL_SEQ_DICT.get(x, {}).get('CpG_chrm', 'Unknown'))
    test_matrix = matrix_df[matrix_df['Chromosome'] == 'chr1'].copy()
    test_dataset = MultiOmicsDataset(test_matrix, GLOBAL_SEQ_DICT, REGION_VOCAB, ISLAND_VOCAB, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # =========================================
    # Model Rebuilding and Loading
    # =========================================
    print("[*] Rebuilding model and loading weights...")
    local_model_dir = perform_triton_surgery()
    config = AutoConfig.from_pretrained(local_model_dir, trust_remote_code=True)
    model = SilentMethylModel(config, len(REGION_VOCAB), len(ISLAND_VOCAB)).to(device)
    model = inject_lora_adapters(model)
    model = load_lora_weights(model, args.weights_path, device)
    model.eval()

    # =========================================
    # Evaluation
    # =========================================
    test_true = []
    test_pred = []
    
    print(f"[*] Evaluating on Chromosome 1 (Never seen during training!)...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[BLIND TEST]"):
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            num_feats = batch['numerical_features'].to(device)
            reg_idx, isl_idx = batch['region_idx'].to(device), batch['island_idx'].to(device)
            tata_idx = batch['tata_idx'].to(device)
            true_beta = batch['targets'].to(device)

            _, pred_reg_logits = model(input_ids, attention_mask, num_feats, reg_idx, isl_idx, tata_idx)
            pred_beta = torch.sigmoid(pred_reg_logits)
            clamped_beta = torch.clamp(pred_beta, 0.0, 1.0)

            test_true.extend(true_beta.cpu().numpy().flatten())
            test_pred.extend(clamped_beta.cpu().numpy().flatten())

    test_true = np.array(test_true)
    test_pred = np.array(test_pred)

    # =========================================
    # Calculate and Display Metrics
    # =========================================
    test_mae = mean_absolute_error(test_true, test_pred)
    test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    test_pearson, _ = pearsonr(test_true, test_pred) if np.std(test_pred) > 0 else (0.0, 0.0)

    try:
        test_auroc = roc_auc_score((test_true > 0.5).astype(int), test_pred)
    except ValueError:
        test_auroc = 0.5

    print(f"{'='*50}")
    print(f"--- BLIND TEST METRICS (CHROMOSOME 1 - {len(test_true)} Probes) ---")
    print(f"-> Test RMSE:       {test_rmse:.4f}")
    print(f"-> Test MAE:        {test_mae:.4f}")
    print(f"-> Test Pearson R:  {test_pearson:.4f}")
    print(f"-> Test AUROC:      {test_auroc:.4f}")
    print(f"==================================================")

    # =========================================
    # Visualization
    # =========================================
    plt.figure(figsize=(10, 5))
    sns.kdeplot(test_true, color='blue', label='True Biology (Chr1)', fill=True, alpha=0.3)
    sns.kdeplot(test_pred, color='red', label='AI Prediction (Chr1)', fill=True, alpha=0.3)
    plt.title(f"Blind Test Generalization: Chromosome 1 (n={len(test_true)} probes) RMSE: {test_rmse:.4f} | AUROC: {test_auroc:.4f}")
    plt.xlim(0, 1)
    plt.xlabel("Methylation Beta Value")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "blind_test_evaluation.png"))
    
if __name__ == "__main__":
    main()
