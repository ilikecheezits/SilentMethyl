import os
import sys
import argparse
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.preprocess import clean_training_data, preprocess_data, build_vocabularies

def main():
    parser = argparse.ArgumentParser(description="Build and preprocess data.")
    parser.add_argument("--matrix_path", type=str, required=True, help="Path to the raw matrix data.")
    parser.add_argument("--dict_path", type=str, required=True, help="Path to the raw sequence dictionary.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory to save preprocessed data.")
    args = parser.parse_args()

    # Create base_dir if it doesn't exist
    os.makedirs(args.base_dir, exist_ok=True)
    
    # 1. Clean Data
    cleaned_matrix_path = os.path.join(args.base_dir, "cleaned_training_data.csv")
    matrix_df = clean_training_data(args.matrix_path, cleaned_matrix_path)

    # 2. Load dictionary
    dict_df = pd.read_csv(args.dict_path)

    # 3. Preprocess Data
    matrix_df, dict_df = preprocess_data(matrix_df, dict_df, args.base_dir)
    
    # 4. Build Vocabularies
    build_vocabularies(dict_df, args.base_dir)

    print("[✓] Data build process complete.")

if __name__ == "__main__":
    main()
