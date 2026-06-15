import pandas as pd
import time

INPUT_CSV = "train.csv"
OUTPUT_CSV = "train_sorted.csv"

def main():
    print(f"[*] Loading {INPUT_CSV} into memory... (This may take a minute depending on file size)")
    start_time = time.time()
    df = pd.read_csv(INPUT_CSV)
    print(f"[+] Loaded {len(df):,} rows.")
    print("[*] Sorting by Chromosome and Position...")
    df['pos'] = pd.to_numeric(df['pos'], errors='coerce')
    df_sorted = df.sort_values(by=['chr', 'pos'], ascending=[True, True])
    print(f"[*] Saving sorted dataset to {OUTPUT_CSV}...")
    df_sorted.to_csv(OUTPUT_CSV, index=False)
    elapsed = time.time() - start_time
    print(f"[✓] Done! Dataset sorted in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
