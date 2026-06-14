import pandas as pd
import time

INPUT_CSV = "test.csv"
OUTPUT_CSV = "test_sorted.csv"

def main():
    print(f"[*] Loading {INPUT_CSV} into memory... (This may take a minute depending on file size)")
    start_time = time.time()
    
    # Load the full dataset
    df = pd.read_csv(INPUT_CSV)
    
    print(f"[+] Loaded {len(df):,} rows.")
    print("[*] Sorting by Chromosome and Position...")
    
    # Ensure pos is treated as an integer for proper numerical sorting
    df['pos'] = pd.to_numeric(df['pos'], errors='coerce')
    
    # Sort values. 
    # (Even though 'chr10' will sort alphabetically before 'chr2', it doesn't matter 
    # for I/O optimization as long as they are grouped together and sequential!)
    df_sorted = df.sort_values(by=['chr', 'pos'], ascending=[True, True])
    
    print(f"[*] Saving sorted dataset to {OUTPUT_CSV}...")
    df_sorted.to_csv(OUTPUT_CSV, index=False)
    
    elapsed = time.time() - start_time
    print(f"[✓] Done! Dataset sorted in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
