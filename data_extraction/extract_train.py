import pandas as pd

print("[*] Loading training_data.csv...")
df = pd.read_csv("training_data.csv")

with open("train_healthy_101bp.fasta", "w") as f_wt:
    for idx, row in df.iterrows():
        header = f">{row['probeID']}"
        wt_101 = str(row['Healthy_5000bp_DNA'])[2450:2551]
        
        f_wt.write(f"{header}\n{wt_101}\n")

print("[✓] Generated train FASTAs using probeID headers.")
