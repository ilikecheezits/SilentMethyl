import pandas as pd

print("[*] Loading train.csv...")
df = pd.read_csv("train.csv")

with open("train_100p.fasta", "w") as f_wt:
    for idx, row in df.iterrows():
        header = f">{row['probeID']}"
        seq_100 = str(row['Healthy_5000bp_DNA'])[2450:2550]
        f_wt.write(f"{header}\n{seq_100}\n")

print("[*] Loading val.csv...")
df = pd.read_csv("val.csv")

with open("val_100p.fasta", "w") as f_wt:
    for idx, row in df.iterrows():
        header = f">{row['probeID']}"
        seq_100 = str(row['Healthy_5000bp_DNA'])[2450:2550]
        f_wt.write(f"{header}\n{seq_100}\n")

print("[*] Loading test.csv...")
df = pd.read_csv("test.csv")

with open("test_100p.fasta", "w") as f_wt:
    for idx, row in df.iterrows():
        header = f">{row['probeID']}"
        seq_100 = str(row['Healthy_5000bp_DNA'])[2450:2550]
        f_wt.write(f"{header}\n{seq_100}\n")

print("[✓] Generated FASTAs using probeID headers.")