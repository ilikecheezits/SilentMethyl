import pandas as pd

print("[*] Loading actual_testing_data.csv...")
df = pd.read_csv("actual_testing_data.csv")

with open("test_healthy_100p.fasta", "w") as f_wt, open("test_mutated_100bp.fasta", "w") as f_mut:
    for idx, row in df.iterrows():
        header = f">{row['probeID']}"
        wt_100 = str(row['Healthy_5000bp_DNA'])[2450:2550]
        mut_100 = str(row['Mutated_5000bp_DNA'])[2450:2550]
        
        f_wt.write(f"{header}\n{wt_100}\n")
        f_mut.write(f"{header}\n{mut_100}\n")

print("[✓] Generated test FASTAs using probeID headers.")
