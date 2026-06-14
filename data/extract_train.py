import pandas as pd

df = pd.read_csv("train.csv")

with open("test_healthy_100bp.fasta", "w") as f_wt, open("test_mutated_101bp.fasta", "w") as f_mut:
    for idx, row in df.iterrows():
        header = f">{row['probeID']}"
        wt_101 = str(row['Healthy_5000bp_DNA'])[2450:2551]
        mut_101 = str(row['Mutated_5000bp_DNA'])[2450:2551]

        f_wt.write(f"{header}\n{wt_101}\n")
        f_mut.write(f"{header}\n{mut_101}\n")

df = pd.read_csv("train.csv")

with open("test_healthy_101bp.fasta", "w") as f_wt, open("test_mutated_101bp.fasta", "w") as f_mut:
    for idx, row in df.iterrows():
        header = f">{row['probeID']}"
        wt_101 = str(row['Healthy_5000bp_DNA'])[2450:2551]
        mut_101 = str(row['Mutated_5000bp_DNA'])[2450:2551]

        f_wt.write(f"{header}\n{wt_101}\n")
        f_mut.write(f"{header}\n{mut_101}\n")
df = pd.read_csv("train.csv")

with open("test_healthy_101bp.fasta", "w") as f_wt, open("test_mutated_101bp.fasta", "w") as f_mut:
    for idx, row in df.iterrows():
        header = f">{row['probeID']}"
        wt_101 = str(row['Healthy_5000bp_DNA'])[2450:2551]
        mut_101 = str(row['Mutated_5000bp_DNA'])[2450:2551]

        f_wt.write(f"{header}\n{wt_101}\n")
        f_mut.write(f"{header}\n{mut_101}\n")

