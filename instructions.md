```python
md_content = """# SilentMethyl: Pipeline Execution Commands

This document contains the step-by-step terminal commands to execute the complete training, testing, and interpretability pipeline for the Sequence + Epigenomic fusion model.

---

## 1. Training Phase

Since training requires significant compute time and GPU memory, these should be submitted via SLURM scripts.

**1a. Train the Baseline DNABERT-2 Model**

```

```text
Traceback (most recent call last):
  File "<xbox-string>", line 89, in <module>
    f.write(content)
NameError: name 'content' is not defined

```bash
sbatch run_baseline.sh

```

**1b. Train the Pure Epigenomic Network**

```bash
sbatch run_epi.sh

```

*(Wait for this to finish to generate `checkpoints_epi_only/best_weights.pth` before proceeding to 1c)*

**1c. Train the Final Seq+Epi Fusion Model**

```bash
sbatch run_fusion.sh

```

---

## 2. Testing Phase

Extract the final performance metrics (RMSE, MAE, AUC) and generate the manuscript figures (Scatter, Bimodal, ROC, Calibration). These can be run directly from the terminal.

**2a. Test the Baseline Model**

```bash
python scripts/02_test_baseline.py \
  --test_data_path "data/datafiles/test.csv" \
  --weights_path "checkpoints_baseline/best_weights.pth" \
  --batch_size 64

```

**2b. Test the Final Fusion Model**

```bash
python scripts/02_test_model.py \
  --test_csv_path "data/datafiles/test.csv" \
  --weights_path "checkpoints_seq_epi_fusion/best_weights.pth" \
  --batch_size 64

```

---

## 3. Interpretability & Robustness Experiments

These scripts analyze the trained model's decision-making process.

**3a. Matched Synonymous Null Experiment (Statistical Significance)**
Evaluates variants against an empirical FDR distribution of matched synonymous controls.

```bash
python scripts/03_matched_null.py

```

*(Note: Ensure `matched_synonymous_null.py` is in the same directory as this script).*

**3b. JASPAR Motif Scan (Biological Mechanism)**
Calculates precise *In Silico* Mutagenesis (ISM) disruption scores for known transcription factor motifs.

```bash
python scripts/04_jaspar.py

```

**3c. Stability & MC Dropout Diagnostics**
Runs 50 paired Monte Carlo dropout passes and reverse-complement checks to verify predictive robustness. Because this takes 1-2 hours, it is best run as a SLURM job, but can be run interactively:

```bash
python scripts/06_stability.py

```

**3d. HOMER Saliency Extraction (Optional)**
Extracts high-attention tokens to FASTA files for *de novo* motif discovery.

```bash
python scripts/05_homer.py

```

*(After this completes, run the generated `findMotifs.pl` commands via the HOMER suite).*

**3e. Fourier Periodicity Test (Optional)**
Mathematically evaluates if the model detects structural 147bp nucleosome wrapping.

```bash
python scripts/06_fourier.py

```

"""

with open("silentmethyl_commands.md", "w") as f:
f.write(md_content)

print("silentmethyl_commands.md")

```
```python?code_reference&code_event_index=6
md_content = """# SilentMethyl: Pipeline Execution Commands

This document contains the step-by-step terminal commands to execute the complete training, testing, and interpretability pipeline for the Sequence + Epigenomic fusion model.

---

## 1. Training Phase

Since training requires significant compute time and GPU memory, these should be submitted via SLURM scripts.

**1a. Train the Baseline DNABERT-2 Model**

```

```text
silentmethyl_commands.md

```bash
sbatch run_baseline.sh

```

**1b. Train the Pure Epigenomic Network**

```bash
sbatch run_epi.sh

```

*(Wait for this to finish to generate `checkpoints_epi_only/best_weights.pth` before proceeding to 1c)*

**1c. Train the Final Seq+Epi Fusion Model**

```bash
sbatch run_fusion.sh

```

---

## 2. Testing Phase

Extract the final performance metrics (RMSE, MAE, AUC) and generate the manuscript figures (Scatter, Bimodal, ROC, Calibration). These can be run directly from the terminal.

**2a. Test the Baseline Model**

```bash
python scripts/02_test_baseline.py \\
  --test_data_path "data/datafiles/test.csv" \\
  --weights_path "checkpoints_baseline/best_weights.pth" \\
  --batch_size 64

```

**2b. Test the Final Fusion Model**

```bash
python scripts/02_test_model.py \\
  --test_csv_path "data/datafiles/test.csv" \\
  --weights_path "checkpoints_seq_epi_fusion/best_weights.pth" \\
  --batch_size 64

```
**2c. Test the Epigenomic Model**
```bash
python scripts/02_test_epi.py \\
  --test_csv_path "data/datafiles/test.csv" \\
  --weights_path "checkpoints_epi_only/best_weights.pth" \\
  --batch_size 512
```
---

## 3. Interpretability & Robustness Experiments

These scripts analyze the trained model's decision-making process.

**3a. Matched Synonymous Null Experiment (Statistical Significance)**
Evaluates variants against an empirical FDR distribution of matched synonymous controls.

```bash
python scripts/03_matched_null.py

```

*(Note: Ensure `matched_synonymous_null.py` is in the same directory as this script).*

**3b. JASPAR Motif Scan (Biological Mechanism)**
Calculates precise *In Silico* Mutagenesis (ISM) disruption scores for known transcription factor motifs.

```bash
python scripts/04_jaspar.py

```

**3c. Stability & MC Dropout Diagnostics**
Runs 50 paired Monte Carlo dropout passes and reverse-complement checks to verify predictive robustness. Because this takes 1-2 hours, it is best run as a SLURM job, but can be run interactively:

```bash
python scripts/06_stability.py

```

**3d. HOMER Saliency Extraction (Optional)**
Extracts high-attention tokens to FASTA files for *de novo* motif discovery.

```bash
python scripts/05_homer.py

```

*(After this completes, run the generated `findMotifs.pl` commands via the HOMER suite).*

**3e. Fourier Periodicity Test (Optional)**
Mathematically evaluates if the model detects structural 147bp nucleosome wrapping.

```bash
python scripts/06_fourier.py

```

"""

with open("silentmethyl_commands.md", "w") as f:
f.write(md_content)

print("silentmethyl_commands.md")

```
Your Markdown file containing the full list of execution commands is ready:
[file-tag: code-generated-file-0-1786039089430369234]

You can view or download this `.md` file directly. It contains all the exact `sbatch` and `python` commands in order, completely updated for your current architecture and filenames.

```