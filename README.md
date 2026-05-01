# SilentMethyl

This project is a deep learning pipeline for predicting DNA methylation from sequence data, based on the "SilentMethyl" model.

## Project Structure

- `data/`: Contains raw and preprocessed data. (Ignored by git)
- `scripts/`: Contains scripts for data processing, training, evaluation, and inference.
- `src/`: Contains the source code for the model, dataset, and training components.
- `*.ipynb`: Jupyter notebooks used for development and experimentation (will be removed).

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

### 1. Build Data
This will preprocess the raw data and save the necessary scalers and vocabularies.
```bash
python scripts/01_build_data.py --matrix_path path/to/training_data.csv --dict_path path/to/DNA_Sequence_Dictionary.csv --base_dir ./data/processed
```

### 2. Train Model
This will train the SilentMethyl model using the preprocessed data.
```bash
python scripts/02_train_model.py --matrix_path ./data/processed/cleaned_training_data.csv --dict_path ./data/processed/SilentMethyl_SeqDict.pkl --region_vocab_path ./data/processed/SilentMethyl_RegionVocab.pkl --island_vocab_path ./data/processed/SilentMethyl_IslandVocab.pkl --save_dir ./checkpoints
```

### 3. Evaluate Model
This will evaluate the trained model on the blind test set (chromosome 1).
```bash
python scripts/04_evaluate_model.py --matrix_path ./data/processed/cleaned_training_data.csv --dict_path ./data/processed/SilentMethyl_SeqDict.pkl --region_vocab_path ./data/processed/SilentMethyl_RegionVocab.pkl --island_vocab_path ./data/processed/SilentMethyl_IslandVocab.pkl --weights_path ./checkpoints/SilentMethyl_Best_Weights.pth --save_dir ./evaluation
```

### 4. Run Mass Inference
This will run the inference engine on a discovery dataset to predict the impact of mutations. It also performs statistical analysis and generates a volcano plot.
```bash
python scripts/03_run_mass_inference.py --data_path path/to/Final_Discovery_Dataset_MultiOmics.csv --base_dir ./data/processed --weights_path ./checkpoints/SilentMethyl_Best_Weights.pth
```

## Additional Analyses

These scripts are for more in-depth analysis and are not part of the main training/inference pipeline.

### Monte Carlo Stability Analysis
This script runs a Monte Carlo simulation to analyze the stability of the model's predictions for a single sample.
```bash
python scripts/05_run_monte_carlo.py --data_path path/to/Final_Discovery_Dataset_MultiOmics.csv --base_dir ./data/processed --weights_path ./checkpoints/SilentMethyl_Best_Weights.pth --sample_index 0 --n_runs 200 --save_dir ./analysis
```

### JASPAR Motif Scan
This script scans for broken transcription factor binding motifs using the JASPAR database.
```bash
python scripts/07_run_jaspar_scan.py --chromosome 2 --mutation_pos 60461250 --ref_allele G --alt_allele T
```

### Silent Permutation Test
This script runs a rigorous statistical test by generating random synonymous mutations to calculate a Z-score for a given mutation.
```bash
python scripts/08_run_silent_permutation_test.py --wt_sequence <WILD_TYPE_SEQUENCE> --real_delta <REAL_DELTA> --base_dir ./data/processed --weights_path ./checkpoints/SilentMethyl_Best_Weights.pth
```
