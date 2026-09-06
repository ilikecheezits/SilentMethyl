.
├── MELODY_COMPARISON.md
├── PAPER_FRAMEWORK.md
├── README.md
├── REQUIREMENTS.md
├── _archive
│   ├── logs_genoa_scoring.tar.gz
│   ├── logs_harmonize_genoa.tar.gz
│   ├── logs_pre_renumber.tar.gz
│   ├── logs_testing.tar.gz
│   ├── logs_training.tar.gz
│   ├── rc_uncertainty_s20.tar.gz
│   ├── rc_uncertainty_s50.tar.gz
│   ├── supplementary_package_stale.tar.gz
│   └── tensorboard_events.tar.gz
├── a.txt
├── build_data.sh
├── checkpoints_folds
│   ├── fold1
│   │   ├── sequence_seed42
│   │   │   ├── best_weights.pth
│   │   │   ├── latest_checkpoint.pt
│   │   │   ├── run_config.json
│   │   │   └── tensorboard
│   │   │       └── events.out.tfevents.1788651417.v022.ib.bridges2.psc.edu.9307.0
│   │   ├── split_checksums.txt
│   │   └── splits_summary_at_launch.json
│   ├── fold2
│   │   ├── sequence_seed42
│   │   │   ├── best_weights.pth
│   │   │   ├── latest_checkpoint.pt
│   │   │   ├── run_config.json
│   │   │   └── tensorboard
│   │   │       └── events.out.tfevents.1788651423.v002.ib.bridges2.psc.edu.98700.0
│   │   ├── split_checksums.txt
│   │   └── splits_summary_at_launch.json
│   └── fold3
│       ├── sequence_seed42
│       │   ├── best_weights.pth
│       │   ├── latest_checkpoint.pt
│       │   ├── run_config.json
│       │   └── tensorboard
│       │       └── events.out.tfevents.1788651423.v007.ib.bridges2.psc.edu.22214.0
│       ├── split_checksums.txt
│       └── splits_summary_at_launch.json
├── checkpoints_journal
│   ├── seed42
│   │   ├── epi
│   │   │   ├── best_weights.pth
│   │   │   └── run_config.json
│   │   ├── fusion
│   │   │   ├── best_validation_gates.csv
│   │   │   ├── best_weights.pth
│   │   │   └── run_config.json
│   │   └── sequence
│   │       ├── best_weights.pth
│   │       └── run_config.json
│   ├── seed43
│   │   ├── epi
│   │   │   ├── best_weights.pth
│   │   │   └── run_config.json
│   │   ├── fusion
│   │   │   ├── best_validation_gates.csv
│   │   │   ├── best_weights.pth
│   │   │   └── run_config.json
│   │   └── sequence
│   │       ├── best_weights.pth
│   │       └── run_config.json
│   └── seed44
│       ├── epi
│       │   ├── best_weights.pth
│       │   └── run_config.json
│       ├── fusion
│       │   ├── best_validation_gates.csv
│       │   ├── best_weights.pth
│       │   └── run_config.json
│       └── sequence
│           ├── best_weights.pth
│           └── run_config.json
├── cleanup_workspace.sh
├── data
│   ├── HM450.hg38.manifest.CpGIsland.tsv.gz
│   ├── HM450.hg38.manifest.tsv.gz
│   ├── TCGA-BRCA.methylation450.tsv.gz
│   ├── __pycache__
│   │   └── build_genoa_scoring_input.cpython-312.pyc
│   ├── _test_egtex_harmonizer.py
│   ├── acquire_external_cohorts.py
│   ├── acquire_healthy_breast_cohort.py
│   ├── audit_data_purity.py
│   ├── audit_training_data.py
│   ├── build_genoa_scoring_input.py
│   ├── build_tcga_ancestry_labels.py
│   ├── build_tcga_tumor_cohort.py
│   ├── build_testing_data.py
│   ├── build_training_data.py
│   ├── datafiles
│   │   ├── candidate_cohort_manifest.json
│   │   ├── data_purity_audit.json
│   │   ├── feature_imputation.json
│   │   ├── gdc_tcga_brca_synonymous_raw.json.gz
│   │   ├── split_manifest.json
│   │   ├── splits
│   │   │   ├── fold0
│   │   │   │   ├── test.csv
│   │   │   │   ├── train.csv
│   │   │   │   └── val.csv
│   │   │   ├── fold1
│   │   │   │   ├── test.csv
│   │   │   │   ├── train.csv
│   │   │   │   └── val.csv
│   │   │   ├── fold2
│   │   │   │   ├── test.csv
│   │   │   │   ├── train.csv
│   │   │   │   └── val.csv
│   │   │   ├── fold3
│   │   │   │   ├── test.csv
│   │   │   │   ├── train.csv
│   │   │   │   └── val.csv
│   │   │   └── splits_summary.json
│   │   ├── tcga_normal_sample_ids.json
│   │   ├── test.csv
│   │   ├── test_100bp.fasta
│   │   ├── test_100bp_rc.fasta
│   │   ├── test_healthy_100bp.fasta
│   │   ├── test_healthy_100bp_rc.fasta
│   │   ├── test_mutated_100bp.fasta
│   │   ├── test_mutated_100bp_rc.fasta
│   │   ├── testing_data.csv
│   │   ├── testing_data_test_only.csv
│   │   ├── train.csv
│   │   ├── train_100bp.fasta
│   │   ├── train_100bp_rc.fasta
│   │   ├── training_data_manifest.json
│   │   ├── val.csv
│   │   ├── val_100bp.fasta
│   │   └── val_100bp_rc.fasta
│   ├── egtex_breast_mqtl_heldout.csv
│   ├── egtex_breast_mqtl_heldout_qc.csv
│   ├── egtex_breast_mqtl_model_visible.csv
│   ├── egtex_significance_threshold.py
│   ├── exteral
│   │   └── melody
│   │       └── Melody_repo
│   │           └── scripts
│   ├── external
│   │   ├── baselines
│   │   │   ├── CpGenie
│   │   │   │   ├── Dockerfiles
│   │   │   │   │   ├── cuda7.0
│   │   │   │   │   │   └── Dockerfile
│   │   │   │   │   ├── cuda8.0
│   │   │   │   │   │   └── Dockerfile
│   │   │   │   │   └── cuda9.0
│   │   │   │   │       └── Dockerfile
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.md
│   │   │   │   ├── cnn
│   │   │   │   │   └── seq_128x3_5_5_2f_simple.template
│   │   │   │   ├── copyover.py
│   │   │   │   ├── data
│   │   │   │   │   └── hg19.size
│   │   │   │   ├── example
│   │   │   │   │   ├── test.fa
│   │   │   │   │   └── test.vcf
│   │   │   │   ├── helper
│   │   │   │   │   ├── embedH5.py
│   │   │   │   │   ├── splitVCF.py
│   │   │   │   │   ├── utility.R
│   │   │   │   │   └── vcf2fasta2.R
│   │   │   │   ├── main.py
│   │   │   │   └── variant.py
│   │   │   ├── RESTORE.md
│   │   │   └── deepcpg
│   │   │       ├── LICENSE
│   │   │       ├── MANIFEST.in
│   │   │       ├── R
│   │   │       │   ├── README.md
│   │   │       │   ├── eval_perf_mult.Rmd
│   │   │       │   └── eval_perf_single.Rmd
│   │   │       ├── README.rst
│   │   │       ├── deepcpg
│   │   │       │   ├── __init__.py
│   │   │       │   ├── callbacks.py
│   │   │       │   ├── data
│   │   │       │   │   ├── __init__.py
│   │   │       │   │   ├── annotations.py
│   │   │       │   │   ├── dna.py
│   │   │       │   │   ├── fasta.py
│   │   │       │   │   ├── feature_extractor.py
│   │   │       │   │   ├── hdf.py
│   │   │       │   │   ├── stats.py
│   │   │       │   │   └── utils.py
│   │   │       │   ├── evaluation.py
│   │   │       │   ├── metrics.py
│   │   │       │   ├── models
│   │   │       │   │   ├── __init__.py
│   │   │       │   │   ├── cpg.py
│   │   │       │   │   ├── dna.py
│   │   │       │   │   ├── joint.py
│   │   │       │   │   └── utils.py
│   │   │       │   ├── motifs.py
│   │   │       │   └── utils.py
│   │   │       ├── docs
│   │   │       │   ├── Makefile
│   │   │       │   ├── make.bat
│   │   │       │   └── source
│   │   │       │       ├── conf.py
│   │   │       │       ├── data.rst
│   │   │       │       ├── fig1.png
│   │   │       │       ├── index.rst
│   │   │       │       ├── lib
│   │   │       │       │   ├── data
│   │   │       │       │   │   └── index.rst
│   │   │       │       │   ├── deepcpg.rst
│   │   │       │       │   ├── index.rst
│   │   │       │       │   └── models
│   │   │       │       │       └── index.rst
│   │   │       │       ├── models.rst
│   │   │       │       ├── scripts
│   │   │       │       │   └── index.rst
│   │   │       │       ├── train.rst
│   │   │       │       └── zoo.md
│   │   │       ├── examples
│   │   │       │   ├── README.md
│   │   │       │   ├── notebooks
│   │   │       │   │   ├── basics
│   │   │       │   │   │   └── index.ipynb
│   │   │       │   │   ├── fine_tune
│   │   │       │   │   │   └── index.ipynb
│   │   │       │   │   ├── motifs
│   │   │       │   │   │   └── index.ipynb
│   │   │       │   │   ├── snp
│   │   │       │   │   │   └── index.ipynb
│   │   │       │   │   └── stats
│   │   │       │   │       └── index.ipynb
│   │   │       │   ├── scripts
│   │   │       │   │   ├── data.sh
│   │   │       │   │   ├── eval.sh
│   │   │       │   │   ├── lib.sh
│   │   │       │   │   └── train.sh
│   │   │       │   └── setup.sh
│   │   │       ├── requirements.txt
│   │   │       ├── scripts
│   │   │       │   ├── dcpg_data.py
│   │   │       │   ├── dcpg_data_show.py
│   │   │       │   ├── dcpg_data_stats.py
│   │   │       │   ├── dcpg_download.py
│   │   │       │   ├── dcpg_eval.py
│   │   │       │   ├── dcpg_eval_export.py
│   │   │       │   ├── dcpg_eval_perf.py
│   │   │       │   ├── dcpg_filter_act.py
│   │   │       │   ├── dcpg_filter_motifs.py
│   │   │       │   ├── dcpg_snp.py
│   │   │       │   ├── dcpg_train.py
│   │   │       │   └── dcpg_train_viz.py
│   │   │       ├── setup.py
│   │   │       └── tests
│   │   │           ├── deepcpg
│   │   │           │   ├── data
│   │   │           │   │   ├── test_annos.py
│   │   │           │   │   ├── test_feature_extractor.py
│   │   │           │   │   └── test_hdf.py
│   │   │           │   └── models
│   │   │           │       └── test_utils.py
│   │   │           └── integration_tests
│   │   │               ├── data
│   │   │               │   ├── annos
│   │   │               │   │   ├── CGI.bed.gz
│   │   │               │   │   └── exons.bed.gz
│   │   │               │   ├── cpg_profiles
│   │   │               │   │   ├── BS27_4_SER.bed.gz
│   │   │               │   │   └── BS28_2_SER.bed.gz
│   │   │               │   └── test_data_values.py
│   │   │               ├── setup.sh
│   │   │               ├── test_data.py
│   │   │               └── test_train.py
│   │   ├── clinvar
│   │   │   ├── clinvar_GRCh38.vcf.gz
│   │   │   └── clinvar_GRCh38.vcf.gz.tbi
│   │   ├── egtex_breast
│   │   │   ├── BreastMammaryTissue.mQTLs.regular.txt.gz
│   │   │   ├── BreastMammaryTissue.regular.perm.fdr.txt
│   │   │   ├── egtex_breast_within600.tsv.gz
│   │   │   └── scoring
│   │   │       ├── egtex_probe_significance.csv
│   │   │       ├── egtex_scoring_input_heldout.csv
│   │   │       ├── egtex_scoring_input_model_visible.csv
│   │   │       ├── egtex_scoring_summary.json
│   │   │       └── egtex_significance_summary.json
│   │   ├── egtex_multitissue
│   │   │   ├── BreastMammaryTissue.mQTLs.conditional.txt.gz
│   │   │   ├── BreastMammaryTissue.regular.perm.fdr.txt -> ../egtex_breast/BreastMammaryTissue.regular.perm.fdr.txt
│   │   │   ├── ColonTransverse.mQTLs.conditional.txt.gz
│   │   │   ├── ColonTransverse.regular.perm.fdr.txt
│   │   │   ├── KidneyCortex.mQTLs.conditional.txt.gz
│   │   │   ├── KidneyCortex.regular.perm.fdr.txt
│   │   │   ├── Lung.mQTLs.conditional.txt.gz
│   │   │   ├── Lung.regular.perm.fdr.txt
│   │   │   ├── MuscleSkeletal.mQTLs.conditional.txt.gz
│   │   │   ├── MuscleSkeletal.regular.perm.fdr.txt
│   │   │   ├── Ovary.mQTLs.conditional.txt.gz
│   │   │   ├── Ovary.regular.perm.fdr.txt
│   │   │   ├── Prostate.mQTLs.conditional.txt.gz
│   │   │   ├── Prostate.regular.perm.fdr.txt
│   │   │   ├── README
│   │   │   ├── Testis.mQTLs.conditional.txt.gz
│   │   │   ├── Testis.regular.perm.fdr.txt
│   │   │   ├── WholeBlood.mQTLs.conditional.txt.gz
│   │   │   ├── WholeBlood.regular.perm.fdr.txt
│   │   │   ├── _scratch
│   │   │   │   ├── cond_cpgs.txt
│   │   │   │   └── sig_cpgs.txt
│   │   │   ├── scoring
│   │   │   │   ├── BreastMammaryTissue
│   │   │   │   │   ├── egtex_scoring_input_heldout.csv
│   │   │   │   │   ├── egtex_scoring_input_model_visible.csv
│   │   │   │   │   └── egtex_scoring_summary.json
│   │   │   │   ├── ColonTransverse
│   │   │   │   │   ├── egtex_scoring_input_heldout.csv
│   │   │   │   │   ├── egtex_scoring_input_model_visible.csv
│   │   │   │   │   └── egtex_scoring_summary.json
│   │   │   │   ├── KidneyCortex
│   │   │   │   │   ├── egtex_scoring_input_heldout.csv
│   │   │   │   │   ├── egtex_scoring_input_model_visible.csv
│   │   │   │   │   └── egtex_scoring_summary.json
│   │   │   │   ├── Lung
│   │   │   │   │   ├── egtex_scoring_input_heldout.csv
│   │   │   │   │   ├── egtex_scoring_input_model_visible.csv
│   │   │   │   │   └── egtex_scoring_summary.json
│   │   │   │   ├── MuscleSkeletal
│   │   │   │   │   ├── egtex_scoring_input_heldout.csv
│   │   │   │   │   ├── egtex_scoring_input_model_visible.csv
│   │   │   │   │   └── egtex_scoring_summary.json
│   │   │   │   ├── Ovary
│   │   │   │   │   ├── egtex_scoring_input_heldout.csv
│   │   │   │   │   ├── egtex_scoring_input_model_visible.csv
│   │   │   │   │   └── egtex_scoring_summary.json
│   │   │   │   ├── Prostate
│   │   │   │   │   ├── egtex_scoring_input_heldout.csv
│   │   │   │   │   ├── egtex_scoring_input_model_visible.csv
│   │   │   │   │   └── egtex_scoring_summary.json
│   │   │   │   ├── Testis
│   │   │   │   │   ├── egtex_scoring_input_heldout.csv
│   │   │   │   │   ├── egtex_scoring_input_model_visible.csv
│   │   │   │   │   └── egtex_scoring_summary.json
│   │   │   │   ├── WholeBlood
│   │   │   │   │   ├── egtex_scoring_input_heldout.csv
│   │   │   │   │   ├── egtex_scoring_input_model_visible.csv
│   │   │   │   │   └── egtex_scoring_summary.json
│   │   │   │   └── union_scoring_input_heldout.csv
│   │   │   └── within600
│   │   │       ├── ColonTransverse_within600.tsv.gz
│   │   │       ├── KidneyCortex_within600.tsv.gz
│   │   │       ├── Lung_within600.tsv.gz
│   │   │       ├── MuscleSkeletal_within600.tsv.gz
│   │   │       ├── Ovary_within600.tsv.gz
│   │   │       ├── Prostate_within600.tsv.gz
│   │   │       ├── Testis_within600.tsv.gz
│   │   │       └── WholeBlood_within600.tsv.gz
│   │   ├── external_manifest.json
│   │   ├── genoa_meqtl
│   │   │   ├── GENOA_meQTL_README.txt
│   │   │   ├── harmonized
│   │   │   │   ├── genoa_model_visible_pairs.csv.gz
│   │   │   │   └── harmonization_summary.json
│   │   │   └── scoring
│   │   │       ├── genoa_scoring_input_heldout.csv
│   │   │       ├── genoa_scoring_input_model_visible.csv
│   │   │       └── genoa_scoring_summary.json
│   │   ├── gwas_catalog
│   │   │   ├── SHA256SUMS.txt
│   │   │   └── gwas-catalog-download-associations-alt-full.tsv
│   │   ├── hocomoco
│   │   │   └── hocomoco_core_pwm.tar.gz
│   │   ├── jaspar
│   │   │   └── JASPAR_CORE_vertebrates_non-redundant_pfms_jaspar.txt
│   │   ├── melody
│   │   │   ├── Melody_repo
│   │   │   │   ├── --input-csv
│   │   │   │   ├── --limit
│   │   │   │   ├── --output
│   │   │   │   ├── --tracks
│   │   │   │   ├── LICENSE
│   │   │   │   ├── README.assets
│   │   │   │   │   └── image-20251125003954614.png
│   │   │   │   ├── README.md
│   │   │   │   ├── __pycache__
│   │   │   │   │   ├── basic_blocks.cpython-310.pyc
│   │   │   │   │   ├── global_constants.cpython-310.pyc
│   │   │   │   │   ├── models.cpython-310.pyc
│   │   │   │   │   ├── selene_util.cpython-310.pyc
│   │   │   │   │   └── stateless.cpython-310.pyc
│   │   │   │   ├── _config.py
│   │   │   │   ├── basic_blocks.py
│   │   │   │   ├── cell_embedding.py
│   │   │   │   ├── cell_zones
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── test_data_dict.json
│   │   │   │   │   └── train_data.json
│   │   │   │   ├── check_utils.py
│   │   │   │   ├── data
│   │   │   │   │   ├── data_readme
│   │   │   │   │   └── sample.csv
│   │   │   │   ├── dataset_util.py
│   │   │   │   ├── drive
│   │   │   │   │   ├── Melody-MT-39.pth
│   │   │   │   │   ├── Melody-ST-Breast-Basal.pth
│   │   │   │   │   ├── Melody-ST-Breast-Luminal.pth
│   │   │   │   │   └── Melody-ST-Lung-Alveolar.pth
│   │   │   │   ├── embeddings
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── scgpt
│   │   │   │   │       ├── Adipocytes.npy
│   │   │   │   │       ├── Aorta-Endothel.npy
│   │   │   │   │       ├── Aorta-Smooth-Muscle.npy
│   │   │   │   │       ├── Bladder-Epithelial.npy
│   │   │   │   │       ├── Blood-B.npy
│   │   │   │   │       ├── Blood-Granulocytes.npy
│   │   │   │   │       ├── Blood-Monocytes.npy
│   │   │   │   │       ├── Blood-NK.npy
│   │   │   │   │       ├── Blood-T-CD3.npy
│   │   │   │   │       ├── Bone-Osteoblasts.npy
│   │   │   │   │       ├── Bone_marrow-Erythrocyte_progenitors.npy
│   │   │   │   │       ├── Breast-Basal-Epithelial.npy
│   │   │   │   │       ├── Breast-Luminal-Epithelial.npy
│   │   │   │   │       ├── Colon-Fibroblasts.npy
│   │   │   │   │       ├── Colon-Right-Epithelial.npy
│   │   │   │   │       ├── Cortex-Neuron.npy
│   │   │   │   │       ├── Dermal-Fibroblasts.npy
│   │   │   │   │       ├── Epidermal-Keratinocytes.npy
│   │   │   │   │       ├── Fallopian-Epithelial.npy
│   │   │   │   │       ├── Gallbladder-Epithelial.npy
│   │   │   │   │       ├── Gastric-fundus-Epithelial.npy
│   │   │   │   │       ├── Heart-Cardiomyocyte.npy
│   │   │   │   │       ├── Heart-Fibroblasts.npy
│   │   │   │   │       ├── Kidney-Tubular-Endothel.npy
│   │   │   │   │       ├── Liver-Hepatocytes.npy
│   │   │   │   │       ├── Lung-Alveolar-Epithelial.npy
│   │   │   │   │       ├── Lung-Bronchus-Epithelial.npy
│   │   │   │   │       ├── Oligodendrocytes.npy
│   │   │   │   │       ├── Ovary-Epithelial.npy
│   │   │   │   │       ├── Pancreas-Acinar.npy
│   │   │   │   │       ├── Pancreas-Alpha.npy
│   │   │   │   │       ├── Pancreas-Beta.npy
│   │   │   │   │       ├── Pancreas-Delta.npy
│   │   │   │   │       ├── Pancreas-Duct.npy
│   │   │   │   │       ├── Prostate-Epithelial.npy
│   │   │   │   │       ├── Skeletal-Muscle.npy
│   │   │   │   │       ├── Small-int-Epithelial.npy
│   │   │   │   │       ├── Thyroid-Epithelial.npy
│   │   │   │   │       ├── Tonsil-Palatine-Epithelial.npy
│   │   │   │   │       └── __init__.py
│   │   │   │   ├── eqtl_pure_util.py
│   │   │   │   ├── examples
│   │   │   │   │   ├── README.md
│   │   │   │   │   ├── example_predictions_regions.csv
│   │   │   │   │   ├── example_regions.bed
│   │   │   │   │   └── example_sequences.fa
│   │   │   │   ├── global_constants.py
│   │   │   │   ├── install.bash
│   │   │   │   ├── melodyG1
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── check_utils.py
│   │   │   │   │   ├── run.py
│   │   │   │   │   └── test_regions.py
│   │   │   │   ├── melodyG2
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── check_utils.py
│   │   │   │   │   ├── run.py
│   │   │   │   │   └── test_regions.py
│   │   │   │   ├── meqtl
│   │   │   │   │   └── dataset
│   │   │   │   │       └── processed
│   │   │   │   │           ├── EPIGEN
│   │   │   │   │           │   ├── EPIC.csv
│   │   │   │   │           │   └── skin.csv
│   │   │   │   │           ├── GTEX
│   │   │   │   │           │   ├── GTEX_BreastMammaryTissue.csv
│   │   │   │   │           │   ├── GTEX_ColonTransverse.csv
│   │   │   │   │           │   ├── GTEX_KidneyCortex.csv
│   │   │   │   │           │   ├── GTEX_Lung.csv
│   │   │   │   │           │   ├── GTEX_MuscleSkeletal.csv
│   │   │   │   │           │   ├── GTEX_Ovary.csv
│   │   │   │   │           │   ├── GTEX_Prostate.csv
│   │   │   │   │           │   ├── GTEX_Testis.csv
│   │   │   │   │           │   └── GTEX_WholeBlood.csv
│   │   │   │   │           └── Olafur_2024
│   │   │   │   │               ├── CPG_units.csv
│   │   │   │   │               └── MDSs.csv
│   │   │   │   ├── models.py
│   │   │   │   ├── predict.py
│   │   │   │   ├── run.py
│   │   │   │   ├── selene_sdk
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── __pycache__
│   │   │   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   │   │   └── _stub.cpython-310.pyc
│   │   │   │   │   ├── _stub.py
│   │   │   │   │   ├── predict
│   │   │   │   │   │   └── __init__.py
│   │   │   │   │   ├── samplers
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── file_samplers
│   │   │   │   │   │       └── __init__.py
│   │   │   │   │   ├── sequences
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── __pycache__
│   │   │   │   │   │       └── __init__.cpython-310.pyc
│   │   │   │   │   ├── targets
│   │   │   │   │   │   ├── __init__.py
│   │   │   │   │   │   └── __pycache__
│   │   │   │   │   │       └── __init__.cpython-310.pyc
│   │   │   │   │   └── utils
│   │   │   │   │       └── __init__.py
│   │   │   │   ├── selene_test.py
│   │   │   │   ├── selene_util.py
│   │   │   │   ├── seq_preds_check.csv
│   │   │   │   ├── stateless.py
│   │   │   │   ├── supplemental_data
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   └── supplemental_data.xlsx
│   │   │   │   ├── timing_50.csv
│   │   │   │   ├── timing_50.fa
│   │   │   │   ├── variable_blocks_util.py
│   │   │   │   └── wandb_worker.py
│   │   │   └── melody_env_freeze.txt
│   │   ├── tcga_ancestry
│   │   │   ├── Admixture_by_sample.txt
│   │   │   ├── Broad_ancestry_PCA.txt
│   │   │   ├── UCSF_Ancestry_Calls.csv
│   │   │   ├── ancestry_summary.json
│   │   │   └── tcga_ancestry_labels.csv
│   │   └── tcga_tumor
│   │       ├── tcga_tumor_all.csv
│   │       ├── tcga_tumor_summary.json
│   │       └── tcga_tumor_unpaired.csv
│   ├── fetch_colon_only.sh
│   ├── fetch_egtex_multitissue.sh
│   ├── fetch_egtex_regular_slices.sh
│   ├── harmonize_egtex_mqtl.py
│   ├── harmonize_egtex_multitissue.sh
│   ├── harmonize_genoa_meqtl.py
│   ├── harmonize_multitissue.sbatch
│   ├── hg38.fa
│   ├── hg38.fa.fai
│   ├── literature_variant_screen
│   │   ├── SHA256SUMS.txt
│   │   ├── curated_seed_grch38_preview.csv
│   │   └── literature_breast_variant_seeds.csv
│   ├── merge_genoa_harmonized.py
│   ├── reference
│   │   ├── ATAC_seq.bw
│   │   ├── H3K27ac.bw
│   │   ├── H3K27me3.bw
│   │   ├── H3K36me3.bw
│   │   ├── H3K4me1.bw
│   │   ├── H3K4me3.bw
│   │   ├── H3K9me3.bw
│   │   ├── gencode.v44.annotation.gtf.gz
│   │   ├── hg19ToHg38.over.chain.gz
│   │   └── hg38.phyloP100way.bw
│   ├── run_harmonize_genoa.sh
│   └── split_predictions_by_tissue.py
├── dnabert2_local
│   ├── LICENSE
│   ├── README.md
│   ├── bert_layers.py
│   ├── bert_padding.py
│   ├── config.json
│   ├── configuration_bert.py
│   ├── flash_attn_triton.py
│   ├── generation_config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── logs
│   ├── ctxperm
│   │   ├── 45258873.err
│   │   ├── 45258873.out
│   │   ├── 45259005.err
│   │   └── 45259005.out
│   ├── data_build
│   │   ├── acquire_clinvar.log
│   │   └── acquire_genoa.log
│   ├── egtex_mt_scoring
│   │   ├── score_0_45179943.err
│   │   ├── score_0_45179943.out
│   │   ├── score_1_45192174.err
│   │   ├── score_1_45192174.out
│   │   ├── score_1_45198766.err
│   │   ├── score_1_45198766.out
│   │   ├── score_2_45192174.err
│   │   ├── score_2_45192174.out
│   │   ├── score_3_45192174.err
│   │   ├── score_3_45192174.out
│   │   ├── score_3_45198766.err
│   │   ├── score_3_45198766.out
│   │   ├── score_4_45192174.err
│   │   ├── score_4_45192174.out
│   │   ├── score_5_45192174.err
│   │   └── score_5_45192174.out
│   ├── experiments
│   │   ├── 04_mqtl_probe_sensitivity.log
│   │   ├── 07_candidate_model_comparison.log
│   │   ├── 08_paired_model_bootstrap.log
│   │   ├── 09_mqtl_matched_negative.log
│   │   ├── 09_mqtl_matched_negative_optimal.log
│   │   ├── 09_mqtl_matched_negative_smoke.log
│   │   ├── 12_biological_context_analysis.log
│   │   ├── 13_build_manuscript_figures.log
│   │   ├── 14_known_variant_application.log
│   │   ├── 15_literature_variant_screen.log
│   │   ├── 15_literature_variant_screen_prepare.log
│   │   ├── 17_uncertainty_conditional.log
│   │   ├── 17_uncertainty_strata20.log
│   │   ├── 17_uncertainty_strata50.log
│   │   ├── 20_genoa_variant_evaluation.log
│   │   ├── 21_tcga_tumor_domain_shift.log
│   │   └── 22_motif_disruption.log
│   ├── fetch_colon.log
│   ├── fetch_egtex_multitissue.log
│   ├── fetch_egtex_slices.log
│   ├── folds
│   │   ├── 45290628_0.err
│   │   ├── 45290628_0.out
│   │   ├── 45290628_1.err
│   │   ├── 45290628_1.out
│   │   ├── 45290628_2.err
│   │   ├── 45290628_2.out
│   │   ├── 45290628_3.err
│   │   ├── 45290628_3.out
│   │   ├── 45290715_1.err
│   │   ├── 45290715_1.out
│   │   ├── 45290715_2.err
│   │   ├── 45290715_2.out
│   │   ├── 45290715_3.err
│   │   └── 45290715_3.out
│   ├── h2h_BreastMammaryTissue.log
│   ├── h2h_ColonTransverse.log
│   ├── h2h_KidneyCortex.log
│   ├── h2h_MuscleSkeletal.log
│   ├── h2h_Ovary.log
│   ├── h2h_Prostate.log
│   ├── h2h_WholeBlood.log
│   ├── harmonize_BreastMammaryTissue.log
│   ├── harmonize_ColonTransverse.log
│   ├── harmonize_KidneyCortex.log
│   ├── harmonize_Lung.log
│   ├── harmonize_MuscleSkeletal.log
│   ├── harmonize_Ovary.log
│   ├── harmonize_Prostate.log
│   ├── harmonize_Testis.log
│   ├── harmonize_WholeBlood.log
│   ├── harmonize_multitissue.log
│   ├── harmonize_run.log
│   ├── melody
│   │   ├── union_45258872.err
│   │   ├── union_45258872.out
│   │   ├── union_45259004.err
│   │   ├── union_45259004.out
│   │   ├── union_45270023.err
│   │   └── union_45270023.out
│   ├── melody_st
│   │   ├── 45289799.err
│   │   └── 45289799.out
│   ├── merge_verify.log
│   ├── r4_nine_tissue.log
│   ├── r4_smoke.log
│   ├── reproducibility
│   │   ├── 10_tcga_participant_matrix_audit.log
│   │   ├── 11_build_supplement_package.log
│   │   ├── 11_supplement_checksum_verification.log
│   │   └── data_purity_audit.log
│   ├── st_h2h_BreastMammaryTissue.log
│   ├── st_h2h_ColonTransverse.log
│   ├── st_h2h_KidneyCortex.log
│   ├── st_h2h_Lung.log
│   ├── st_h2h_MuscleSkeletal.log
│   ├── st_h2h_Ovary.log
│   ├── st_h2h_Prostate.log
│   ├── st_h2h_Testis.log
│   ├── st_h2h_WholeBlood.log
│   ├── st_matched_lung.log
│   ├── transfer_BreastMammaryTissue.log
│   ├── transfer_ColonTransverse.log
│   ├── transfer_KidneyCortex.log
│   ├── transfer_MuscleSkeletal.log
│   ├── transfer_Ovary.log
│   ├── transfer_Prostate.log
│   ├── transfer_Testis.log
│   ├── transfer_WholeBlood.log
│   ├── transfer_all.log
│   └── transfer_lung.log
├── main.tex
├── main_revised.pdf
├── main_revised.tex
├── presentation_outline.md
├── presentation_outline_long.md
├── reproducibility
│   ├── analysis_audit.txt
│   ├── data_purity_audit.json
│   ├── environment_snapshot.txt
│   ├── gdc_query_audit_sample.json
│   ├── git_storage_audit.txt
│   ├── hm450_manifest_audit.json
│   ├── pre_additional_analysis_sha256.txt
│   ├── processed_data_sha256.txt
│   ├── reference_audit.txt
│   ├── tcga_methylation_input_audit.json
│   ├── tcga_participant_matrix_audit.json
│   └── tcga_selected_normal_samples.csv
├── requirements.txt
├── results
│   └── journal
│       ├── baseline_variant_evaluation
│       │   ├── distance_bins.csv
│       │   ├── fusion_vs_sequence_paired.csv
│       │   ├── gate_modulation.csv
│       │   ├── matched_negative_auroc.csv
│       │   ├── matching_balance.csv
│       │   ├── plots
│       │   │   ├── discrimination_vs_distance.pdf
│       │   │   ├── discrimination_vs_distance.png
│       │   │   ├── significance_gradient.pdf
│       │   │   └── significance_gradient.png
│       │   ├── primary_metrics.csv
│       │   ├── run_summary.json
│       │   └── significance_gradient.csv
│       ├── biological_context
│       │   ├── fusion_gain_by_context.csv
│       │   ├── genomic_region_classification_audit.csv
│       │   ├── heldout_cpg_context_assignments.csv
│       │   ├── locus_metrics_by_context.csv
│       │   ├── plots
│       │   │   ├── fusion_gain_by_context.png
│       │   │   ├── fusion_gain_by_genomic_region.png
│       │   │   ├── information_source_gains.png
│       │   │   └── variant_response_by_distance.png
│       │   ├── run_summary.json
│       │   ├── variant_response_by_context.csv
│       │   └── variant_response_by_distance.csv
│       ├── candidates
│       │   ├── candidate_analysis_summary.json
│       │   ├── candidate_matched_background_statistics.csv
│       │   ├── candidate_seed_scores_long.csv
│       │   ├── eligible_heldout_model_visible_cohort.csv
│       │   ├── model_comparison
│       │   │   ├── candidate_sequence_vs_fusion.csv
│       │   │   ├── eligible_candidate_cohort.csv
│       │   │   ├── pairwise_seed_stability_by_model.csv
│       │   │   ├── sequence
│       │   │   │   ├── candidate_matched_background_statistics.csv
│       │   │   │   ├── candidate_seed_scores_long.csv
│       │   │   │   ├── seed42
│       │   │   │   │   └── candidate_scores.csv
│       │   │   │   ├── seed43
│       │   │   │   │   └── candidate_scores.csv
│       │   │   │   ├── seed44
│       │   │   │   │   └── candidate_scores.csv
│       │   │   │   └── top_candidates.csv
│       │   │   └── sequence_vs_fusion_summary.json
│       │   ├── plots
│       │   │   ├── matched_background_rank1.png
│       │   │   ├── matched_background_rank2.png
│       │   │   └── matched_background_rank3.png
│       │   ├── seed42
│       │   │   └── candidate_scores.csv
│       │   ├── seed43
│       │   │   └── candidate_scores.csv
│       │   ├── seed44
│       │   │   └── candidate_scores.csv
│       │   ├── stability
│       │   │   ├── candidate_run_consensus.csv
│       │   │   ├── pairwise_run_stability.csv
│       │   │   ├── per_run_orientation_stability.csv
│       │   │   └── stability_summary.json
│       │   ├── top_candidate_case_study.csv
│       │   ├── top_candidate_matched_comparators_long.csv
│       │   └── top_candidates.csv
│       ├── context_permutation
│       │   ├── agreement_with_identity.csv
│       │   ├── pair_scores_identity.csv
│       │   ├── pair_scores_median.csv
│       │   ├── pair_scores_shuffle.csv
│       │   └── run_summary.json
│       ├── egtex_mqtl_matched_negative
│       │   ├── eligible_significant_and_nonsignificant_leads.csv
│       │   ├── fusion
│       │   │   ├── seed42
│       │   │   │   └── matched_lead_predictions.csv
│       │   │   ├── seed43
│       │   │   │   └── matched_lead_predictions.csv
│       │   │   └── seed44
│       │   │       └── matched_lead_predictions.csv
│       │   ├── match_sets.csv
│       │   ├── matched_lead_cohort.csv
│       │   ├── matched_lead_predictions_all_models_seeds.csv
│       │   ├── matched_lead_predictions_seed_aggregate.csv
│       │   ├── matched_negative_metrics.csv
│       │   ├── matched_negative_summary.txt
│       │   ├── matching_balance.csv
│       │   ├── plots
│       │   │   ├── fusion_matched_negative_discrimination.png
│       │   │   └── sequence_matched_negative_discrimination.png
│       │   ├── run_summary.json
│       │   └── sequence
│       │       ├── seed42
│       │       │   └── matched_lead_predictions.csv
│       │       ├── seed43
│       │       │   └── matched_lead_predictions.csv
│       │       └── seed44
│       │           └── matched_lead_predictions.csv
│       ├── egtex_mqtl_positive_control
│       │   ├── fusion
│       │   │   ├── seed42
│       │   │   │   └── mqtl_predictions.csv
│       │   │   ├── seed43
│       │   │   │   └── mqtl_predictions.csv
│       │   │   └── seed44
│       │   │       └── mqtl_predictions.csv
│       │   ├── hm450_probe_overlap_audit.json
│       │   ├── mqtl_leave_one_variant_out.csv
│       │   ├── mqtl_leave_one_variant_out_summary.txt
│       │   ├── mqtl_positive_control_metrics.csv
│       │   ├── mqtl_positive_control_summary.txt
│       │   ├── mqtl_predictions_all_models_seeds.csv
│       │   ├── mqtl_predictions_seed_aggregate.csv
│       │   ├── mqtl_probe_overlap_sensitivity_metrics.csv
│       │   ├── plots
│       │   │   ├── fusion_effect_rank_scatter.png
│       │   │   ├── fusion_vs_sequence_delta_m.png
│       │   │   └── sequence_effect_rank_scatter.png
│       │   ├── run_summary.json
│       │   ├── sequence
│       │   │   ├── seed42
│       │   │   │   └── mqtl_predictions.csv
│       │   │   ├── seed43
│       │   │   │   └── mqtl_predictions.csv
│       │   │   └── seed44
│       │   │       └── mqtl_predictions.csv
│       │   └── validated_heldout_cohort.csv
│       ├── egtex_multitissue_scoring
│       │   ├── by_tissue
│       │   │   ├── BreastMammaryTissue
│       │   │   │   └── heldout
│       │   │   │       ├── fusion
│       │   │   │       │   ├── seed42
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   ├── seed43
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   └── seed44
│       │   │   │       │       └── pair_scores.csv
│       │   │   │       └── sequence
│       │   │   │           ├── seed42
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           ├── seed43
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           └── seed44
│       │   │   │               └── pair_scores.csv
│       │   │   ├── ColonTransverse
│       │   │   │   └── heldout
│       │   │   │       ├── fusion
│       │   │   │       │   ├── seed42
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   ├── seed43
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   └── seed44
│       │   │   │       │       └── pair_scores.csv
│       │   │   │       └── sequence
│       │   │   │           ├── seed42
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           ├── seed43
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           └── seed44
│       │   │   │               └── pair_scores.csv
│       │   │   ├── KidneyCortex
│       │   │   │   └── heldout
│       │   │   │       ├── fusion
│       │   │   │       │   ├── seed42
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   ├── seed43
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   └── seed44
│       │   │   │       │       └── pair_scores.csv
│       │   │   │       └── sequence
│       │   │   │           ├── seed42
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           ├── seed43
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           └── seed44
│       │   │   │               └── pair_scores.csv
│       │   │   ├── Lung
│       │   │   │   └── heldout
│       │   │   │       ├── fusion
│       │   │   │       │   ├── seed42
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   ├── seed43
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   └── seed44
│       │   │   │       │       └── pair_scores.csv
│       │   │   │       └── sequence
│       │   │   │           ├── seed42
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           ├── seed43
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           └── seed44
│       │   │   │               └── pair_scores.csv
│       │   │   ├── MuscleSkeletal
│       │   │   │   └── heldout
│       │   │   │       ├── fusion
│       │   │   │       │   ├── seed42
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   ├── seed43
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   └── seed44
│       │   │   │       │       └── pair_scores.csv
│       │   │   │       └── sequence
│       │   │   │           ├── seed42
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           ├── seed43
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           └── seed44
│       │   │   │               └── pair_scores.csv
│       │   │   ├── Ovary
│       │   │   │   └── heldout
│       │   │   │       ├── fusion
│       │   │   │       │   ├── seed42
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   ├── seed43
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   └── seed44
│       │   │   │       │       └── pair_scores.csv
│       │   │   │       └── sequence
│       │   │   │           ├── seed42
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           ├── seed43
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           └── seed44
│       │   │   │               └── pair_scores.csv
│       │   │   ├── Prostate
│       │   │   │   └── heldout
│       │   │   │       ├── fusion
│       │   │   │       │   ├── seed42
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   ├── seed43
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   └── seed44
│       │   │   │       │       └── pair_scores.csv
│       │   │   │       └── sequence
│       │   │   │           ├── seed42
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           ├── seed43
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           └── seed44
│       │   │   │               └── pair_scores.csv
│       │   │   ├── Testis
│       │   │   │   └── heldout
│       │   │   │       ├── fusion
│       │   │   │       │   ├── seed42
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   ├── seed43
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   └── seed44
│       │   │   │       │       └── pair_scores.csv
│       │   │   │       └── sequence
│       │   │   │           ├── seed42
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           ├── seed43
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           └── seed44
│       │   │   │               └── pair_scores.csv
│       │   │   ├── WholeBlood
│       │   │   │   └── heldout
│       │   │   │       ├── fusion
│       │   │   │       │   ├── seed42
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   ├── seed43
│       │   │   │       │   │   └── pair_scores.csv
│       │   │   │       │   └── seed44
│       │   │   │       │       └── pair_scores.csv
│       │   │   │       └── sequence
│       │   │   │           ├── seed42
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           ├── seed43
│       │   │   │           │   └── pair_scores.csv
│       │   │   │           └── seed44
│       │   │   │               └── pair_scores.csv
│       │   │   └── split_summary.json
│       │   └── heldout
│       │       ├── fusion
│       │       │   ├── seed42
│       │       │   │   └── pair_scores.csv
│       │       │   ├── seed43
│       │       │   │   └── pair_scores.csv
│       │       │   └── seed44
│       │       │       └── pair_scores.csv
│       │       ├── run_summary_fusion_seed42.json
│       │       ├── run_summary_fusion_seed43.json
│       │       ├── run_summary_fusion_seed44.json
│       │       ├── run_summary_sequence_seed42.json
│       │       ├── run_summary_sequence_seed43.json
│       │       ├── run_summary_sequence_seed44.json
│       │       └── sequence
│       │           ├── seed42
│       │           │   └── pair_scores.csv
│       │           ├── seed43
│       │           │   └── pair_scores.csv
│       │           └── seed44
│       │               └── pair_scores.csv
│       ├── egtex_variant_evaluation
│       │   ├── distance_bins.csv
│       │   ├── fusion_vs_sequence_paired.csv
│       │   ├── gate_modulation.csv
│       │   ├── matched_negative_auroc.csv
│       │   ├── matching_balance.csv
│       │   ├── plots
│       │   │   ├── discrimination_vs_distance.pdf
│       │   │   ├── discrimination_vs_distance.png
│       │   │   ├── significance_gradient.pdf
│       │   │   └── significance_gradient.png
│       │   ├── primary_metrics.csv
│       │   ├── run_summary.json
│       │   └── significance_gradient.csv
│       ├── egtex_variant_scoring
│       │   └── heldout
│       │       ├── fusion
│       │       │   ├── seed42
│       │       │   │   └── pair_scores.csv
│       │       │   ├── seed43
│       │       │   │   └── pair_scores.csv
│       │       │   └── seed44
│       │       │       └── pair_scores.csv
│       │       ├── run_summary_fusion_seed42.json
│       │       ├── run_summary_fusion_seed43.json
│       │       ├── run_summary_fusion_seed44.json
│       │       ├── run_summary_sequence_seed42.json
│       │       ├── run_summary_sequence_seed43.json
│       │       ├── run_summary_sequence_seed44.json
│       │       └── sequence
│       │           ├── seed42
│       │           │   └── pair_scores.csv
│       │           ├── seed43
│       │           │   └── pair_scores.csv
│       │           └── seed44
│       │               └── pair_scores.csv
│       ├── genoa_variant_evaluation
│       │   ├── distance_bins.csv
│       │   ├── fusion_vs_sequence_paired.csv
│       │   ├── gate_modulation.csv
│       │   ├── matched_negative_auroc.csv
│       │   ├── matching_balance.csv
│       │   ├── plots
│       │   │   ├── discrimination_vs_distance.pdf
│       │   │   ├── discrimination_vs_distance.png
│       │   │   ├── significance_gradient.pdf
│       │   │   └── significance_gradient.png
│       │   ├── primary_metrics.csv
│       │   ├── run_summary.json
│       │   └── significance_gradient.csv
│       ├── genoa_variant_scoring
│       │   └── heldout
│       │       ├── fusion
│       │       │   ├── seed42
│       │       │   │   └── pair_scores.csv
│       │       │   ├── seed43
│       │       │   │   └── pair_scores.csv
│       │       │   └── seed44
│       │       │       └── pair_scores.csv
│       │       ├── run_summary_fusion_seed42.json
│       │       ├── run_summary_fusion_seed43.json
│       │       ├── run_summary_fusion_seed44.json
│       │       ├── run_summary_sequence_seed42.json
│       │       ├── run_summary_sequence_seed43.json
│       │       ├── run_summary_sequence_seed44.json
│       │       └── sequence
│       │           ├── seed42
│       │           │   └── pair_scores.csv
│       │           ├── seed43
│       │           │   └── pair_scores.csv
│       │           └── seed44
│       │               └── pair_scores.csv
│       ├── gwas_enrichment_egtex_nominal
│       │   ├── labelled_pairs.csv
│       │   ├── preregistration.json
│       │   └── run_summary.json
│       ├── gwas_enrichment_genoa_nominal
│       │   ├── labelled_pairs.csv
│       │   ├── preregistration.json
│       │   └── run_summary.json
│       ├── known_variant_application
│       │   ├── known_variant_distal_target_audit.csv
│       │   ├── known_variant_input.csv
│       │   ├── known_variant_model_visible_cpgs.csv
│       │   ├── known_variant_predictions_all_seeds.csv
│       │   ├── known_variant_predictions_ensemble.csv
│       │   ├── known_variant_primary_case.csv
│       │   ├── known_variant_primary_case.png
│       │   ├── known_variant_visibility_audit.csv
│       │   └── run_summary.json
│       ├── literature_variant_screen
│       │   ├── all_resolved_literature_candidates.csv
│       │   ├── candidate_breast_mqtl_hits.csv
│       │   ├── candidate_exclusion_audit.csv
│       │   ├── eligible_published_candidates.csv
│       │   ├── literature_variant_predictions_all_pairs.csv
│       │   ├── literature_variant_predictions_ranked.csv
│       │   ├── publication_link_audit.csv
│       │   ├── run_summary.json
│       │   ├── silentmethyl_scoring
│       │   │   ├── known_variant_input.csv
│       │   │   ├── known_variant_model_visible_cpgs.csv
│       │   │   ├── known_variant_predictions_all_seeds.csv
│       │   │   ├── known_variant_predictions_ensemble.csv
│       │   │   ├── known_variant_primary_case.csv
│       │   │   ├── known_variant_primary_case.png
│       │   │   ├── known_variant_visibility_audit.csv
│       │   │   └── run_summary.json
│       │   ├── silentmethyl_variant_input.csv
│       │   ├── source_query_audit.csv
│       │   ├── stk11_case_study_figure_values.csv
│       │   └── top_case_per_evidence_tier.csv
│       ├── manuscript_figures
│       │   ├── candidate_response_by_context.png
│       │   ├── fusion_gain_by_epigenomic_context.png
│       │   ├── model_incremental_performance.png
│       │   ├── mqtl_combined_rank.png
│       │   ├── run_summary.json
│       │   ├── stk11_variants_vs_nonsynonymous_screen.png
│       │   └── top_candidate_matched_background.png
│       ├── melody
│       │   ├── by_tissue
│       │   │   ├── BreastMammaryTissue
│       │   │   │   └── heldout
│       │   │   │       └── melody
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── ColonTransverse
│       │   │   │   └── heldout
│       │   │   │       └── melody
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── KidneyCortex
│       │   │   │   └── heldout
│       │   │   │       └── melody
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── Lung
│       │   │   │   └── heldout
│       │   │   │       └── melody
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── MuscleSkeletal
│       │   │   │   └── heldout
│       │   │   │       └── melody
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── Ovary
│       │   │   │   └── heldout
│       │   │   │       └── melody
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── Prostate
│       │   │   │   └── heldout
│       │   │   │       └── melody
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── WholeBlood
│       │   │   │   └── heldout
│       │   │   │       └── melody
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   └── melody_by_tissue_summary.json
│       │   ├── smoke_wholeblood.csv
│       │   ├── smoke_wholeblood.summary.json
│       │   ├── union_breastluminal_only.csv
│       │   ├── union_breastluminal_only.summary.json
│       │   ├── union_pair_scores_15track.csv
│       │   ├── union_pair_scores_15track.summary.json
│       │   ├── wb_bloodB.csv
│       │   ├── wb_bloodB.summary.json
│       │   ├── wb_m100.csv
│       │   ├── wb_m100.summary.json
│       │   ├── wb_m1000.csv
│       │   ├── wb_m1000.summary.json
│       │   ├── wb_m20.csv
│       │   ├── wb_m20.summary.json
│       │   ├── wb_m200.csv
│       │   ├── wb_m200.summary.json
│       │   ├── wb_m40.csv
│       │   ├── wb_m40.summary.json
│       │   ├── wb_m500.csv
│       │   ├── wb_m500.summary.json
│       │   ├── wb_m75.csv
│       │   ├── wb_m75.summary.json
│       │   ├── wb_probe.csv
│       │   ├── wb_probe.summary.json
│       │   ├── wb_sig_m1.csv
│       │   ├── wb_sig_m1.summary.json
│       │   ├── wb_sig_m100.csv
│       │   ├── wb_sig_m100.summary.json
│       │   ├── wb_sig_m20.csv
│       │   ├── wb_sig_m20.summary.json
│       │   ├── wb_sig_m40.csv
│       │   └── wb_sig_m40.summary.json
│       ├── melody_head_to_head
│       │   ├── BreastMammaryTissue
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── ColonTransverse
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── KidneyCortex
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── Lung
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── MuscleSkeletal
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── Ovary
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── Prostate
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   └── WholeBlood
│       │       ├── paired_differences.csv
│       │       ├── per_model_on_shared_cohort.csv
│       │       ├── run_summary.json
│       │       └── tail_enrichment.csv
│       ├── melody_st
│       │   ├── by_tissue
│       │   │   ├── BreastMammaryTissue
│       │   │   │   └── heldout
│       │   │   │       └── melody_st_breast
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── ColonTransverse
│       │   │   │   └── heldout
│       │   │   │       └── melody_st_breast
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── KidneyCortex
│       │   │   │   └── heldout
│       │   │   │       └── melody_st_breast
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── Lung
│       │   │   │   └── heldout
│       │   │   │       ├── melody_st_breast
│       │   │   │       │   └── seed42
│       │   │   │       │       └── pair_scores.csv
│       │   │   │       └── melody_st_lung
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── MuscleSkeletal
│       │   │   │   └── heldout
│       │   │   │       └── melody_st_breast
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── Ovary
│       │   │   │   └── heldout
│       │   │   │       └── melody_st_breast
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── Prostate
│       │   │   │   └── heldout
│       │   │   │       └── melody_st_breast
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── Testis
│       │   │   │   └── heldout
│       │   │   │       └── melody_st_breast
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   ├── WholeBlood
│       │   │   │   └── heldout
│       │   │   │       └── melody_st_breast
│       │   │   │           └── seed42
│       │   │   │               └── pair_scores.csv
│       │   │   └── melody_st_by_tissue_summary.json
│       │   ├── st_breast_BreastMammaryTissue.csv
│       │   ├── st_breast_BreastMammaryTissue.summary.json
│       │   ├── st_breast_ColonTransverse.csv
│       │   ├── st_breast_ColonTransverse.summary.json
│       │   ├── st_breast_KidneyCortex.csv
│       │   ├── st_breast_KidneyCortex.summary.json
│       │   ├── st_breast_Lung.csv
│       │   ├── st_breast_Lung.summary.json
│       │   ├── st_breast_MuscleSkeletal.csv
│       │   ├── st_breast_MuscleSkeletal.summary.json
│       │   ├── st_breast_Ovary.csv
│       │   ├── st_breast_Ovary.summary.json
│       │   ├── st_breast_Prostate.csv
│       │   ├── st_breast_Prostate.summary.json
│       │   ├── st_breast_Testis.csv
│       │   ├── st_breast_Testis.summary.json
│       │   ├── st_breast_WholeBlood.csv
│       │   ├── st_breast_WholeBlood.summary.json
│       │   ├── st_lung_Lung.csv
│       │   └── st_lung_Lung.summary.json
│       ├── melody_st_head_to_head
│       │   ├── BreastMammaryTissue
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── ColonTransverse
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── KidneyCortex
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── Lung
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── MuscleSkeletal
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── Ovary
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── Prostate
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── Testis
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   └── WholeBlood
│       │       ├── paired_differences.csv
│       │       ├── per_model_on_shared_cohort.csv
│       │       ├── run_summary.json
│       │       └── tail_enrichment.csv
│       ├── melody_st_matched_vs_unmatched
│       │   └── Lung
│       │       ├── paired_differences.csv
│       │       ├── per_model_on_shared_cohort.csv
│       │       ├── run_summary.json
│       │       └── tail_enrichment.csv
│       ├── meqtl_class_chromatin
│       │   ├── chromatin_by_class.csv
│       │   └── run_summary.json
│       ├── motif_disruption
│       │   ├── coupling_null_summary.json
│       │   ├── meqtl_discrimination_by_motif_status.csv
│       │   ├── motif_vs_background.csv
│       │   ├── per_motif_coupling.csv
│       │   ├── plots
│       │   │   ├── motif_disruption.pdf
│       │   │   └── motif_disruption.png
│       │   ├── run_summary.json
│       │   └── top_motif_overlap.csv
│       ├── motif_disruption_kmer_baseline
│       │   ├── coupling_null_summary.json
│       │   ├── meqtl_discrimination_by_motif_status.csv
│       │   ├── motif_vs_background.csv
│       │   ├── per_motif_coupling.csv
│       │   ├── plots
│       │   │   ├── motif_disruption.pdf
│       │   │   └── motif_disruption.png
│       │   ├── run_summary.json
│       │   └── top_motif_overlap.csv
│       ├── paired_model_bootstrap
│       │   ├── model_metrics_recomputed.csv
│       │   ├── paired_model_difference_bootstrap.csv
│       │   └── run_summary.json
│       ├── paired_model_comparison_egtex
│       │   ├── paired_differences.csv
│       │   ├── per_model_on_shared_cohort.csv
│       │   └── run_summary.json
│       ├── paired_model_comparison_genoa
│       │   ├── paired_differences.csv
│       │   ├── per_model_on_shared_cohort.csv
│       │   └── run_summary.json
│       ├── published_baselines
│       │   ├── cpgenie
│       │   │   ├── grid_search.csv
│       │   │   ├── seed42
│       │   │   │   ├── metrics.json
│       │   │   │   ├── predictions.csv
│       │   │   │   └── weights.pth
│       │   │   ├── seed43
│       │   │   │   ├── metrics.json
│       │   │   │   ├── predictions.csv
│       │   │   │   └── weights.pth
│       │   │   ├── seed44
│       │   │   │   ├── metrics.json
│       │   │   │   ├── predictions.csv
│       │   │   │   └── weights.pth
│       │   │   └── selected_hyperparameters.json
│       │   ├── deepcpg
│       │   │   ├── grid_search.csv
│       │   │   ├── seed42
│       │   │   │   ├── metrics.json
│       │   │   │   ├── predictions.csv
│       │   │   │   └── weights.pth
│       │   │   ├── seed43
│       │   │   │   ├── metrics.json
│       │   │   │   ├── predictions.csv
│       │   │   │   └── weights.pth
│       │   │   ├── seed44
│       │   │   │   ├── metrics.json
│       │   │   │   ├── predictions.csv
│       │   │   │   └── weights.pth
│       │   │   └── selected_hyperparameters.json
│       │   ├── variant_evaluation_genoa
│       │   │   ├── distance_bins.csv
│       │   │   ├── fusion_vs_sequence_paired.csv
│       │   │   ├── gate_modulation.csv
│       │   │   ├── matched_negative_auroc.csv
│       │   │   ├── matching_balance.csv
│       │   │   ├── plots
│       │   │   │   ├── discrimination_vs_distance.pdf
│       │   │   │   ├── discrimination_vs_distance.png
│       │   │   │   ├── significance_gradient.pdf
│       │   │   │   └── significance_gradient.png
│       │   │   ├── primary_metrics.csv
│       │   │   ├── run_summary.json
│       │   │   └── significance_gradient.csv
│       │   └── variant_scoring
│       │       └── heldout
│       │           ├── cpgenie
│       │           │   ├── scoring_summary.json
│       │           │   ├── seed42
│       │           │   │   └── pair_scores.csv
│       │           │   ├── seed43
│       │           │   │   └── pair_scores.csv
│       │           │   └── seed44
│       │           │       └── pair_scores.csv
│       │           └── deepcpg
│       │               ├── scoring_summary.json
│       │               ├── seed42
│       │               │   └── pair_scores.csv
│       │               ├── seed43
│       │               │   └── pair_scores.csv
│       │               └── seed44
│       │                   └── pair_scores.csv
│       ├── published_baselines_egtex
│       │   ├── variant_evaluation
│       │   │   ├── distance_bins.csv
│       │   │   ├── fusion_vs_sequence_paired.csv
│       │   │   ├── gate_modulation.csv
│       │   │   ├── matched_negative_auroc.csv
│       │   │   ├── matching_balance.csv
│       │   │   ├── plots
│       │   │   │   ├── discrimination_vs_distance.pdf
│       │   │   │   ├── discrimination_vs_distance.png
│       │   │   │   ├── significance_gradient.pdf
│       │   │   │   └── significance_gradient.png
│       │   │   ├── primary_metrics.csv
│       │   │   ├── run_summary.json
│       │   │   └── significance_gradient.csv
│       │   └── variant_scoring
│       │       └── heldout
│       │           ├── cpgenie
│       │           │   ├── scoring_summary.json
│       │           │   ├── seed42
│       │           │   │   └── pair_scores.csv
│       │           │   ├── seed43
│       │           │   │   └── pair_scores.csv
│       │           │   └── seed44
│       │           │       └── pair_scores.csv
│       │           └── deepcpg
│       │               ├── scoring_summary.json
│       │               ├── seed42
│       │               │   └── pair_scores.csv
│       │               ├── seed43
│       │               │   └── pair_scores.csv
│       │               └── seed44
│       │                   └── pair_scores.csv
│       ├── rc_uncertainty
│       │   ├── disagreement_decile_table.csv
│       │   ├── disagreement_error_correlations.csv
│       │   ├── estimator_comparison.csv
│       │   ├── plots
│       │   │   ├── error_by_disagreement_decile_epi_seed42.png
│       │   │   ├── error_by_disagreement_decile_epi_seed43.png
│       │   │   ├── error_by_disagreement_decile_epi_seed44.png
│       │   │   ├── error_by_disagreement_decile_fusion_seed42.png
│       │   │   ├── error_by_disagreement_decile_fusion_seed43.png
│       │   │   ├── error_by_disagreement_decile_fusion_seed44.png
│       │   │   ├── error_by_disagreement_decile_sequence_seed42.png
│       │   │   ├── error_by_disagreement_decile_sequence_seed43.png
│       │   │   ├── error_by_disagreement_decile_sequence_seed44.png
│       │   │   ├── risk_coverage_epi_seed42.png
│       │   │   ├── risk_coverage_epi_seed43.png
│       │   │   ├── risk_coverage_epi_seed44.png
│       │   │   ├── risk_coverage_fusion_seed42.png
│       │   │   ├── risk_coverage_fusion_seed43.png
│       │   │   ├── risk_coverage_fusion_seed44.png
│       │   │   ├── risk_coverage_sequence_seed42.png
│       │   │   ├── risk_coverage_sequence_seed43.png
│       │   │   └── risk_coverage_sequence_seed44.png
│       │   ├── rc_uncertainty_summary.txt
│       │   ├── run_summary.json
│       │   └── selective_prediction_curves.csv
│       ├── rc_uncertainty_conditional
│       │   ├── conditional_summary.txt
│       │   ├── incremental_value.csv
│       │   ├── partial_correlations.csv
│       │   ├── run_summary.json
│       │   ├── stratified_selective_prediction.csv
│       │   └── within_stratum_correlations.csv
│       ├── rc_uncertainty_figure
│       │   ├── beats_boundary_by_strata.csv
│       │   ├── run_summary.json
│       │   ├── uncertainty_figure_summary.txt
│       │   ├── uncertainty_scale_stability.pdf
│       │   ├── uncertainty_scale_stability.png
│       │   └── within_stratum_by_strata.csv
│       ├── seed42
│       │   ├── epi
│       │   │   ├── fig_1_density_scatter.png
│       │   │   ├── fig_2a_signed_error.png
│       │   │   ├── fig_2b_absolute_error.png
│       │   │   ├── fig_3_beta_distribution.png
│       │   │   ├── fig_4_roc.png
│       │   │   ├── fig_5_calibration.png
│       │   │   ├── metrics.json
│       │   │   └── predictions.csv
│       │   ├── fusion
│       │   │   ├── fig_1_density_scatter.png
│       │   │   ├── fig_2a_signed_error.png
│       │   │   ├── fig_2b_absolute_error.png
│       │   │   ├── fig_3_beta_distribution.png
│       │   │   ├── fig_4_roc.png
│       │   │   ├── fig_5_calibration.png
│       │   │   ├── fig_6_gate_share_distribution.png
│       │   │   ├── fig_7_gate_rc_consistency.png
│       │   │   ├── metrics.json
│       │   │   └── predictions.csv
│       │   └── sequence
│       │       ├── fig_1_density_scatter.png
│       │       ├── fig_2a_signed_error.png
│       │       ├── fig_2b_absolute_error.png
│       │       ├── fig_3_beta_distribution.png
│       │       ├── fig_4_roc.png
│       │       ├── fig_5_calibration.png
│       │       ├── metrics.json
│       │       └── predictions.csv
│       ├── seed43
│       │   ├── epi
│       │   │   ├── fig_1_density_scatter.png
│       │   │   ├── fig_2a_signed_error.png
│       │   │   ├── fig_2b_absolute_error.png
│       │   │   ├── fig_3_beta_distribution.png
│       │   │   ├── fig_4_roc.png
│       │   │   ├── fig_5_calibration.png
│       │   │   ├── metrics.json
│       │   │   └── predictions.csv
│       │   ├── fusion
│       │   │   ├── fig_1_density_scatter.png
│       │   │   ├── fig_2a_signed_error.png
│       │   │   ├── fig_2b_absolute_error.png
│       │   │   ├── fig_3_beta_distribution.png
│       │   │   ├── fig_4_roc.png
│       │   │   ├── fig_5_calibration.png
│       │   │   ├── fig_6_gate_share_distribution.png
│       │   │   ├── fig_7_gate_rc_consistency.png
│       │   │   ├── metrics.json
│       │   │   └── predictions.csv
│       │   └── sequence
│       │       ├── fig_1_density_scatter.png
│       │       ├── fig_2a_signed_error.png
│       │       ├── fig_2b_absolute_error.png
│       │       ├── fig_3_beta_distribution.png
│       │       ├── fig_4_roc.png
│       │       ├── fig_5_calibration.png
│       │       ├── metrics.json
│       │       └── predictions.csv
│       ├── seed44
│       │   ├── epi
│       │   │   ├── fig_1_density_scatter.png
│       │   │   ├── fig_2a_signed_error.png
│       │   │   ├── fig_2b_absolute_error.png
│       │   │   ├── fig_3_beta_distribution.png
│       │   │   ├── fig_4_roc.png
│       │   │   ├── fig_5_calibration.png
│       │   │   ├── metrics.json
│       │   │   └── predictions.csv
│       │   ├── fusion
│       │   │   ├── fig_1_density_scatter.png
│       │   │   ├── fig_2a_signed_error.png
│       │   │   ├── fig_2b_absolute_error.png
│       │   │   ├── fig_3_beta_distribution.png
│       │   │   ├── fig_4_roc.png
│       │   │   ├── fig_5_calibration.png
│       │   │   ├── fig_6_gate_share_distribution.png
│       │   │   ├── fig_7_gate_rc_consistency.png
│       │   │   ├── metrics.json
│       │   │   └── predictions.csv
│       │   └── sequence
│       │       ├── fig_1_density_scatter.png
│       │       ├── fig_2a_signed_error.png
│       │       ├── fig_2b_absolute_error.png
│       │       ├── fig_3_beta_distribution.png
│       │       ├── fig_4_roc.png
│       │       ├── fig_5_calibration.png
│       │       ├── metrics.json
│       │       └── predictions.csv
│       ├── sequence_baselines
│       │   ├── absolute_prediction_metrics.csv
│       │   ├── run_summary.json
│       │   └── variant_scoring
│       │       └── heldout
│       │           ├── composition
│       │           │   └── seed-1
│       │           │       └── pair_scores.csv
│       │           └── kmer_ridge
│       │               └── seed-1
│       │                   └── pair_scores.csv
│       ├── target_qc
│       │   ├── coverage_bin_metrics.csv
│       │   ├── coverage_error_correlations.csv
│       │   ├── coverage_sensitivity_summary.txt
│       │   ├── coverage_threshold_metrics.csv
│       │   ├── hm450_manifest_audit.json
│       │   └── test_coverage_per_probe.csv
│       ├── tissue_shared_meqtls
│       │   ├── by_class.csv
│       │   ├── decision.json
│       │   ├── run_summary.json
│       │   ├── summary_direction_agreement.csv
│       │   └── summary_signed_rho.csv
│       ├── tissue_specificity_smoke
│       │   ├── by_class.csv
│       │   └── run_summary.json
│       ├── training_data_audit
│       │   ├── cross_split_sequence_similarity.csv
│       │   ├── probe_qc_composition.csv
│       │   ├── run_summary.json
│       │   └── split_comparability.csv
│       ├── transfer_discrimination
│       │   ├── BreastMammaryTissue
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── ColonTransverse
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── KidneyCortex
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── Lung
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── MuscleSkeletal
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── Ovary
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── Prostate
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── Testis
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   ├── WholeBlood
│       │   │   ├── paired_differences.csv
│       │   │   ├── per_model_on_shared_cohort.csv
│       │   │   ├── run_summary.json
│       │   │   └── tail_enrichment.csv
│       │   └── nine_tissue_summary.csv
│       └── variant_effect_synthesis
│           ├── all_strata.csv
│           ├── cross_cohort_meta_analysis.csv
│           └── run_summary.json
├── run_baseline.sh
├── run_epi.sh
├── run_experiments.sh
├── run_multimodal.sh
└── scripts
    ├── 01_target_qc.py
    ├── 02_tcga_participant_audit.py
    ├── 10_train_sequence.py
    ├── 11_train_epi.py
    ├── 12_train_fusion.py
    ├── 13_test_model.py
    ├── 14_baselines_simple.py
    ├── 15_baselines_published.py
    ├── 16_paired_model_bootstrap.py
    ├── 17_chromosome_splits.py
    ├── 18_ancestry_stratified_error.py
    ├── 20_variant_scoring.py
    ├── 21_variant_evaluation.py
    ├── 22_context_stratification.py
    ├── 23_context_permutation.py
    ├── 30_transfer_synthesis.py
    ├── 31_transfer_discrimination.py
    ├── 32_transfer_summary.py
    ├── 33_melody_scoring.py
    ├── 34_melody_by_tissue.py
    ├── 35_melody_st_by_tissue.py
    ├── 40_meqtl_tissue_specificity.py
    ├── 41_tissue_specificity_summary.py
    ├── 50_motif_disruption.py
    ├── 51_rc_uncertainty.py
    ├── 52_gwas_enrichment.py
    ├── 60_candidate_background.py
    ├── 61_candidate_stability.py
    ├── 62_candidate_comparison.py
    ├── 63_known_variant_application.py
    ├── 64_literature_variant_screen.py
    ├── 70_mqtl_positive_control.py
    ├── 71_mqtl_matched_negative.py
    ├── 90_build_supplement_package.py
    ├── 91_build_manuscript_figures.py
    ├── __pycache__
    │   ├── 20_variant_scoring.cpython-310.pyc
    │   ├── 21_variant_evaluation.cpython-310.pyc
    │   ├── matched_background_utils.cpython-310.pyc
    │   └── training_common.cpython-310.pyc
    ├── _test_sequence_baselines.py
    ├── literature_breast_variant_seeds.csv
    ├── matched_background_utils.py
    ├── run_baseline_grid.sh
    ├── run_baseline_seeds.sh
    ├── run_ctxperm.sbatch
    ├── run_egtex_multitissue_scoring.sh
    ├── run_egtex_scoring.sh
    ├── run_folds.sbatch
    ├── run_genoa_scoring.sh
    ├── run_melody_st.sbatch
    ├── run_melody_union.sbatch
    ├── run_transfer_all.sh
    ├── testing_common.py
    └── training_common.py

487 directories, 1285 files
