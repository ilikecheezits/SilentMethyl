#!/usr/bin/env Rscript

if (!requireNamespace("DNAshapeR", quietly = TRUE)) {
    stop("[!] Error: DNAshapeR library not installed.")
}

library(DNAshapeR)
library(Biostrings)

SHAPE_TYPES <- c(
    "Shift", "Slide", "Rise", "Tilt", "Roll", "HelT",
    "Shear", "Stretch", "Stagger", "Buckle", "ProT", "Opening",
    "MGW", "EP"
)

extract_3d_tensor <- function(fasta_path, output_filename) {
    cat(paste0("[*] Processing FASTA file: ", fasta_path, "\n"))
    
    # DNAshapeR calculates this in highly-optimized C++
    pred <- getShape(fasta_path, shapeType = SHAPE_TYPES)
    sample_count <- nrow(pred[[1]])
    
    # Pre-allocating the matrix makes R lightning fast
    compiled_data <- matrix(0, nrow = sample_count, ncol = 14 * 101)

    for (i in 1:length(SHAPE_TYPES)) {
        shape_name <- SHAPE_TYPES[i]
        shape_matrix <- pred[[shape_name]]

        if (ncol(shape_matrix) == 100) {
            shape_matrix <- cbind(shape_matrix, NA)
        }
	
        col_start <- ((i - 1) * 101) + 1
        col_end <- i * 101
        compiled_data[, col_start:col_end] <- shape_matrix

        unlink(paste0(fasta_path, ".", shape_name))
    }

    write.table(compiled_data, file = output_filename, sep = "\t",
                row.names = FALSE, col.names = FALSE)
    cat(paste0("[+] Exported matrix to: ", output_filename, "\n\n"))
}

# --- EXECUTION PIPELINE ---

# 1. Process Training Data
if (file.exists("train_healthy_101bp.fasta")) {
    extract_3d_tensor("train_healthy_101bp.fasta", "training_wild_type_3d_shapes.tsv")
    # Force RAM garbage collection to free memory and keep the script fast
    gc() 
}

# 2. Process Testing Data
if (file.exists("test_healthy_101bp.fasta")) {
    extract_3d_tensor("test_healthy_101bp.fasta", "testing_wild_type_3d_shapes.tsv")
    gc()
}

if (file.exists("test_mutated_101bp.fasta")) {
    extract_3d_tensor("test_mutated_101bp.fasta", "testing_mutated_3d_shapes.tsv")
    gc()
}

cat("[*] Phase 2 Geometry Engine tasks complete.\n")

