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
    compiled_data <- matrix(0, nrow = sample_count, ncol = 14 * 100)

    for (i in 1:length(SHAPE_TYPES)) {
        shape_name <- SHAPE_TYPES[i]
        shape_matrix <- pred[[shape_name]]

        if (ncol(shape_matrix) == 100) {
            shape_matrix <- cbind(shape_matrix, NA)
        }
	
        col_start <- ((i - 1) * 100) + 1
        col_end <- i * 100
        compiled_data[, col_start:col_end] <- shape_matrix

        unlink(paste0(fasta_path, ".", shape_name))
    }

    write.table(compiled_data, file = output_filename, sep = "\t",
                row.names = FALSE, col.names = FALSE)
    cat(paste0("[+] Exported matrix to: ", output_filename, "\n\n"))
}


if (file.exists("train_val_healthy_101bp.fasta")) {
    extract_3d_tensor("train_val_healthy_101bp.fasta", "train_val_3d_shapes.tsv")
    gc() 
}

if (file.exists("held_out_test_healthy_101bp.fasta")) {
    extract_3d_tensor("held_out_test_healthy_101bp.fasta", "held_out_test_3d_shapes.tsv")
    gc()
}


cat("[*] Phase 2 Geometry Engine tasks complete.\n")

