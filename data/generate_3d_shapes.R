#!/usr/bin/env Rscript

if (!requireNamespace("DNAshapeR", quietly = TRUE)) {
    stop("[!] Error: DNAshapeR library not installed.")
}
if (!requireNamespace("data.table", quietly = TRUE)) {
    install.packages("data.table", repos = "http://cran.us.r-project.org")
}

library(DNAshapeR)
library(Biostrings)
library(data.table)

SHAPE_TYPES <- c(
    "Shift", "Slide", "Rise", "Tilt", "Roll", "HelT",
    "Shear", "Stretch", "Stagger", "Buckle", "ProT", "Opening",
    "MGW", "EP"
)

extract_3d_tensor <- function(fasta_path, output_filename) {
    cat(paste0("[*] Processing FASTA file: ", fasta_path, "\n"))
    
    pred <- getShape(fasta_path, shapeType = SHAPE_TYPES)
    sample_count <- nrow(pred[[1]])
    
    compiled_data <- matrix(0, nrow = sample_count, ncol = 14 * 100)

    for (i in 1:length(SHAPE_TYPES)) {
        shape_name <- SHAPE_TYPES[i]
        shape_matrix <- pred[[shape_name]]

        if (ncol(shape_matrix) == 99) {
            shape_matrix <- cbind(shape_matrix, NA)
        }
    
        col_start <- ((i - 1) * 100) + 1
        col_end <- i * 100
        compiled_data[, col_start:col_end] <- shape_matrix

        unlink(paste0(fasta_path, ".", shape_name))
    }

    fwrite(as.data.table(compiled_data), file = output_filename, sep = "\t",
           row.names = FALSE, col.names = FALSE)
    rm(pred, compiled_data)
    cat(paste0("[+] Exported matrix to: ", output_filename, "\n\n"))
}

if (file.exists("train_100bp.fasta")) {
    extract_3d_tensor("train_100bp.fasta", "train_3d_shapes.tsv")
    gc() 
}

if (file.exists("val_100bp.fasta")) {
    extract_3d_tensor("val_100bp.fasta", "val_3d_shapes.tsv")
    gc() 
}
if (file.exists("test_100bp.fasta")) {
    extract_3d_tensor("test_100bp.fasta", "test_3d_shapes.tsv")
    gc() 
}

cat("[*] Phase 2 Geometry Engine tasks complete.\n")


