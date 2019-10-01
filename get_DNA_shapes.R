#########################################################################################
#
# DESCRIPTION
# This R script iterates through all FASTA files in current directory and calculates
# 3D DNA shapes for for each sequence in the file. Ideally, the current directory only
# contains foreground or background sequence FASTA files.
#
# AUTHOR
# Akshay Paropkari
#
# VERSION
# 0.0.1
#########################################################################################

library(DNAshapeR)

# add in the appropriate directory
setwd("path/to/directory/with/bkg/FASTA/files")

fasta.files <- list.files(path=".", pattern="*.fasta",
                          full.names=TRUE)

for (f in fasta.files) {
  getShape(filename = f)
}
