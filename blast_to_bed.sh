#!/bin/bash

#########################################################################################
# DESCRIPTION
# Convert tabular BLAST output to BED format. Users need to supply input and output file
# with the script.
#
# USAGE
# sh blast_to_bed.sh input_blast_file genome_file
#
# from bedtools reference -
# The genome file should tab delimited and structured as follows
# <chromName><TAB><chromSize>
#
# SCRIPT INFORMATION
#
# Author: Akshay Paropkari
# Version: 0.0.4
#########################################################################################

# define input and output file variables
INFILE=$(basename "$1")
OUTFILE=$(basename "$INFILE" .blout).bed
CHRFILE=$(realpath "$2")

# change directory to input folder
if cd "$(dirname "$1")"; then echo -e "\n$(date "+%a %D %r"): cd into input directory successful"; else echo -e "\n$(date "+%a %D %r"): cd into input directory failed. Exiting."; exit 1; fi

# convert BLAST output to BED format
echo -e "\n$(date "+%a %D %r"): PROCESSING $INFILE"
awk 'BEGIN {FS="\t"; OFS=FS} {if ($9 < $10) {print $2, $9 - 1, $10, "", "0", "+"} else {print $2, $10 - 1, $9, "", "0", "-"}}' < "$INFILE" > "$OUTFILE"
echo -e "\n$(date "+%a %D %r"): $INFILE converted to $OUTFILE"

# define files for flankBed command
FLANKEDFILE=$(basename "$OUTFILE" .bed)_1000bp_flanking.bed
FOLDERNAME="flanking_regions"

# get 1000 bp flanking regions around TFBS
echo -e "\n$(date "+%a %D %r"): Getting 1000bp flanking interval around TFBS"
flankBed -i "$OUTFILE" -g "$CHRFILE" -b 1000 -s -header > "$FLANKEDFILE"
if [[ -d "$FOLDERNAME" ]]
then
  mv "$FLANKEDFILE" "$FOLDERNAME" 
else
  mkdir -p "$FOLDERNAME"
  mv "$FLANKEDFILE" "$FOLDERNAME"
fi
echo -e "\n$(date "+%a %D %r"): BED file with flanking intervals saved in $(realpath "$FOLDERNAME")"
