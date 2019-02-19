#!/bin/bash

#########################################################################################
# DESCRIPTION
# Convert tabular BLAST output to BED format. Users need to supply input and output file
# with the script.
#
# USAGE
# sh blast_to_bed.sh input_blast_file genome_file chromosomal_sequences_fasta
#
# from bedtools reference -
# The genome file should tab delimited and structured as follows
# <chromName><TAB><chromSize>
#
# SCRIPT INFORMATION
#
# Author: Akshay Paropkari
# Version: 0.0.6
#########################################################################################

# define input and output file variables
INFILE=$(basename "$1")
OUTFILE=$(basename "$INFILE" .blout).bed
GENOMEFILE=$(realpath "$2")
CHROMFASTA=$(realpath "$3")

# change directory to input folder
if cd "$(dirname "$1")"; then echo -e "$(date "+%a %D %r"): cd into input directory successful"; else echo -e "$(date "+%a %D %r"): cd into input directory failed. Exiting."; exit 1; fi

# convert BLAST output to BED format
echo -e "$(date "+%a %D %r"): PROCESSING $INFILE"
awk 'BEGIN {FS="\t"; OFS=FS} {if ($9 < $10) {print $2, $9 - 1, $10, ".", "0", "+"} else {print $2, $10 - 1, $9, ".", "0", "-"}}' < "$INFILE" > "$OUTFILE"
sort -k1,1 -k2,2n "$OUTFILE" > tmp.bed && mv tmp.bed "$OUTFILE"
echo -e "$(date "+%a %D %r"): $INFILE converted to $OUTFILE"

# define files for flankBed command
FLANKEDFILE=$(basename "$OUTFILE" .bed)_1000bp_flanking.bed
FOLDERNAME="flanking_regions"

# get 1000 bp flanking regions around TFBS
echo -e "$(date "+%a %D %r"): Getting 1000bp flanking interval around TFBS"
flankBed -i "$OUTFILE" -g "$GENOMEFILE" -b 1000 -s -header > "$FLANKEDFILE"
if [[ -d "$FOLDERNAME" ]]
then
  mv "$FLANKEDFILE" "$FOLDERNAME" 
else
  mkdir -p "$FOLDERNAME"
  mv "$FLANKEDFILE" "$FOLDERNAME"
fi
echo -e "$(date "+%a %D %r"): BED file with flanking intervals saved in $(realpath "$FOLDERNAME")"

# get sequences of flaking regions identified in previous step
echo -e "$(date "+%a %D %r"): Getting 1000bp flanking sequences around TFBS"
BEDFILE=$(realpath "$FOLDERNAME"/"$FLANKEDFILE")
FASTAOUT=$(basename "$BEDFILE" .bed).fasta
bedtools getfasta -s -fi "$CHROMFASTA" -bed "$BEDFILE" > "$FOLDERNAME/$FASTAOUT"
echo -e "$(date "+%a %D %r"): Flanking FASTA sequences collected in $(realpath "$FOLDERNAME")"
