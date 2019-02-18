#! /usr/bin.bash


#########################################################################################
# DESCRIPTION
# Convert tabular BLAST output to BED format. Users need to supply input and output file
# with the script.
#
# USAGE
# sh blast_to_bed.sh input_blast_file output_bed_file genome_file
#
# from bedtools reference -
# The genome file should tab delimited and structured as follows
# <chromName><TAB><chromSize>
#
# SCRIPT INFORMATION
#
# Author: Akshay Paropkari
# Version: 0.0.2
#########################################################################################

# load Anaconda
module load anaconda3

# define input and output file variables
INFILE=$(realpath "$1")
OUTFILE=$(realpath "$2")

# convert BLAST output to BED format
echo -e "\nPROCESSING $INFILE"
awk 'BEGIN {FS="\t"; OFS=FS} {if ($9 < $10) {print $2, $9 - 1, $10, "", "0", "+"} else {print $2, $10 - 1, $9, "", "0", "-"}}' < "$INFILE" > "$OUTFILE"
echo -e "$INFILE converted to $OUTFILE\n"

# define files for flankBed command
FLANKEDFILE=$(sed "s/.bed/_1000bp_flanking.bed/" "$OUTFILE")
CHRFILE=$(realpath "$3")

# get 1000 bp flanking regions around TFBS
flankBed -i "$OUTFILE" -g "$CHRFILE" -b 1000 -s -header > "$FLANKEDFILE"
echo -e "BED file with flanking intervals created from $OUTFILE\n"
