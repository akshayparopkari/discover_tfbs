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

module load anaconda3

INFILE=$(realpath "$1")
OUTFILE=$(realpath "$2")

echo -e "\nPROCESSING $INFILE"
awk 'BEGIN {FS="\t"; OFS=FS} {if ($9 < $10) {print $2, $9 - 1, $10, "", "0", "+"} else {print $2, $10 - 1, $9, "", "0", "-"}}' < "$INFILE" > "$OUTFILE"
echo -e "$INFILE converted to $OUTFILE\n"

FLANKEDFILE=$(sed "s/.bed/_1000bp_flanking.bed/" "$OUTFILE")
CHRFILE=$(realpath "$3")
flankBed -i "$OUTFILE" -g "$CHRFILE" -b 1000 -s -header > "$FLANKEDFILE"
echo -e "BED file with flanking intervals created from $OUTFILE\n"

