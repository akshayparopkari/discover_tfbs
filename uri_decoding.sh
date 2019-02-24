#!/bin/bash

#########################################################################################
# DESCRIPTION
#
# For an input file, decode percent encoding characters and replace with standard human
# readable chracters. This script is specifically written to decode GFF file information
# from Candida Genome Database (CGD). The following encodings are decoded -
# %20: '_' (underscore)
# %22: '"' (double quotation mark)
# %23: '#' (hash sign)
# %25: '%' (percent sign)
# %26: '&' (ampersand)
# %27: "'" (single quotation mark)
# %28: '(' (open parenthesis)
# %29: ')' (close parenthesis)
# %2B: '+' (plus sign)
# %2C: ',' (comma)
# %2F: '/' (forward slash)
# %3A: ':' (colon)
# %3B: ';' (semicolon)
# %3E: '>' (greater than)
# %5B: '[' (open square bracket)
# %5D: ']' (close square bracket)
#
#########################################################################################
#
# USAGE
# sh path/to/uri_decofing.sh input_file
# input_file contains percent encoding which will be decoded inplace with this script via
# sed -i
#
#########################################################################################
#
# AUTHOR: Akshay Paropkari
# VERSION: 0.0.2
#
#########################################################################################

# Testing for input
if [[ -z "$1" ]] ; then
    echo -e "\n$(date "+%a %D %r"): Input file not supplied. Please provide an input."
    exit 1
fi

INPUT_FILE=$(realpath "$1")
sed -i -e 's/%20/_/g' -e 's/%22/"/g' -e 's/%23/#/g' -e 's/%25/%/g' -e 's/%26/\&/g' -e 's/%27/'\''/g'  -e 's/%28/(/g' -e 's/%29/)/g' -e 's/%2B/+/g' -e 's/%2C/,/g' -e 's/%2F/\//g' -e 's/%3A/:/g' -e 's/%3B/;/g' -e 's/%3E/>/g' -e 's/%5B/[/g' -e 's/%5D/]/g' "$INPUT_FILE"
sort -k1,1 -k4,4n "$INPUT_FILE"
echo -e "$(date "+%a %D %r"): $INPUT_FILE decoded and sorted"
