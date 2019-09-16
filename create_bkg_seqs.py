#!/usr/bin/env python3

"""
Build feature table from input FASTA files.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.1.0"


from sys import exit
from os.path import isfile, abspath
import argparse
from random import sample
from collections import defaultdict
from time import localtime, strftime
from pprint import pprint
from utils import parse_fasta, calculate_gc_percent, pac
err = []
try:
    import numpy as np
except ImportError:
    err.append("numpy")
try:
    import pandas as pd
except ImportError:
    err.append("pandas")
try:
    assert len(err) == 0
except AssertionError:
    for error in err:
        print("Please install {}".format(error))
    exit()


def handle_program_options():
    parser = argparse.ArgumentParser(
        description="Using foreground sequences, generate background sequences. "
        "Background sequences will be generated firstly by matching foreground motif "
        "GC-percent and length. Secondly, the foregound sequences will be shuffled to "
        "keep dinucleotide composition constant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("fg_fasta_file", type=str,
                        help="Path to foreground/true positive sequence dataset FASTA "
                        "format file [REQUIRED]")
    parser.add_argument("protein_name", type=str,
                        help="Name of transcription factor protein. [REQUIRED]")
    parser.add_argument("genome_fasta_files", type=str, nargs="+",
                        help="Specify path to one or more genome files to use as template"
                        " from which to generae random background sequences. These genome"
                        "file(s) must be FASTA file with very low probability of "
                        "containing sequences with binding motifs. For example, these "
                        "files can be FASTA file of exonic regions of non-related species")
    parser.add_argument("-o", "--output_file", type=str,
                        help="Specify location and filename to save background sequence "
                        "data.")
    return parser.parse_args()


def main():

    args = handle_program_options()

    try:
        assert isfile(args.fg_fasta_file)
    except AssertionError as e:
        print("Error with input foreground FASTA file(s). Please check supplied FASTA "
              "file - {}".format(e))
        exit()

    # generate random length matched sequences
    print("\n", strftime("%x %X".format(localtime)),
          ": Generating random length-matched sequences")
    print("="*63, sep="\n")
    fg_seqs = {header: seq for header, seq in parse_fasta(args.fg_fasta_file)}
    seq_length = len(list(fg_seqs.values())[0])
    print(fg_seqs, seq_length, sep="\n\n")



if __name__ == "__main__":
    exit(main())
