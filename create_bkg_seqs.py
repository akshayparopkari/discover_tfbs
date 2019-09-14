#!/usr/bin/env python3

"""
Build feature table from input FASTA files.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.1.6"


from sys import exit
from os.path import isfile, abspath
import argparse
from random import sample
from collections import defaultdict
from itertools import product, starmap
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
    parser.add_argument("fg_fasta_file", help="Path to foreground/true positive sequence "
                        "dataset FASTA format file [REQUIRED]")
    parser.add_argument("protein_name", type=str,
                        help="Name of transcription factor. Please see the list of valid "
                        "choices for this parameter [REQUIRED]")
    parser.add_argument("-o", "--output_file", type=str,
                        help="Specify location and filename to save consolidated data to "
                        "this tab-separated file")
    return parser.parse_args()







if __name__ == "__main__":
    exit(main())
