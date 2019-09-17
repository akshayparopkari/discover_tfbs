#!/usr/bin/env python3

"""
Build feature table from input FASTA files.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.1.1"


import argparse
from sys import exit
from os import mkdir
from os.path import isfile, join, abspath, exists
from random import choice
from collections import defaultdict
from time import localtime, strftime
from utils import parse_fasta, calculate_gc_percent, get_kmers


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
    parser.add_argument("-g", "--genome_fasta_files", type=str, nargs="+",
                        help="Specify path to one or more genome files to use as template"
                        " from which to generae random background sequences. These genome"
                        "file(s) must be FASTA file with very low probability of "
                        "containing sequences with binding motifs. For example, these "
                        "files can be FASTA file of exonic regions of non-related "
                        "species. Please do not supply gzipped files.")
    parser.add_argument("-o", "--output_dir", type=str,
                        help="Specify a directory to save background sequence data.")
    return parser.parse_args()


def main():

    args = handle_program_options()

    try:
        assert isfile(args.fg_fasta_file)
    except AssertionError as e:
        print("Error with input foreground FASTA file(s). Please check supplied FASTA "
              "file - {}".format(e))
        exit()
    try:
        outdir = abspath(args.output_dir)
        assert exists(outdir)
    except AssertionError:
        # output directory doesn't exist, create it
        mkdir(outdir)

    #########################################################
    # GC% AND LENGTH MATCHED BACKGROUND SEQUENCE GENERATION #
    #########################################################
    print("\n", strftime("%x %H:%M: ".format(localtime)),
          "Generating random length-matched sequences")
    print("="*60, sep="\n")
    fg_seqs = {header: seq for header, seq in parse_fasta(args.fg_fasta_file)}
    seq_length = len(list(fg_seqs.values())[0])
    fg_gc = {header: round(calculate_gc_percent(seq)) for header, seq in fg_seqs.items()}

    # parse genome CDS/exonic regions and collect GC% and length-matched sequences
    cds_exons_len_matched_gc = defaultdict(list)
    for f in args.genome_fasta_files:
        for header, seq in parse_fasta(f):
            for kmer in get_kmers(seq, k=seq_length):
                gc_content = round(calculate_gc_percent(kmer))
                dict_key = "gc_{:d}_pc".format(gc_content)
                cds_exons_len_matched_gc[dict_key].append(kmer)

    # write to output FASTA file
    outfnh = join(outdir,
                  "{}_cds_exon_len_matched_bkg_seqs.fasta".format(args.protein_name))
    with open(outfnh, "w") as outf:
        for header, gc_pc in fg_gc.items():
            random_header = "gc_len_matched_bkg_for_{}".format(header)
            dict_key = "gc_{:d}_pc".format(gc_pc)
            random_seq = choice(cds_exons_len_matched_gc[dict_key])
            outf.write(">{0}\n{1}\n".format(random_header, random_seq))

    ########################################################
    # DINUCLEOTIDE SHUFFLED BACKGROUND SEQUENCE GENERATION #
    ########################################################
    for header, seq in parse_fasta(args.fg_fasta_file):
        dinuc_shuff_header = "dinucleotide_shuffled_bkg_seq_for_{}".format(header)



if __name__ == "__main__":
    exit(main())
