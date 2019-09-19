#!/usr/bin/env python3

"""
Build feature table from input FASTA files.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.1.2"


import argparse
from sys import exit
from os import mkdir
from os.path import isfile, join, abspath, exists
from random import choices
from itertools import product
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
    print(strftime("\n%x %H:%M:".format(localtime)),
          "Generating random length-matched background sequences from CDS/exonic regions")
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
            random_seq = choices(cds_exons_len_matched_gc[dict_key])[0]
            outf.write(">{0}\n{1}\n".format(random_header, random_seq))

    ########################################################
    # DINUCLEOTIDE SHUFFLED BACKGROUND SEQUENCE GENERATION #
    ########################################################
    print(strftime("\n%x %H:%M:".format(localtime)),
          "Generating length-matched background sequences using 2nd order Markov model")
    k = 2
    all_dinuc_combination = {"".join(list(entry)): 0
                             for entry in product("ATGC", repeat=2)}
    trans_mat = {key: {nuc: 0 for nuc in "ATGC"} for key in all_dinuc_combination.keys()}
    dinuc_freq = defaultdict(int)
    num_seq = 0
    dinuc_shuff_header = set()
    for header, seq in parse_fasta(args.fg_fasta_file):
        dinuc_shuff_header.add("dinucleotide_shuffled_bkg_seq_for_{}".format(header))
        num_seq += 1
        for twomer in get_kmers(seq, k=k):
            dinuc_freq[twomer] += 1
        for i in range(0, seq_length - k):
            try:
                first_two = seq[i: i + k]
                third = seq[i + k + 1]
            except IndexError:
                first_two = seq[i: i + k]
                third = seq[-1]
            trans_mat[first_two][third] += 1
    dinuc_prob = {k: v / sum(dinuc_freq.values()) for k, v in dinuc_freq.items()}
    trans_mat_prob = defaultdict(lambda: defaultdict(int))
    for dinuc, data in trans_mat.items():
        for third, freq in data.items():
            try:
                numerator = trans_mat[dinuc][third] + (dinuc_prob[dinuc] * 8)
                denominator = 8 * sum(dinuc_freq.values())
                trans_mat_prob[dinuc][third] = numerator / denominator
            except Exception as err:
                exit(err)

    markov_model_seqs = set()
    for _ in range(num_seq):
        sample = choices(list(dinuc_prob.keys()), weights=list(dinuc_prob.values()))[0]
        for _ in range(seq_length):
            markov_state = sample[-2:]
            sample += choices(population=list(trans_mat_prob[markov_state].keys()),
                              weights=list(trans_mat_prob[markov_state].values()))[0]
        markov_model_seqs.add(sample)

    outfnh = join(outdir,
                  "{}_markovmodel_len_matched_bkg_seqs.fasta".format(args.protein_name))
    with open(outfnh, "w") as outf:
        for header, seq in zip(dinuc_shuff_header, markov_model_seqs):
            outf.write(">{0}\n{1}\n".format(header, seq))


if __name__ == "__main__":
    exit(main())
