#! /usr/bin/env python

"""
:Abstract: File containing utility functions.
:Author: Akshay Paropkari
"""


def dna_iupac_codes(sequence):
    """
    Given a DNA sequence, return all possible degenerate sequences.

    :type sequence: str
    :param sequence: a valid single DNA nucleotide sequence
    """
    # import
    from itertools import product

    # initialize IUPAC codes in a dict
    iupac_codes = {"A": ["A"], "C": ["C"], "G": ["G"], "T": ["T"], "R": ["A", "G"],
                   "Y": ["C", "T"], "S": ["G", "C"], "W": ["A", "T"], "K": ["G", "T"],
                   "M": ["A", "C"], "B": ["C", "G", "T"], "D": ["A", "G", "T"],
                   "H": ["A", "C", "T"], "V": ["A", "C", "G"], "N": ["A", "C", "G", "T"]}
    return list(map("".join, product(*[iupac_codes[nt] for nt in sequence])))


def slice_fasta(fasta_file):
    """
    Parse a FASTA file for easy pythonic access.

    :type fasta_file: str
    :param fasta_file: file path and name handle
    """
    # imports
    from sys import exit
    from os.path import isfile
    from pyfaidx import Fasta

    try:
        assert isfile(fasta_file)
    except AssertionError:
        # file doesn't exist
        exit("\n{} does not exist. Please provide a valid FASTA file.\n".
             format(fasta_file))
    else:
        # valid FASTA file
        return Fasta(fasta_file, one_based_attributes=False)


def parse_blastn_results(f):
    """
    Get blastn results in a dict.
    """
    # imports
    from time import localtime, strftime
    from os.path import basename

    print("{}: PARSING BLASTn RESULTS".format(strftime("%m/%d/%Y %H:%M:%S", localtime())))
    blastn_results = dict()
    tf_name = basename(f).split("_")[0]
    print("PROCESSING {}".format(tf_name.upper()))

    with open(f) as inblast:
        for line in inblast:
            line = line.strip().split("\t")
            try:
                # confirm  pident is 100%
                assert float(line[2]) == 100.0
                # confirm alignment length is full motif length
                assert int(line[3]) == int(line[12])
            except AssertionError:
                # not valid blastn match
                continue
            else:
                # 100% identical match found over 100% motif sequence length
                # get match direction form blastn result
                if int(line[8]) < int(line[9]):
                    # positive direction
                    strand = "+"
                else:
                    # negative direction
                    strand = "-"

                if line[0] in blastn_results.keys():
                    # query exists in dict, add elements
                    blastn_results[line[0]]["sseqid"].append(line[1])
                    blastn_results[line[0]]["pident"].append(float(line[2]))
                    blastn_results[line[0]]["length"].append(int(line[3]))
                    blastn_results[line[0]]["qstart"].append(int(line[6]))
                    blastn_results[line[0]]["qend"].append(int(line[7]))
                    blastn_results[line[0]]["sstart"].append(int(line[8]))
                    blastn_results[line[0]]["send"].append(int(line[9]))
                    blastn_results[line[0]]["evalue"].append(float(line[10]))
                    blastn_results[line[0]]["bitscore"].append(float(line[11])),
                    blastn_results[line[0]]["strand"].append(strand)
                else:
                    # initialize query in dict
                    blastn_results[line[0]] = {
                        "sseqid": [line[1]],
                        "pident": [float(line[2])],
                        "length": [int(line[3])],
                        "qstart": [int(line[6])],
                        "qend": [int(line[7])],
                        "sstart": [int(line[8])],
                        "send": [int(line[9])],
                        "evalue": [float(line[10])],
                        "bitscore": [float(line[11])],
                        "strand": [strand]
                    }
    return blastn_results


def get_kmers(seq, k=6):
    """
    Generate kmers from a given sequence. By default, this function will generate 6-mers
    from the given sequence. Function will return an error if kmer length is greater than
    sequence length.

    :type seq: str/list of str
    :param seq: a single nucleotide sequence or a list of nucleotide sequences

    :type k: int
    :param k: length of kmers to generate, default is 6mers under the constraint that
              length(seq) > k
    """
    # import
    import numpy as np

    try:
        # check is seq is single sequence or list of sequences
        assert isinstance(seq, str)
    except AssertionError:
        # `seq` is not a str, iterate through the list of nucleotide sequences
        kmers = dict()
        for s in seq:
            kmers[s] = np.array([])
            for i in range(0, len(s) - (k - 1), 1):
                kmers[s] = np.append(kmers[s], s[i: i + k])
        return kmers  # dict of seq and their kmers {sequence: [kmer1, kmer2, ...]}
    else:
        kmers = np.array([])
        # `seq` is a single sequence
        for i in range(0, len(seq) - (k - 1), 1):
            kmers = np.append(kmers, seq[i: i + k])
        return kmers  # list of kmers [kmer1, kmer2, ...]


def parse_gff_fasta(gff_file, parsed_fasta, out_fasta="Ca22_CDS_seqs.fasta", genome="22",
                    feature="CDS"):
    """
    Parses a GFF and fasta data (output from parse_fasta()) of a genome and collect
    sequences of certain feature in GFF file. By default, this function will return all
    CDS sequences for all entries in GFF file.

    :type gff_file: str
    :param gff_file: file path and name handle

    :type parsed_fasta: dict-like
    :param parsed_fasta: Output from parse_fasta()

    :type out_fasta: str
    :param out_fasta: Output FASTA file path and name. By default, the file name will be
                     Ca22_CDS_seqs.fasta. I recommend changing file name based on genome
                     and feature used for your run.

    :type genome: str
    :param genome: Assembly version for Candida albicans genome. Default value is 22
                       for Assembly 22, but can be changed to 21 for Assembly 21.

    :type feature: str
    :param feature: Single feature annotation from input GFF file. Default is 'CDS', but
                    can be changed by user to other feature types such as 'exon',
                    'intergenic_region', 'start_codon', 'stop_codon', etc.
    """
    # imports
    from sys import exit
    from os.path import isfile
    from urllib.parse import parse_qs

    # Testing GFF file
    try:
        assert isfile(gff_file)
    except AssertionError:
        # file doesn't exist
        exit("\n{} does not exist. Please provide a valid GFF file.\n".
             format(gff_file))

    # Go through GFF file and write out feature sequences to output FASTA file
    with open(gff_file) as infile:
        with open(out_fasta, "w") as outfile:
            for line in infile:
                try:
                    assert line.startswith("Ca22chr")
                except AssertionError:
                    # invalid line, skip processing
                    continue
                else:
                    # valid line, parse contents of line into an object
                    line = line.strip().split("\t")
                    start, end = int(line[3]), int(line[4])
                    try:
                        assert (end - start) > 15
                    except AssertionError:
                        # feature length is not positive value, skip
                        continue
                    else:
                        attributes = parse_qs(line[8])
                        seq_name = attributes["ID"][0] + "|" +\
                            attributes["Note"][0].replace(" ", "_")
                        outfile.write(">{0}\n{1}\n".format(seq_name,
                                      parsed_fasta[line[0]][start: end].seq))
    return None


def get_start_prob(fasta_file, verbose=False):
    """
    From a list of sequences, get the background probabilities of adenine(A),
    cytosine (C), guanine(G) and thymine (T)

    :type fasta_file: str
    :param fasta_file: FASTA file path and name handle

    :type verbose: bool
    :param verbose: Set `True` to get GC content of input sequence. By default, the
                    information will not be displayed to user.
    """
    # imports
    from sys import exit
    from os.path import isfile
    from collections import Counter as cnt
    from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp

    try:
        assert isfile(fasta_file)
    except AssertionError:
        # file doesn't exist
        exit("\n{} does not exist. Please provide a valid FASTA file.\n".
             format(fasta_file))
    else:
        # parse FASTA file and collect nucleotide frequencies
        bkg_freq = cnt()
        with open(fasta_file) as infile:
            for name, seq in sfp(infile):
                bkg_freq.update(seq)

    # helpful message about input sequences - optional
    try:
        assert verbose
    except AssertionError:
        # no calculations requested
        pass
    else:
        # print information
        gc_content = 100 * ((bkg_freq["G"] + bkg_freq["G"]) / sum(bkg_freq.values()))
        print("GC content of sequences in {}: {:0.2f}%".format(fasta_file, gc_content))

    # calculate background probabilities
    start_prob = {nt: freq / sum(bkg_freq.values()) for nt, freq in bkg_freq.items()}

    return start_prob


def get_transmat(fasta_file, n=5):
    """
    From a FASTA file, generate nth order Markov chain transition probability matrix. By
    default, a 5th order Markov chain transition matrix will be calculated and returned as
    Pandas dataframe.

    :type fasta_file: str
    :param fasta_file: FASTA file path and name handle
    """
    # imports
    from sys import exit
    from os.path import isfile
    from collections import Counter as cnt
    import pandas as pd
    from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp

    try:
        assert isfile(fasta_file)
    except AssertionError:
        # file doesn't exist
        exit("\n{} does not exist. Please provide a valid FASTA file.\n".
             format(fasta_file))
    else:
        # parse FASTA file and collect nucleotide frequencies
        with open(fasta_file) as infile:
            pentamer_counts = dict()
            for title, seq in sfp(infile):
                for hexamer in get_kmers(seq, k=n + 1):
                    hexamer = dna_iupac_codes(hexamer)
                    for mer in hexamer:
                        try:
                            pentamer_counts[mer[:-1]].update(mer[-1])
                        except KeyError:
                            pentamer_counts[mer[:-1]] = cnt(mer[-1])
                intergenic_hexamer_prob = dict()
                for k1, v1 in pentamer_counts.items():
                    intergenic_hexamer_prob[k1] = dict()
                    for k2, v2 in v1.items():
                        intergenic_hexamer_prob[k1][k2] = v2 / sum(v1.values())
    return pd.DataFrame.from_dict(intergenic_hexamer_prob, orient="index").fillna(0.)
