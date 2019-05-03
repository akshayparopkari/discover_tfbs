#! /usr/bin/env python

"""
File containing utility functions.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.1.1"


# imports
from sys import exit
from random import choices
from itertools import product
from urllib.parse import parse_qs
from os.path import isfile, basename
from time import localtime, strftime
from collections import Counter as cnt
from os.path import join, realpath, isfile
err = set()
try:
    import pandas as pd
except ImportError:
    err.add("pandas")
try:
    import numpy as np
except ImportError:
    err.add("numpy")
try:
    from Bio import Seq
    from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
except ImportError:
    err.add("biopython")
if len(err) > 0:
    for e in err:
        print("Please install {} package".format(e))
    exit()


def random_dna(n=25, ambiguous=True):
    """
    Return a random DNA sequence with ambiguous bases of length 'n'

    :type n: int
    :param n: length of the sequence to be returned, 25bp sequence is the default

    :type ambiguous: bool
    :param ambiguous: set to False to get sequence without ambiguous bases [A, T, G, C]
    """
    if ambiguous:
        samples = "ACGTRYSWKMBDHVN"
        weights = [25, 25, 25, 25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10]
        seq = "".join(choices(samples, weights=weights, k=n))
        return seq
    else:
        samples = "ATGC"
        seq = "".join(choices(samples, k=n))  # all bases will be equally likely
        return seq


def reverse_complement(seq):
    """
    For an input sequence, this function returns its reverse complement.

    :type seq: str
    :param seq: valid DNA nucleotide sequence without any ambiguous bases
    """
    base_complements = Seq.IUPAC.IUPACData.ambiguous_dna_complement
    return "".join([base_complements[nt] for nt in seq[::-1]])


def dna_iupac_codes(seq):
    """
    Given a DNA sequence, return all possible degenerate sequences.

    :type sequence: str
    :param sequence: a valid single DNA nucleotide sequence
    """
    # initialize IUPAC codes in a dict
    iupac_codes = {"A": ["A"], "C": ["C"], "G": ["G"], "T": ["T"], "R": ["A", "G"],
                   "Y": ["C", "T"], "S": ["G", "C"], "W": ["A", "T"], "K": ["G", "T"],
                   "M": ["A", "C"], "B": ["C", "G", "T"], "D": ["A", "G", "T"],
                   "H": ["A", "C", "T"], "V": ["A", "C", "G"], "N": ["A", "C", "G", "T"]}
    return list(map("".join, product(*[iupac_codes[nt] for nt in seq])))


def parse_fasta(fasta_file):
    """
    Parse a FASTA file for easy access.

    :type fasta_file: str
    :param fasta_file: file path and name handle
    """
    try:
        assert isfile(fasta_file)
    except AssertionError:
        # file doesn't exist
        exit("\n{} does not exist. Please provide a valid FASTA file.\n".
             format(fasta_file))
    else:
        # valid FASTA file
        with open(fasta_file) as input_fasta:
            for header, sequence in sfp(input_fasta):
                yield header, sequence


def bkg_gc(bkg_fasta, outdir):
    """
    Parse input FASTA file, spread out sequences based on GC percent content
    into separate file. If seqA is 35% and seqB is 89%, they will saved in
    outdir/bkg_gc_35.txt and outdir/bkg_gc_89.txt, respectively.

    :type bkg_fasta: str/file name handle
    :param bkg_fasta: FASTA file containing background sequences

    :type outdir: str/file name handle
    :param outdir: Folder to save binned files in
    """
    for header, seq in parse_fasta(bkg_fasta):
        bseqs = dna_iupac_codes(seq)
        gc = {s: 100 * (s.count("G") + s.count("C")) /\
              (s.count("G") + s.count("C") + s.count("A") + s.count("T"))
              for s in bseqs}
        for sequence, gc in gc.items():
            outfnh = realpath(join(outdir, "bkg_gc_{}.txt".format(round(gc))))
            if isfile(outfnh):
                with open(outfnh, "a") as outfile:
                    outfile.write(">{0}|gc:{1:.2f}\n{2}\n".format(header, gc, sequence))
            else:
                with open(outfnh, "w") as outfile:
                    outfile.write(">{0}|gc:{1:.2f}\n{2}\n".format(header, gc, sequence))


def parse_blastn_results(f):
    """
    Get blastn results in a dict.
    """
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

    :type seq: str
    :param seq: a single nucleotide sequence or a list of nucleotide sequences

    :type k: int
    :param k: length of kmers to generate, default is 6mers under the constraint that
              length(seq) > k
    """
    try:
        # check is seq is single sequence or list of sequences
        assert isinstance(seq, str)
    except AssertionError:
        # `seq` is not a str, iterate through the list of nucleotide sequences
        # dict of seq and their kmers {sequence: [kmer1, kmer2, ...]}
        return  {s: [s[i: i + k] for i in range(0, len(s) - (k - 1), 1)] for s in seq}
    else:
        # `seq` is a single sequence
        # list of kmers [kmer1, kmer2, ...]
        return [seq[i: i + k] for i in range(0, len(seq) - (k - 1), 1)]


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
    # Testing GFF file
    try:
        assert isfile(gff_file)
    except AssertionError:
        # file doesn't exist
        exit("\n{} does not exist. Please provide a valid GFF file.\n".format(gff_file))

    # Go through GFF file and write out feature sequences to output FASTA file
    with open(gff_file) as infile:
        with open(out_fasta, "w") as outfile:
            for line in infile:
                try:
                    assert line.startswith("Ca{}chr".format(genome))
                    assert line.strip().split("\t")[2] == feature
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
                        try:
                            seq_name = "|".join([attributes["ID"][0],
                                                attributes["orf_classification"][0],
                                                attributes["parent_feature_type"][0],
                                                attributes["Parent"][0], line[2]])
                            seq_name = seq_name.replace(" ", "_")
                        except KeyError:
                            # sparse GFF attributes
                            try:
                                seq_name = "|".join([attributes["Parent"][0], line[1],
                                                     line[2]])
                                seq_name = seq_name.replace(" ", "_")
                            except KeyError:
                                # even more sparse GFF attributes
                                seq_name = "|".join([attributes["ID"][0], line[1],
                                                     line[2]])
                                seq_name = seq_name.replace(" ", "_")
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
    if verbose:
        gc_content = 100 * ((bkg_freq["G"] + bkg_freq["C"]) /\
            sum([bkg_freq["C"], bkg_freq["T"], bkg_freq["A"], bkg_freq["G"]]))
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
    try:
        assert isfile(fasta_file)
    except AssertionError:
        # file doesn't exist
        exit("\n{} does not exist. Please provide a valid FASTA file.\n".
             format(fasta_file))
    else:
        # parse FASTA file and collect nucleotide frequencies
        with open(fasta_file) as infile:
            kmer_counts = dict()
            for title, seq in sfp(infile):

                # get a list of all unambiguous kmers for FASTA sequence
                seqs = [dna_iupac_codes(seq) for seq in get_kmers(seq, n + 1)]

                # iterate through seqs and get counts of nt following kmer
                for kmer in seqs:
                    try:
                        assert len(kmer) == 1
                    except AssertionError:
                        # more than 1 unambiguous sequence in kmer list
                        for mer in kmer:
                            try:
                                kmer_counts[mer[:-1]].update(mer[-1])
                            except KeyError:
                                kmer_counts[mer[:-1]] = cnt(mer[-1])
                    else:
                        # single unambiguous sequence in kmer list
                        try:
                            kmer_counts[kmer[0][:-1]].update(kmer[0][-1])
                        except KeyError:
                            kmer_counts[kmer[0][:-1]] = cnt(kmer[0][-1])

            # calculate probabilities for all nt after kmer
            kmer_prob = {k1: {k2: v2 / sum(v1.values()) for k2, v2 in v1.items()}
                         for k1, v1 in kmer_counts.items()}
    return pd.DataFrame.from_dict(kmer_prob, orient="index").fillna(0.)


def pac(seqA, seqB):
    """
    Poisson based similarity measure, PAC.
    Adopted from Jacques van Helden BIOINFORMATICS (2002)

    :type seqA: str
    :param seqA: Nucleotide sequence, preferably unambiguous DNA sequence

    :type seqA: str
    :param seqA: Nucleotide sequence, preferably unambiguous DNA sequence
    """

    # length of kmer is half if seqA length
    k = round(len(seqA) / 3)

    # kmer word count for seqA
    seqA_kmers = get_kmers(seqA, k)
    seqA_wc = {kmer: seqA_kmers.count(kmer) for kmer in seqA_kmers}
    # prior probability of kmer 'w'
    fw_A = {word: freq / len(seqA_kmers) for word, freq in seqA_wc.items()}
    # expected number of occurrences
    mw_A = {word: pp * (len(seqA) - k + 1) for word, pp in fw_A.items()}

    # kmer word count for seqB
    seqB_kmers = get_kmers(seqB, k)
    seqB_wc = {kmer: seqB_kmers.count(kmer) for kmer in seqB_kmers}
    # prior probability of kmer 'w'
    fw_B = {word: freq / len(seqB_kmers) for word, freq in seqB_wc.items()}
    # expected number of occurrences
    mw_B = {word: pp * (len(seqB) - k + 1) for word, pp in fw_B.items()}

    similarity = 0
    prod = 1
    total_patterns = set(seqA_kmers + seqB_kmers)
    for word in total_patterns:
        Cw_AB = min([seqA_wc.get(word, 0), seqB_wc.get(word, 0)])
        if Cw_AB > 0:
            prob = (1 - poisson.cdf(Cw_AB - 1, mw_A[word])) *\
                   (1 - poisson.cdf(Cw_AB - 1, mw_B[word]))
        else:
            prob = 1
        prod *= prob
        similarity += (1 - prob)
    prod = 1 - (prod ** (1 / len(total_patterns)))
    similarity = similarity / len(total_patterns)
    return similarity, prod
