#! /usr/bin/env python3

"""
File containing utility functions.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.2.5"


# imports
from sys import exit
import subprocess as sp
from time import strftime
from random import choices
from functools import lru_cache
from urllib.parse import parse_qs
from urllib.request import urlopen
from itertools import product, starmap
from collections import defaultdict, Counter as cnt
from os.path import join, abspath, realpath, isfile, basename
err = set()
try:
    from pybedtools import BedTool
except ImportError:
    err.add("pybedtools")
try:
    from sklearn.model_selection import permutation_test_score
except ImportError:
    err.add("scikit-learn")
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
except ImportError:
    err.add("matplotlib")
try:
    import shlex as sh
except ImportError:
    err.add("shlex")
try:
    import numpy as np
    from numpy.random import choice as npchoice
except ImportError:
    err.add("numpy")
try:
    import pandas as pd
except ImportError:
    err.add("pandas")
try:
    from scipy.stats import poisson
except ImportError:
    err.add("scipy")
try:
    from Bio import Seq
    from Bio.SeqIO.FastaIO import SimpleFastaParser as sfp
except ImportError:
    err.add("biopython")
if len(err) > 0:
    for e in err:
        print("Please install {} package".format(e))
    exit()

try:
    profile
except NameError:
    profile = lambda x: x


@profile
def random_dna(n=25, ambiguous=True) -> str:
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


@profile
def reverse_complement(seq: str) -> str:
    """
    For an input sequence, this function returns its reverse complement.

    :type seq: str
    :param seq: valid DNA nucleotide sequence without any ambiguous bases
    """
    base_complements = Seq.IUPAC.IUPACData.ambiguous_dna_complement
    return "".join([base_complements[nt] for nt in seq[::-1]])


@profile
def dna_iupac_codes(seq: str) -> list:
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
    return list(map("".join, product(*[iupac_codes[nt.upper()] for nt in seq])))


@profile
def get_ca_chrom_seq(url="http://www.candidagenome.org/download/sequence/C_albicans_SC5314/Assembly22/current/C_albicans_SC5314_A22_current_chromosomes.fasta.gz",
                     outfile="./C_albicans_SC5314_A22_current_chromosomes.fasta.gz"):
    """
    Download Candida albicans chromosome sequence FASTA file from Candida genome
    database (CGD)

    :param url: URL of Gzipped file from CGD

    :param outfile: File name and location to save chromosomal sequences from CGD
    """
    with urlopen(url) as response, open(outfile, "wb") as outf:
        for chunk in response.stream():
            outf.write(chunk)


def parse_fasta(fasta_file: str):
    """
    Efficiently parse a FASTA file.

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
        try:
            with open(fasta_file) as input_fasta:
                for header, sequence in sfp(input_fasta):
                    yield header, sequence
        except StopIteration:
            # end of FASTA file, exit
            return


@profile
def bkg_gc(bkg_fasta: str, outdir: str):
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
        gc = {s: 100 * (s.count("G") + s.count("C")) /
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


@profile
def calculate_gc_percent(sequence: str) -> float:
    """
    Compute the GC percent of unambiguous input nucleotide sequence.

    :type sequence: str
    :param sequence: sequence consisting of A, T, G, C letters
    """

    # get length of input sequence
    seq_len = len(sequence)

    # get number of G and C in input sequence
    sequence = sequence.upper()  # convert to upper case
    gc = 100 * (sequence.count("G") + sequence.count("C")) / seq_len
    return gc


@profile
def download_ca_a22_genome_annot(outfile="C_albicans_SC5314_A22_current_features.gff"):
    """
    Download the latest genome GFF file for Candida albicans SC5314 from Candida Genome
    Database. Data is located at
    http://www.candidagenome.org/download/gff/C_albicans_SC5314/Assembly22/C_albicans_SC5314_A22_current_features.gff
    As per CGD - This file contains the current CGD annotation of all features in GFF
    based on Assembly 22 of the C. albicans SC5314 genome sequence.

    :type outfile: str
    :param outfile: File name used to save the CGD genome annotations. Be default, the
                 file is saved in current directory as
                 C_albicans_SC5314_A22_current_features.gff
    """
    try:
        from urllib3 import PoolManager
    except AssertionError:
        exit("Please install urllib3 package")
    # set download URL
    gff_url = "http://www.candidagenome.org/download/gff/C_albicans_SC5314/Assembly22/C_albicans_SC5314_A22_current_features.gff"

    print("Getting GFF file content")
    gff_data = PoolManager().request("GET", gff_url, preload_content=False)

    # settle output file name handle
    outfnh = abspath(join("./", outfile))

    # write GFF content to file
    print("Saving GFF file")
    with open(outfnh, "wb") as outf:
        for chunk in gff_data.stream():
            outf.write(chunk)
    print("GFF file saved at {0}".format(outfnh))


@profile
def split_file(fnh: str, nlines=50000):
    """
    Split FASTA file into multiple smaller FASTA files. Each of the smaller FASTA file
    will have at least 50000 lines.

    :param fnh: File name handle of the FASTA file to be split into smaller files.

    :type nlines: int
    :param nlines: Split fnh into smaller files with nlines number of lines.
    """
    prefix = fnh.split(".")[0] + "_"
    additional_suffix = ".fasta"
    split_func = "split -d -a 5 --additional-suffix {0} -l {1} {2} {3}".\
        format(additional_suffix, nlines, fnh, prefix)
    kwargs = sh.split(split_func)
    print("Running {0}".format(kwargs))
    sp.run(kwargs)


@profile
def parse_blastn_results(f: str) -> dict:
    """
    Get blastn results in a dict.

    :param f: Blastn output file
    """
    print(strftime("%x %X | PARSING BLASTn RESULTS"))
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


@profile
def get_kmers(seq: str, k=6):
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
        return ({s: [s[i: i + k] for i in range(0, len(s) - (k - 1), 1)] for s in seq})
    else:
        # `seq` is a single sequence
        # list of kmers [kmer1, kmer2, ...]
        return ([seq[i: i + k] for i in range(0, len(seq) - (k - 1), 1)])


@profile
def get_kmer_counts(seq: str, k=6) -> dict:
    """
    For an input sequence, return counts for all kmers in dict keyed on kmer,
    and its counts as the value

    :type seq: str
    :param seq: a single nucleotide sequence or a list of nucleotide sequences

    :type k: int
    :param k: length of kmers to generate, default is 6mers under the constraint that
               length(seq) > k
    """
    counts = defaultdict(lambda: 0)
    n_kmers = len(seq) - k + 1
    try:
        for i in range(n_kmers):
            counts[seq[i: i + k]] += 1
    except Exception as e:
        print("Exception occurred: {}".format(e))
        return
    else:
        return counts


@profile
def blast_to_bed(infile: str, outfile: str):
    """
    Convert a tabular BLAST format to BED6 format.

    :type infile: str, file name handle
    :param infile: input tabular BLAST file

    :type outfile: str, file name handle
    :param outfile: tab-separated BED6 format
                    chrom    chromStart    chromEnd    name    score    strand
    """
    with open(infile) as inblast:
        with open(outfile, "w") as outbed:
            for line in inblast:
                line_list = line.strip().split("\t")
                chrom = line_list[1].split("_")[0]
                motif_name = line_list[0]
                if int(line_list[8]) < int(line_list[9]):
                    # alignment is on positive strand
                    chrom_start = int(line_list[8]) - 1
                    chrom_end = int(line_list[9])
                    strand = "+"
                else:
                    # alignment is on negative strand
                    chrom_start = int(line_list[9]) - 1
                    chrom_end = int(line_list[8])
                    strand = "-"
                outbed.write("{0}\t{1}\t{2}\t{3}\t.\t{4}\n".format(chrom, chrom_start,
                                                                   chrom_end, motif_name,
                                                                   strand))


@profile
def sort_bed_file(inbed: str, outbed: str):
    """
    Sort input BED6 file first by chromosome and then by starting position.

    :type inbed: str
    :param inbed: Input BED6 file to be sorted

    :type outbed: str
    :param outbed: Sorted output BED6 file
    """
    import sys
    import subprocess as sp
    try:
        import shlex
    except ImportError as ie:
        sys.exit("Please install {} module before executing this script.".format(ie))

    sort_function = "sort -k1,1 -k2,2n {0}".format(inbed)
    kwargs = shlex.split(sort_function)
    print("Running {0}".format(kwargs))
    outfile = open(outbed, "w")
    sp.run(kwargs, stdout=outfile, check=True)
    outfile.close()


@profile
def bed_to_fasta(inbed: str, genome_file: str, outfasta: str):
    """
    Convert a BED6 file to FASTA file using bedtools' getfasta function'.

    From bedtools website -
    bedtools getfasta will extract the sequence defined by the coordinates in a BED
    interval and create a new FASTA entry in the output file for each extracted sequence.
    By default, the FASTA header for each extracted sequence will be formatted as follows:
    “<chrom>:<start>-<end>”.
    The headers in the input FASTA file must exactly match the chromosome column in the
    BED file.

    :type inbed: str
    :param inbed: Input BED6 file to convert into FASTA file

    :type genome_file: str
    :param genome_file: Genome sequences for all chromosomes

    :type outfasta: str
    :param outfasta: Output file name of FASTA file
    """
    import sys
    import subprocess as sp
    try:
        import shlex
    except ImportError as ie:
        sys.exit("Please install {} module before executing this script.".format(ie))

    get_fasta = "bedtools getfasta -fi {0} -bed {1} -s -fo {2}".format(genome_file,
                                                                       inbed, outfasta)
    kwargs = shlex.split(get_fasta)
    print("Running {0}".format(get_fasta))
    sp.run(kwargs, check=True)


@profile
def get_shape_data(bedfile: str, shapefiles: list) -> dict:
    """
    Using binding data BED file and shape files, retrieve DNA shape of each bound or
    potential binding site.

    :param bedfile: Input BED file containing true or potential binding site
                 Ca22chr1 29234233 29346234 protein104 3.0295 -

    :param shapefile: Input shape files in FASTA format, which are output from DNAshapeR
                   getShape() function
                   >Ca22chr1A
                   -2.44,-3.63,-1.59,4.15,-2.36,4.19,-1.99, ...
    """
    # read all shape files into a dict
    print(strftime("%x %X | Parsing DNA shape files"))
    shape_data = defaultdict(lambda: defaultdict())
    for file in shapefiles:
        whichshape = file.split(".")[-1]
        with open(file, "r") as infasta:
            shape_data[whichshape] = {header: np.array(seq.split(","))
                                      for header, seq in sfp(infasta)}

    # parse BED file and get DNA shape of features listed in BED file
    print(strftime("%x %X | Retrieving DNA shape data"))
    with open(bedfile, "r") as inbed:
        genome_shape = dict()
        for line in inbed:
            chrom, start, end, name, score, strand = tuple(line.strip().split("\t"))
            name = chrom + ":" + start + "-" + end + "(" + strand + ")"
            if not genome_shape.get(name):
                # key doesn't exist, create key as new entry
                genome_shape[name] = dict()
            for shape, data in shape_data.items():
                if strand == "-":
                    # negative strand, get shape data in reverse order
                    seq = data[chrom][int(end): int(start) - 1: -1]
                else:
                    # positive strand
                    seq = data[chrom][int(start): int(end) + 1]
                for i in range(1, len(seq)):
                    position = "{0}_pos_{1}".format(shape, i)
                    if seq[i] == "NA":
                        # no shape data calculated, use 0.0
                        genome_shape[name][position] = 0.0
                    else:
                        # retrieve shape value
                        genome_shape[name][position] = float(seq[i])
    return genome_shape


@profile
def parse_gff_fasta(gff_file: str, parsed_fasta, out_fasta="Ca22_CDS_seqs.fasta",
                    genome="22", feature="CDS"):
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


@profile
def get_start_prob(fasta_file, verbose=False) -> dict:
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
        gc_content = 100 * ((bkg_freq["G"] + bkg_freq["C"]) /
                            sum([bkg_freq["C"], bkg_freq["T"], bkg_freq["A"],
                                 bkg_freq["G"]]))
        print("GC content of sequences in {}: {:0.2f}%".format(fasta_file, gc_content))

    # calculate background probabilities
    start_prob = {nt: freq / sum(bkg_freq.values()) for nt, freq in bkg_freq.items()}

    return start_prob


@profile
def get_transmat(fasta_file: str, n=5):
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


@profile
def markov_next_state(current: str, transmat) -> str:
    """
    Using current state of sequence, give the next state based on transition probability
    matrix

    :type current: str
    :param current: Current state of the sequence. For nth degree Markov chain, this
                 parameter must be n-character long DNA nucleotide

    :type transmat: pandas dataframe
    :param transmat: Transition probability matrix dataframe. Ideally, this dataframe is
                  the output from get_transmat() function
    """
    states = transmat.columns
    p = transmat.loc[current]
    return npchoice(states, p=p)


@profile
def markov_seq(seq_len: int, bkg_seq: str, transmat) -> str:
    """
    Using transition probability matrix and a starting sequence, generate seq_len length
    sequence from a 2nd order Markov model. The final output sequence will be padded by
    randomly generated dinucleotides.

    :type seq_len: int
    :param seq_len: Length of the final sequence

    :type bkg_seq: str
    :param bkg_seq: Starting DNA nucleotide sequence. A minimum of two nucleotides must be
                    supplied.

    :type transmat: pandas dataframe
    :param transmat: Transition probability matrix dataframe. Ideally, this dataframe is
                     the output from get_transmat() function
    """
    for _ in range(seq_len):
        current = bkg_seq[-2:]
        bkg_seq += markov_next_state(current, transmat)
    return bkg_seq + random_dna(2, False)


@profile
def compute_overlap(bedA: str, bedB: str):
    """
    With two input BED6 format files and return the number of overlapping features
    between them.

    :param bedA: True TF binding sites  in BED6 file format

    :param bedB: Genomic regions classified as TF binding site by model results in BED6
                 file format
    """
    a = BedTool(bedA)
    b = BedTool(bedB)
    overlap_count = len(a.intersect(b, u=True))
    overlap_pct = 100 * (overlap_count / len(a))
    print("{0} out of {1} ({2:0.2f}%) overlaps found".format(overlap_count,
                                                             len(a),
                                                             overlap_pct))
    return None


@profile
def gc_len_matched_bkg_seq_gen(fg_seqs: dict, transmat: dict, tol=5) -> dict:
    """
    Using 2nd order Markov model trained on Drosophila melanogaster and Escherichia coli
    coding sequence regions, generate GC content and length matched background sequence.
    GC content is allowed a controllable difference of 5% around the GC content of
    foreground sequence.

    :type fg_seqs: dict
    :param fg_seqs: Parsed FASTA file of foreground sequences


    :type transmat: dict
    :param transmat: A dict containing all transition probability matrix dataframe.
                     Ideally, the dataframes will be output from get_transmat()
    """
    bkg_seqs = dict()
    for header, seq in fg_seqs.items():
        for entry in dna_iupac_codes(seq):
            gc = round(calculate_gc_percent(seq))
            seq_len = len(seq)
            random_key = choices(list(transmat.keys()))[0]
            while True:
                bkg_seqs[header] = markov_seq(seq_len,
                                              random_dna(2, False),
                                              transmat[random_key])
                core_seq = bkg_seqs[header][2:-2]
                if core_seq not in fg_seqs.values():
                    gc_diff = abs(gc - round(calculate_gc_percent(core_seq)))
                    if gc_diff <= tol:
                        break
    return bkg_seqs


@lru_cache(512)
def _cdf(k: int, mu: float):
    """
    From Python3 docs - Decorator to wrap a function with a memoizing callable that saves
    up to the maxsize most recent calls. This decreases computation time for Poisson CDF
    function, speeding up pac()
    """
    return poisson.cdf(k, mu)


@profile
def pac(seqA: str, seqB: str):
    """
    Poisson based similarity measure, PAC. Adopted from Jacques van Helden
    Bioinformatics (2002)

    :type seqA: str
    :param seqA: Nucleotide sequence, preferably unambiguous DNA sequence

    :type seqA: str
    :param seqA: Nucleotide sequence, preferably unambiguous DNA sequence
    """
    # length of kmer is half if seqA length
    k = round(len(seqA) / 3)

    # kmer word count for seqA
    seqA_kmers = get_kmers(seqA, k)
    n_seqA_kmers = len(seqA_kmers)
    seqA_wc = {kmer: seqA_kmers.count(kmer) for kmer in seqA_kmers}
    # prior probability of kmer 'w'
    fw_A = {word: freq / n_seqA_kmers for word, freq in seqA_wc.items()}
    # expected number of occurrences
    mw_A = {word: pp * n_seqA_kmers for word, pp in fw_A.items()}

    seqB_kmers = get_kmers(seqB, k)
    n_seqB_kmers = len(seqB_kmers)
    seqB_wc = {kmer: seqB_kmers.count(kmer) for kmer in seqB_kmers}
    # prior probability of kmer 'w'
    fw_B = {word: freq / n_seqB_kmers for word, freq in seqB_wc.items()}
    # expected number of occurrences
    mw_B = {word: pp * n_seqB_kmers for word, pp in fw_B.items()}

    total_patterns = set(seqA_kmers + seqB_kmers)
    n_patterns = len(total_patterns)
    sga = seqA_wc.get
    sgb = seqB_wc.get
    for word in total_patterns:
        wc_seqA = sga(word, 0)
        wc_seqB = sgb(word, 0)
        Cw_AB = wc_seqB if wc_seqA > wc_seqB else wc_seqA
        if Cw_AB > 0:
            prob = (1 - _cdf(Cw_AB - 1, mw_A[word])) *\
                   (1 - _cdf(Cw_AB - 1, mw_B[word]))
        else:
            prob = 1
        try:
            prod
        except NameError:
            # prod doesn't exist, initialize it
            prod = prob
        else:
            # prod exists, update its value
            prod *= prob
        try:
            similarity
        except NameError:
            # prod doesn't exist, initialize it
            similarity = 1 - prob
        else:
            # prod exists, update its value
            similarity += 1 - prob
    prod = 1 - (prod ** (1 / n_patterns))
    similarity = similarity / n_patterns
    return similarity, prod


@profile
def permutation_result(estimator, X, y, cv, file: str, n_permute=5000, random_state=0):
    """
    Run permutation tests for classifier and assess significance of accuracy score. This
    is a wrapper around sklearn.model_selection.permutation_test_score

    :type estimator: scikit-learn classifier object
    :param estimator: Instance of scikit-learn initialized classifier which has a 'fit'
                      method

    :type X: array-like, list or numpy array
    :param X: Numpy array of features - columns of feature table

    :type y: array-like, list or numpy array
    :param y: Class labels of each row in X

    :type cv: int, iterable
    :param cv: If integer, those many cross validations are run. User can also supply an
               iterable to create (train, test) splits using indices.

    :type random_state: numpy random object
    :param random_state: Seed to use for multiple reproducible runs

    :type file: str
    :param file: Path and file name to save the bar plot
    """
    print(strftime("%x %X | Starting permutation testing"))
    score, permutation_score, p_value = permutation_test_score(
        estimator,
        X,
        y,
        scoring="balanced_accuracy",
        cv=cv,
        n_permutations=n_permute,
        n_jobs=-1,
        random_state=random_state,
    )
    print(
        strftime(
            "%x %X | Linear SVM classification score {0:0.3f} (pvalue : {1:0.5f})".format(
                score, p_value
            )
        )
    )
    with mpl.style.context("ggplot"):
        plt.figure(figsize=(10, 8))
        plt.hist(
            permutation_score,
            bins=25,
            alpha=0.5,
            hatch="//",
            edgecolor="w",
            label="Permutation scores for shuffled labels",
        )
        ylim = plt.ylim()[1]
        plt.vlines(
            2 * [1.0 / np.unique(y).size], 0, ylim, linewidth=3, label="50/50 chance"
        )
        plt.vlines(2 * [score], 0, ylim, linewidth=3, colors="g")
        plt.title("Model Accuracy Score = {:0.2f}*".format(100 * score), color="g")
        plt.xlim(0.0, 1.0)
        plt.legend(loc=2)
        plt.xlabel("Accuracy score as proportion")
        plt.ylabel("Frequency of calculated accuracy score")
        plt.tight_layout()
        plt.savefig(file, dpi=300, format="pdf", bbox_inches="tight")


@profile
def build_feature_table(infasta: str, truefasta: str, fg_bed_file: str,
                        genome_wide_shape_fasta_file: list):
    """
    Build feature table by calculating/retrieving all relevant characteristics.

    :infasta: Input FASTA file with foreground/true sequences

    :truefasta: FASTA file with true/foreground sequences to be used for calculating
                Poisson based similarity metrics

    :fg_bed_file: Path to foreground/true binding event BED file. This file must have a
                  minimum of BED6 format - i.e. chrom start end name score strand columns

    :genome_wide_shape_fasta_file: A list with paths to genome-wide 3D DNA shape
                                   (DNAShapeR output files) data single-line FASTA format
                                   files associated with '--predict' parameters
    """
    # read in FASTA data
    print(strftime("%x %X | Reading input FASTA file"))
    fg_data = {header: seq for header, seq in parse_fasta(infasta)}

    # calculate GC percent for all foreground seqs
    print(strftime("%x %X | Calculating GC percent"))
    fg_gc = dict(zip(list(fg_data.keys()),
                 map(calculate_gc_percent, list(fg_data.values()))))
    fg_gc_df = pd.DataFrame.from_dict(fg_gc, orient="index", columns=["GC_percent"])

    # calculate poisson based metric
    print(strftime("%x %X | Calculating Poisson based metrics"))
    true_seqs = {header: seq for header, seq in parse_fasta(truefasta)}
    seq_pairs = (list(product([seq], list(true_seqs.values())))
                 for seq in fg_data.values())
    fg_poisson_metrics = np.asarray([np.asarray(list(starmap(pac, pair_set))).
                                     mean(axis=0, dtype=np.float64)
                                     for pair_set in seq_pairs])
    fg_pac = dict(zip(fg_data.keys(), fg_poisson_metrics))
    fg_pac_df = pd.DataFrame.from_dict(fg_pac, orient="index", columns=["PAS", "PPS"])

    # calculate/retrieve shape values
    print(strftime("%x %X | Processing DNA shape data"))
    fg_shapes = get_shape_data(fg_bed_file, genome_wide_shape_fasta_file)
    fg_shapes_df = pd.DataFrame.from_dict(fg_shapes, orient="index")

    # create pandas dataframe and return it
    training_data_df = fg_gc_df.merge(fg_pac_df, how="outer", left_index=True,
                                      right_index=True)
    training_data_df = training_data_df.merge(fg_shapes_df, how="outer", left_index=True,
                                              right_index=True)
    return training_data_df
