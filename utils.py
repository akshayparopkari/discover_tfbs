#! /usr/bin/env python3

"""
File containing utility functions.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.3.7"


import shlex as sh
import subprocess as sp
from collections import Counter as cnt
from collections import defaultdict
from functools import lru_cache
from itertools import product, starmap
from math import ceil, log
from os import remove
from os.path import abspath, basename, isfile, join, realpath
from random import choice, choices
from sys import exit
from time import localtime, strftime
from urllib.parse import parse_qs
from urllib.request import urlopen

err = set()
try:
    from pybedtools import BedTool
except ImportError:
    err.add("pybedtools")
try:
    import mmh3
except ImportError:
    err.add("mmh3")
try:
    from sklearn.model_selection import permutation_test_score
except ImportError:
    err.add("scikit-learn")
try:
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    plt.switch_backend("agg")
except ImportError:
    err.add("matplotlib")
try:
    import numpy as np
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
    base_complements = Seq.ambiguous_dna_complement
    return "".join([base_complements[nt] for nt in seq[::-1]])


@profile
def dna_iupac_codes(seq: str) -> list:
    """
    Given a DNA sequence, return all possible degenerate sequences.

    :type sequence: str
    :param sequence: a valid single DNA nucleotide sequence
    """
    # initialize IUPAC codes in a dict
    iupac_codes = {
        "A": ["A"],
        "C": ["C"],
        "G": ["G"],
        "T": ["T"],
        "R": ["A", "G"],
        "Y": ["C", "T"],
        "S": ["G", "C"],
        "W": ["A", "T"],
        "K": ["G", "T"],
        "M": ["A", "C"],
        "B": ["C", "G", "T"],
        "D": ["A", "G", "T"],
        "H": ["A", "C", "T"],
        "V": ["A", "C", "G"],
        "N": ["A", "C", "G", "T"],
    }
    return list(map("".join, product(*[iupac_codes[nt.upper()] for nt in seq])))


@profile
def FASTA_header_to_bed6(header: str) -> str:
    """
    Given a FASTA header with format "chrom:start-end(strand)" like "Ca22chr1A:293970-293981(-)",
    return the genome location in BED6 format.

    :param header: FASTA header like Ca22chr1A:293970-293981(-)
    """
    chrom, coordinates = tuple(header.split(":"))
    start = coordinates.split("-")[0]
    end = coordinates.split("(")[0].split("-")[1]
    strand = coordinates.split("(")[1][0]
    return "\t".join([chrom, start, end, ".", ".", strand])


@profile
def get_ca_chrom_seq(
    url="http://www.candidagenome.org/download/sequence/C_albicans_SC5314/Assembly22/current/C_albicans_SC5314_A22_current_chromosomes.fasta.gz",
    outfile="./C_albicans_SC5314_A22_current_chromosomes.fasta.gz",
):
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
        exit(
            "\n{} does not exist. Please provide a valid FASTA file.\n".format(
                fasta_file
            )
        )
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
    :param bkg_fasta: FASTA file containing false (background) sequences

    :type outdir: str/file name handle
    :param outdir: Folder to save binned files in
    """
    for header, seq in parse_fasta(bkg_fasta):
        bseqs = dna_iupac_codes(seq)
        gc = {s: 100 * calculate_gc_content(s) for s in dna_iupac_codes(seq)}
        for sequence, gc in gc.items():
            outfnh = realpath(join(outdir, "bkg_gc_{}.txt".format(round(gc))))
            if isfile(outfnh):
                with open(outfnh, "a") as outfile:
                    outfile.write(">{0}|gc:{1:.2f}\n{2}\n".format(header, gc, sequence))
            else:
                with open(outfnh, "w") as outfile:
                    outfile.write(">{0}|gc:{1:.2f}\n{2}\n".format(header, gc, sequence))


@profile
def calculate_gc_content(sequence: str) -> float:
    """
    Compute the GC proportion of unambiguous input nucleotide sequence {A, T, G, C}.

    :type sequence: str
    :param sequence: sequence consisting of A, T, G, C letters
    """

    # get length of input sequence
    seq_len = len(sequence)

    # get number of G and C in input sequence
    sequence = sequence.upper()  # convert to upper case
    gc = (sequence.count("G") + sequence.count("C")) / seq_len
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
        from re import compile, escape

        from urllib3 import PoolManager
    except AssertionError:
        exit("Please install urllib3 package")
    else:
        # set replacement strings and set up pattern
        uri_decode = {
            "%20": "_",
            "%22": '"',
            "%23": "#",
            "%25": "%",
            "%26": "&",
            "%27": "'",
            "%28": "(",
            "%29": ")",
            "%2B": "+",
            "%2C": ",",
            "%2F": "/",
            "%3A": ":",
            "%3B": ";",
            "%3E": ">",
            "%5B": "[",
            "%5D": "]",
            "_C_albicans_SC5314": "",
        }
        uri_decode_esc = {escape(k): v for k, v in uri_decode.items()}
        pattern = compile("|".join(uri_decode_esc.keys()))
        # set download URL
        gff_url = "http://www.candidagenome.org/download/gff/C_albicans_SC5314/Assembly22/C_albicans_SC5314_A22_current_features.gff"

        print("Getting GFF file content")
        gff_data = PoolManager().request("GET", gff_url, preload_content=False)

        # settle output file name handle
        outfnh = abspath(join("./", outfile))

        # write GFF content to file
        with open(outfnh, "w") as outf:
            for chunk in gff_data.stream():
                text = pattern.sub(
                    lambda m: uri_decode_esc[m.group(0)], chunk.decode("utf-8")
                )
                outf.write(text)

        # remove extraneous information from feature file - i.e. first 26 lines
        today = strftime("%Y%m%d", localtime())
        outfnh_fixed = abspath(outfnh).replace(".gff", "_{}.gff".format(today))
        print("Writing GFF file to {}".format(outfnh_fixed))
        with open(outfnh_fixed, "w") as outfile:
            with open(outfnh) as infile:
                for line in infile:
                    if line.startswith("#"):
                        continue
                    elif line.split("\t")[2] != "chromosome":
                        outfile.write(line)
                    else:
                        continue
        remove(outfnh)  # remove older file version
        print("GFF file saved at {0}".format(outfnh_fixed))


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
    split_func = "split -d -a 5 --additional-suffix {0} -l {1} {2} {3}".format(
        additional_suffix, nlines, fnh, prefix
    )
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
                        "strand": [strand],
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
        return {s: [s[i : i + k] for i in range(0, len(s) - (k - 1), 1)] for s in seq}
    else:
        # `seq` is a single sequence
        # list of kmers [kmer1, kmer2, ...]
        return [seq[i : i + k] for i in range(0, len(seq) - (k - 1), 1)]


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
            counts[seq[i : i + k]] += 1
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
                outbed.write(
                    "{0}\t{1}\t{2}\t{3}\t.\t{4}\n".format(
                        chrom, chrom_start, chrom_end, motif_name, strand
                    )
                )


@profile
def sort_bed_file(inbed: str, outbed: str):
    """
    Sort input BED6 file first by chromosome and then by starting position.

    :type inbed: str
    :param inbed: Input BED6 file to be sorted

    :type outbed: str
    :param outbed: Sorted output BED6 file
    """
    import subprocess as sp
    import sys

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
    import subprocess as sp
    import sys

    try:
        import shlex
    except ImportError as ie:
        sys.exit("Please install {} module before executing this script.".format(ie))

    get_fasta = "bedtools getfasta -fi {0} -bed {1} -s -fo {2}".format(
        genome_file, inbed, outfasta
    )
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
            shape_data[whichshape] = {
                header: np.array(seq.strip().split(",")) for header, seq in sfp(infasta)
            }

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
                    seq = data[chrom][int(end) : int(start) - 1 : -1]
                else:
                    # positive strand
                    seq = data[chrom][int(start) : int(end) + 1]
                for i in range(1, len(seq)):
                    position = "{0}_pos_{1}".format(shape, i)
                    try:
                        # retrieve shape value
                        genome_shape[name][position] = float(seq[i])
                    except ValueError:
                        # no shape data calculated, use 0.0
                        genome_shape[name][position] = 0.0
    return genome_shape


@profile
def parse_gff_fasta(
    gff_file: str,
    parsed_fasta,
    out_fasta="Ca22_CDS_seqs.fasta",
    genome="22",
    feature="CDS",
):
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
                            seq_name = "|".join(
                                [
                                    attributes["ID"][0],
                                    attributes["orf_classification"][0],
                                    attributes["parent_feature_type"][0],
                                    attributes["Parent"][0],
                                    line[2],
                                ]
                            )
                            seq_name = seq_name.replace(" ", "_")
                        except KeyError:
                            # sparse GFF attributes
                            try:
                                seq_name = "|".join(
                                    [attributes["Parent"][0], line[1], line[2]]
                                )
                                seq_name = seq_name.replace(" ", "_")
                            except KeyError:
                                # even more sparse GFF attributes
                                seq_name = "|".join(
                                    [attributes["ID"][0], line[1], line[2]]
                                )
                                seq_name = seq_name.replace(" ", "_")
                        outfile.write(
                            ">{0}\n{1}\n".format(
                                seq_name, parsed_fasta[line[0]][start:end].seq
                            )
                        )
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
        exit(
            "\n{} does not exist. Please provide a valid FASTA file.\n".format(
                fasta_file
            )
        )
    else:
        # parse FASTA file and collect nucleotide frequencies
        bkg_freq = cnt()
        with open(fasta_file) as infile:
            for name, seq in sfp(infile):
                bkg_freq.update(seq)

    # helpful message about input sequences - optional
    if verbose:
        gc_content = 100 * (
            (bkg_freq["G"] + bkg_freq["C"])
            / sum([bkg_freq["C"], bkg_freq["T"], bkg_freq["A"], bkg_freq["G"]])
        )
        print("GC content of sequences in {}: {:0.2f}%".format(fasta_file, gc_content))

    # calculate background probabilities
    start_prob = {nt: freq / sum(bkg_freq.values()) for nt, freq in bkg_freq.items()}

    return start_prob


@profile
def get_transmat(fasta_file: str, n=2):
    """
    From a FASTA file, generate nth order Markov chain transition probability matrix. By
    default, a 2nd order Markov chain transition matrix will be calculated and returned as
    Pandas dataframe.

    :type fasta_file: str
    :param fasta_file: FASTA file path and name handle
    """
    # parse FASTA file and collect nucleotide frequencies
    kmer_counts = dict()
    for header, seq in parse_fasta(fasta_file):

        # get a list of all unambiguous kmers for FASTA sequence
        seqs = [dna_iupac_codes(seq) for seq in get_kmers(seq, n + 1)]
        seqs = [s for seq in seqs for s in seq]

        # iterate through seqs and get counts of nt following kmer
        for kmer in seqs:
            try:
                kmer_counts[kmer[:-1]].update(kmer[-1])
            except KeyError:
                kmer_counts[kmer[:-1]] = cnt(kmer[-1])

    # calculate probabilities for all nt after kmer
    kmer_prob = {
        k1: {k2: v2 / sum(v1.values()) for k2, v2 in v1.items()}
        for k1, v1 in kmer_counts.items()
    }
    return pd.DataFrame.from_dict(kmer_prob, orient="index").fillna(0.0)


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
    return choices(states, weights=p)[0]


@profile
def markov_seq(seq_len: int, bkg_seq: str, transmat, degree: int) -> str:
    """
    Using transition probability matrix and a starting sequence, generate seq_len length
    sequence from a 2nd order Markov model. The final output sequence will be padded by
    five randomly generated dinucleotides.

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
        current = bkg_seq[-degree:]
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
    print(
        "{0} out of {1} ({2:0.2f}%) overlaps found".format(
            overlap_count, len(a), overlap_pct
        )
    )
    return None


@profile
def gc_len_matched_bkg_seq_gen(
    fg_seq: str, transmat: dict, fg_seqs: list, degree=2, tol=1
) -> dict:
    """
    Using a markov model trained on non-Candida genome coding sequence regions,
    generate GC content and length matched false (background) sequence. GC content is
    allowed a controllable difference of 5% around the GC content of true (foreground)
    sequence.

    :type fg_seqs: dict
    :param fg_seqs: Parsed FASTA file of true (foreground) sequences

    :type transmat: dict
    :param transmat: A dict containing all transition probability matrix dataframe.
                     Ideally, the dataframes will be output from get_transmat()
    """
    try:
        fg_iupac_seqs = dna_iupac_codes(fg_seq)
        assert len(fg_iupac_seqs) == 1
    except AssertionError as exc:
        fg_seq = choice(fg_iupac_seqs)
    finally:
        gc = round(100 * calculate_gc_content(fg_seq))
        seq_len = len(fg_seq)
        random_key = choice(list(transmat.keys()))
        while True:
            gc_len_matched_bkg_seq = markov_seq(
                seq_len, random_dna(degree, False), transmat[random_key], degree
            )
            start, end = (
                degree,
                0 - degree,
            )
            core_seq = gc_len_matched_bkg_seq[start:end]
            gc_diff = abs(gc - round(100 * calculate_gc_content(core_seq)))
            if core_seq not in fg_seqs:
                if gc_diff <= tol and jaccard_similarity(fg_seq, core_seq) < 0.1:
                    return gc_len_matched_bkg_seq


@lru_cache(256)
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
    # length of kmer is one-third of seqA length
    n = len(seqA)
    q = 0.01
    k = ceil(log((n * (1 - q)) / q, 4))  # Based on MASH's methods recommendation

    # kmer word count for seqA
    seqA_kmers = get_kmers(seqA, k)
    n_seqA_kmers = len(seqA_kmers)
    seqA_wc = {kmer: seqA_kmers.count(kmer) for kmer in seqA_kmers}
    # prior probability of kmer 'w'
    ppw_A = {word: freq / n_seqA_kmers for word, freq in seqA_wc.items()}
    # expected number of occurrences
    mw_A = {word: pp * n_seqA_kmers for word, pp in ppw_A.items()}

    # kmer word count for seqB
    seqB_kmers = get_kmers(seqB, k)
    n_seqB_kmers = len(seqB_kmers)
    seqB_wc = {kmer: seqB_kmers.count(kmer) for kmer in seqB_kmers}
    # prior probability of kmer 'w'
    ppw_B = {word: freq / n_seqB_kmers for word, freq in seqB_wc.items()}
    # expected number of occurrences
    mw_B = {word: pp * n_seqB_kmers for word, pp in ppw_B.items()}

    n_patterns = len(seqA_kmers)
    sga = seqA_wc.get
    sgb = seqB_wc.get
    for word in seqA_kmers:
        wc_seqA = sga(word, 0)
        wc_seqB = sgb(word, 0)
        Cw_AB = min(wc_seqA, wc_seqB)
        prob = (
            (1 - _cdf(Cw_AB - 1, mw_A[word])) * (1 - _cdf(Cw_AB - 1, mw_B[word]))
            if Cw_AB > 0
            else 1
        )
        try:
            prod *= prob  # update poisson product similarity
        except NameError:
            prod = prob
        try:
            sim = 1 - prob
            similarity += sim  # update poisson additive similarity
        except NameError:
            similarity = 1 - prob
    product = 1 - (prod ** (1 / n_patterns))
    similarity = similarity / n_patterns
    return similarity, product


@profile
def permutation_result(
    protein, estimator, X, y, cv, file: str, n_permute=1000, random_state=39
):
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
        scoring="average_precision",
        cv=cv,
        n_permutations=n_permute,
        n_jobs=-1,
        random_state=random_state,
    )
    print(
        strftime(
            "%x %X | Linear SVM classification score {0:0.3f} (pvalue: {1:0.5f})".format(
                score, p_value
            )
        )
    )
    with mpl.style.context("fast"):
        plt.figure(figsize=(10, 7), edgecolor="k", tight_layout=True)
        plt.hist(
            permutation_score,
            bins=25,
            alpha=0.5,
            hatch="//",
            edgecolor="w",
            label="Precision scores for shuffled labels",
        )
        ylim = plt.ylim()[1]
        plt.vlines(
            2 * [1.0 / np.unique(y).size], 0, ylim, linewidth=3, label="50/50 chance"
        )
        plt.vlines(2 * [score], 0, ylim, linewidth=3, colors="g")
        plt.title(
            "Model Average Precision Score = {:0.2f}*".format(100 * score),
            color="g",
            fontsize=20,
        )
        plt.xlim(0.0, 1.1)
        plt.legend(loc=2)
        plt.figtext(
            0.15,
            0.815,
            f"{protein.capitalize()}",
            c="w",
            backgroundcolor="k",
            size=20,
            weight="bold",
            ha="center",
            va="center",
        )
        plt.xlabel("Average precision score", fontsize=20, color="k")
        plt.ylabel("Frequency", fontsize=20, color="k")
        plt.savefig(
            file,
            dpi=300,
            format="pdf",
            bbox_inches="tight",
            edgecolor="k",
            pad_inches=0.2,
        )


def _minhash_similarity(seqA: str, true_seqs: list) -> float:
    """
    Given a test sequence (seqA)  and list of reference sequences (true_seqs), calculate
    the average MinHash similarity score for seqA.
    """
    return np.mean(list(starmap(jaccard_similarity, product(true_seqs, [seqA]))))


def _pac(seqA: str, true_seqs: list) -> float:
    """
    Given a test sequence (seqA)  and list of reference sequences (true_seqs), calculate
    the average Poisson similarity score for seqA.
    """
    pas = [pac(*pair)[0] for pair in product(true_seqs, [seqA])]
    pps = [pac(*pair)[1] for pair in product(true_seqs, [seqA])]
    return {"PAS": np.mean(pas), "PPS": np.mean(pps)}


def _get_DNA_shape(loc: str, shapedata: dict, shape: str) -> dict:
    """
    Get DNA shape data from genomic coordinates.
    """
    chrom, location = tuple(loc.split(":"))
    strand = location.split("(")[1][0]
    start = int(location.split("-")[0])
    end = int(location.split("(")[0].split("-")[1])

    # get shape values
    shapes = {}
    if strand == "+":
        for i, value in enumerate(shapedata[chrom][start : end + 1]):
            try:
                shapes[f"{shape}_{i + 1:02d}"] = float(value)
            except ValueError:
                shapes[f"{shape}_{i + 1:02d}"] = 0.0
    else:
        for i, value in enumerate(shapedata[chrom][end : start - 1 : -1]):
            try:
                shapes[f"{shape}_{i + 1:02d}"] = float(value)
            except ValueError:
                shapes[f"{shape}_{i + 1:02d}"] = 0.0
    return shapes


@profile
def build_feature_table(
    infasta: str, truefasta: str, genome_wide_shape_data=None, minhash=True
):
    """
    Build feature table by calculating/retrieving all relevant characteristics -
    1. sequence GC percent value
    2. Poisson additive similarity score
    3. Poisson multiplicative similarity score
    4. MinHash based similarity score
    5. sequence DNA shape values (EP, HelT, MGM, Prot, Roll)
    and return a Pandas dataframe

    :infasta: Input FASTA file with true (foreground) or genome-wide BLASTn match sequences

    :truefasta: FASTA file with true (foreground) sequences to be used for calculating
                Poisson based similarity metrics

    :genome_wide_shape_data: A dict with genome-wide 3D DNA shape (DNAShapeR output files)
                             data single-line. The format would be ->
                             {"shape_name": {chromosome: ['0.0', '0.0', ..., '0.0']}}
    """
    print(strftime("%x %X | Begin building feature table"))
    tf_data = pd.DataFrame.from_dict(
        {header: seq for header, seq in parse_fasta(infasta)},
        orient="index",
        columns=["sequence"],
    )
    tf_data = tf_data.reset_index().rename(columns={"index": "location"})
    true_seqs = {seq for header, seq in parse_fasta(truefasta)}

    # Add GC percent column
    print(strftime("%x %X | Calculating GC content of input sequences"))
    tf_data["GC_content"] = tf_data["sequence"].apply(calculate_gc_content)

    if minhash:
        # Add MinHash similarity score column
        print(strftime("%x %X | Calculating MinHash similarity score"))
        tf_data["MinHash"] = tf_data["sequence"].apply(
            _minhash_similarity, args=(true_seqs,)
        )

    # Add Poisson similarity score columns
    print(strftime("%x %X | Calculating Poisson similarity scores"))
    tf_data["PBS"] = tf_data["sequence"].apply(_pac, args=(true_seqs,))
    tf_data = tf_data.join(pd.json_normalize(tf_data["PBS"])).drop(columns=["PBS"])

    if genome_wide_shape_data:
        # Retrieve DNA shapes for input sequences
        print(strftime("%x %X | Retrieving Electrostatic potential shape values"))
        tf_data["EP"] = tf_data["location"].apply(
            _get_DNA_shape, args=(genome_wide_shape_data["EP"], "EP",)
        )
        tf_data = tf_data.join(pd.json_normalize(tf_data["EP"])).drop(columns=["EP"])

        print(strftime("%x %X | Retrieving Helix Twist shape values"))
        tf_data["HelT"] = tf_data["location"].apply(
            _get_DNA_shape, args=(genome_wide_shape_data["HelT"], "HelT",)
        )
        tf_data = tf_data.join(pd.json_normalize(tf_data["HelT"])).drop(
            columns=["HelT"]
        )

        print(strftime("%x %X | Retrieving Minor groove width shape values"))
        tf_data["MGW"] = tf_data["location"].apply(
            _get_DNA_shape, args=(genome_wide_shape_data["MGW"], "MGW",)
        )
        tf_data = tf_data.join(pd.json_normalize(tf_data["MGW"])).drop(columns=["MGW"])

        print(strftime("%x %X | Retrieving Propeller twist shape values"))
        tf_data["ProT"] = tf_data["location"].apply(
            _get_DNA_shape, args=(genome_wide_shape_data["ProT"], "ProT",)
        )
        tf_data = tf_data.join(pd.json_normalize(tf_data["ProT"])).drop(
            columns=["ProT"]
        )

        print(strftime("%x %X | Retrieving Roll shape values"))
        tf_data["Roll"] = tf_data["location"].apply(
            _get_DNA_shape, args=(genome_wide_shape_data["Roll"], "Roll",)
        )
        tf_data = tf_data.join(pd.json_normalize(tf_data["Roll"])).drop(
            columns=["Roll"]
        )

    print(strftime("%x %X | Finished building feature table"))
    return tf_data


@profile
def plot_coefficients(feature_score, feature_names, protein, file: str):
    """
    Using the coefficient weights, plot the contribution of each feature in
    classification. Currently, this function is set up for binary classification.

    :type feature_score: array-like, list or numpy array
    :param feature_score: SVM/LDA weights assigned to each feature.

    :type feature_names: list
    :param feature_names: List of feature names to use for plotting

    :type file: str
    :param file: Path and name of file to save feature contribution bar plot. The file
                 will be saved in PDF format.
    """
    feature_names = np.asarray([name.replace("_", " ") for name in feature_names])
    feature_rank = feature_score.importances_mean.argsort()[::-1]
    res = list(
        zip(
            feature_names[feature_rank],
            feature_score.importances_mean[feature_rank][
                feature_score.importances_mean[feature_rank] > 0
            ],
            feature_score.importances_std[feature_rank],
        )
    )
    res = pd.DataFrame(res, columns=["Feature", "Mean_score", "Std_score"]).sort_values(
        "Mean_score", ascending=False
    )
    res = res.head(10)

    # create plot
    with mpl.style.context("fast"):
        plt.figure(figsize=(10, 6), edgecolor="k", tight_layout=True)
        plt.barh(
            range(len(res)),
            res["Mean_score"][::-1],
            color="#c8c8c8",
            edgecolor="k",
            tick_label=res["Feature"][::-1],
        )
        plt.yticks(fontsize=14)
        plt.figtext(
            0.925,
            0.165,
            f"{protein.capitalize()}",
            c="w",
            backgroundcolor="k",
            size=20,
            weight="bold",
            ha="center",
            va="center",
        )
        plt.xlabel("Mean contribution score", fontsize=16, color="k")
        plt.grid(axis="x", ls=":")
        plt.savefig(
            file,
            dpi=300,
            format="pdf",
            bbox_inches="tight",
            edgecolor="k",
            pad_inches=0.1,
        )


@profile
def jaccard_similarity(a: str, b: str) -> float:
    """
    Given two sequences, calculate their Jaccard similarity score using minhash algorithm.
    """
    k = ceil(log((len(a) * 0.99) / 0.01, 4))
    seqA_kmers_hashed = set([_minhash_kmer(kmer) for kmer in get_kmers(a, k)])

    seqB_kmers_hashed = set([_minhash_kmer(kmer) for kmer in get_kmers(b, k)])

    numerator = len(seqA_kmers_hashed.intersection(seqB_kmers_hashed))
    denominator = len(seqA_kmers_hashed.union(seqB_kmers_hashed))

    return numerator / denominator


@profile
def _minhash_kmer(kmer: str) -> int:
    """
    Given a k length string of unambiguous nucleotide, return its 32-bit hash value.
    Both kmer and its reverse complement are compared and hash value for lexicographically
    smaller sequence is returned.
    """
    kmer_rc = reverse_complement(kmer)
    if kmer < kmer_rc:
        return mmh3.hash(kmer, seed=39, signed=False)
    else:
        return mmh3.hash(kmer_rc, seed=39, signed=False)


@profile
def get_closest_genes(bedfile: str, genome_file: str):
    """
    Given a genome file and TFBS data in BED format, return closest gene to each TFBS and
    counts of TFBS for closest gene.

    :param bedfile: True/foreground or model predicted TFBS in BED6 format
    :param genome_file: Candida albicans or other model organism ORF feature only GFF file
    """
    # Read in files with BedTool
    bedfile = BedTool(bedfile)
    genome_file = BedTool(genome_file)
    orfs = {}

    # Iterate over the closest gene to each TFBS and get information of the closest gene
    for line in bedfile.closest(genome_file, D="b", stream=True):
        orf_info = line.fields[12].split(";")
        for entry in orf_info:
            if entry.startswith("Name"):
                name = "_".join(entry.split("=")[1].split("_")[:-1])
                if orfs.get(name, False):
                    # Gene entry exists in orfs dict, increment the TFBS count for gene
                    orfs[name]["Closest_TFBS_Counts"] += 1
                    continue
                else:
                    # Gene entry does not exist in orfs dict, initialize gene entry
                    orfs[name] = {
                        "Gene_Function": "NA",
                        "Gene_Name": "NA",
                        "Closest_TFBS_Counts": 1,
                    }

            if entry.startswith("Gene"):
                # Get human readable gene name
                orfs[name]["Gene_Name"] = entry.split("=")[1]
            elif entry.startswith("Note"):
                # Get human readable gene function
                try:
                    orfs[name]["Gene_Function"] = (
                        entry.split("=")[1].replace("_", " ").strip()
                    )
                except Exception as exc:
                    print(
                        exc,
                        "\n",
                        entry,
                        "\n",
                        line.strip().split("\t")[12].split(";"),
                        "\n",
                    )
                    break
            elif entry.startswith("_"):
                # Append additional human readable gene function entries
                orfs[name]["Gene_Function"] += ";" + entry.replace("_", " ").rstrip()
            else:
                continue

        # How far is TFBS away from closest gene/TSS?
        try:
            orfs[name]["Distance_from_gene"] += int(line.fields[13])
        except Exception:
            orfs[name]["Distance_from_gene"] = int(line.fields[13])

    # Convert dict to pandas dataframe and make it pretty
    orfs_df = (
        pd.DataFrame.from_dict(orfs, orient="index")
        .reset_index()
        .rename(columns={"index": "Systematic_Name"})
        .sort_values("Closest_TFBS_Counts", ascending=False)
    )
    orfs_df["Mean_dist_from_gene"] = (
        orfs_df["Distance_from_gene"] / orfs_df["Closest_TFBS_Counts"]
    )
    orfs_df.drop(columns="Distance_from_gene", inplace=True)
    orfs_df = orfs_df.loc[
        :,
        [
            "Systematic_Name",
            "Gene_Name",
            "Closest_TFBS_Counts",
            "Mean_dist_from_gene",
            "Gene_Function",
        ],
    ]
    return orfs_df
