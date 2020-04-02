#!/usr/bin/env python3

"""
Create background sequences from input FASTA files.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.2.5"


import argparse
from os import mkdir
from os.path import abspath, exists, isfile, join
from random import choices, randint, random
from sys import exit
from time import localtime, strftime

from utils import (calculate_gc_percent, dna_iupac_codes, get_transmat, markov_seq,
                   parse_fasta, random_dna)

try:
    from rpy2.robjects.packages import importr
except ImportError:
    exit("\nPlease install rpy2 package.\n")
else:
    try:
        dnashaper = importr("DNAshapeR")
    except Exception:
        exit("\nPlease install bioconductor-dnashaper package.\n")


def handle_program_options():
    parser = argparse.ArgumentParser(
        description="Using foreground sequences, generate background sequences. "
        "Background sequences will be generated firstly by matching foreground motif "
        "GC-percent and length. Secondly, the foregound sequences will be shuffled to "
        "keep dinucleotide composition constant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "fg_fasta_file",
        type=str,
        help="Path to foreground/true positive sequence dataset FASTA "
        "format file [REQUIRED]",
    )
    parser.add_argument(
        "protein_name",
        type=str,
        help="Name of transcription factor protein. [REQUIRED]",
    )
    parser.add_argument(
        "-g",
        "--genome_fasta_files",
        type=str,
        nargs="+",
        help="Specify path to one or more genome files to use as template"
        " from which to generae random background sequences. These genome"
        "file(s) must be FASTA file with very low probability of "
        "containing sequences with binding motifs. For example, these "
        "files can be FASTA file of exonic regions of non-related "
        "species. Please do not supply gzipped files.",
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=int,
        default=2,
        help="Percent tolerance allowed for matching GC content of "
        "background sequence with foreground sequence. The default value "
        "is 2 percent difference between background and foreground "
        "sequence. A value of zero will increase eexecution time for this"
        " script.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Specify a directory to save background sequence data.",
    )
    return parser.parse_args()


##########################################################################################
# altschulEriksonDinuclShuffle.py
# P. Clote, Oct 2003


def computeCountAndLists(s):
    # WARNING: Use of function count(s,"UU") returns 1 on word UUU since it apparently
    # counts only nonoverlapping words UU. For this reason, we work with the indices.

    # Initialize lists and mono- and dinucleotide dictionaries
    nuclList = ["A", "C", "G", "T"]
    List = {nt: [] for nt in nuclList}  # List is a dictionary of lists
    s = s.upper()
    nuclCnt = {}  # empty dictionary
    dinuclCnt = {}  # empty dictionary
    for x in nuclList:
        nuclCnt[x] = 0
        dinuclCnt[x] = {y: 0 for y in nuclList}

    # Compute count and lists
    nuclCnt[s[0]] = 1
    nuclTotal = 1
    dinuclTotal = 0
    for i in range(len(s) - 1):
        x = s[i]
        y = s[i + 1]
        List[x].append(y)
        nuclCnt[y] += 1
        nuclTotal += 1
        dinuclCnt[x][y] += 1
        dinuclTotal += 1
    assert nuclTotal == len(s)
    assert dinuclTotal == len(s) - 1
    return nuclCnt, dinuclCnt, List


def chooseEdge(x, dinuclCnt):
    numInList = 0
    for y in ["A", "C", "G", "T"]:
        numInList += dinuclCnt[x][y]
    z = random()
    denom = (
        dinuclCnt[x]["A"] + dinuclCnt[x]["C"] + dinuclCnt[x]["G"] + dinuclCnt[x]["T"]
    )
    numerator = dinuclCnt[x]["A"]
    if z < float(numerator) / float(denom):
        dinuclCnt[x]["A"] -= 1
        return "A"
    numerator += dinuclCnt[x]["C"]
    if z < float(numerator) / float(denom):
        dinuclCnt[x]["C"] -= 1
        return "C"
    numerator += dinuclCnt[x]["G"]
    if z < float(numerator) / float(denom):
        dinuclCnt[x]["G"] -= 1
        return "G"
    dinuclCnt[x]["T"] -= 1
    return "T"


def connectedToLast(edgeList, nuclList, lastCh):
    D = {x: 0 for x in nuclList}
    for edge in edgeList:
        a = edge[0]
        b = edge[1]
        if b == lastCh:
            D[a] = 1
    for i in range(2):
        for edge in edgeList:
            a = edge[0]
            b = edge[1]
            if D[b] == 1:
                D[a] = 1
    # ok = 0
    for x in nuclList:
        if x != lastCh and D[x] == 0:
            return 0
    return 1


def eulerian(s):
    nuclCnt, dinuclCnt, List = computeCountAndLists(s)
    # compute nucleotides appearing in s
    nuclList = []
    for x in ["A", "C", "G", "T"]:
        if x in s:
            nuclList.append(x)
    # compute numInList[x] = number of dinucleotides beginning with x
    numInList = {}
    for x in nuclList:
        numInList[x] = 0
        for y in nuclList:
            numInList[x] += dinuclCnt[x][y]
    # create dinucleotide shuffle L
    # firstCh = s[0]   # start with first letter of s
    lastCh = s[-1]
    edgeList = []
    for x in nuclList:
        if x != lastCh:
            edgeList.append([x, chooseEdge(x, dinuclCnt)])
    ok = connectedToLast(edgeList, nuclList, lastCh)
    return ok, edgeList, nuclList, lastCh


def shuffleEdgeList(L):
    n = len(L)
    barrier = n
    for i in range(n - 1):
        z = int(random() * barrier)
        tmp = L[z]
        L[z] = L[barrier - 1]
        L[barrier - 1] = tmp
        barrier -= 1
    return L


def dinuclShuffle(s):
    ok = 0
    while not ok:
        ok, edgeList, nuclList, lastCh = eulerian(s)
    nuclCnt, dinuclCnt, List = computeCountAndLists(s)

    # remove last edges from each vertex list, shuffle, then add back
    # the removed edges at end of vertex lists.
    for [x, y] in edgeList:
        List[x].remove(y)
    for x in nuclList:
        shuffleEdgeList(List[x])
    for [x, y] in edgeList:
        List[x].append(y)

    # construct the eulerian path
    L = [s[0]]
    prevCh = s[0]
    for i in range(len(s) - 2):
        ch = List[prevCh][0]
        L.append(ch)
        del List[prevCh][0]
        prevCh = ch
    L.append(s[-1])
    t = "".join(L)
    return t


##########################################################################################


def main():

    args = handle_program_options()

    try:
        assert isfile(args.fg_fasta_file)
    except AssertionError as e:
        print(
            "Error with input foreground FASTA file(s). Please check supplied FASTA "
            "file - {0}".format(e)
        )
        exit()
    else:
        # parse foreground sequence FASTA file
        print(
            strftime("%x %X:".format(localtime)),
            "1. Parsing foreground sequence FASTA file",
        )
        fg_seqs = {header: seq for header, seq in parse_fasta(args.fg_fasta_file)}

    try:
        outdir = abspath(args.output_dir)
        assert exists(outdir)
    except AssertionError:
        # output directory doesn't exist, create it
        mkdir(outdir)
    else:
        # get output file path and name
        outfnh = join(outdir, "{}_bkg_seqs.fasta".format(args.protein_name))

    # parse unrelated genome FASTA files containing CDS sequences
    print(
        strftime("%x %X:".format(localtime)),
        "2. Calculating transition probability from FASTA files",
    )
    cds_transmat = dict()
    degree = 2
    for f in args.genome_fasta_files:
        cds_transmat[f.split("/")[-1]] = get_transmat(f, degree)

    print(
        strftime("%x %X:".format(localtime)),
        "3. Writing background sequences to {0}".format(outfnh),
    )
    with open(outfnh, "w") as outf:
        for header, seq in fg_seqs.items():
            for entry in dna_iupac_codes(seq):

                ################################################
                # MONONUCLEOTIDE SHUFFLED FOREGROUND SEQUENCES #
                # Durstenfeld shuffle                          #
                ################################################
                outf.write(">mononuc_shuffled_bkg_for_{}\n".format(header))
                shuff_seq = []
                for nt in entry:
                    j = randint(0, len(shuff_seq))
                    if j == len(shuff_seq):
                        shuff_seq.append(nt)
                    else:
                        shuff_seq.append(shuff_seq[j])
                        shuff_seq[j] = nt
                # pad with 2 random nucleotides
                shuff_seq = (
                    random_dna(2, False) + "".join(shuff_seq) + random_dna(2, False)
                )
                outf.write("{}\n".format("".join(shuff_seq)))

                ##############################################
                # DINUCLEOTIDE SHUFFLED FOREGROUND SEQUENCES #
                ##############################################
                outf.write(">dinuc_shuffled_bkg_for_{}\n".format(header))
                # pad with 2 random nucleotides
                shuff_seq = (
                    random_dna(2, False) + dinuclShuffle(entry) + random_dna(2, False)
                )
                outf.write("{}\n".format(shuff_seq))

                ################################################################
                # GC CONTENT AND LENGTH MATCHED BACKGROUND SEQUENCE GENERATION #
                ################################################################
                outf.write(">gc_len_matched_bkg_for_{}\n".format(header))
                gc = round(calculate_gc_percent(seq))
                seq_len = len(seq)
                random_key = choices(list(cds_transmat.keys()))[0]
                while True:
                    gc_len_seq = markov_seq(
                        seq_len, random_dna(2, False), cds_transmat[random_key]
                    )
                    core_seq = gc_len_seq[2:-2]
                    if core_seq not in fg_seqs.values():
                        gc_diff = abs(gc - round(calculate_gc_percent(core_seq)))
                        if gc_diff <= args.tolerance:
                            outf.write("{}\n".format(gc_len_seq))
                            break

    ######################################################
    # CALCULATE DNA SHAPE VALUE FOR BACKGROUND SEQUENCES #
    ######################################################
    for shape in ["MGW", "Roll", "HelT", "ProT", "EP"]:
        dnashaper.getDNAShape(outfnh, shape)


if __name__ == "__main__":
    exit(main())
