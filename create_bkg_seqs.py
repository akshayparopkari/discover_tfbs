#!/usr/bin/env python3

"""
Create background sequences from input FASTA files.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.1.4"


import argparse
from sys import exit
from os import mkdir
from random import choices, random
from collections import defaultdict
from time import localtime, strftime
from os.path import isfile, join, abspath, exists
from utils import parse_fasta, calculate_gc_percent, get_kmers


def handle_program_options():
    parser = argparse.ArgumentParser(
        description="Using foreground sequences, generate background sequences. "
        "Background sequences will be generated firstly by matching foreground motif "
        "GC-percent and length. Secondly, the foregound sequences will be shuffled to "
        "keep dinucleotide composition constant.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

##########################################################################################
    # altschulEriksonDinuclShuffle.py
    # P. Clote, Oct 2003


def computeCountAndLists(s):
    # WARNING: Use of function count(s,"UU") returns 1 on word UUU since it apparently
    # counts only nonoverlapping words UU. For this reason, we work with the indices.

    # Initialize lists and mono- and dinucleotide dictionaries
    List = {}  # List is a dictionary of lists
    List["A"] = []
    List["C"] = []
    List["G"] = []
    List["T"] = []
    nuclList = ["A", "C", "G", "T"]
    s = s.upper()
    s = s.replace("T", "T")
    nuclCnt = {}       # empty dictionary
    dinuclCnt = {}     # empty dictionary
    for x in nuclList:
        nuclCnt[x] = 0
        dinuclCnt[x] = {}
        for y in nuclList:
            dinuclCnt[x][y] = 0

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
    assert (nuclTotal == len(s))
    assert (dinuclTotal == len(s) - 1)
    return nuclCnt, dinuclCnt, List


def chooseEdge(x, dinuclCnt):
    numInList = 0
    for y in ["A", "C", "G", "T"]:
        numInList += dinuclCnt[x][y]
    z = random()
    denom = dinuclCnt[x]["A"] + dinuclCnt[x]["C"] + dinuclCnt[x]["G"] +\
        dinuclCnt[x]["T"]
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
    D = {}
    for x in nuclList:
        D[x] = 0
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

    ##############################################
    # DINUCLEOTIDE SHUFFLED FOREGROUND SEQUENCES #
    ##############################################
    print(strftime("\n%x %H:%M:".format(localtime)),
          "Generating length-matched shuffled sequences from foreground sequences")

    # write to output FASTA file
    outfnh = join(outdir,
                  "{}_dinuc_shuffled_len_matched_bkg_seqs.fasta".
                  format(args.protein_name))
    with open(outfnh, "w") as outf:
        for header, seq in fg_seqs.items():
            outf.write(">dinuc_shuffled_len_matched_bkg_for_{}\n".format(header))
            shuffled = dinuclShuffle(seq)
            outf.write("{}\n".format(shuffled))


if __name__ == "__main__":
    exit(main())
