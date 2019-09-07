#!/usr/bin/env python3

"""
Build feature table from input FASTA files.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.1.2"


from sys import exit
from os.path import isfile, abspath
import argparse
from collections import defaultdict
from itertools import product, starmap
from time import localtime, strftime
from pprint import pprint
from utils import parse_fasta, calculate_gc_percent, pac
err = []
# try:
#     import matplotlib as mpl
#     import matplotlib.pyplot as plt
#     mpl.rc("font", family="Arial")
# except ImportError:
#     err.append("matplotlib")
try:
    import numpy as np
except ImportError:
    err.append("numpy")
# try:
#     import pandas as pd
# except ImportError:
#     err.append("pandas")
# try:
#     from sklearn.svm import SVC
#     from sklearn.preprocessing import MaxAbsScaler
#     from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
#     from sklearn.decomposition import PCA
#     from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, average_precision_score
# except ImportError:
#     err.append("scikit-learn")
try:
    assert len(err) == 0
except AssertionError:
    for error in err:
        print("Please install {}".format(error))
    exit()


def handle_program_options():
    parser = argparse.ArgumentParser(
        description="Build feature table from input FASTA files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("fg_fasta_file", help="Path to foreground/true positive sequence "
                        "dataset FASTA format file [REQUIRED]")
    parser.add_argument("bkg_fasta_file", help="Path to background/negative sequence "
                        "dataset FASTA format file [REQUIRED]")
    parser.add_argument("-fsff", "--fg_shape_fasta_file", nargs="+", help="Path to 3D DNA"
                        " shape (DNAShapeR output files) data FASTA format files "
                        "associated with 'fg_fasta_file' parameters [REQUIRED]")
    parser.add_argument("-bsff", "--bkg_shape_fasta_file", nargs="+", help="Path to 3D "
                        "DNA shape (DNAShapeR output files) data FASTA format file "
                        "associated with 'bkg_fasta_file' parameters [REQUIRED]")
    parser.add_argument("protein_name", type=str, help="Name of transcription factor "
                        "denoted by fasta_file parameter. Name must only contain "
                        "alphanumeric characters [REQUIRED]")
    parser.add_argument("-s", "--savefile", action="store_true",
                        help="Save plots as SVG files.")
    parser.add_argument("-o", "--output_file", action="store_true",
                        help="Save consolidated data to this tab-separated file")
    return parser.parse_args()


def map_headers_to_values(fasta_header, values) -> dict:
    """
    Given equal length lists of FASTA header lines and calculated numerical values,
    return a dict mapping using zip()

    :type fasta_header: array-like, list or numpy array
    :param fasta_header: List or numpy array of FASTA header lines

    :type values: array-like, list or numpy array
    :param values: List or numpy array of numerical values equal to length of
                   fasta_header
    """
    try:
        assert len(fasta_header) == len(values)
    except AssertionError:
        exit("Could not create a mapping between FASTA headers and input numerical "
             "array.")
    else:
        return dict(zip(fasta_header, values))


def all_possible_seq_pairs(list1, fg_seqs):
    """
    Get all possible pairs of foreground and background sequences for calculating metrics
    and return all possible background-foreground sequence pairs

    :type list1: array-like
    :param list1: Array of background sequences

    :type fg_seqs: array-like
    :param fg_seqs: Array of foreground sequences
    """
    return (list(product([seq], fg_seqs)) for seq in list1)


def main():
    args = handle_program_options()

    try:
        assert isfile(args.fg_fasta_file)
        assert isfile(args.bkg_fasta_file)
        # assert isfile(args.test_fasta_file)
    except AssertionError as e:
        print("Input FASTA file(s) does not exist. Please check supplied FASTA file.\n{}"
              .format(e))
        exit()

    #######################################
    # FOREGROUND SEQUENCE PROCESSING #
    #######################################
    print(strftime("%x %X".format(localtime)), ": Processing foreground FASTA file")
    print("="*52, sep="\n")

    # get all foreground sequences
    print(strftime("%x %X".format(localtime)), ": Reading FASTA file")
    fg_seqs = defaultdict(dict)
    fg_seqs[args.protein_name]["header"] = np.asarray([name
                                                       for name, seq in
                                                       parse_fasta(args.fg_fasta_file)])
    fg_seqs[args.protein_name]["seqs"] = np.asarray([seq
                                                     for name, seq in
                                                     parse_fasta(args.fg_fasta_file)])

    # get GC percent for all foreground sequences
    print(strftime("%x %X".format(localtime)), ": Calculating GC percent")
    fg_gc = {tf: map_headers_to_values(data["header"],
                                       list(map(calculate_gc_percent, data["seqs"])))
             for tf, data in fg_seqs.items()}

    # calculating poisson based metrics
    print(strftime("%x %X".format(localtime)), ": Calculating Poisson based metrics")
    fg_seq_pairs = all_possible_seq_pairs(fg_seqs[args.protein_name]["seqs"],
                                          fg_seqs[args.protein_name]["seqs"])
    fg_poisson_metrics = np.asarray([np.asarray(list(starmap(pac, pair_set))).
                                     mean(axis=0, dtype=np.float64)
                                     for pair_set in fg_seq_pairs])

    fg_pac = map_headers_to_values(fg_seqs[args.protein_name]["header"],
                                   fg_poisson_metrics)

    # collate all DNA shape values
    print(strftime("%x %X".format(localtime)), ": Processing DNA shape data\n")
    fg_shapes = dict()
    for shapefile in args.fg_shape_fasta_file:
        whichshape = shapefile.split(".")[-1]
        if whichshape in ["MGW", "ProT", "EP"]:
            for name, shape in parse_fasta(abspath(shapefile)):
                shape = shape.split(",")
                if not fg_shapes.get(name):
                    fg_shapes[name] = dict()
                for i in range(2, len(shape) - 2):
                    position = "{0}_pos_{1}".format(whichshape, i + 1)
                    fg_shapes[name][position] = float(shape[i])
        else:
            # shape is Roll or HelT
            for name, shape in parse_fasta(abspath(shapefile)):
                shape = shape.split(",")
                if not fg_shapes.get(name):
                    fg_shapes[name] = dict()
                for i in range(1, len(shape) - 1):
                    position = "{0}_pos_{1}".format(whichshape, i + 1)
                    fg_shapes[name][position] = float(shape[i])

    ##################################
    # BACKGROUND SEQUENCE PROCESSING #
    ##################################
    print(strftime("%x %X".format(localtime)), ": Processing background FASTA file")
    print("="*52, sep="\n")

    # get all background sequences
    print(strftime("%x %X".format(localtime)), ": Reading FASTA file")
    bkg_seqs = defaultdict(dict)
    bkg_seqs[args.protein_name]["header"] = np.asarray([name
                                                       for name, seq in
                                                       parse_fasta(args.bkg_fasta_file)])
    bkg_seqs[args.protein_name]["seqs"] = np.asarray([seq
                                                     for name, seq in
                                                     parse_fasta(args.bkg_fasta_file)])

    # get GC percent for all background sequences
    print(strftime("%x %X".format(localtime)), ": Calculating GC percent")
    bkg_gc = {tf: map_headers_to_values(data["header"],
                                        list(map(calculate_gc_percent, data["seqs"])))
              for tf, data in bkg_seqs.items()}

    # calculating poisson based metrics
    print(strftime("%x %X".format(localtime)), ": Calculating Poisson based metrics")
    bkg_seq_pairs = all_possible_seq_pairs(bkg_seqs[args.protein_name]["seqs"],
                                           bkg_seqs[args.protein_name]["seqs"])
    bkg_poisson_metrics = np.asarray([np.asarray(list(starmap(pac, pair_set))).
                                      mean(axis=0, dtype=np.float64)
                                      for pair_set in bkg_seq_pairs])

    bkg_pac = map_headers_to_values(bkg_seqs[args.protein_name]["header"],
                                    bkg_poisson_metrics)

    # collate all DNA shape values
    print(strftime("%x %X".format(localtime)), ": Processing DNA shape data\n")
    bkg_shapes = dict()
    for shapefile in args.bkg_shape_fasta_file:
        whichshape = shapefile.split(".")[-1]
        if whichshape in ["MGW", "ProT", "EP"]:
            for name, shape in parse_fasta(abspath(shapefile)):
                shape = shape.split(",")
                if not bkg_shapes.get(name):
                    bkg_shapes[name] = dict()
                for i in range(2, len(shape) - 2):
                    position = "{0}_pos_{1}".format(whichshape, i + 1)
                    bkg_shapes[name][position] = float(shape[i])
        else:
            # shape is Roll or HelT
            for name, shape in parse_fasta(abspath(shapefile)):
                shape = shape.split(",")
                if not bkg_shapes.get(name):
                    bkg_shapes[name] = dict()
                for i in range(1, len(shape) - 1):
                    position = "{0}_pos_{1}".format(whichshape, i + 1)
                    bkg_shapes[name][position] = float(shape[i])


if __name__ == "__main__":
    exit(main())
