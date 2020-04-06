#!/usr/bin/env python3

"""
Build feature table from input FASTA files.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.3.2"


import argparse
from collections import defaultdict
from itertools import product, starmap
from os.path import abspath, isfile
from sys import exit
from time import strftime

from utils import build_feature_table, calculate_gc_percent, pac, parse_fasta

err = []
try:
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")
except ImportError:
    err.append("matplotlib")
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
        print("Please install {0}".format(error))
    exit()


def handle_program_options():
    parser = argparse.ArgumentParser(
        description="Build feature table from input FASTA files and save"
        "feature table/training data to feather format file. Feather files "
        "will then be used for downstream hyperparameter optimization and "
        "classification tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "protein_name",
        choices=["bcr1", "brg1", "efg1", "ndt80", "rob1", "tec1"],
        type=str,
        help="Name of transcription factor. Please see the list of valid "
        "choices for this parameter [REQUIRED]",
    )
    parser.add_argument(
        "fg_fasta_file",
        metavar="/path/to/true_binding_site_sequences.fasta",
        type=str,
        help="Path to true positive sequence dataset FASTA format file [REQUIRED]",
    )
    parser.add_argument(
        "fg_bed_file",
        metavar="/path/to/true_binding_site_sequences.bed",
        type=str,
        help="Path to true binding event BED file. This file contains sequence "
        "information listed in `bkg_fasta_file`. This file must have a "
        "minimum of BED6 format - i.e. chrom start end name score strand "
        " columns [REQUIRED]",
    )
    parser.add_argument(
        "bkg_fasta_file",
        metavar="/path/to/background_binding_site_sequences.fasta",
        type=str,
        help="Path to background sequence dataset FASTA format file. "
        "This file is created using create_bkg_seqs.py script [REQUIRED]",
    )
    parser.add_argument(
        "bkg_shape_fasta_file",
        metavar="/path/to/background_sequences_shape.fasta.*",
        nargs=5,
        help="Path to 3D DNA shape (DNAShapeR output files) data FASTA format file "
        "associated with '--bkg_fasta_file' parameters [REQUIRED]",
    )
    parser.add_argument(
        "genome_wide_shape_fasta_file",
        nargs=5,
        help="Path to genome-wide 3D DNA shape (DNAShapeR output files) "
        "data single-line FASTA format files associated with '--predict' "
        "parameters [REQUIRED]",
    )
    parser.add_argument(
        "save_training_data",
        metavar="/path/to/tf_training_dataset.feather",
        type=str,
        help="Specify location and name of the file to save training data"
        " table. Training data will be saved in feather format. "
        "For more details about feather format, please check "
        "https://github.com/wesm/feather/tree/master/python [REQUIRED]",
    )
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
    except AssertionError as e:
        exit(
            "Could not create a mapping between FASTA headers and input numerical array."
            "\n{0}".format(e)
        )
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

    print("#" * 90, strftime("%x %X | BUILD FEATURE TABLE"), sep="\n")
    args = handle_program_options()

    try:
        assert isfile(args.fg_fasta_file)
        assert isfile(args.bkg_fasta_file)
        assert len(args.genome_wide_shape_fasta_file) == 5
        assert len(args.bkg_shape_fasta_file) == 5
    except AssertionError as err:
        print("Error: Please check supplied FASTA file\n{0}".format(err))
        exit()

    ###################################
    # Processing foreground sequences #
    ###################################
    print("\n", strftime("%x %X | Processing foreground FASTA file"))
    print("=" * 90, sep="\n")
    positive_data_df = build_feature_table(
        args.fg_fasta_file,
        args.fg_fasta_file,
        args.fg_bed_file,
        args.genome_wide_shape_fasta_file,
    )
    positive_data_df.insert(0, "seq_type", "True")

    ###################################
    # Processing background sequences #
    ###################################
    print(strftime("%x %X | Processing background FASTA file"))
    print("=" * 90, sep="\n")

    # get all background sequences
    print(strftime("%x %X | Reading FASTA file"))
    bkg_seqs = defaultdict(dict)
    bkg_seqs[args.protein_name]["header"] = np.asarray(
        [name for name, seq in parse_fasta(args.bkg_fasta_file)]
    )
    bkg_seqs[args.protein_name]["seqs"] = np.asarray(
        [seq for name, seq in parse_fasta(args.bkg_fasta_file)]
    )

    # get GC percent for all background sequences
    print(strftime("%x %X | Calculating GC percent"))
    bkg_gc = {
        tf: map_headers_to_values(
            data["header"], list(map(calculate_gc_percent, data["seqs"]))
        )
        for tf, data in bkg_seqs.items()
    }

    # calculating poisson based metrics
    print(strftime("%x %X | Calculating Poisson based metrics"))
    fg_seqs = defaultdict(dict)
    fg_seqs[args.protein_name]["header"] = np.asarray(
        [name for name, seq in parse_fasta(args.fg_fasta_file)]
    )
    fg_seqs[args.protein_name]["seqs"] = np.asarray(
        [seq for name, seq in parse_fasta(args.fg_fasta_file)]
    )
    bkg_seq_pairs = all_possible_seq_pairs(
        bkg_seqs[args.protein_name]["seqs"], fg_seqs[args.protein_name]["seqs"]
    )
    bkg_poisson_metrics = np.asarray(
        [
            np.asarray(list(starmap(pac, pair_set))).mean(axis=0, dtype=np.float64)
            for pair_set in bkg_seq_pairs
        ]
    )

    bkg_pac = map_headers_to_values(
        bkg_seqs[args.protein_name]["header"], bkg_poisson_metrics
    )

    # collate all DNA shape values
    print(strftime("%x %X | Processing DNA shape data"))
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

    print(strftime("%x %X | Creating background training dataset"))
    gc_data_df = pd.DataFrame.from_dict(
        bkg_gc[args.protein_name], orient="index", columns=["GC_percent"]
    )
    pac_data_df = pd.DataFrame.from_dict(
        bkg_pac, orient="index", columns=["PAS", "PPS"]
    )
    shapes_data_df = pd.DataFrame.from_dict(bkg_shapes, orient="index")
    negative_data_df = gc_data_df.merge(
        pac_data_df, how="outer", left_index=True, right_index=True
    )
    negative_data_df = negative_data_df.merge(
        shapes_data_df, how="outer", left_index=True, right_index=True
    )
    negative_data_df.insert(0, "seq_type", "Not_True")

    ###########################################
    # Save training dataset to a feather file #
    ###########################################
    if args.save_training_data:
        training_data = pd.concat([positive_data_df, negative_data_df], sort=False)
        training_data = training_data.dropna(axis=1)  # drop columns with any NaN
        # convert row index to a column called 'index', since feather format doesn't
        # support row indexing
        training_data = training_data.reset_index()
        print(
            strftime(
                "%x %X | Saving {0} background training dataset to {1}".format(
                    args.protein_name, args.save_training_data
                )
            )
        )
        training_data.to_feather(args.save_training_data)

    print(strftime("%x %X | END BUILD FEATURE TABLE"))


if __name__ == "__main__":
    exit(main())
