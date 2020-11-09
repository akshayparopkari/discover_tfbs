#!/usr/bin/env python3

"""
Build feature table from input FASTA files.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.4.0"


import argparse
from itertools import product
from os.path import abspath, isfile
from sys import exit
from time import strftime

from utils import build_feature_table, parse_fasta

err = []
try:
    from matplotlib import pyplot as plt

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
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    plt.switch_backend("agg")
except ImportError:
    err.append("matplotlib")
try:
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
except ImportError:
    err.append("sklearn")
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
    parser.add_argument(
        "save_pca_plot",
        type=str,
        metavar="/path/to/pca_plot.pdf",
        help="Specify location and name of the file to save the principal"
        " component plot. By default, the plots will be saved in the format "
        "specified in the file ending by the user. E.g. the 'pca_plot.pdf' "
        "file will be saved as PDF file. For more information about file types"
        ", please read the 'format' attribute of figure.savefig function "
        "on matplotlib's documentation [REQUIRED]",
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
    return (list(product(fg_seqs, [seq])) for seq in list1)


def main():

    print("#" * 90, strftime("%x %X | BUILD FEATURE TABLE\n"), sep="\n\n")
    args = handle_program_options()

    try:
        assert isfile(args.fg_fasta_file)
        assert isfile(args.bkg_fasta_file)
        assert len(args.genome_wide_shape_fasta_file) == 5
        assert len(args.bkg_shape_fasta_file) == 5
    except AssertionError as err:
        print("Error: Please check supplied FASTA file\n{0}".format(err))
        exit()
    else:
        output_format = args.save_pca_plot.split("/")[-1].split(".")[-1]
        try:
            assert output_format in ["pdf", "svg", "png", "jpg", "tiff", "eps", "ps"]
        except AssertionError:
            print(
                "Error: Please check the output file format provided. '{0}' format is"
                " not supported in {1}.".format(output_format, args.output_format)
            )

    ###################################
    # Processing foreground sequences #
    ###################################
    print(strftime("%x %X | Processing foreground FASTA file"))
    shape_data = {
        file.split(".")[-1]: {
            header: seq.strip().split(",") for header, seq in parse_fasta(file)
        }
        for file in args.genome_wide_shape_fasta_file
    }
    print("=" * 90, sep="\n")
    positive_data_df = build_feature_table(
        args.fg_fasta_file, args.fg_fasta_file, shape_data, minhash=True
    )
    positive_data_df.insert(1, "seq_type", "True")

    ###################################
    # Processing background sequences #
    ###################################
    print(strftime("\n%x %X | Processing background FASTA file"))
    print("=" * 90, sep="\n")
    negative_data_df = build_feature_table(
        args.bkg_fasta_file, args.fg_fasta_file, minhash=True
    )

    # collate all DNA shape values
    print(strftime("%x %X | Processing DNA shape data"))
    bkg_shapes = dict()
    for shapefile in args.bkg_shape_fasta_file:
        whichshape = shapefile.split(".")[-1]
        if whichshape in ["MGW", "ProT", "EP"]:
            for name, shape in parse_fasta(abspath(shapefile)):
                shape = shape.strip().split(",")
                if not bkg_shapes.get(name):
                    bkg_shapes[name] = dict()
                for i, val in enumerate(shape[2:-2]):
                    position = "{0}_{1:02d}".format(whichshape, i + 1)
                    try:
                        bkg_shapes[name][position] = float(val)
                    except Exception:
                        bkg_shapes[name][position] = 0.0
        else:
            # shape is Roll or HelT
            for name, shape in parse_fasta(abspath(shapefile)):
                shape = shape.split(",")
                if not bkg_shapes.get(name):
                    bkg_shapes[name] = dict()
                for i, val in enumerate(shape[1:-1]):
                    position = "{0}_{1:02d}".format(whichshape, i + 1)
                    try:
                        bkg_shapes[name][position] = float(val)
                    except Exception:
                        bkg_shapes[name][position] = 0.0

    print(strftime("%x %X | Creating background training dataset"))
    shapes_data_df = (
        pd.DataFrame.from_dict(bkg_shapes, orient="index")
        .reset_index()
        .rename(columns={"index": "location"})
    )
    negative_data_df = negative_data_df.merge(
        shapes_data_df, how="left", left_on="location", right_on="location"
    )
    negative_data_df.insert(1, "seq_type", "Not_True")

    ###########################################
    # Save training dataset to a feather file #
    ###########################################
    training_data = pd.concat([positive_data_df, negative_data_df], sort=False)
    training_data = training_data.dropna(axis=1)  # drop columns with any NaN
    # convert row index to a column called 'index', since feather format doesn't
    # support row indexing
    training_data = training_data.reset_index()
    print(
        strftime(
            f"%x %X | Saving {args.protein_name} background training dataset to {args.save_training_data}"
        )
    )
    training_data.to_feather(args.save_training_data)
    # switch back
    training_data = training_data.drop(columns=["index"]).set_index(
        "location", verify_integrity=True
    )

    ##################################
    # Plot PCA for training data set #
    ##################################
    print(
        strftime(
            "%x %X | Saving {0} foreground vs background PCA plot to {1}".format(
                args.protein_name, args.save_pca_plot
            )
        )
    )
    X = training_data.iloc[:, 2:].to_numpy()
    y = training_data["seq_type"].tolist()
    y_encoded = np.ravel(label_binarize(y, classes=["Not_True", "True"]))
    marker = ["x" if m == 0 else "o" for m in y_encoded]
    size = ["#0066FF" if m == 0 else "#FFFFFF" for m in y_encoded]
    colors = []
    pipe = Pipeline(
        [
            ("scale", MinMaxScaler(copy=False)),
            ("standardize", StandardScaler(copy=False)),
        ]
    )
    X_scaled = pipe.fit_transform(X)
    pca = PCA(n_components=2, whiten=True, random_state=39)
    X_transformed = pca.fit_transform(X_scaled)
    pc1, pc2 = tuple(pca.explained_variance_ratio_)
    with mpl.style.context("fast"):
        plt.figure(figsize=(10, 7), edgecolor="k", tight_layout=True)
        for entry, label in zip(X_transformed, y_encoded):
            if label == 0:
                marker = "x"
                size = 50
                color = "#0066FF"
            else:
                marker = "o"
                size = 100
                color = "#000000"
            plt.scatter(
                entry[0], entry[1], s=size, c=color, marker=marker, alpha=0.5,
            )
        plt.figtext(
            0.135,
            0.935,
            f"{args.protein_name.capitalize()}",
            c="w",
            backgroundcolor="k",
            size=20,
            weight="bold",
            ha="center",
            va="center",
        )
        plt.xlabel(f"PC1 (explained variance = {pc1:0.2%})", fontsize=20, color="k")
        plt.ylabel(f"PC2 (explained variance = {pc2:0.2%})", fontsize=20, color="k")
        plt.savefig(
            args.save_pca_plot,
            dpi=300.0,
            format=output_format,
            edgecolor="k",
            bbox_inches="tight",
            pad_inches=0.2,
        )

    print(strftime("\n%x %X | END BUILD FEATURE TABLE\n"))


if __name__ == "__main__":
    exit(main())
