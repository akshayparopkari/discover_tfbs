#!/usr/bin/env python3

"""
Using trained model and input FASTA file, classify each sequence as a true or false TF
binding site.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.1.7"


import argparse
from collections import defaultdict
from itertools import product, starmap
from os.path import isfile
from sys import exit
from time import strftime

from joblib import load

from utils import calculate_gc_percent, get_shape_data, pac, parse_fasta

err = []
try:
    import matplotlib as mpl
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
    assert len(err) == 0
except AssertionError:
    for error in err:
        print("Please install {0}".format(error))
    exit()


def handle_program_options():
    parser = argparse.ArgumentParser(
        description="Using trained model and an input FASTA file of potential binding "
        "sites, predict True vs. False binding sequence from input FASTA file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "prediction_FASTA",
        type=str,
        metavar="path/to/sequences_to_predict.fasta",
        help="Specify location and name of FASTA file whose sequences "
        "are to be classified by the model [REQUIRED]",
    )
    parser.add_argument(
        "prediction_BED",
        type=str,
        metavar="path/to/entries_to_predict.bed",
        help="Specify location and name of BED6 file whose entries "
        "are to be classified by the model. This file contains same "
        "information as the file for `read_prediction_FASTA` option "
        "[REQUIRED]",
    )
    parser.add_argument(
        "fg_fasta_file",
        type=str,
        metavar="/path/to/true_binding_seqeuences.fasta",
        help="Specify location and name of foreground/true positive "
        "sequence FASTA file [REQUIRED]",
    )
    parser.add_argument(
        "model_file",
        type=str,
        metavar="/path/to/model_file.pkl.z",
        help="Specify location and name of model pickle file associated "
        "with 'protein_name'. This file can(should) be generated using "
        "cross_validate.py script [REQUIRED]",
    )
    parser.add_argument(
        "read_training_data",
        type=str,
        metavar="/path/to/tf_training_dataset.feather",
        help="Specify location and name of training data used to build "
        "`model_file` feather format file. [REQUIRED]",
    )
    parser.add_argument(
        "protein_name",
        type=str,
        choices=["bcr1", "brg1", "efg1", "ndt80", "rob1", "tec1"],
        help="Specify the name of transcription factor. Please see the "
        "list of valid choices for this parameter [REQUIRED]",
    )
    parser.add_argument(
        "genome_wide_shape_fasta_file",
        nargs="+",
        metavar="/path/to/organism_genome_shape.fasta.*",
        help="Path to Candida albicans genome-wide 3D DNA shape "
        "(DNAShapeR output files) data single-line FASTA format files "
        "[REQUIRED]",
    )
    parser.add_argument(
        "save_feature_table",
        type=str,
        metavar="/path/to/tf_feature_table.feather",
        help="Specify location and name of the file to save the feature table "
        "to be used as input for classification. The results will be saved in feather "
        "format [REQUIRED]",
    )
    parser.add_argument(
        "prediction_results",
        type=str,
        metavar="/path/to/true_binding_site_predictions.bed",
        help="Specify location and name of the file to save the results "
        "of classification prediction. The results will be saved in BED6 "
        "format [REQUIRED]",
    )
    parser.add_argument(
        "plot_feature_contribution",
        type=str,
        metavar="/path/to/feature_contribution.pdf",
        help="Specify location and name of the file to save the contribution "
        "of all feature towards classification. By default, the plots will be saved in "
        "the format specified in the file ending by the user. E.g. the "
        "'feature_contribution.pdf' file will be saved as PDF "
        "file. For more information about file types, please read the "
        "'format' attribute of figure.savefig function on matplotlib's "
        "documentation [REQUIRED]",
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


def plot_coefficients(coef, feature_names, file: str):
    """
    Using the coefficient weights, plot the contribution of each feature in
    classification. Currently, this function is set up for binary classification.

    :type coef: array-like, list or numpy array
    :param coef: SVM weights assigned to each feature.

    :type names: list
    :param names: List of feature names to use for plotting

    :type file: str
    :param file: Path and name of file to save feature contribution bar plot. The file
                 will be saved in PDF format.
    """
    coef = coef.ravel()
    feature_names = np.array(feature_names)
    top_coefficients = np.argsort(coef)

    # create plot
    with mpl.style.context("ggplot"):
        plt.figure(figsize=(10, 10))
        colors = [
            "#b20000" if c < 0 else "#008000" for c in coef[top_coefficients][::-1]
        ]
        plt.barh(
            range(len(coef)),
            coef[top_coefficients][::-1],
            color=colors,
            hatch="//",
            tick_label=feature_names[top_coefficients][::-1],
        )
        plt.savefig(file, dpi=300, format="pdf", bbox_inches="tight")


def main():

    print("#" * 90, strftime("%x %X | START CLASSIFICATION\n"), sep="\n\n")
    args = handle_program_options()

    # Check input validity
    try:
        assert isfile(args.prediction_FASTA)
        assert isfile(args.prediction_BED)
        assert isfile(args.fg_fasta_file)
        assert isfile(args.model_file)
        assert isfile(args.read_training_data)
        assert len(set(args.genome_wide_shape_fasta_file)) == 5
    except Exception as e:
        print("Error with input file(s). Please review the error\n{0}".format(e))
        exit()
    else:
        file_formats = ["pdf", "svg", "png", "jpg", "tiff", "eps", "ps"]
        output_format = args.plot_feature_contribution.split("/")[-1].split(".")[-1]
        try:
            assert output_format in file_formats
        except AssertionError:
            print(
                "\nError: Please check the output file format provided. '{0}' format is "
                "not supported in {1}.".format(
                    output_format, args.plot_feature_contribution
                )
            )
        protein_name = args.protein_name.capitalize()

        ######################################
        # Read in training data feather file #
        ######################################
        try:
            training_data = pd.read_feather(args.read_training_data)
        except Exception:
            print("Error: Please check input file {0}".format(args.read_training_data))
            exit()
        else:
            # feather file reading successful, collect feature set
            training_features = list(training_data.columns[2:])

        ######################
        # Read in FASTA file #
        ######################
        prediction_data = defaultdict(str)
        prediction_data = {
            name: seq for name, seq in parse_fasta(args.prediction_FASTA)
        }

    ####################################################
    # Calculating GC content and poisson based metrics #
    ####################################################
    print(strftime("%x %X | Calculating GC percent"))
    pred_gc = map_headers_to_values(
        prediction_data.keys(),
        np.fromiter(map(calculate_gc_percent, list(prediction_data.values())), float),
    )

    print(strftime("%x %X | Calculating Poisson based metrics"))
    fg_seqs = [seq for header, seq in parse_fasta(args.fg_fasta_file)]
    prediction_data_pairs = all_possible_seq_pairs(prediction_data.values(), fg_seqs)
    pred_poisson_metrics = np.asarray(
        [
            np.asarray(list(starmap(pac, pair_set))).mean(axis=0, dtype=np.float64)
            for pair_set in prediction_data_pairs
        ]
    )
    pred_pac = map_headers_to_values(prediction_data.keys(), pred_poisson_metrics)

    #####################################################
    # Calculating 3d DNA shape data for input sequences #
    #####################################################
    print(strftime("%x %X | Calculating DNA shape data"))
    pred_shapes = get_shape_data(args.prediction_BED, args.genome_wide_shape_fasta_file)

    #############################
    # Collect data in DataFrame #
    #############################
    print(strftime("%x %X | Creating prediction dataset"))
    gc_data_df = pd.DataFrame.from_dict(pred_gc, orient="index", columns=["GC_percent"])
    pac_data_df = pd.DataFrame.from_dict(
        pred_pac, orient="index", columns=["PAS", "PPS"]
    )
    shapes_data_df = pd.DataFrame.from_dict(pred_shapes, orient="index")
    prediction_data_df = gc_data_df.merge(
        pac_data_df, how="outer", left_index=True, right_index=True
    )
    prediction_data_df = prediction_data_df.merge(
        shapes_data_df, how="outer", left_index=True, right_index=True
    )
    try:
        prediction_data_df = prediction_data_df.loc[:, training_features]
    except KeyError as ke:
        # misalignment of feature space
        prediction_data_features = list(prediction_data_df.columns)
        print(
            strftime(
                "%x %X | Some feature labels were not found in prediction "
                "dataset\n{0}".format(ke)
            )
        )
        if len(prediction_data_features) > len(training_features):
            additional_features = set(prediction_data_features).difference(
                training_features
            )
            print("Features model is unaware of -\n{0}".format(additional_features))
        else:
            missing_features = set(training_features).difference(
                prediction_data_features
            )
            print("Missing features -\n{0}".format(missing_features))
            exit()
    else:
        print(
            strftime(
                "%x %X | Saving feature table (input for model) in {0}".format(
                    args.save_feature_table
                )
            )
        )
        save_predict_data = prediction_data_df.reset_index()
        save_predict_data.to_feather(args.save_feature_table)

        #################################
        # Interpolate / drop NaN values #
        #################################
        dropped_entries = list(
            prediction_data_df[prediction_data_df.isnull().any(axis=1)].index
        )

        original_row_count = prediction_data_df.shape[0]
        prediction_data_df = prediction_data_df.dropna()
        #         new_rows = prediction_data_df.shape[0]
        lost_entries_pct = 100 * (len(dropped_entries) / original_row_count)
        print(
            strftime(
                "%x %X | Losing {0} ({1:0.2f}%) entries due to prescence of NaNs".format(
                    len(dropped_entries), lost_entries_pct
                )
            )
        )
        prediction_data_features = prediction_data_df.to_numpy()
        model = load(args.model_file)
        prediction_data_features = model["scaler"].transform(prediction_data_features)

        ############################################
        # Classify sequences as True vs False TFBS #
        ############################################
        print(strftime("%x %X | Classifying sequences"))
        clf = model["search"].best_estimator_
        try:
            pred_results = clf.predict(prediction_data_features)
        except Exception as e:
            exit("\nError in classification\n{0}".format(e))
        else:
            positive_pred_orfs = prediction_data_df.index.to_numpy()[
                np.where(pred_results)
            ]

        ###########################################################
        #        Save positive predictions in BED format          #
        # 1.chrom 2.chromStart 3.chromEnd 4.name 5.score 6.strand #
        ###########################################################
        print(
            strftime(
                "%x %X | Writing positive prediction results to {0}".format(
                    args.prediction_results
                )
            )
        )
        with open(args.prediction_results, "w") as pred_out:
            for genome_loc in positive_pred_orfs:
                chrom = genome_loc.strip().split(":")[0]
                chromStart = genome_loc.strip().split(":")[1].split("-")[0]
                chromEnd = genome_loc.strip().split(":")[1].split("-")[1].split("(")[0]
                pred_name = "{0}_TFBS".format(protein_name)
                score = "."
                strand = genome_loc.strip().split("(")[1][0]
                pred_out.write(
                    "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
                        chrom, chromStart, chromEnd, pred_name, score, strand
                    )
                )

        ###############################################
        # Feature contribution towards classification #
        ###############################################
        print(
            strftime(
                "%x %X | Saving feature importance ranking plot to {0}".format(
                    args.plot_feature_contribution
                )
            )
        )
        feature_names = [
            entry.replace("_", " ").replace("pos ", "Pos")
            for entry in prediction_data_df.columns.to_numpy()
        ]
        plot_coefficients(clf.coef_, feature_names, args.plot_feature_contribution)

        print(strftime("\n%x %X | END CLASSIFICATION\n"), sep="\n")


if __name__ == "__main__":
    exit(main())
