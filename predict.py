#!/usr/bin/env python3

"""
Using trained model and input FASTA file, classify each sequence as a true or false TF
binding site.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.3.5"


import argparse
from itertools import product
from os.path import isfile
from sys import exit
from time import strftime

from joblib import load
from utils import build_feature_table, parse_fasta

err = []
try:
    import numpy as np
    from numpy.linalg import norm
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
        "model_stats",
        type=str,
        metavar="/path/to/tf_model_stats/folder",
        help="Specify location and name of folder to save SVM model statistics "
        "like distance of hyperplane and log probability of classification [REQUIRED]",
    )
    parser.add_argument(
        "roc_like_curve",
        type=str,
        metavar="/path/to/roc_like_curve.pdf",
        help="Specify location and name of the file to save the ROC-like curve "
        "of positive classification. X-axis is positive TFBS predictions and Y-axis "
        "denotes true TFBS [REQUIRED]",
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

    print("#" * 90, "\n\n", strftime("%x %X | START CLASSIFICATION\n"))
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
        protein_name = args.protein_name.capitalize()
        try:
            output_format = args.roc_like_curve.split("/")[-1].split(".")[-1]
            assert output_format in ["pdf", "svg", "png", "jpg", "tiff", "eps", "ps"]
        except AssertionError:
            print(
                f"Error: Please check the output file format provided. '{output_format}' format is not supported in {args.roc_like_curve}."
            )

        ######################################
        # Read in training data feather file #
        ######################################
        try:
            training_data = (
                pd.read_feather(args.read_training_data)
                .drop(columns=["index"])
                .set_index("location", verify_integrity=True)
            )
        except Exception:
            print("Error: Please check input file {0}".format(args.read_training_data))
            exit()
        else:
            # feather file reading successful, collect feature set
            # starting at 2, since location is now row index
            training_features = list(training_data.columns[2:])
            X_train = training_data.iloc[:, 2:].to_numpy()

        ######################
        # Read in shape data #
        ######################
        shape_data = {
            file.split(".")[-1]: {
                header: seq.strip().split(",") for header, seq in parse_fasta(file)
            }
            for file in args.genome_wide_shape_fasta_file
        }

    ##########################################################
    # Calculating GC content and sequence similarity metrics #
    ##########################################################
    prediction_data_df = build_feature_table(
        args.prediction_FASTA, args.fg_fasta_file, shape_data, minhash=True
    ).set_index("location", verify_integrity=True)

    try:
        prediction_data_df = prediction_data_df.loc[:, training_features]
    except KeyError as ke:
        # misalignment of feature space
        prediction_data_features = list(prediction_data_df.columns)
        print(
            strftime(
                f"%x %X | Some feature labels were not found in prediction dataset\n{ke}"
            )
        )
        if len(prediction_data_features) > len(training_features):
            additional_features = set(prediction_data_features).difference(
                training_features
            )
            print(f"Features model is unaware of -\n{additional_features}")
            exit()
        else:
            missing_features = set(training_features).difference(
                prediction_data_features
            )
            print(f"Missing features -\n{missing_features}")
            exit()
    else:
        print(
            strftime(
                f"%x %X | Saving feature table (input for model) in {args.save_feature_table}"
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
        prediction_data_features = (
            model["scaler"].fit(X_train).transform(prediction_data_features)
        )
        X_train_transformed = model["scaler"].fit_transform(X_train)

        ###################################################
        # Classify sequences as either True or False TFBS #
        ###################################################
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
            print(
                strftime(
                    f"%x %X | Model predicted {len(positive_pred_orfs):,}{len(positive_pred_orfs) / len(prediction_data_df.index): .2%} positive TFBS hits"
                )
            )
            ######################################
            # Calculate distance from hyperplane #
            ######################################
            try:
                prediction_decision = np.asarray(
                    clf.decision_function(prediction_data_features) / norm(clf.coef_)
                )
                training_decision = np.asarray(
                    clf.decision_function(X_train_transformed) / norm(clf.coef_)
                )
            except Exception:
                print(
                    f"{'': >20}Error while calculating exact distance from hyperplane. "
                    "Reverting to using relative distance from hyperplane."
                )
                prediction_decision = np.asarray(
                    clf.decision_function(prediction_data_features)
                )
                training_decision = np.asarray(
                    clf.decision_function(X_train_transformed)
                )

            print(
                strftime(f"%x %X | Saving SVM model statistics to {args.model_stats}")
            )
            training_model_stats = pd.DataFrame.from_records(
                np.column_stack(
                    [
                        training_data.index,
                        clf.predict(X_train_transformed),
                        training_decision,
                        clf.predict_log_proba(X_train_transformed),
                    ]
                ),
                columns=[
                    "location",
                    "prediction",
                    "distance_from_hyperplane",
                    "log_probability_of_True_TFBS",
                    "log_probability_of_False_TFBS",
                ],
            ).set_index("location", verify_integrity=True)
            fnh = f"{args.model_stats}{args.protein_name}_training_model_statistics.txt"
            training_model_stats.to_csv(f"{fnh}", sep="\t", na_rep="NA")
            prediction_model_stats = pd.DataFrame.from_records(
                np.column_stack(
                    [
                        prediction_data_df.index,
                        pred_results,
                        prediction_decision,
                        clf.predict_log_proba(prediction_data_features),
                    ]
                ),
                columns=[
                    "location",
                    "prediction",
                    "distance_from_hyperplane",
                    "log_probability_of_True_TFBS",
                    "log_probability_of_False_TFBS",
                ],
            ).set_index("location", verify_integrity=True)
            fnh = (
                f"{args.model_stats}{args.protein_name}_prediction_model_statistics.txt"
            )
            prediction_model_stats.to_csv(f"{fnh}", sep="\t", na_rep="NA")

            # Collect positive training and prediction data
            positive_pred_data = prediction_decision[
                np.where(prediction_decision > 0.0)
            ]
            positive_train_data = training_decision[np.where(training_decision > 0.0)]

            # Collect high-confidence positive TFBS prediction
            min_dist_true_TFBS_from_hyperplane = min(
                training_model_stats["distance_from_hyperplane"][
                    training_model_stats["prediction"] == 1
                ].tolist()
            )
            positive_high_confidence_pred_data = prediction_decision[
                np.where(prediction_decision > min_dist_true_TFBS_from_hyperplane)
            ]

            ########################################################################
            # Plot ROC-like curves for true positive and positively predicted data #
            ########################################################################
            max_dist = max(
                np.r_[positive_high_confidence_pred_data, positive_train_data]
            )
            new_predictions = np.fromiter(
                [
                    np.count_nonzero(positive_high_confidence_pred_data <= dist)
                    for dist in np.linspace(
                        min_dist_true_TFBS_from_hyperplane, max_dist, 100
                    )
                ],
                "int32",
            )
            # transform new_predictions for cleaner axes
            new_predictions = np.sqrt(new_predictions)
            new_predictions = np.true_divide(new_predictions, new_predictions.max())
            true_predictions = np.fromiter(
                [
                    np.count_nonzero(positive_train_data <= dist)
                    for dist in np.linspace(
                        min_dist_true_TFBS_from_hyperplane, max_dist, 100
                    )
                ],
                "int32",
            )
            # transform true_predictions for cleaner axes
            true_predictions = np.sqrt(true_predictions)
            true_predictions = np.true_divide(true_predictions, true_predictions.max())

            print(strftime(f"%x %X | Saving ROC-like plot to {args.roc_like_curve}"))
            with mpl.style.context("fast"):
                plt.figure(figsize=(7, 7), edgecolor="k", tight_layout=True)
                plt.step(
                    new_predictions, true_predictions, lw=2, alpha=1, where="post",
                )
                plt.fill_between(
                    new_predictions, true_predictions, alpha=0.5, step="post"
                )
                plt.xlabel("Positive prediction rate", color="k", size=20)
                plt.ylabel("True positives rate", color="k", size=20)
                plt.figtext(
                    0.2,
                    0.935,
                    f"{protein_name}",
                    c="w",
                    backgroundcolor="k",
                    size=20,
                    weight="bold",
                    ha="center",
                    va="center",
                )
                plt.tight_layout()
                plt.savefig(
                    args.roc_like_curve,
                    dpi=300.0,
                    format=output_format,
                    edgecolor="k",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )

        ###########################################################
        #        Save positive predictions in BED format          #
        # 1.chrom 2.chromStart 3.chromEnd 4.name 5.score 6.strand #
        ###########################################################
        print(
            strftime(
                f"%x %X | Writing positive prediction results to {args.prediction_results}"
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

        print(strftime("\n%x %X | END CLASSIFICATION\n"), sep="\n")


if __name__ == "__main__":
    exit(main())
