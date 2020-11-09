#!/usr/bin/env python3

"""
Cross validate linear SVC model using training data.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.3.0"


import argparse
from os.path import isfile
from sys import exit
from time import strftime

from joblib import dump
from utils import permutation_result, plot_coefficients

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
    from sklearn.decomposition import PCA
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        fbeta_score,
        make_scorer,
        matthews_corrcoef,
        precision_recall_curve,
    )
    from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
    from sklearn.svm import SVC
except ImportError:
    err.append("scikit-learn")
try:
    assert len(err) == 0
except AssertionError:
    for error in err:
        print(f"Please install {error}")
    exit()


def handle_program_options():
    parser = argparse.ArgumentParser(
        description="Using training data saved in 'feather' files, this script will "
        "create a SVC classifier model, tune the hyperparameters using "
        "RandomizedSearchCV, and plot the confusion matrix for test data. The input data "
        "will be divided into training, validation and testing data. The SVC model will "
        "be optimized on training and validation data, while the testing data will be "
        "used for generating the confusion matrix. Please refer to scikit-learn's user "
        "guide for additional information on model selection and cross-validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "read_training_data",
        type=str,
        metavar="/path/to/tf_training_dataset.feather",
        help="Specify location and name of training data feather format "
        "file. This file can(should) be generated using build_features.py"
        " script [REQUIRED]",
    )
    parser.add_argument(
        "protein_name",
        type=str,
        choices=["bcr1", "brg1", "efg1", "ndt80", "rob1", "tec1"],
        help="Specify the name of transcription factor. Please see the "
        "list of valid choices for this parameter [REQUIRED]",
    )
    parser.add_argument(
        "save_model_file",
        type=str,
        metavar="path/to/tf_model.pkl.z",
        help="Specify location and name of the model file. Model file "
        "will be saved as 'proteinname_model.pkl.z' as a compressed "
        "pickle file [REQUIRED]",
    )
    parser.add_argument(
        "save_prc_plot",
        type=str,
        metavar="/path/to/precision_recall_plot.pdf",
        help="Specify location and name of the file to save the precision "
        "recall plot. By default, the plots will be saved in the format "
        "specified in the file ending by the user. E.g. the "
        "'precision_recall_plot.pdf' file will be saved as PDF "
        "file. For more information about file types, please read the "
        "'format' attribute of figure.savefig function on matplotlib's "
        "documentation [REQUIRED]",
    )
    parser.add_argument(
        "save_permute_test",
        type=str,
        metavar="/path/to/permutation_test_plot.svg",
        help="Specify location and name of the file to save the results "
        "of permutation testing. By default, the plots will be saved in "
        "the format specified in the file ending by the user. E.g. the "
        "'permutation_test_plot.svg' file will be saved as SVG "
        "file. For more information about file types, please read the "
        "'format' attribute of figure.savefig function on matplotlib's "
        "documentation [REQUIRED]",
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


def main():

    print("#" * 90, "\n\n", strftime("%x %X | START CROSS VALIDATION\n"))
    args = handle_program_options()

    # Check input validity
    try:
        assert isfile(args.read_training_data)
    except AssertionError as e:
        print(f"Input feather file do not exist. Please check supplied file - {e}")
        exit()
    else:
        file_formats = ["pdf", "svg", "png", "jpg", "tiff", "eps", "ps"]
        for argument in [
            args.save_prc_plot,
            args.save_permute_test,
            args.plot_feature_contribution,
        ]:
            output_format = argument.split("/")[-1].split(".")[-1]
            try:
                assert output_format in file_formats
            except AssertionError:
                print(
                    f"Error: Please check the output file format provided. '{output_format}' format is not supported in {argument}."
                )
        protein_name = args.protein_name.capitalize()
        #######################################
        # Read in training data feather files #
        #######################################
        try:
            print(strftime("%x %X | Reading input feather data file"))
            training_data = (
                pd.read_feather(args.read_training_data)
                .drop(columns=["index"])
                .set_index("location", verify_integrity=True)
            )
        except Exception as e:
            print(f"Error: Please check input file {args.read_training_data}\n{e}")
            exit()

    ######################################
    # Set up inputs for cross validation #
    ######################################
    X = training_data.iloc[:, 2:].to_numpy()
    y = training_data["seq_type"].to_numpy()

    # Encode y labels and power transform X to make it more Gaussian
    y_encoded = np.ravel(label_binarize(y, classes=["Not_True", "True"]))
    pipe = Pipeline(
        [
            ("scale", MinMaxScaler(copy=False)),
            ("standardize", StandardScaler(copy=False)),
        ]
    )
    dump_file = {"scaler": pipe}
    X_transformed = pipe.fit_transform(X)

    ###################################################################
    # Split data into training and testing set for RandomizedSearchCV #
    ###################################################################
    print(strftime("%x %X | Tuning the hyper-parameters for classifier"))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=39)
    train_indices = list(sss.split(X_transformed, y_encoded))[0][0]
    test_indices = list(sss.split(X_transformed, y_encoded))[0][1]
    X_train, X_test = X_transformed[train_indices], X_transformed[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    #################################
    # Build an optimized classifier #
    #################################
    tuning_parameters = {
        "kernel": ["linear", "rbf"],
        "C": np.logspace(-2, 3, 1000),
        "gamma": np.logspace(-3, 2, 5).tolist() + ["scale", "auto"],
    }
#     tuning_parameters = {
#         "kernel": ["linear"],
#         "C": np.logspace(-2, 3, 1000),
#     }
    cv = StratifiedShuffleSplit(n_splits=25, test_size=0.2, random_state=39)
    scorers = {
        "Cohens_kappa": make_scorer(cohen_kappa_score),
        "Average_precision": make_scorer(average_precision_score),
        "F_beta_2": make_scorer(fbeta_score, beta=2),
        "MCC": make_scorer(matthews_corrcoef),
    }
    clf = SVC(
        cache_size=500, probability=True, class_weight="balanced", random_state=39,
    )
    grid_search = GridSearchCV(
        clf,
        tuning_parameters,
        scoring=scorers,
        n_jobs=-1,
        cv=cv,
        refit="Average_precision",
    )
    search = grid_search.fit(X_train, y_train)
    print(
        strftime(
            f"%x %X | Best parameters set found on development set: {search.best_params_}"
        )
    )
    y_pred = search.predict(X_test)
    y_score = search.decision_function(X_test)
    print(
        strftime(
            f"%x %X | Classification scores for {protein_name} - (higher percent is better)"
        )
    )
    print(
        strftime(
            f"%x %X | Mean cross-validated score of the best_estimator: {search.best_score_:.2%}"
        )
    )
    for scorer in scorers.keys():
        cv_key = f"mean_test_{scorer}"
        mean_score = 100 * np.mean(search.cv_results_[cv_key])
        cv_key = f"std_test_{scorer}"
        std_score = np.mean(search.cv_results_[cv_key])
        print(f"{'': >20}Mean {scorer} score: {mean_score: 0.2F} +/- {std_score: 0.2%}")

    ###############################################
    # Feature contribution towards classification #
    ###############################################
    print(
        strftime(
            f"%x %X | Saving feature importance ranking plot to {args.plot_feature_contribution}"
        )
    )
    res = permutation_importance(
        search.best_estimator_, X_test, y_test, n_repeats=25, n_jobs=-1, random_state=39
    )
    plot_coefficients(
        res,
        training_data.columns[2:].tolist(),
        protein_name,
        args.plot_feature_contribution,
    )

    ##############################
    # Save trained model to file #
    ##############################
    try:
        assert args.save_model_file.endswith(".z")
    except AssertionError:
        # add compression file ending
        args.save_model_file += ".z"
    else:
        print(strftime(f"%x %X | Saving model file to {args.save_model_file}"))
        dump_file["search"] = search
        dump(dump_file, args.save_model_file, compress=9, protocol=-1)

    ############################################################
    # Plot precision recall curve for training data (80% data) #
    ############################################################
    print(strftime(f"%x %X | Saving precision recall curves to {args.save_prc_plot}"))
    acc_score = balanced_accuracy_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_pred, average="weighted")
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    with mpl.style.context("fast"):
        plt.figure(figsize=(8, 8), edgecolor="k", tight_layout=True)
        plt.step(recall, precision, where="post", lw=2, color="b", alpha=1)
        plt.fill_between(recall, precision, alpha=0.2, color="b", step="post")
        plt.xlabel("Recall (Proportion of true samples recovered)", color="k", size=20)
        plt.ylabel(
            "Precision (Proportion of correct classification)", color="k", size=20
        )
        plt.ylim([0.0, 1.1])
        plt.xlim([0.0, 1.1])
        plt.title(
            f"{avg_precision: 0.2%} precision | {acc_score: 0.2%} accuracy",
            fontsize=20,
            color="k",
        )
        plt.figtext(
            0.91,
            0.91,
            f"{args.protein_name.capitalize()}",
            c="w",
            backgroundcolor="k",
            size=20,
            weight="bold",
            ha="center",
            va="center",
        )
        plt.savefig(
            args.save_prc_plot,
            dpi=300.0,
            format=output_format,
            edgecolor="k",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    ##############################################################
    # Test significance of classification using permutation test #
    ##############################################################
    print(strftime("%x %X | Running permutation tests to assess model accuracy"))
    permutation_result(
        protein_name,
        search.best_estimator_,
        X_transformed,
        y_encoded,
        cv=cv,
        file=args.save_permute_test,
    )

    print(strftime("\n%x %X | END CROSS VALIDATION\n"))


if __name__ == "__main__":
    exit(main())
