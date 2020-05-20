#!/usr/bin/env python3

"""
Create confusion matrices from training data.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.1.6"


import argparse
from os.path import isfile
from sys import exit
from time import strftime

from joblib import dump
from utils import permutation_result

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
    from sklearn.svm import SVC
    from sklearn.model_selection import (
        RandomizedSearchCV,
        StratifiedShuffleSplit,
    )
    from sklearn.preprocessing import LabelEncoder, PowerTransformer
    from sklearn.metrics import (
        balanced_accuracy_score,
        matthews_corrcoef,
        make_scorer,
        plot_confusion_matrix,
    )
except ImportError:
    err.append("scikit-learn")
try:
    assert len(err) == 0
except AssertionError:
    for error in err:
        print("Please install {0}".format(error))
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
        "save_conf_mat",
        type=str,
        metavar="/path/to/confusion_matrix_plot.pdf",
        help="Specify location and name of the file to save the confusion"
        " matrix plot. By default, the plots will be saved in the format "
        "specified in the file ending by the user. E.g. the "
        "'confusion_matrix_plot.pdf' file will be saved as PDF "
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
    return parser.parse_args()


def main():

    print("#" * 90, strftime("%x %X | START CROSS VALIDATION\n"), sep="\n\n")
    args = handle_program_options()

    # Check input validity
    try:
        assert isfile(args.read_training_data)
    except AssertionError as e:
        print(
            "Input feather file do not exist. Please check supplied file - {0}".format(
                e
            )
        )
        exit()
    else:
        file_formats = ["pdf", "svg", "png", "jpg", "tiff", "eps", "ps"]
        for argument in [args.save_conf_mat, args.save_permute_test]:
            output_format = argument.split("/")[-1].split(".")[-1]
            try:
                assert output_format in file_formats
            except AssertionError:
                print(
                    "Error: Please check the output file format provided. '{0}' format is"
                    " not supported in {1}.".format(output_format, argument)
                )
        protein_name = args.protein_name.capitalize()
        #######################################
        # Read in training data feather files #
        #######################################
        try:
            print(strftime("%x %X | Reading input feather data file"))
            training_data = pd.read_feather(args.read_training_data)
        except Exception as e:
            print(
                "Error: Please check input file {0}\\n{1}".format(
                    args.read_training_data, e
                )
            )
            exit()
        else:
            # feather file reading successful
            training_data.set_index("index", inplace=True, verify_integrity=True)

    ######################################
    # Set up inputs for cross validation #
    ######################################
    X = training_data.iloc[:, 1 : training_data.shape[1]].to_numpy()
    y = training_data.iloc[:, 0].to_numpy()

    # Encode y labels and power transform X to make it more Gaussian
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    scaler = PowerTransformer()
    scaler_fit = scaler.fit(X)
    dump_file = {"scaler": scaler_fit}
    X_transformed = scaler.transform(X)

    ###################################################################
    # Split data into training and testing set for RandomizedSearchCV #
    ###################################################################
    print(strftime("%x %X | Tuning the hyper-parameters for classifier"))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_indices = list(sss.split(X_transformed, y_encoded))[0][0]
    test_indices = list(sss.split(X_transformed, y_encoded))[0][1]
    X_train, X_test = X_transformed[train_indices], X_transformed[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    #################################
    # Build an optimized classifier #
    #################################
    clf = SVC(kernel="linear", cache_size=500, class_weight="balanced")
    params = {"C": np.geomspace(0.1, 100, num=1000)}
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    scorers = {
        "Balanced_Accuracy": "balanced_accuracy",
        "MCC": make_scorer(matthews_corrcoef),
    }
    random_search = RandomizedSearchCV(
        clf,
        params,
        n_iter=100,
        scoring=scorers,
        n_jobs=-1,
        cv=cv,
        refit="Balanced_Accuracy",
        random_state=0,
    )
    search = random_search.fit(X_train, y_train)
    y_pred = search.predict(X_test)
    mean_ba = 100 * np.mean(search.cv_results_["mean_test_Balanced_Accuracy"])
    std_ba = np.mean(search.cv_results_["std_test_Balanced_Accuracy"])
    mean_mcc = 100 * np.mean(search.cv_results_["mean_test_MCC"])
    std_mcc = np.mean(search.cv_results_["std_test_MCC"])
    print(
        strftime(
            "%x %X | Classification scores for {0} - (higher percent is better)".format(
                protein_name
            )
        ),
        "{0:>20}Mean accuracy score: {1:0.2f} +/- {2:0.2f}%".format(
            "", mean_ba, std_ba
        ),
        "{0:>20}Mean Matthews correlation coefficient: {1:0.2f} +/- {2:0.2f}%".format(
            "", mean_mcc, std_mcc
        ),
        sep="\n",
    )

    ##############################
    # Save trained model to file #
    ##############################
    try:
        assert args.save_model_file.endswith(".z")
    except AssertionError:
        args.save_model_file += ".z"
    else:
        print(strftime("%x %X | Saving model file to {0}".format(args.save_model_file)))
        dump_file["search"] = search
        dump(dump_file, args.save_model_file, compress=9, protocol=-1)

    ##############################################################
    # Test significance of classification using permutation test #
    ##############################################################
    print(strftime("%x %X | Running permutation tests to assess model accuracy"))
    permutation_result(
        search.best_estimator_,
        X_transformed,
        y_encoded,
        cv=cv,
        file=args.save_permute_test,
    )

    #########################
    # Plot confusion matrix #
    #########################
    print(strftime("%x %X | Saving confusion matrix to {0}".format(args.save_conf_mat)))
    y_pred = search.best_estimator_.predict(X_test)
    acc_score = balanced_accuracy_score(y_test, y_pred)
    disp = plot_confusion_matrix(
        search.best_estimator_,
        X_test,
        y_test,
        display_labels=["Not True", "True"],
        values_format="3d",
        cmap=plt.cm.Purples,
    )
    disp.ax_.set_title(
        "{0} ({1:0.2f}% classification accuracy)".format(protein_name, 100 * acc_score)
    )
    disp.figure_.savefig(
        args.save_conf_mat,
        dpi=300.0,
        format=output_format,
        edgecolor="k",
        bbox_inches="tight",
        pad_inches=0.1,
    )

    print(strftime("\n%x %X | END CROSS VALIDATION\n"))


if __name__ == "__main__":
    exit(main())
