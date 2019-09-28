#!/usr/bin/env python3

"""
Build feature table from input FASTA files.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.1.7"


import argparse
from sys import exit
from random import sample
from collections import defaultdict
from os.path import isfile, abspath
from time import localtime, strftime
from itertools import product, starmap
from utils import parse_fasta, calculate_gc_percent, pac
err = []
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    mpl.rc("font", family="Arial")
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
    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc, precision_recall_curve,\
        average_precision_score
except ImportError:
    err.append("scikit-learn")
try:
    assert len(err) == 0
except AssertionError:
    for error in err:
        print("Please install {}".format(error))
    exit()


def handle_program_options():
    parser = argparse.ArgumentParser(
        description="Build feature table from input FASTA files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument("protein_name", type=str,
                        choices=["bcr1", "brg1", "efg1", "ndt80", "rob1", "tec1"],
                        help="Name of transcription factor. Please see the list of valid "
                        "choices for this parameter [REQUIRED]")
    parser.add_argument("-cv", "--cross_validation", type=str, default="roc",
                        choices=["roc", "prc"],
                        help="Specify which type pf cross validation curves to output. By"
                        " default, ROC plots will be saved.")
    parser.add_argument("-s", "--savefile", type=str,
                        help="Specify location and filename to save plots as SVG files.")
    parser.add_argument("-o", "--output_file", type=str,
                        help="Specify location and filename to save consolidated data to "
                        "this tab-separated file")
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
        print("Input FASTA file(s) does not exist. Please check supplied FASTA file - {}"
              .format(e))
        exit()

    #######################################
    # FOREGROUND SEQUENCE PROCESSING #
    #######################################
    print("\n", strftime("%x %X".format(localtime)), ": Processing foreground FASTA file")
    print("=" * 53, sep="\n")

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
    print(strftime("%x %X".format(localtime)), ": Processing DNA shape data")
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

    # create dataframe of all features for positive training data
    print(strftime("%x %X".format(localtime)), ": Creating positive training dataset\n")
    gc_data_df = pd.DataFrame.from_dict(fg_gc[args.protein_name], orient="index",
                                        columns=["gc_percent"])
    pac_data_df = pd.DataFrame.from_dict(fg_pac, orient="index",
                                         columns=["poisson_add_sim", "poisson_prod_sim"])
    shapes_data_df = pd.DataFrame.from_dict(fg_shapes, orient="index")
    positive_data_df = gc_data_df.merge(pac_data_df, how="outer",
                                        left_index=True, right_index=True)
    positive_data_df = positive_data_df.merge(shapes_data_df, how="outer",
                                              left_index=True, right_index=True)
    positive_data_df.insert(0, "seq_type", "True")

    ##################################
    # BACKGROUND SEQUENCE PROCESSING #
    ##################################
    print(strftime("%x %X".format(localtime)), ": Processing background FASTA file")
    print("=" * 52, sep="\n")

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
    print(strftime("%x %X".format(localtime)), ": Processing DNA shape data")
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

    # collect balanced dataset for training and prediction
    print(strftime("%x %X".format(localtime)), ": Creating negative training dataset\n")
    if args.cross_validation == "roc":
        sample_count = len(fg_seqs[args.protein_name]["seqs"])  # number of fg seqs
        negative_sample_list = sample(list(bkg_seqs[args.protein_name]["header"]),
                                      sample_count)
        # create a dict which is subset for all features
        gc_subset = {entry: bkg_gc[args.protein_name].get(entry, None)
                     for entry in negative_sample_list}
        pac_subset = {entry: bkg_pac.get(entry, None)
                      for entry in negative_sample_list}
        shapes_subset = {entry: bkg_shapes.get(entry, None)
                         for entry in negative_sample_list}
        gc_data_df = pd.DataFrame.from_dict(gc_subset, orient="index",
                                            columns=["gc_percent"])
        pac_data_df = pd.DataFrame.from_dict(pac_subset, orient="index",
                                             columns=["poisson_add_sim",
                                                      "poisson_prod_sim"])
        shapes_data_df = pd.DataFrame.from_dict(shapes_subset, orient="index")
        negative_data_df = gc_data_df.merge(pac_data_df, how="outer",
                                            left_index=True, right_index=True)
        negative_data_df = negative_data_df.merge(shapes_data_df, how="outer",
                                                  left_index=True, right_index=True)
        negative_data_df.insert(0, "seq_type", "Not_True")
    else:
        # use all background data for PRC calculations
        gc_data_df = pd.DataFrame.from_dict(bkg_gc[args.protein_name], orient="index",
                                            columns=["gc_percent"])
        pac_data_df = pd.DataFrame.from_dict(bkg_pac, orient="index",
                                             columns=["poisson_add_sim",
                                                      "poisson_prod_sim"])
        shapes_data_df = pd.DataFrame.from_dict(bkg_shapes, orient="index")
        negative_data_df = gc_data_df.merge(pac_data_df, how="outer",
                                            left_index=True, right_index=True)
        negative_data_df = negative_data_df.merge(shapes_data_df, how="outer",
                                                  left_index=True, right_index=True)
        negative_data_df.insert(0, "seq_type", "Not_True")

    ############################
    # TRAINING DATA PROCESSING #
    ############################
    print("*" * 53, sep="\n")
    print(strftime("%x %X".format(localtime)), ": Starting 10-fold cross-validation")
    random_state = np.random.RandomState(0)
    training_data = pd.concat([positive_data_df, negative_data_df])
    training_data = shuffle(training_data, random_state=random_state)
    X = training_data.iloc[:, 1: training_data.shape[1]].values
    y = training_data["seq_type"].values
    y = np.ravel(label_binarize(y, classes=["True", "Not_True"]))

    # tuning parameters for SVC
    C_range = np.logspace(-10, 10, base=2)
    param_grid = dict(C=C_range)
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=random_state)
    grid = GridSearchCV(SVC(kernel="linear"), param_grid=param_grid, cv=cv)
    grid.fit(X, y)
    svc = SVC(C=grid.best_params_["C"], kernel="linear", probability=True)

    if args.cross_validation == "roc":
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        print(strftime("%x %X".format(localtime)), ": Plotting and saving ROC data\n")
        with mpl.style.context("ggplot"):
            plt.figure(figsize=(7, 7))
            for train, test in cv.split(X, y):
                probas_ = svc.fit(X[train], y[train]).predict_proba(X[test])
                fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)
            plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Random",
                     alpha=0.8)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=0.8,
                     label=r'Mean ROC (AUC = {0:0.1f} $\pm$ {1:0.2f})'.format(mean_auc,
                                                                              std_auc))
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color="b", alpha=0.2,
                             label=r"$\pm$ 1 std. dev.")
            plt.figtext(0.35, 0.28, args.protein_name.capitalize(), color="k",
                        fontsize=16)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.xlabel("False Positive Rate", color="k", size=12)
            plt.ylabel("True Positive Rate", color="k", size=12)
            plt.title("{} ROC".format(args.protein_name.capitalize()))
            plt.legend(loc="lower right", fontsize=12)
            plt.savefig(args.savefile, dpi=300, format="pdf", bbox_inches="tight",
                        pad_inches=0.1)
    else:
        # return PRC plots
        print(strftime("%x %X".format(localtime)), ": Plotting and saving PRC data\n")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=random_state)
        svc.fit(X_train, y_train)
        y_score = svc.decision_function(X_test)
        precision = dict()
        recall = dict()
        average_precision = dict()
        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
                                                                        y_score.ravel())
        average_precision["micro"] = average_precision_score(y_test, y_score,
                                                             average="micro")
        with mpl.style.context("ggplot"):
            plt.figure(figsize=(7, 7))
            plt.step(recall["micro"], precision["micro"], color="b", alpha=1,
                     where="post")
            plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color="b",
                             step="post")
            plt.xlabel("Recall", color="k", size=12)
            plt.ylabel("Precision", color="k", size=12)
            plt.ylim([0.0, 1.0])
            plt.xlim([0.0, 1.0])
            plt.title("Average precision score for {}: {:0.2f}".
                      format(args.protein_name.capitalize(), average_precision["micro"]))
            plt.savefig(args.savefile, dpi=300, format="pdf", bbox_inches="tight",
                        pad_inches=0.1)


if __name__ == "__main__":
    exit(main())
