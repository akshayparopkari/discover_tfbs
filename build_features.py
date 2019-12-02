#!/usr/bin/env python3

"""
Build feature table from input FASTA files.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.2.3"


import argparse
from sys import exit
from os import cpu_count
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
    from sklearn.model_selection import StratifiedShuffleSplit, permutation_test_score
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
        description="Build feature table from input FASTA files, train a SVC classifier "
        " and perform cross-validation of prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("fg_fasta_file", help="Path to foreground/true positive sequence "
                        "dataset FASTA format file [REQUIRED]")
    parser.add_argument("bkg_fasta_file", help="Path to background/negative sequence "
                        "dataset FASTA format file [REQUIRED]")
    parser.add_argument("-fsff", "--fg_shape_fasta_file", nargs="+", help="Path to 3D DNA"
                        " shape (DNAShapeR output files) data FASTA format files "
                        "associated with '--fg_fasta_file' parameters [REQUIRED]")
    parser.add_argument("-bsff", "--bkg_shape_fasta_file", nargs="+", help="Path to 3D "
                        "DNA shape (DNAShapeR output files) data FASTA format file "
                        "associated with '--bkg_fasta_file' parameters [REQUIRED]")
    parser.add_argument("protein_name", type=str,
                        choices=["bcr1", "brg1", "efg1", "ndt80", "rob1", "tec1"],
                        help="Name of transcription factor. Please see the list of valid "
                        "choices for this parameter [REQUIRED]")
    parser.add_argument("-cv", "--cross_validation", type=str, default=None,
                        choices=["roc", "prc"],
                        help="Specify which type pf cross validation curves to output. By"
                        " default, ROC plots will be saved.")
    parser.add_argument("-p", "--predict", default=None,
                        help="Supply a FASTA file to predict if the sequences in it are "
                        "binding site or not.")
    parser.add_argument("-per", "--permute", type=int, default=None,
                        help="Number of permutations to run for assessing model bias. "
                        "Supply an integer value, and value of 10000 is considered as a "
                        "good estimate. Higher values take longer to run the permutation"
                        "tests.")
    parser.add_argument("-psff", "--predict_shape_fasta_file", nargs="+", help="Path to "
                        "3D DNA shape (DNAShapeR output files) data FASTA format files "
                        "associated with '--predict' parameters [REQUIRED]")
    parser.add_argument("-s", "--savefile", type=str,
                        help="Specify location and filename to save plots as PDF files.")
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
    except AssertionError as e:
        exit("Could not create a mapping between FASTA headers and input numerical array."
             "\n{}".format(e))
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


def f_importances(coef, names: list, file: str, top=-1):
    """
    Using the coefficient weights, plot the contribution of each or subset of features in
    classification. Currently, this function is set up for binary classification.

    :type coef: array-like, list or numpy array
    :param coef: SVM weights assigned to each feature.

    :type names: list
    :param names: List of feature names to use for plotting

    :type file: str
    :param file: Path and name of file to save feature contribution bar plot. The file
                 will be saved in PDF format.

    :type top: int
    :param top: Number of features to plot for visualizing their contribution. Default
                value of -1 will plot all features.
    """
    imp = coef.ravel()
    imp, names = zip(*sorted(list(zip(imp, names))))

    if top == -1:
        # Show all features
        top = len(names)
    colors = ["#008000" if c < 0.0 else "#b20000" for c in imp]
    with mpl.style.context("ggplot"):
        plt.figure(figsize=(10, 8))
        plt.barh(range(top), imp[::-1][0:top], align="center", color=colors)
        plt.yticks(range(top), names[::-1][0:top], fontsize=10)
        plt.tight_layout()
        plt.savefig(file, dpi=300, format="pdf", bbox_inches="tight")


def permutation_result(estimator, X, y, cv, n_permute, random_state, file: str):
    """
    Run permutation tests for classifier and assess significance of accuracy score. This
    is a wrapper around sklearn.model_selection.permutation_test_score

    :type estimator: scikit-learn classifier object
    :param estimator: Instance of scikit-learn initialized classifier which has a 'fit'
                      method

    :type X: array-like, list or numpy array
    :param X: Numpy array of features - columns of feature table

    :type y: array-like, list or numpy array
    :param y: Class labels of each row in X

    :type cv: int, iterable
    :param cv: If integer, those many cross validations are run. User can also supply an
               iterable to create (train, test) splits using indices.

    :type random_state: numpy random object
    :param random_state: Seed to use for multiple reproducible runs

    :type file: str
    :param file: Path and file name to save the bar plot
    """
    score, permutation_score, p_value = permutation_test_score(estimator, X, y,
                                                               scoring="average_precision",
                                                               cv=cv,
                                                               n_permutations=n_permute,
                                                               n_jobs=-1,
                                                               random_state=random_state)
    print(strftime("%x %X:".format(localtime)),
          "Linear SVM classification score {0:0.03f} (pvalue : {1:0.05f})".
          format(score, p_value))
    with mpl.style.context("ggplot"):
        plt.figure(figsize=(10, 8))
        plt.hist(permutation_score, bins=25, alpha=0.5, hatch="//", edgecolor="k",
                 label="Precision scores for shuffled labels")
        ylim = plt.ylim()[1]
        plt.vlines(2 * [1. / np.unique(y).size], 0, ylim, linestyle="dashdot",
                   linewidth=2, label='50/50 chance')
        plt.vlines(2 * [score], 0, ylim, linewidth=3, colors="g")
        score_text = "Model Score\n{:0.03f}*".format(score)
        plt.text(score - 0.05, ylim + 0.075, score_text, ma="center")
        plt.xlim(0.0, 1.0)
        plt.legend(loc=2)
        plt.xlabel("Average precision scores")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(file, dpi=300, format="pdf", bbox_inches="tight")


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
    print("\n", strftime("%x %X:".format(localtime)), "Processing foreground FASTA file")
    print("=" * 53, sep="\n")

    # get all foreground sequences
    print(strftime("%x %X:".format(localtime)), "Reading FASTA file")
    fg_seqs = defaultdict(dict)
    fg_seqs[args.protein_name]["header"] = np.asarray([name
                                                       for name, seq in
                                                       parse_fasta(args.fg_fasta_file)])
    fg_seqs[args.protein_name]["seqs"] = np.asarray([seq
                                                     for name, seq in
                                                     parse_fasta(args.fg_fasta_file)])

    # get GC percent for all foreground sequences
    print(strftime("%x %X:".format(localtime)), "Calculating GC percent")
    fg_gc = {tf: map_headers_to_values(data["header"],
                                       list(map(calculate_gc_percent, data["seqs"])))
             for tf, data in fg_seqs.items()}

    # calculating poisson based metrics
    print(strftime("%x %X:".format(localtime)), "Calculating Poisson based metrics")
    fg_seq_pairs = all_possible_seq_pairs(fg_seqs[args.protein_name]["seqs"],
                                          fg_seqs[args.protein_name]["seqs"])
    fg_poisson_metrics = np.asarray([np.asarray(list(starmap(pac, pair_set))).
                                     mean(axis=0, dtype=np.float64)
                                     for pair_set in fg_seq_pairs])

    fg_pac = map_headers_to_values(fg_seqs[args.protein_name]["header"],
                                   fg_poisson_metrics)

    # collate all DNA shape values
    print(strftime("%x %X:".format(localtime)), "Processing DNA shape data")
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
    print(strftime("%x %X:".format(localtime)), "Creating positive training dataset\n")
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
    print(strftime("%x %X:".format(localtime)), "Processing background FASTA file")
    print("=" * 52, sep="\n")

    # get all background sequences
    print(strftime("%x %X:".format(localtime)), "Reading FASTA file")
    bkg_seqs = defaultdict(dict)
    bkg_seqs[args.protein_name]["header"] = np.asarray([name
                                                       for name, seq in
                                                       parse_fasta(args.bkg_fasta_file)])
    bkg_seqs[args.protein_name]["seqs"] = np.asarray([seq
                                                     for name, seq in
                                                     parse_fasta(args.bkg_fasta_file)])

    # get GC percent for all background sequences
    print(strftime("%x %X:".format(localtime)), "Calculating GC percent")
    bkg_gc = {tf: map_headers_to_values(data["header"],
                                        list(map(calculate_gc_percent, data["seqs"])))
              for tf, data in bkg_seqs.items()}

    # calculating poisson based metrics
    print(strftime("%x %X:".format(localtime)), "Calculating Poisson based metrics")
    bkg_seq_pairs = all_possible_seq_pairs(bkg_seqs[args.protein_name]["seqs"],
                                           fg_seqs[args.protein_name]["seqs"])
    bkg_poisson_metrics = np.asarray([np.asarray(list(starmap(pac, pair_set))).
                                      mean(axis=0, dtype=np.float64)
                                      for pair_set in bkg_seq_pairs])

    bkg_pac = map_headers_to_values(bkg_seqs[args.protein_name]["header"],
                                    bkg_poisson_metrics)

    # collate all DNA shape values
    print(strftime("%x %X:".format(localtime)), "Processing DNA shape data")
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
    print(strftime("%x %X:".format(localtime)), "Creating negative training dataset\n")
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
    print(strftime("%x %X:".format(localtime)), "Starting data training")
    random_state = np.random.RandomState(0)
    training_data = pd.concat([positive_data_df, negative_data_df])
    training_data = shuffle(training_data, random_state=random_state)
    if args.predict:
        # exclude seq type for prediction_data
        X = training_data.iloc[:, 2: training_data.shape[1]].values
    else:
        # don't exclude seq type for accuracy check
        X = training_data.iloc[:, 1: training_data.shape[1]].values
    y = training_data["seq_type"].values
    y = np.ravel(label_binarize(y, classes=["Not_True", "True"]))

    # tuning parameters for SVC
    C_range = np.logspace(-10, 10, base=2)
    param_grid = dict(C=C_range)
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=random_state)
    grid = GridSearchCV(SVC(kernel="linear"), param_grid=param_grid, cv=cv, n_jobs=-1)
    grid.fit(X, y)
    svc = SVC(C=grid.best_params_["C"], kernel="linear", probability=True)

    # Permutation test to calculate significance of model accuracy
    if args.permute:
        print(strftime("%x %X:".format(localtime)),
              "Performing permutation test to assess model accuracy for {}".
              format(args.protein_name))
        permutation_result(svc, X, y, cv, args.permute, random_state, args.savefile)
        exit()

    # predict class of input FASTA data
    if args.predict:
        prediction_data = defaultdict(str)
        prediction_data = {name: seq for name, seq in parse_fasta(args.predict)}

        # calculating GC content and poisson based metrics
        print(strftime("%x %X:".format(localtime)), "Calculating GC percent")
        pred_gc = map_headers_to_values(prediction_data.keys(),
                                        np.fromiter(map(calculate_gc_percent,
                                                        prediction_data.values()),
                                                    float))

        print(strftime("%x %X:".format(localtime)), "Calculating Poisson based metrics")
        prediction_data_pairs = all_possible_seq_pairs(prediction_data.values(),
                                                       fg_seqs[args.protein_name]["seqs"])
        pred_poisson_metrics = np.asarray([np.asarray(list(starmap(pac, pair_set))).
                                          mean(axis=0, dtype=np.float64)
                                          for pair_set in prediction_data_pairs])
        pred_pac = map_headers_to_values(prediction_data.keys(),
                                         pred_poisson_metrics)

        print(strftime("%x %X:".format(localtime)), "Processing DNA shape data")
        pred_shapes = dict()
        for shapefile in args.predict_shape_fasta_file:
            whichshape = shapefile.split(".")[-1]
            if whichshape in ["MGW", "ProT", "EP"]:
                for name, shape in parse_fasta(abspath(shapefile)):
                    shape = shape.split(",")
                    if not pred_shapes.get(name):
                        pred_shapes[name] = dict()
                    for i in range(2, len(shape) - 2):
                        position = "{0}_pos_{1}".format(whichshape, i + 1)
                        if shape[i] == "NA":
                            # if DNAshapeR couldn't calculate the shape, use a value of 0
                            pred_shapes[name][position] = 0.0
                        else:
                            # use DNAshapeR calculated value
                            pred_shapes[name][position] = float(shape[i])
            else:
                # shape is Roll or HelT
                for name, shape in parse_fasta(abspath(shapefile)):
                    shape = shape.split(",")
                    if not pred_shapes.get(name):
                        pred_shapes[name] = dict()
                    for i in range(1, len(shape) - 1):
                        position = "{0}_pos_{1}".format(whichshape, i + 1)
                        if shape[i] == "NA":
                            # if DNAshapeR couldn't calculate the shape, use a value of 0
                            pred_shapes[name][position] = 0.0
                        else:
                            # use DNAshapeR calculated value
                            pred_shapes[name][position] = float(shape[i])

        # collect data in DataFrame
        print(strftime("%x %X:".format(localtime)), "Creating prediction dataset")
        gc_data_df = pd.DataFrame.from_dict(pred_gc, orient="index",
                                            columns=["gc_percent"])
        pac_data_df = pd.DataFrame.from_dict(pred_pac, orient="index",
                                             columns=["poisson_add_sim",
                                                      "poisson_prod_sim"])
        shapes_data_df = pd.DataFrame.from_dict(pred_shapes, orient="index")
        prediction_data_df = gc_data_df.merge(pac_data_df, how="outer",
                                              left_index=True, right_index=True)
        prediction_data_df = prediction_data_df.merge(shapes_data_df, how="outer",
                                                      left_index=True, right_index=True)
        try:
            assert not prediction_data_df.isnull().values.any()
        except AssertionError:
            # NaNs detected in input dataset, remove rows with NaNs
            prediction_data_df = prediction_data_df.dropna()
        prediction_data_features = prediction_data_df.iloc[:, 1: training_data.shape[1]].values
        pred_results = svc.fit(X, y).predict(prediction_data_features)

        print(strftime("%x %X:".format(localtime)),
              "Writing feature importance ranking to {}\n".
              format(abspath(args.savefile)))
        f_importances(svc.coef_,
                      [entry.replace("_", " ").replace("pos ", "P")
                       for entry in prediction_data_df.columns.values],
                      args.savefile)
        positive_pred_orfs = prediction_data_df.index.values[np.where(pred_results)]

        # print positive predictions in BED format
        # 1.chrom 2.chromStart 3.chromEnd 4.name 5.score 6.strand
        print(strftime("%x %X:".format(localtime)),
              "Writing positive prediction results to {}\n".
              format(abspath(args.output_file)))
        with open(args.output_file, "w") as pred_out:
            for genome_loc in positive_pred_orfs:
                chrom = genome_loc.strip().split(":")[0]
                chromStart = genome_loc.strip().split(":")[1].split("-")[0]
                chromEnd = genome_loc.strip().split(":")[1].split("-")[1].split("(")[0]
                pred_name = "{0}_TFBS".format(args.protein_name)
                score = "."
                strand = genome_loc.strip().split("(")[1][0]
                pred_out.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".
                               format(chrom, chromStart, chromEnd, pred_name, score,
                                      strand))
        exit()

    # perform cross-validation
    if args.cross_validation == "roc":
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        print(strftime("%x %X:".format(localtime)), "Plotting and saving ROC data\n")
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
    elif args.cross_validation == "prc":
        # return PRC plots
        print(strftime("%x %X:".format(localtime)), "Plotting and saving PRC data\n")

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
