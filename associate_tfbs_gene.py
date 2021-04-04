#!/usr/bin/env python3

"""
Parse positive prediction from model classification and associate gene to model
predicted TFBS.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.3.2"


import argparse
from itertools import product
from os.path import isfile
from sys import exit
from time import strftime

from utils import get_closest_genes

err = []
try:
    from pybedtools import BedTool
except ImportError:
    err.append("pybedtools")
try:
    import pandas as pd
except ImportError:
    err.append("pandas")
try:
    import numpy as np
    from numpy.linalg import norm
except ImportError:
    err.append("numpy")
try:
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    plt.switch_backend("agg")
except ImportError:
    err.append("matplotlib")
try:
    import networkx as nx
except ImportError:
    err.append("networkx")
if len(err) > 0:
    for e in err:
        print(f"Please install {e}")
    exit()


def handle_program_options():
    parser = argparse.ArgumentParser(
        description="Parse positive prediction from model classification and associate "
        "gene to model predicted TFBS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "protein_name",
        type=str,
        choices=["bcr1", "brg1", "efg1", "ndt80", "rob1", "tec1"],
        help="Specify the name of transcription factor. Please see the "
        "list of valid choices for this parameter [REQUIRED]",
    )
    parser.add_argument(
        "prediction_results",
        type=str,
        metavar="/path/to/true_binding_site_predicted_by_model.bed",
        help="Specify location and name of the file to read the results "
        "of classification prediction. The results will be saved in BED6 "
        "format [REQUIRED]",
    )
    parser.add_argument(
        "genome_intergenic_feature_file",
        type=str,
        metavar="/path/to/organism_genome_integenic_features.gff",
        help="Specify location and name of the file listing the intergenic "
        "regions between coding regions in the chromosomes. This file can be "
        "a either GFF or BED file [REQUIRED]",
    )
    parser.add_argument(
        "genome_feature_file",
        type=str,
        metavar="/path/to/organism_genome_features.gff",
        help="Specify location and name of the file containing unique "
        "genome features. This file can be a either GFF or BED file [REQUIRED]",
    )
    parser.add_argument(
        "fg_bed_file",
        metavar="/path/to/true_binding_site_sequences.bed",
        type=str,
        help="Path to true binding sites in a BED file. This file must have a "
        "minimum of BED6 format - i.e. chrom start end name score strand "
        " columns [REQUIRED]",
    )
    parser.add_argument(
        "model_stats",
        type=str,
        metavar="/path/to/model_stats",
        help="Specify location of folder to save SVM model statistics "
        "like distance of hyperplane and log probability of classification [REQUIRED]",
    )
    parser.add_argument(
        "temporal_rnaseq",
        metavar="/path/to/temporal_rnaseq.xlsx",
        type=str,
        help="Specify location and name of supplementary Excel file from Fox et al. "
        "2015 publication found here - https://onlinelibrary.wiley.com/doi/full/"
        "10.1111/mmi.13002"
        " [REQUIRED]",
    )
    parser.add_argument(
        "output_file",
        metavar="/path/to/closest_gene_to_model_predicted_tfbs.bed",
        type=str,
        help="Specify location and name of file to save the closest gene to all positive "
        "TFBS predictions from model. This file will be saved in BED file. [REQUIRED]",
    )
    parser.add_argument(
        "orfs_output_file",
        metavar="/path/to/unique_orfs_associated_to_model_predicted_tfbs.txt",
        type=str,
        help="Specify location and name of file to save the unique closest gene to all "
        "positive TFBS predictions from model. [REQUIRED]",
    )
    parser.add_argument(
        "stage_activity",
        metavar="/path/to/tfbs_gene_activity_per_stage.txt",
        type=str,
        help="Specify location and name of file to save TFBS-target gene activity. "
        "For each of the four biofilm developement stages, all TFBS-gene "
        "interactions will be assigned either UP, DOWN or NONE values based "
        "on Fox et al. 2015 temporal gene expression data set [REQUIRED]",
    )
    parser.add_argument(
        "tf_gene_network_data",
        metavar="/path/to/tf_gene_network/",
        type=str,
        help="Specify folder location to save TFBS-target gene network for each stage "
        " in GEXF format. Positive and negative regulation will be shown in "
        "green and red colors, respectively. [REQUIRED]",
    )
    parser.add_argument(
        "roc_like_curve",
        type=str,
        metavar="/path/to/roc_like_curve.pdf",
        help="Specify location and name of the file to save the ROC-like curve "
        "of positive classification. X-axis is predicted target ORFs and Y-axis "
        "denotes experimentally verified target ORFs [REQUIRED]",
    )
    return parser.parse_args()


def str_split(instr: str) -> str:
    """
    Given an input string of format ""(orf19.2823) HMG domain transcriptional repres...""
    return the orf ID "orf19.2823" from the input.
    """
    return instr.split(")")[0].replace("(", "")


def gene_regulation(value: float) -> str:
    """
    Given an expression value for a gene, return if it is UP (>1 LFC), DOWN (< -1 LFC)
    or "None" (-1 LFC > x > 1 LFC) regulated.
    """
    if value > 1.0:
        return "UP"
    elif value < -1.0:
        return "DOWN"
    else:
        return "None"


def combine_tfbs_gene_svm_stats(bedtool_closest: list):
    """
    Combine TFBS and gene locations away from SVM hyperplane
    """
    output = []
    for tf_gene in bedtool_closest:
        tf_gene = tf_gene.fields
        header = (
            tf_gene[0] + ":" + tf_gene[1] + "-" + tf_gene[2] + "(" + tf_gene[5] + ")"
        )
        gene = "_".join(tf_gene[12].split(";")[6].split("=")[1].split("_")[:2])
        output.append((header, gene, int(tf_gene[-1])))
    return pd.DataFrame.from_records(
        output, columns=["location", "Systematic_name", "Distance_from_gene"]
    )


def main():

    args = handle_program_options()

    print("#" * 90, "\n\n", strftime("%x %X | START TFBS-GENE ASSOCIATION\n"))

    # Check input validity
    try:
        assert isfile(args.prediction_results)
        assert isfile(args.genome_feature_file)
        assert isfile(args.fg_bed_file)
    except Exception as e:
        print(f"Error with input file(s). Please review the error\n{e}")
        exit()
    else:
        protein_name = args.protein_name.capitalize()
        try:
            output_format = args.roc_like_curve.split("/")[-1].split(".")[-1]
            assert output_format in ["pdf", "svg", "png", "jpg", "tiff", "eps", "ps"]
        except AssertionError:
            print(
                f"Error: Please check the output file format provided. '{output_format}' "
                f"format is not supported in {args.roc_like_curve}."
            )

    ###########################################
    # Signal recovery from Nobile et al. 2012 #
    ###########################################
    true_tfbs = BedTool(args.fg_bed_file)
    predicted_tfbs = BedTool(args.prediction_results)
    overlap_count = len(true_tfbs.intersect(predicted_tfbs, u=True))
    print(
        strftime(
            f"%x %X | {overlap_count} ({overlap_count / len(true_tfbs): .2%}) were "
            "TFBS predicted true out of {len(true_tfbs)} entries from {args.fg_bed_file}"
        )
    )
    genome_file = BedTool(args.genome_feature_file)
    true_tfbs_closest_orfs_df = get_closest_genes(true_tfbs, genome_file)
    true_tfbs_target_orf_counts = len(true_tfbs_closest_orfs_df["Systematic_Name"])

    ##############################
    # Gather all intergenic hits #
    ##############################
    genome_intergenic_regions = BedTool(args.genome_intergenic_feature_file)
    try:
        predicted_intergenic_hits = predicted_tfbs.intersect(
            genome_intergenic_regions, f=1.0, wa=True
        ).sort()
    except Exception as err:
        exit(err)
    else:
        #######################################################
        # Gathering closest gene for each positive prediction #
        #######################################################
        try:
            closest_gene = predicted_intergenic_hits.closest(
                genome_file, D="b", output=args.output_file,
            )
        except Exception as err:
            exit(err)
        else:
            print(
                strftime(
                    f"%x %X | Generating ROC-like plots using genes, instead of TFBS"
                )
            )

        ##########################################################
        # Plot ROC-like plots for true genes and predicted genes #
        ##########################################################
        training_data_model_stats = pd.read_csv(
            f"{args.model_stats}{args.protein_name}_training_model_statistics.txt",
            sep="\t",
        )
        positive_training_data_model_stats = training_data_model_stats[
            training_data_model_stats["prediction"] == 1
        ]
        training_closest_gene = true_tfbs.closest(genome_file, D="b")
        positive_training_data_gene = combine_tfbs_gene_svm_stats(training_closest_gene)
        positive_training_data_gene = positive_training_data_gene.merge(
            positive_training_data_model_stats, left_on="location", right_on="location",
        )

        predicted_data_model_stats = pd.read_csv(
            f"{args.model_stats}{args.protein_name}_prediction_model_statistics.txt",
            sep="\t",
        )
        positive_predicted_data_model_stats = predicted_data_model_stats[
            predicted_data_model_stats["prediction"] == 1
        ]
        positive_predicted_tfbs_gene = combine_tfbs_gene_svm_stats(closest_gene)
        positive_predicted_tfbs_gene = positive_predicted_tfbs_gene.merge(
            positive_predicted_data_model_stats,
            left_on="location",
            right_on="location",
        )

        # collect data for plotting ROC curves
        min_dist_true_TFBS_from_hyperplane = min(
            positive_training_data_gene["distance_from_hyperplane"]
        )
        positive_high_confidence_pred_data = positive_predicted_tfbs_gene[
            "distance_from_hyperplane"
        ][
            positive_predicted_tfbs_gene["distance_from_hyperplane"]
            > min_dist_true_TFBS_from_hyperplane
        ]
        max_dist = max(
            np.r_[
                positive_high_confidence_pred_data,
                positive_training_data_gene["distance_from_hyperplane"],
            ]
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
                np.count_nonzero(
                    positive_training_data_gene["distance_from_hyperplane"] <= dist
                )
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
            plt.fill_between(new_predictions, true_predictions, alpha=0.5, step="post")
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

        ################################################################
        # Get unique feature information for each closest gene feature #
        ################################################################
        print(
            strftime(
                "%x %X | Writing unique closest orfs to model predicted TFBS to "
                f"{args.orfs_output_file}"
            )
        )
        orfs_df = get_closest_genes(predicted_intergenic_hits, genome_file)
        common_target_orfs = len(
            set(orfs_df["Systematic_Name"]).intersection(
                set(true_tfbs_closest_orfs_df["Systematic_Name"])
            )
        )
        print(
            strftime(
                f"%x %X | {common_target_orfs} "
                f"({common_target_orfs / true_tfbs_target_orf_counts: .2%}) common "
                "target ORFs between true TFBS and model predicted TFBS"
            )
        )

        # Read in Fox 2015 data and merge predicted tfbs genes with their expression value
        orfs_df["ORF"] = orfs_df["Gene_Function"].apply(str_split)
        fox_temporal_data = pd.read_excel(args.temporal_rnaseq,
            index_col=None,
            usecols=[
                "ORF",
                "Name",
                "Description",
                "Ad vs Sat Avg",
                "8h vs Sat Avg",
                "24h vs Sat Avg",
                "48h vs Sat Avg",
            ],
        )
        orfs_df = orfs_df.merge(fox_temporal_data, left_on="ORF", right_on="ORF")
        orfs_df = orfs_df.loc[
            :,
            [
                "ORF",
                "Systematic_Name",
                "Gene_Name",
                "Closest_TFBS_Counts",
                "Mean_dist_from_gene",
                "Name",
                "Ad vs Sat Avg",
                "8h vs Sat Avg",
                "24h vs Sat Avg",
                "48h vs Sat Avg",
                "Gene_Function",
                "Description",
            ],
        ]
        predicted_high_confidance_tfbs_gene = positive_predicted_tfbs_gene[
            positive_predicted_tfbs_gene["distance_from_hyperplane"]
            > min_dist_true_TFBS_from_hyperplane
        ]["Systematic_name"]
        orfs_df.loc[orfs_df["Systematic_Name"].isin(predicted_high_confidance_tfbs_gene)]

        print(
            strftime(
                "%x %X | Saving high-confidance TF-gene interactions to "
                f"{args.orfs_output_file}"
            )
        )
        orfs_df.to_csv(args.orfs_output_file, sep="\t", index=False, na_rep="NA")

        ##################################
        # Write TF-gene activity results #
        ##################################
        orfs_df["Gene_expression_stage_1"] = orfs_df["Ad vs Sat Avg"].apply(
            gene_regulation
        )
        orfs_df["Gene_expression_stage_2"] = orfs_df["8h vs Sat Avg"].apply(
            gene_regulation
        )
        orfs_df["Gene_expression_stage_3"] = orfs_df["24h vs Sat Avg"].apply(
            gene_regulation
        )
        orfs_df["Gene_expression_stage_4"] = orfs_df["48h vs Sat Avg"].apply(
            gene_regulation
        )
        orfs_df = orfs_df.loc[
            :,
            [
                "ORF",
                "Systematic_Name",
                "Gene_Name",
                "Gene_expression_stage_1",
                "Gene_expression_stage_2",
                "Gene_expression_stage_3",
                "Gene_expression_stage_4",
                "Description",
            ],
        ]
        print(
            strftime(
                f"%x %X | Writing stage-dependent TF-gene activity to {args.stage_activity}"
            )
        )
        orfs_df.to_csv(args.stage_activity, sep="\t", index=False, na_rep="NA")

        print(
            strftime(
                "%x %X | Writing TF-gene interaction network for each stage to "
                f"{args.tf_gene_network_data}"
            )
        )
        for stage in [1, 2, 3, 4]:
            # create directed graph
            G = nx.Graph()
            positive_edges = list(
                product(
                    [protein_name],
                    orfs_df["ORF"][orfs_df[f"Gene_expression_stage_{stage}"] == "UP"],
                )
            )
            negative_edges = list(
                product(
                    [protein_name],
                    orfs_df["ORF"][orfs_df[f"Gene_expression_stage_{stage}"] == "DOWN"],
                )
            )
            G.add_edges_from(positive_edges)
            G.add_edges_from(negative_edges)
            G.nodes[protein_name]["size"] = 25
            G.nodes[protein_name]["color"] = "#dc32d9"
            upreg_node_attrs = {
                node: {"color": "#3532dc"}
                for node in orfs_df["ORF"][
                    orfs_df[f"Gene_expression_stage_{stage}"] == "UP"
                ].tolist()
            }
            downreg_node_attrs = {
                node: {"color": "#d9dc32"}
                for node in orfs_df["ORF"][
                    orfs_df[f"Gene_expression_stage_{stage}"] == "DOWN"
                ].tolist()
            }
            nx.set_node_attributes(G, upreg_node_attrs)
            nx.set_node_attributes(G, downreg_node_attrs)
            nx.write_gexf(
                G,
                f"{args.tf_gene_network_data}{args.protein_name}_stage_{stage}_network.gexf",
            )

            # calculate graph centrality metrics
            betweenness = pd.DataFrame.from_dict(
                nx.betweenness_centrality(G),
                orient="index",
                columns=["Betweenness_centrality"],
            )
            popular_target_orf = pd.DataFrame.from_dict(
                nx.degree_centrality(G), orient="index", columns=["Degree_centrality"],
            )
            information_centrality = pd.DataFrame.from_dict(
                nx.current_flow_closeness_centrality(G),
                orient="index",
                columns=["Information_flow_centrality"],
            )
            centrality = betweenness.merge(
                popular_target_orf, left_index=True, right_index=True
            )
            centrality = centrality.merge(
                information_centrality, left_index=True, right_index=True
            )
            centrality = centrality.merge(
                orfs_df, left_index=True, right_on="ORF"
            ).set_index("ORF")
            print(
                strftime(
                    f"%x %X | {protein_name} stage {stage} network efficiency: "
                    f"{nx.global_efficiency(G)}"
                )
            )
            centrality.to_csv(
                f"{args.tf_gene_network_data}{args.protein_name}_stage_{stage}_"
                "centralities.txt",
                sep="\t",
                index=False,
            )
            G.clear()

    print(strftime("\n%x %X | END TFBS-GENE ASSOCIATION\n"), sep="\n")


if __name__ == "__main__":
    exit(main())
