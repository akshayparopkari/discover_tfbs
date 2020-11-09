#!/usr/bin/env python3

"""
Parse positive prediction from model classification and associate gene to model
predicted TFBS.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.2.5"


import argparse
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
        "temporal_rnaseq",
        metavar="/path/to/temporal_rnaseq.xlsx",
        type=str,
        help="Specify location and name of supplementary Excel file from Fox et al. "
        "2015 publication found here - https://onlinelibrary.wiley.com/doi/full/10.1111/mmi.13002"
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
        "gexf_network_file",
        metavar="/path/to/tf_gene_network.gexf",
        type=str,
        help="Specify location and name of file to save TFBS-target gene network "
        "file in GEXF format. Positive and negative regulation will be shown in "
        "green and red colors, respectively. [REQUIRED]",
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
    Given an expression value for a gene, return if it is UP (>1 LFC), DOWN (< -1 LFC) or "None" (-1 LFC > x > 1 LFC) regulated.
    """
    if value > 1.0:
        return "UP"
    elif value < -1.0:
        return "DOWN"
    else:
        return "None"


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

    ###########################################
    # Signal recovery from Nobile et al. 2012 #
    ###########################################
    true_tfbs = BedTool(args.fg_bed_file)
    predicted_tfbs = BedTool(args.prediction_results)
    overlap_count = len(true_tfbs.intersect(predicted_tfbs, u=True))
    print(
        strftime(
            f"%x %X | {overlap_count} ({overlap_count / len(true_tfbs): .2%}) were TFBS predicted true out of {len(true_tfbs)} entries "
            "from {args.fg_bed_file}"
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
                    f"%x %X | Writing all closest gene ID to BED file {args.output_file}"
                )
            )

        ################################################################
        # Get unique feature information for each closest gene feature #
        ################################################################
        print(
            strftime(
                f"%x %X | Writing unique closest orfs to model predicted TFBS to {args.orfs_output_file}"
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
                f"%x %X | {common_target_orfs} ({common_target_orfs / true_tfbs_target_orf_counts: .2%}) common target ORFs between true TFBS and model predicted TFBS"
            )
        )

        # Read in Fox 2015 data and merge predicted tfbs genes with their expression value
        orfs_df["ORF"] = orfs_df["Gene_Function"].apply(str_split)
        fox_temporal_data = pd.read_excel(
            "/home/aparopkari/endor/fox_supplementals/MMI_13002_supp-0002-Dataset1_temporalexp.xlsx",
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
                f"%x %X | Writing TF-gene activity network file to {args.gexf_network_file}"
            )
        )
        positive_edges = list(product([tf.capitalize()], tf_gene_activity_data["ORF"][tf_gene_activity_data["Gene_expression_stage_1"] == "UP"], [{"color": "#3532dc"}]))
        positive_edges = list(product([tf.capitalize()], tf_gene_activity_data["ORF"][tf_gene_activity_data["Gene_expression_stage_1"] == "DOWN"], [{"color": "#d9dc32"}]))
    print(strftime("\n%x %X | END TFBS-GENE ASSOCIATION\n"), sep="\n")


if __name__ == "__main__":
    exit(main())
