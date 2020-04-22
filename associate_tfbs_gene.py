#!/usr/bin/env python3

"""
Parse positive prediction from model classification and associate gene to model
predicted TFBS.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.1.2"


import argparse
from os.path import isfile
from sys import exit
from time import strftime

try:
    from pybedtools import BedTool
except ImportError:
    err.append("Please install pybedtools")


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
    return parser.parse_args()


def main():

    args = handle_program_options()

    print("#" * 90, strftime("%x %X | START TFBS-GENE ASSOCIATION\n"), sep="\n\n")

    # Check input validity
    try:
        assert isfile(args.prediction_results)
        assert isfile(args.genome_feature_file)
        assert isfile(args.fg_bed_file)
    except Exception as e:
        print("Error with input file(s). Please review the error\n{0}".format(e))
        exit()
    else:
        protein_name = args.protein_name.capitalize()

    ######################################
    # Signal recovery from previous data #
    ######################################
    true_tfbs = BedTool(args.fg_bed_file)
    predicted_tfbs = BedTool(args.prediction_results)
    overlap_count = len(true_tfbs.intersect(predicted_tfbs, u=True))
    overlap_pct = 100 * (overlap_count / len(true_tfbs))
    print(
        strftime(
            "%x %X | {0} ({1:0.2f}%) were entries predicted true out of {2} entries "
            "from {3}".format(
                overlap_count, overlap_pct, len(true_tfbs), args.fg_bed_file
            )
        )
    )

    ##############################
    # Gather all intergenic hits #
    ##############################
    genome_intergenic_regions = BedTool(args.genome_intergenic_feature_file)
    try:
        predicted_intergenic_hits = predicted_tfbs.intersect(
            genome_intergenic_regions, f=1.0, wa=True
        ).sort()
    except Exception as err:
        exit(print(err))
    else:
        #######################################################
        # Gathering closest gene for each positive prediction #
        #######################################################
        genome_file = BedTool(args.genome_feature_file)
        try:
            closest_gene = predicted_intergenic_hits.closest(
                genome_file,
                D="b",
                output=args.output_file,
            )
        except Exception as err:
            exit(print(err))
        else:
            print(
                strftime(
                    "%x %X | Writing all closest gene ID to BED file {0}".format(
                        args.output_file
                    )
                )
            )

        ################################################################
        # Get unique feature information for each closest gene feature #
        ################################################################
        orfs = {}
        with open(args.output_file) as infile:
            for line in infile:
                line = line.strip().split("\t")
                for entry in line[12].split(";"):
                    if entry.startswith("Gene="):
                        try:
                            orfs[orfid]["gene"] = entry.split("=")[1]
                        except KeyError:
                            orfs[orfid] = {
                                "counts": 1,
                                "function": orf_function,
                                "gene": entry.split("=")[1],
                                "distance": 0,
                            }
                    if entry.startswith("Note="):
                        feat = entry.split("Note=")[1]
                        orfid = feat.split("_")[0][1:-1]
                        orf_function = " ".join(feat.split("_")[1:])
                        try:
                            orfs[orfid]["counts"] += 1
                            orfs[orfid]["function"] = orf_function
                        except KeyError:
                            orfs[orfid] = {
                                "counts": 1,
                                "function": orf_function,
                                "gene": "NA",
                                "distance": 0,
                            }
                    if entry.startswith("_"):
                        orfs[orfid]["function"] += entry.replace("_", " ")
                try:
                    orfs[orfid]["distance"] += int(line[13])
                except KeyError:
                    # no orfid found since there was no match from
                    continue
        print(
            strftime(
                "%x %X | Writing unique closest orfs to model predicted TFBS to {0}".format(
                    args.orfs_output_file
                )
            )
        )
        with open(args.orfs_output_file, "w") as outfile:
            outfile.write("ORF_ID\tGene\tOccurrences\tMean distance\tFunction\n")
            for orfid, data in orfs.items():
                outfile.write(
                    "{0}\t{1}\t{2}\t{3:.2f}\t{4}\n".format(
                        orfid,
                        data["gene"],
                        data["counts"],
                        data["distance"] / data["counts"],
                        data["function"],
                    )
                )

    print(strftime("\n%x %X | END TFBS-GENE ASSOCIATION\n"), "#" * 90, sep="\n")


if __name__ == "__main__":
    exit(main())
