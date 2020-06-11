#!/usr/bin/env python3

"""
Calculate the expected counts of TF binding sites in the upstream intergenic region of
Candida albicans CDS regions. Then, compare the null distribution to model predicted TF
binding counts. This will allow us to assess if the number of TF binding sites predicted
are higher, as expected or lower than the mean null expected count.
"""

__author__ = "Akshay Paropkari"
__version__ = "0.1.5"


import argparse
from collections import Counter
from os.path import isfile, join, realpath
from sys import exit
from time import strftime

err = []
try:
    from scipy.stats import describe, ks_2samp, poisson
except ImportError:
    err.append("scipy")
try:
    import numpy as np

    np.random.seed(3914578)
except ImportError:
    err.append("numpy")
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    err.append("matplotlib")
try:
    from pybedtools import BedTool
except ImportError:
    err.append("pybedtools")
try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    err.append("statsmodels")
try:
    assert len(err) == 0
except AssertionError:
    for error in err:
        print("Please install {0}".format(error))
    exit()


def handle_program_options():
    parser = argparse.ArgumentParser(
        description="Calculate the expected counts of TF binding sites in the upstream "
        "intergenic region of Candida albicans CDS regions. Then, compare the null "
        "distribution to model predicted TF binding counts. This will allow us to assess "
        "if the number of TF binding sites predicted are higher, as expected or lower "
        "than the mean null expected count.",
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
        "tf_genome_wide_blastn_matches",
        metavar="/path/to/tf_genome_wide_blastn_matches.bed",
        type=str,
        help="Path to BED file containing genome wide BLASTn matches [REQUIRED]",
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
        "null_tfbs_dist_output_file",
        metavar="/path/to/tfbs_null_dist.pdf",
        type=str,
        help="Specify location and name of file to save the null distribution of TF "
        "binding site distribution. The null counts will be Poisson distributed, based "
        "on approximating the limit of Bernoulli trials [REQUIRED]",
    )
    parser.add_argument(
        "predicted_tfbs_significance",
        metavar="/path/to/predicted_tfbs_significance_output_folder/",
        type=str,
        help="Specify location and name of file to save model predicted TF binding site "
        "density significance. One tailed tests will be saved in individual files - "
        "one for greater and one for less [REQUIRED]",
    )
    return parser.parse_args()


def get_intergenic_ranges(intergenic_genome_file: str):
    """
    Yield intergenic regions of Candida albicans

    :param intergenic_genome_file: GTF file with intergenic region ranges. This file can
                                   be found in GFF files on Candida Genome Database.
    """
    with open(intergenic_genome_file) as genome_file:
        for line in genome_file:
            line = line.strip().split("\t")
            metadata = line[8].split(";")
            try:
                record = {
                    "start": np.uint32(line[3]),
                    "end": np.uint32(line[4]),
                    "id": metadata[0].split("=")[1],
                    "note": metadata[1].split("=")[1],
                    "length": np.uint16(metadata[2].split("=")[1]),
                    "gc": np.float32(metadata[3].split("=")[1]),
                    "at": np.float32(metadata[4].split("=")[1]),
                }
            except Exception as err:
                print(err)
                return None
            else:
                yield record


def main():

    args = handle_program_options()

    try:
        assert isfile(args.prediction_results)
        assert isfile(args.tf_genome_wide_blastn_matches)
        assert isfile(args.genome_intergenic_feature_file)
    except Exception as ex:
        print("Error with input file(s). Please review the error\n{0}".format(ex))
        exit()
    else:
        protein_name = args.protein_name.capitalize()
        for fnh in [args.null_tfbs_dist_output_file]:
            output_format = fnh.split("/")[-1].split(".")[-1]
            try:
                assert output_format in [
                    "pdf",
                    "svg",
                    "png",
                    "jpg",
                    "tiff",
                    "eps",
                    "ps",
                ]
            except AssertionError:
                print(
                    "Error: Please check the output file format provided. '{0}' format is"
                    " not supported in {1}.".format(output_format, fnh)
                )
                exit()

    ###############################
    # Calculate null distribution #
    ###############################
    print(
        strftime(
            "%x %X | Processing {0} TFBS background distribution".format(protein_name)
        )
    )
    empirical_tfbs_cnt = Counter()
    intergenic_len = {}
    intergenic_regions = BedTool(args.genome_intergenic_feature_file)
    tf_blastn_matches = BedTool(args.tf_genome_wide_blastn_matches)
    intergenic_blastn_matches = tf_blastn_matches.intersect(
        intergenic_regions, f=1.0, wb=True, stream=True
    )

    for entry in intergenic_blastn_matches:
        region_id = entry.fields[-1].split(";")[0].split("=")[1]
        empirical_tfbs_cnt.update([region_id])
        intergenic_len[region_id] = np.int16(
            entry.fields[-1].split(";")[2].split("=")[1]
        )

    prior_tfbs_probability = sum(empirical_tfbs_cnt.values()) / sum(
        intergenic_len.values()
    )
    cnt_nobs, cnt_minmax, cnt_mean1, cnt_var, cnt_skew, cnt_kurt = describe(
        list(empirical_tfbs_cnt.values())
    )
    len_nobs, len_minmax, len_mean1, len_var, len_skew, len_kurt = describe(
        list(intergenic_len.values())
    )
    expected_null_tfbs_density = cnt_mean1 / len_mean1
    print(
        "{0:>20}Expected number of TFBS in an average intergenic region (Empirical) = "
        "{1:0.3f} +/- {2:0.3f}".format(
            "", expected_null_tfbs_density, np.sqrt(cnt_var) / np.sqrt(len_var)
        )
    )
    empirical_tfbs_density = [
        cnt / intergenic_len[key] for key, cnt in empirical_tfbs_cnt.items()
    ]

    ###########################################################################
    # Calculate distribution of predicted TFBS in upstream intergenic regions #
    ###########################################################################
    print(
        strftime(
            "%x %X | Processing {0} model predicted TFBS distribution".format(
                protein_name
            )
        )
    )
    predictions = BedTool(args.prediction_results)
    predicted_intergenic_tfbs = predictions.intersect(
        intergenic_regions, f=1.0, wb=True, stream=True
    )
    num_of_hits = Counter()
    intergenic_len = {}
    for entry in predicted_intergenic_tfbs:
        region_id = entry.fields[-1].split(";")[0].split("=")[1]
        num_of_hits.update([region_id])
        intergenic_len[region_id] = np.int16(
            entry.fields[-1].split(";")[2].split("=")[1]
        )
    cnt_nobs, cnt_minmax, cnt_mean2, cnt_var, cnt_skew, cnt_kurt = describe(
        list(num_of_hits.values())
    )
    len_nobs, len_minmax, len_mean2, len_var, len_skew, len_kurt = describe(
        list(intergenic_len.values())
    )
    print(
        "{0:>20}Expected number of TFBS in an average intergenic region (predicted) = "
        "{1:0.3f} +/- {2:0.3f}".format(
            "", cnt_mean2 / len_mean2, np.sqrt(cnt_var) / np.sqrt(len_var)
        )
    )
    predicted_density = [cnt / intergenic_len[key] for key, cnt in num_of_hits.items()]

    try:
        # testing conformity of TFBS simulation and model prediction distributions
        ks_stat, ks_pval = ks_2samp(
            list(empirical_tfbs_cnt.values()), list(num_of_hits.values())
        )
        assert ks_pval > 0.05
    except AssertionError:
        print(
            strftime(
                "%x %X | Model predicted TFBS distribution do not align with empirical "
                "TFBS distribution (KS test pvalue={0:.3f})".format(ks_pval)
            )
        )
    else:
        print(
            strftime(
                "%x %X | Model predicted TFBS distribution align with empirical TFBS "
                "distribution (KS test pvalue={0:.3f})".format(ks_pval)
            )
        )

    ###########################################################
    # Plot density distribution across all intergenic regions #
    ###########################################################
    print(
        strftime(
            "%x %X | Plotting and saving {0} density histograms".format(protein_name)
        )
    )
    with mpl.style.context("ggplot"):
        plt.figure(figsize=(10, 5), edgecolor="k", tight_layout=True)
        plt.hist(
            empirical_tfbs_density,
            histtype="barstacked",
            log=True,
            color="#6471B1",
            alpha=0.5,
            label="Empirical TFBS distribution",
        )
        plt.axvline(
            x=cnt_mean1 / len_mean1,
            color="#6471B1",
            lw=3,
            label="Mean empirical TFBS distribution",
        )
        plt.hist(
            predicted_density,
            histtype="barstacked",
            hatch="..",
            log=True,
            color="#B1A464",
            label="Model predicted TFBS distribution",
            alpha=0.5,
        )
        plt.axvline(
            x=cnt_mean2 / len_mean2,
            color="#AB985D",
            lw=3,
            label="Mean model predicted TFBS distribution",
        )
        plt.ylabel("Frequency", color="k", fontsize=10)
        #         plt.xlabel("{} binding site density".format(protein_name), color="k")
        plt.legend(fontsize=12)
        plt.suptitle(
            "{} binding site density".format(protein_name), y=1.01, fontsize=12
        )
        plt.title(
            "Kolmogorov-Smirnov test ({0:0.3f}, pvalue={1:0.3f})".format(
                ks_stat, ks_pval
            ),
            fontsize=11,
        )
        plt.savefig(
            args.null_tfbs_dist_output_file,
            dpi=300.0,
            format=output_format,
            edgecolor="k",
            bbox_inches="tight",
            pad_inches=0.1,
        )

    ######################################################################################
    # Per intergenic region, get collect regions with unusually high or low TFBS density #
    ######################################################################################
    print(
        strftime(
            "%x %X | Collecting high density intergenic regions for {}".format(
                protein_name
            )
        )
    )
    poisson_prob_greater = {}
    poisson_prob_less = {}
    expected_count = {}
    for intergenic_region, region_length in intergenic_len.items():
        expected_count[intergenic_region] = prior_tfbs_probability * region_length
        poisson_prob_greater[intergenic_region] = 1 - poisson.cdf(
            num_of_hits[intergenic_region] - 1, expected_count[intergenic_region]
        )
        poisson_prob_less[intergenic_region] = poisson.cdf(
            num_of_hits[intergenic_region] - 1, expected_count[intergenic_region]
        )

    # Perform multiple testing correction and write to file
    outfnh_greater = realpath(
        join(
            args.predicted_tfbs_significance,
            "{0}_intergenic_tfbs_sig_greater.txt".format(args.protein_name),
        )
    )
    p_adj_greater = dict(
        zip(
            poisson_prob_greater.keys(),
            multipletests(list(poisson_prob_greater.values()), method="fdr_bh")[1],
        )
    )

    with open(outfnh_greater, "w") as outfile:
        outfile.write(
            "intergenic_region\texpected_tfbs_cnt\tpredicted_tfbs_cnt\tpvalue(greater)\tp_adj(greater)\n"
        )
        for intergenic_region in intergenic_len.keys():
            outfile.write(
                "{0}\t{1:0.3f}\t{2:0.1f}\t{3:0.3f}\t{4:0.3f}\n".format(
                    intergenic_region,
                    expected_count[intergenic_region],
                    num_of_hits[intergenic_region],
                    poisson_prob_greater[intergenic_region],
                    p_adj_greater[intergenic_region],
                )
            )

    outfnh_greater = realpath(
        join(
            args.predicted_tfbs_significance,
            "{0}_intergenic_tfbs_sig_less.txt".format(args.protein_name),
        )
    )
    p_adj_less = dict(
        zip(
            poisson_prob_less.keys(),
            multipletests(list(poisson_prob_less.values()), method="fdr_bh")[1],
        )
    )

    with open(outfnh_greater, "w") as outfile:
        outfile.write(
            "intergenic_region\texpected_tfbs_cnt\tpredicted_tfbs_cnt\tpvalue(less)\tp_adj(less)\n"
        )
        for intergenic_region in intergenic_len.keys():
            outfile.write(
                "{0}\t{1:0.3f}\t{2:0.1f}\t{3:0.3f}\t{4:0.3f}\n".format(
                    intergenic_region,
                    expected_count[intergenic_region],
                    num_of_hits[intergenic_region],
                    poisson_prob_less[intergenic_region],
                    p_adj_less[intergenic_region],
                )
            )


if __name__ == "__main__":
    exit(main())
