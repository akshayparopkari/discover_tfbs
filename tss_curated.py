#! /usr/bin/env python3
"""
docs
"""

__version__ = '0.0.1'

import argparse
import sys
from multiprocessing import Pool, cpu_count, current_process
from os import walk
from os.path import basename, dirname, join

from utils import get_blastn_results

try:
    from numpy import mean
except ImportError:
    sys.exit("NumPy not available, please run `pip install numpy` or `conda install "
             "numpy` before running this script.")



def tss_calc(gtf_data, qseqid, match):
    """
    Calculate distance between binding event and closest transcription start site.
    Distance is based on midpoint of binding event and starting coordinate of the closest
    mRNA start codon.
    """
#     print("{}: Started TSS calculation on CPU #{}".
#           format(strftime("%d/%m/%Y %H:%M:%S", localtime()),
#                  current_process().name))
    motif_center_pos = (match[5] +  match[6]) // 2
    min_dist = float('inf')

    m0 = match[0].split("_")[0]
    m9 = match[9]

    for line in gtf_data:
        ok = False
        if not line[9] == m0 :
            continue
        # valid chromosome identified
        # compare to previous value of min_dist, and. if new distance
        # value is lower than current value of min_dist, replace
        # min_dist with new lower distance value
        if m9 == "+":
            # positive strand
            if line[11] < motif_center_pos and line[6] == "+":
                # for positive strand, star_codon has to be
                # upstream/lower coordinate
                if min_dist > abs(motif_center_pos - line[11]):
                    ok = True
        else:
            # negative strand
            if line[11] > motif_center_pos and line[6] == "-":
                # for negative strand, star_codon has to be
                # greater coordinate position
                if min_dist > abs(motif_center_pos - line[11]):
                    ok = True

        if ok:
            min_dist = abs(motif_center_pos - line[11])
            # update gene_id, strand, start and stop
            # coordinate values associated with
            # closest TSS
            gene_id = line[10]
            strand = line[6]
            start_pos = line[3]
            stop_pos = line[4]
            qseqid = qseqid
            sseqid = line[0]
            sstart = match[5]
            send = match[6]

    # test if TSS calculated or not
    try:
        sseqid
    except Exception:
        return None
    else:
        return [qseqid, sseqid, gene_id, str(min_dist), strand,
                start_pos, stop_pos, str(sstart), str(send)]


def handle_program_options():
    parser = argparse.ArgumentParser(description="Calculate the closest TSS for all "
                                     "BLASTn matches for curated motifs.")
    parser.add_argument("blastn_file", help="Blastn results file [REQUIRED]")
    parser.add_argument("gtf_file", help="Input Candida albicans GTF containing start "
                        "and stop codons coordinate [REQUIRED]")
    parser.add_argument("out_file", help="Output file name and location [REQUIRED]")
    return parser.parse_args()


def main():
    args = handle_program_options()

    # get blastn results in a dict
    blastn_results = get_blastn_results(args.blastn_file)

    # read in GTF data
    print("{}: PARSING GTF FILE".format(strftime("%d/%m/%Y %H:%M:%S", localtime())))
    gtf_data = list()
    with open(args.gtf_file) as gtff:
        for line in gtff:
            line = line.strip().split("\t")
            if line[2] == "start_codon" and abs(int(line[3]) - int(line[4])) >= 2:
                chromosom = line[0].split("_")[0]
                line.append(chromosom)
                gene_id = line[-2].split(";")[0].split()[1].replace("\"", "")
                line.append(gene_id)
                line.append(int(line[3]))
                gtf_data.append(line)

    # writing results to file
    print("{}: INITIALIZING OUTPUT FILE {}".
          format(strftime("%d/%m/%Y %H:%M:%S", localtime()), args.out_file))
    with open(args.out_file, "w") as outf:
        outf.write("qseqid\tsseqid\tgene_id\tdist_to_closest_tss\tstrand\t"
                   "start_codon_coord\tstop_codon_coord\tsstart\tsend\n")

    # iterate over dict to calculate TSS per match
    for qseqid, md in blastn_results.items():
        res = []
        for match in zip(*md.values()):
            res.append(tss_calc(gtf_data, qseqid, match))
        for r in res:
            if isinstance(r, list):
                # write out result to file
                with open(args.out_file, "a") as outf:
                    outf.write("{}\n".format("\t".join(r)))


if __name__ == "__main__":
    sys.exit(main())
