import os
from pathlib import Path
import pandas as pd

from util.helpers import build_cd_diagram


if __name__ == "__main__":

    result_dir = "output/clustering" # change as needed
    sp_distance = "symbolic_l1"
    bop_distance = "hist_euclidean"

    methods = ['sax', 'sfa', 'spartan']
    itrs = ['a4_w8'] # change as needed
    display_names = [model_name.upper() for model_name in methods]
    dist_measure = sp_distance # symbolic_l1 distance for single pattern

    build_cd_diagram(
        data_path='.',  
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measure=dist_measure,
        subdir=result_dir, # change as needed
        metric_col="ri",
        result_csv_name="clustering_results.csv",
        savefile_path='./result/Section5_4/CD_3methods_single_pattern.pdf' # change as needed
    )

    methods = ['sax', 'sfa', 'spartan']
    itrs = ['a4_w4_win0.05'] # change as needed
    display_names = [model_name.upper() for model_name in methods]
    dist_measure = bop_distance # hist_enculidean distance for bag of pattern

    build_cd_diagram(
        data_path='.',  
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measure=dist_measure,
        subdir=result_dir, # change as needed
        metric_col="ri",
        result_csv_name="clustering_results.csv",
        savefile_path='./result/Section5_4/CD_3methods_bag_of_pattern.pdf' # change as needed
    )
