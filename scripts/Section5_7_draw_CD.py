import os
from pathlib import Path
import pandas as pd

from util.helpers import build_cd_diagram


if __name__ == "__main__":

    methods = ['sax', 'sfa', 'spartan']
    itrs = ['a4_w8_exist_dist'] # change as needed
    display_names = [f"{model_name.upper()}+MINDIST" for model_name in methods]
    dist_measure = ['sax_mindist', 'sfa_mindist', 'pca_mindist'] # existing distance measures

    build_cd_diagram(
        data_path='.',  
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measure=dist_measure,
        subdir="output/classification", # change as needed
        savefile_path='./result/Section5_7/CD_3methods_mindist_a4_w8.pdf' # change as needed
    )

    methods = ['sfa', 'spartan', 'sfa', 'spartan']
    itrs = ['a4_w8_exist_dist', 'a4_w8_exist_dist', 'a4_w8', 'a4_w8'] # change as needed
    display_names = ['SFA+MINDIST', 'SPARTAN+MINDIST', 'SFA+SymbolicL1', 'SPARTAN+SymbolicL1']
    dist_measure = ['sfa_mindist', 'pca_mindist', 'symbolic_l1', 'symbolic_l1'] # hist_enculidean distance for bag of pattern

    build_cd_diagram(
        data_path='.',  
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measure=dist_measure,
        subdir="output/classification", # change as needed
        savefile_path='./result/Section5_7/CD_2methods_symbolicl1_vs_mindist_a4w8.pdf' # change as needed
    )
