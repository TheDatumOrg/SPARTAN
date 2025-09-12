import os
from pathlib import Path
import pandas as pd

from util.helpers import build_cd_diagram


if __name__ == "__main__":

    methods = ['spartan', 'spartan', 'spartan', 'spartan']
    itrs = ['a4_w8', 'a4_w8_woDAA', 'a4_w8_naiveDAA', 'a4_w8_naiveC'] # change as needed
    display_names = ['SPARTAN', 'SPARTAN_woDAA', 'SPARTAN_naiveDAA', 'SPARTAN_naiveC']
    sp_measure = 'symbolic_l1' # symbolic_l1 distance for single pattern
    bop_measure = 'hist_euclidean'

    build_cd_diagram(
        data_path='.',  
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measure=sp_measure,
        subdir="output/classification", # change as needed
        savefile_path='./result/Section5_2/CD_daa_ablation_single_pattern.pdf' # change as needed
    )

    methods = ['spartan', 'spartan', 'spartan', 'spartan']
    itrs = ['a4_w4_win0.05', 'a4_w4_win0.05_woDAA', 'a4_w4_win0.05_naiveDAA', 'a4_w4_win0.05_naiveC'] # change as needed
    display_names = ['SPARTAN', 'SPARTAN_woDAA', 'SPARTAN_naiveDAA', 'SPARTAN_naiveC']

    build_cd_diagram(
        data_path='.',  
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measure=bop_measure,
        subdir="output/classification", # change as needed
        savefile_path='./result/Section5_2/CD_daa_ablation_bag_of_pattern.pdf' # change as needed
    )
