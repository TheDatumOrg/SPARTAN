import os
from pathlib import Path
import pandas as pd

from util.helpers import build_cd_diagram


if __name__ == "__main__":

    methods = ['sax', 'sfa', 'esax', 'sax_dr', '1dsax', 'tfsax', 'sax_vfd']
    itrs = ['a4_w12']
    display_names = [model_name.upper() for model_name in methods]
    dist_measure = 'symbolic_l1'

    build_cd_diagram(
        data_path='.',  
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measure=dist_measure,
        subdir="output/classification", # change as needed
        savefile_path='./result/Section5_1/CD_baseline_constrained.pdf' # change as needed
    )

    methods = ['sax', 'sfa', 'esax', 'sax_dr', '1dsax', 'tfsax', 'sax_vfd']
    itrs = ['a4_w12', 'a4_w12', 'a4_w36', 'a4_w24', 'a4_w24', 'a4_w24', 'a4_w48']
    display_names = [model_name.upper() for model_name in methods]
    dist_measure = 'symbolic_l1'

    build_cd_diagram(
        data_path='.',  
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measure=dist_measure,
        subdir="output/classification", # change as needed
        savefile_path='./result/Section5_1/CD_baseline_unconstrained.pdf' # change as needed
    )
