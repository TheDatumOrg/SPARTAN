import os
from pathlib import Path
import pandas as pd

from util.helpers import build_cd_diagram


if __name__ == "__main__":

    methods = ['sax', 'sfa', 'spartan', 'spartan', 'spartan']
    itrs = ['a4_w8', 'a4_w8', 'a4_w8_full', 'a4_w8_rand', 'a4_w8_sample'] # change as needed
    display_names = ['SAX', 'SFA', 'SPARTAN', 'SPARTAN-R', 'SPARTAN-S']
    dist_measure = 'symbolic_l1' # symbolic_l1 distance for single pattern

    build_cd_diagram(
        data_path='.',  
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measure=dist_measure,
        subdir="output/classification", # change as needed
        savefile_path='./result/Section5_6/CD_5methods_single_pattern.pdf' # change as needed
    )

