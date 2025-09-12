from pathlib import Path
from typing import List, Optional, Union, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util.helpers import compare_methods_scatter

if __name__ == "__main__":

    result_dir = "output/classification" # change as needed
    output_dir = "result/Section5_2" # change as needed
    sp_dist = "symbolic_l1" # symbolic_l1 as default

    methods = ['spartan', 'spartan']
    itrs = ['a4_w8_equiwidth', 'a4_w8']
    display_names = ['SPARTAN_Equi-width', 'SPARTAN_Equi-depth']
    dist_measures = sp_dist

    _ = compare_methods_scatter(
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measures=dist_measures,
        result_dir=result_dir,
        output_dir=output_dir,
        save_filename=f"1NN_comparison_plot_{display_names[0]}_{display_names[1]}_{itrs[-1]}_tab_binning_ablation.png",
        metric="acc",  
    )

    methods = ['sfa', 'sfa']
    itrs = ['a4_w8_equiwidth', 'a4_w8']
    display_names = ['SFA_Equi-width', 'SFA_Equi-depth']
    dist_measures = sp_dist

    _ = compare_methods_scatter(
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measures=dist_measures,
        result_dir=result_dir,
        output_dir=output_dir,
        save_filename=f"1NN_comparison_plot_{display_names[0]}_{display_names[1]}_{itrs[-1]}_tab_binning_ablation.png",
        metric="acc",  
    )
