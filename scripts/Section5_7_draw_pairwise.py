from pathlib import Path
from typing import List, Optional, Union, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util.helpers import compare_methods_scatter

if __name__ == "__main__":

    result_dir = "output/classification" # change as needed
    output_dir = "result/Section5_7" # change as needed

    # sax
    methods = ['sax', 'sax']
    itrs = ['a4_w8_exist_dist', 'a4_w8']
    display_names = ['SAX+MINDIST', 'SAX+SymbolicL1']
    dist_measures = ['sax_mindist', 'symbolic_l1']

    _ = compare_methods_scatter(
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measures=dist_measures,
        result_dir=result_dir,
        output_dir=output_dir,
        save_filename=f"1NN_comparison_plot_{display_names[0]}_{display_names[1]}_{itrs[-1]}_tab.png",
        metric="acc",  
    )

    # sfa
    methods = ['sfa', 'sfa']
    itrs = ['a4_w8_exist_dist', 'a4_w8']
    display_names = ['SFA+MINDIST', 'SFA+SymbolicL1']
    dist_measures = ['sfa_mindist', 'symbolic_l1']

    _ = compare_methods_scatter(
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measures=dist_measures,
        result_dir=result_dir,
        output_dir=output_dir,
        save_filename=f"1NN_comparison_plot_{display_names[0]}_{display_names[1]}_{itrs[-1]}_tab.png",
        metric="acc",  
    )

    # spartan
    methods = ['spartan', 'spartan']
    itrs = ['a4_w8_exist_dist', 'a4_w8']
    display_names = ['SPARTAN+MINDIST', 'SPARTAN+SymbolicL1']
    dist_measures = ['pca_mindist', 'symbolic_l1']

    _ = compare_methods_scatter(
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measures=dist_measures,
        result_dir=result_dir,
        output_dir=output_dir,
        save_filename=f"1NN_comparison_plot_{display_names[0]}_{display_names[1]}_{itrs[-1]}_tab.png",
        metric="acc",  
    )

    # sax_dr
    methods = ['sax_dr', 'sax_dr']
    itrs = ['a4_w8_exist_dist', 'a4_w8']
    display_names = ['SAX_DR+OWNDIST', 'SAX_DR+SymbolicL1']
    dist_measures = ['saxdr_mindist', 'symbolic_l1']

    _ = compare_methods_scatter(
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measures=dist_measures,
        result_dir=result_dir,
        output_dir=output_dir,
        save_filename=f"1NN_comparison_plot_{display_names[0]}_{display_names[1]}_{itrs[-1]}_tab.png",
        metric="acc",  
    )