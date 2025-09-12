import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from util.helpers import plot_tlb_3d_for_dataset

if __name__ == "__main__":

    dataset_name = "Ham"
    plot_tlb_3d_for_dataset(
        dataset_name=dataset_name, # change as needed
        csv_path="output/tlb",         # change as needed
        draw_version="ablation",       # change as needed (ablation or normal, ablation also displays with SPARTAN w/o DAA)
        save_dir="result/Section5_3",
        save_name=f"{dataset_name}_a10_w10.jpg", # change as needed
        a_fixed=10,                    # fixed alpha for displaying  
        w_fixed=10,                    # fixed omega for displaying  
        alpha=0.5,                     # transparency for bars
    )
