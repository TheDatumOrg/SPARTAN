import os
import sys
import pandas as pd

from util.helpers import draw_tlb_cd_from_csv

if __name__ == "__main__":

    _ = draw_tlb_cd_from_csv(
        csv_path="result/Section5_3/tlb_results_finished.csv", # change as needed
        a=8, # change as needed
        w=8, # change as needed
        save_dir="result/Section5_3", # change as needed
        save_file_prefix="tlb_CD_plot", # change as needed
        alpha=0.05, # change as needed
    )