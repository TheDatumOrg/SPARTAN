import os
import numpy as np
from util.helpers import draw_tlb_pairwise_plot

if __name__ == "__main__":

    draw_tlb_pairwise_plot(
        csv_path='result/Section5_3/tlb_results_finished.csv', # change as needed
        method1='SPARTAN_woDAA', # change as needed
        method2='SAX', # change as needed
        a=8, # change as needed
        w=8, # change as needed
        output_dir='./result/Section5_3', # change as needed
        save_filename='comparison_plot_SPARTAN_woDAA_SAX_a8w8.png' # change as needed
    )