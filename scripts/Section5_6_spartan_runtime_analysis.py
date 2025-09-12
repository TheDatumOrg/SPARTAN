import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

font = {'size': 25}

matplotlib.rc('font', **font)

# Draw bar plot (runtime analysis of SPARTAN)

# Fill in with runtime experiment results
categories = ['StarLightCurves', 'FordA', 'StarLightCurves', 'FordA']
training_time_full = {
    'pca': [3.4e-04, 5.8e-05],         # pca runtime
    'daa': [1.8e-06, 2.5e-07],        # daa runtime
    'binning': [5.1e-07, 4.3e-07], # binning runtime
    'mapping': [2.7e-07, 1.3e-07],  # mapping runtime
}

training_time_full_total = [3.42e-04, 5.88e-05]

inference_time_full = {
    'pca': [3.0e-06, 1.2e-06],    # pca runtime
    'mapping': [1.4e-07, 1.6e-07],  # mapping runtime
}

inference_time_full_total = [3.14e-06, 1.36e-06]

training_time_rand = {
    'pca': [3.1e-05, 1.4e-05],         # pca runtime
    'daa': [1.7e-06, 2.4e-07],        # daa runtime
    'binning': [5.1e-07, 4.5e-07], # binning runtime
    'mapping': [2.0e-07, 1.4e-07],  # mapping runtime
}

training_time_rand_total = [3.34e-05, 1.48e-05]

inference_time_rand = {
    'pca': [2.6e-06, 9.8e-07],    # pca runtime
    'mapping': [1.4e-07, 1.5e-07],  # mapping runtime
}

inference_time_rand_total = [2.74e-06, 1.13e-06]

bar_width = 0.35  # Width of bars



# Data

# draw_version = 'total runtime' # total runtime or decompose runtime

for draw_version in ['decompose runtime', 'total runtime']:

    # Create figure and axes

    fig, ax = plt.subplots(figsize=(18.5, 7))

    hatch_list = ['/', 'o']
    color_list = ['steelblue', 'peru', 'salmon', 'palegoldenrod']

    if draw_version == 'total runtime':

        # Bar positions
        x = np.arange(4)

        # Training bars (spartan)
        ax.bar(x - bar_width/2, training_time_full_total+inference_time_full_total, bar_width, label='SPARTAN', color="#CCCCCC", edgecolor='black', hatch=hatch_list[0])
        ax.bar(x + bar_width/2, training_time_rand_total+inference_time_rand_total, bar_width, label='SPARTAN-R', color="#CCCCCC", edgecolor='black', hatch=hatch_list[1])

        # Labels and titles
        ax.set_ylabel('Average Runtime per Query (s)', fontsize=30)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.set_ylim([0,4e-4])
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=30)
        # ax.legend(fontsize=25)
        ax.grid(color='gray', linestyle='dashed', axis='y')

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=25)

    elif draw_version == 'decompose runtime':

        # Training bars (spartan)
        x = np.arange(2)
        ax.bar(x - bar_width/2, np.array(training_time_full['mapping']) / np.array(training_time_full_total), bar_width, label='Mapping', color=color_list[0], edgecolor='black', hatch=hatch_list[0])
        ax.bar(x - bar_width/2, np.array(training_time_full['binning']) / np.array(training_time_full_total), bar_width, bottom=np.array(training_time_full['mapping']) / np.array(training_time_full_total), label='Binning', color=color_list[1], edgecolor='black', hatch=hatch_list[0])
        ax.bar(x - bar_width/2, np.array(training_time_full['daa']) / np.array(training_time_full_total), bar_width, bottom=(np.array(training_time_full['mapping']) + np.array(training_time_full['binning'])) / np.array(training_time_full_total), label='DAA', color=color_list[2], edgecolor='black', hatch=hatch_list[0])
        ax.bar(x - bar_width/2, np.array(training_time_full['pca']) / np.array(training_time_full_total), bar_width, bottom=(np.array(training_time_full['mapping']) + np.array(training_time_full['daa']) + np.array(training_time_full['binning'])) / np.array(training_time_full_total), label='Approximation', color=color_list[3], edgecolor='black', hatch=hatch_list[0])

        # Training bars (SPARTAN-R)
        ax.bar(x + bar_width/2, np.array(training_time_rand['mapping']) / np.array(training_time_rand_total), bar_width, label='Mapping', color=color_list[0], edgecolor='black', hatch=hatch_list[1])
        ax.bar(x + bar_width/2, np.array(training_time_rand['binning']) / np.array(training_time_rand_total), bar_width, bottom=np.array(training_time_rand['mapping']) / np.array(training_time_rand_total), label='Binning', color=color_list[1], edgecolor='black', hatch=hatch_list[1])
        ax.bar(x + bar_width/2, np.array(training_time_rand['daa']) / np.array(training_time_rand_total), bar_width, bottom=(np.array(training_time_rand['mapping']) + np.array(training_time_rand['binning'])) / np.array(training_time_rand_total), label='DAA', color=color_list[2], edgecolor='black', hatch=hatch_list[1])
        ax.bar(x + bar_width/2, np.array(training_time_rand['pca']) / np.array(training_time_rand_total), bar_width, bottom=(np.array(training_time_rand['mapping']) + np.array(training_time_rand['daa']) + np.array(training_time_rand['binning'])) / np.array(training_time_rand_total), label='Approximation', color=color_list[3], edgecolor='black', hatch=hatch_list[1])

        
        # Inference bars (spartan)
        x = np.arange(4)[2:]
        ax.bar(x - bar_width/2, np.array(inference_time_full['mapping']) / np.array(inference_time_full_total), bar_width, label='Mapping', color=color_list[0], edgecolor='black', hatch=hatch_list[0])
        ax.bar(x - bar_width/2, np.array(inference_time_full['pca']) / np.array(inference_time_full_total), bar_width, bottom=(np.array(inference_time_full['mapping']) ) / np.array(inference_time_full_total), label='Approximation', color=color_list[3], edgecolor='black', hatch=hatch_list[0])

        ax.bar(x + bar_width/2, np.array(inference_time_rand['mapping']) / np.array(inference_time_rand_total), bar_width, label='Mapping', color=color_list[0], edgecolor='black', hatch=hatch_list[1])
        ax.bar(x + bar_width/2, np.array(inference_time_rand['pca']) / np.array(inference_time_rand_total), bar_width, bottom=(np.array(inference_time_rand['mapping']) ) / np.array(inference_time_rand_total), label='Approximation', color=color_list[3], edgecolor='black', hatch=hatch_list[1])

        # Labels and titles
        ax.set_ylabel('Percentage of Runtime', fontsize=30)
        # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_ylim([0,1.2])
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(categories, fontsize=30)
        # ax.legend(fontsize=25)
        ax.grid(color='gray', linestyle='dashed', axis='y')

        component_legend = ax.legend(loc='upper right', title="Components")
        # Additional legend for SPARTAN and SPARTAN-X using rectangular patches
        method_patches = [
            mpatches.Patch(facecolor='white', edgecolor='black', label='SPARTAN', hatch=hatch_list[0], linewidth=2),
            mpatches.Patch(facecolor='white', edgecolor='black', label='SPARTAN-R', hatch=hatch_list[1], linewidth=2)
        ]
        handles, labels = ax.get_legend_handles_labels()
        combined_handles = method_patches + handles
        combined_labels = ['SPARTAN', 'SPARTAN-R'] + labels

        # print(labels)
        # print(handles)
        # ax.legend(combined_handles[:6], combined_labels[:6], loc='upper right', fontsize=25)
        ax.legend(combined_handles[:2] + combined_handles[2:6][::-1], combined_labels[:2] + combined_labels[2:6][::-1], loc='upper center', bbox_to_anchor=(0.45, 1.25), ncol=3, fontsize=24)

    # Display the plot
    save_dir = "result/Section5_6"
    os.makedirs(save_dir, exist_ok=True)

    if draw_version == 'total runtime':
        plt.savefig(os.path.join(save_dir, 'SPARTAN_RUNTIME_total_runtime.jpg'), dpi=500, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(save_dir, 'SPARTAN_RUNTIME_decompose_runtime_components.jpg'), dpi=500, bbox_inches='tight')
    plt.show()
