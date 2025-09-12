import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# -----------------------
# Global style
# -----------------------
font = {'size': 25}
font_size = 35
matplotlib.rc('font', **font)

linewidth = 8
marker_size = 15

OUTDIR = "result/Section5_2/"
os.makedirs(OUTDIR, exist_ok=True)

# Fill in test results
lambdas = [0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0]

# Figure 1
fig1 = {
    "version": "fig1",
    "spartan": [0.612, 0.627, 0.631, 0.632, 0.632, 0.624, 0.618],
    "sax": 0.515,
    "sfa": 0.606,
}

# Figure 2
fig2 = {
    "version": "fig2",
    "spartan": [0.618, 0.646, 0.652, 0.658, 0.657, 0.667, 0.656],
    "sax": 0.608,
    "sfa": 0.560,
}

# -----------------------
# Plotting helper
# -----------------------
def plot_lambda_curve(lambdas, spartan_vals, sax_val, sfa_val, version):
    plt.figure(figsize=(20, 9))

    # SPARTAN curve
    plt.plot(
        lambdas,
        spartan_vals,
        marker='o',
        label='SPARTAN',
        markersize=marker_size,
        linewidth=linewidth
    )

    # Baseline lines
    plt.axhline(y=sax_val, color='rosybrown', linestyle='-.', label='SAX', linewidth=linewidth)
    plt.axhline(y=sfa_val, color='orange', linestyle='--', label='SFA', linewidth=linewidth)

    # Labels & legend
    plt.xlabel(r'Regularization parameter $\lambda$', fontsize=font_size)
    plt.ylabel('Accuracy', fontsize=font_size)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3)
    plt.grid(True)

    # Save + show
    outpath = os.path.join(OUTDIR, f"lambda_{version}.jpg")
    plt.savefig(outpath, dpi=500, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.show()


# -----------------------
# Make both figures
# -----------------------
plot_lambda_curve(lambdas, fig1["spartan"], fig1["sax"], fig1["sfa"], fig1["version"])
plot_lambda_curve(lambdas, fig2["spartan"], fig2["sax"], fig2["sfa"], fig2["version"])
