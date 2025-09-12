import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 30})
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union, Literal, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from util.helpers import fetch_results_sorted

matplotlib.rcParams.update({'font.size': 30})

AggFn = Literal["mean", "sum"]


def load_cumulative_runtime(csv_path: str, alias: Optional[Dict[str, str]] = None) -> pd.Series:
    """
    Read a CSV with columns: dataset, method, train_time, pred_time
    Return a pandas Series mapping <normalized method> -> cumulative_runtime_sum.

    alias: optional dict mapping raw names to your canonical method IDs
           (e.g., {'saxdr': 'sax_dr', '1d-sax':'1dsax'})
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(p)

    # Normalize method names (lowercase, replace '-' with '_', strip)
    meth = (
        df["method"]
        .astype(str)
        .str.lower()
        .str.replace("-", "_", regex=False)
        .str.strip()
    )

    # Apply alias map if provided
    if alias:
        meth = meth.map(lambda m: alias.get(m, m))

    df = df.assign(
        method_norm=meth,
        train_time=pd.to_numeric(df["train_time"], errors="coerce").fillna(0.0),
        pred_time=pd.to_numeric(df["pred_time"], errors="coerce").fillna(0.0),
    )
    df["cum_time"] = df["train_time"] + df["pred_time"]

    # Sum across datasets for each method
    cum_runtime = df.groupby("method_norm")["cum_time"].sum()

    return cum_runtime  # pandas Series: index=method_norm, value=sum(runtime)


def plot_from_results(
    result_dir: str,
    methods: List[str],
    itrs: Union[List[str], str],
    display_names: List[str],
    dist_measure: str,
    *,
    x_metric: str,
    y_metric: str,
    x_override: Optional[np.ndarray] = None,
    x_agg: AggFn = "mean",
    y_agg: AggFn = "mean",
    x_ticks: Optional[List[Union[int, float]]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (13, 7),
    point_size: int = 600,
    ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    output_dir: Optional[str] = None,
    save_file_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    General scatter plot from experiment results with simple label annotation.

    Parameters
    ----------
    result_dir : str
        Base directory: result_dir/method/dataset/itr_xxx/classification_results.csv
    methods : list[str]
        Method directory names
    itrs : list[str] or str
        Iteration tags (one per method or a single one broadcasted)
    display_names : list[str]
        Names to show in plot (and annotation)
    dist_measure : str
        Metric filter (e.g. 'hist_euclidean')
    x_metric, y_metric : str
        Column names to plot (e.g. "runtime", "accuracy")
    x_override : np.ndarray, optional
        External values for x-axis (e.g. bit-budgets). If provided, skip aggregation for x.
    x_agg, y_agg : {"mean", "sum"}
        Aggregation method to compute classifier-level values from per-dataset results.
    """

    # Fetch results (accuracy + runtime available)
    results_sorted = fetch_results_sorted(
        data_root=result_dir,
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measure=dist_measure,
        subdir="",   # result_dir is already the root
    )

    # Aggregate
    if x_override is None:
        if x_agg == "mean":
            x_vals = results_sorted.groupby("classifier_name")[x_metric].mean().reindex(display_names).to_numpy()
        elif x_agg == "sum":
            x_vals = results_sorted.groupby("classifier_name")[x_metric].sum().reindex(display_names).to_numpy()
        else:
            raise ValueError("x_agg must be 'mean' or 'sum'")
    else:
        x_vals = np.asarray(x_override, dtype=float)

    if y_agg == "mean":
        y_vals = results_sorted.groupby("classifier_name")[y_metric].mean().reindex(display_names).to_numpy()
    elif y_agg == "sum":
        y_vals = results_sorted.groupby("classifier_name")[y_metric].sum().reindex(display_names).to_numpy()
    else:
        raise ValueError("y_agg must be 'mean' or 'sum'")

    # Default aesthetics
    if colors is None:
        colors = ['tab:blue', 'pink', 'green', 'purple', 'orange', 'brown', 'red'][:len(display_names)]
    if markers is None:
        markers = ['o', 'D', '^', 'v', 's', 'P', '*'][:len(display_names)]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    for i, lbl in enumerate(display_names):
        ax.scatter(x_vals[i], y_vals[i], color=colors[i], marker=markers[i], s=point_size)
        ax.text(x_vals[i], y_vals[i], lbl, fontsize=22, ha="center", va="bottom")

    ax.set_xlabel(x_label or x_metric.title())
    ax.set_ylabel(y_label or y_metric.title())

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.grid(color='gray', linestyle='dashed', axis='both')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    if output_dir and save_file_name:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(os.path.join(output_dir, save_file_name), dpi=500, bbox_inches="tight")

    plt.show()

    return results_sorted

if __name__ == "__main__":

    labels = ['1d-SAX', 'ESAX', 'SAX', 'SAX-DR', 'SAX_VFD', 'SFA', 'TFSAX']
    methods = ['1dsax', 'esax', 'sax', 'sax_dr', 'sax_vfd', 'sfa', 'tfsax']
    itrs = ['a4_w24', 'a4_w36', 'a4_w12', 'a4_w24', 'a4_w48', 'a4_w12', 'a4_w24']

    plot_from_results(
        result_dir="output/classification/", # change as needed
        methods=methods,
        itrs=itrs,
        display_names=labels,
        dist_measure="symbolic_l1",
        x_override=np.array([48, 72, 24, 48, 96, 24, 48]),
        x_metric="bit_budget",
        y_metric="accuracy",
        x_ticks=[12, 24, 36, 48, 72, 96],
        x_label="Bit-budget",
        y_label="Average Accuracy",
        output_dir="result/Section5_1/", # change as needed
        save_file_name="acc_vs_bitbudget.jpg", # change as needed
    )

    # Display names and model names
    labels  = ['1d-SAX', 'ESAX', 'SAX', 'SAX-DR', 'SAX_VFD', 'SFA', 'TFSAX']
    methods = ['1dsax',  'esax', 'sax', 'sax_dr',  'sax_vfd', 'sfa', 'tfsax']
    itrs    = ['a4_w12']  

    # Load cumulative runtime from previous run (change as needed)
    csv_path = "result/Section5_1/cumulative_runtime_results_comparison_7methods_128dataset.csv"

    cum_runtime_series = load_cumulative_runtime(csv_path)

    # Build x_override (runtime) in the same order as `methods`
    x_override = np.array([float(cum_runtime_series.get(m, 0.0)) for m in methods], dtype=float)

    # Plot Accuracy vs Cumulative Runtime (runtime is overridden by previous CSV totals)
    plot_from_results(
        result_dir="output/classification/",  # change as needed
        methods=methods,
        itrs=itrs,
        display_names=labels,
        dist_measure="symbolic_l1",
        x_metric="runtime",          # label only; real x comes from x_override
        y_metric="accuracy",
        x_override=x_override,       # use precomputed cumulative runtime
        y_agg="mean",                # average accuracy across datasets
        x_label="Cumulative Runtime",
        y_label="Average Accuracy",
        output_dir="result/Section5_1/", # change as needed
        save_file_name="acc_vs_total_runtime.jpg", # change as needed
    )
