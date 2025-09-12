import os
import numpy as np
import pandas as pd
from pathlib import Path

from functools import reduce
from typing import List, Optional, Union, Tuple

from .Friedman_Nemenyi_test import draw_cd_diagram
from .tools import draw_pairwise_plot

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


# Default dataset list (falls back to tools.univariate if not provided explicitly)
try:
    from .tools import univariate as DEFAULT_DATASETS
except Exception:
    DEFAULT_DATASETS = None


def _metric_filter(df: pd.DataFrame, method_name: str, dist_measure: str) -> pd.DataFrame:
    """Keep only rows for the metric of interest. Special-case 'euclidean'; otherwise use dist_measure."""
    if method_name in ['euclidean']:
        return df[df['metric'] == 'euclidean']
    return df[(df['metric'] == dist_measure)]


def fetch_results_sorted(
    data_root: str,
    methods: List[str],
    itrs: Union[List[str], str],
    display_names: List[str],
    dist_measure: str,
    *,
    datasets: Optional[List[str]] = None,
    subdir: str = "output/classification",
    missing_accuracy_fill: float = -1.0,
    missing_runtime_fill: float = -1.0,
    take_first_row: bool = True,
    metric_col: str = "acc",   
    result_csv_name: str = "classification_results.csv"
) -> pd.DataFrame:
    """
    Fetch per-dataset scores (mapped to 'accuracy') and runtimes for (method, itr, dist_measure).

    Expected file layout:
      <data_root>/<subdir>/<method>/<dataset>/itr_<itr>/<task>_results.csv

    Returns
    -------
    pd.DataFrame
        Sorted table with columns:
        ['classifier_name','dataset_name','accuracy','runtime']
        where 'accuracy' contains values from `metric_col`.
    """
    # Normalize itrs
    if isinstance(itrs, str):
        itrs = [itrs for _ in range(len(methods))]
    if isinstance(dist_measure, str):
        dist_measure = [dist_measure for _ in range(len(methods))]
    if len(itrs) == 1 and len(methods) > 1:
        itrs = [itrs[0] for _ in range(len(methods))]

    assert len(methods) == len(itrs), "methods and itrs must have the same length"
    assert len(methods) == len(display_names), "methods and display_names must have the same length"
    assert len(dist_measure) == len(methods), "methods and dist_measure must have the same length"

    # Datasets
    if datasets is None:
        if DEFAULT_DATASETS is None:
            raise ValueError("No datasets provided and tools.univariate is unavailable. Pass datasets=[...].")
        datasets = DEFAULT_DATASETS

    root = Path(data_root) / subdir
    rows = []

    for method, itr_tag, disp_name, dist in zip(methods, itrs, display_names, dist_measure):
        for dataset in datasets:
            run_dir = root / method / dataset / f"itr_{itr_tag}"
            result_csv = run_dir / result_csv_name

            if not result_csv.exists():
                rows.append({
                    "classifier_name": disp_name,
                    "dataset_name": dataset,
                    "accuracy": float(missing_accuracy_fill),
                    "runtime": float(missing_runtime_fill),
                })
                print(f"missing {disp_name} on {dataset}")
                continue
                

            df = pd.read_csv(result_csv)
            df = _metric_filter(df, method, dist)

            if df.empty or (metric_col not in df.columns):
                rows.append({
                    "classifier_name": disp_name,
                    "dataset_name": dataset,
                    "accuracy": float(missing_accuracy_fill),
                    "runtime": float(missing_runtime_fill),
                })
                print(f"missing {disp_name} on {dataset}")
                continue

            if take_first_row:
                score = float(df[metric_col].values[0])
                runtime = float(df["runtime"].values[0]) if "runtime" in df.columns else float(missing_runtime_fill)
            else:
                score = float(pd.to_numeric(df[metric_col], errors="coerce").astype(float).mean())
                runtime = float(pd.to_numeric(df["runtime"], errors="coerce").astype(float).mean()) if "runtime" in df.columns else float(missing_runtime_fill)

            rows.append({
                "classifier_name": disp_name,
                "dataset_name": dataset,
                "accuracy": score,    # keep column name 'accuracy' for CD diagram compatibility
                "runtime": runtime,
            })

    results = pd.DataFrame(rows)
    results_sorted = results.sort_values(by=["dataset_name", "classifier_name"], ignore_index=True)
    valid_counts = (
    results_sorted[results_sorted["accuracy"] > 0]
        .groupby("classifier_name")["dataset_name"]
        .nunique()
    )
    print(valid_counts)
    return results_sorted

def build_cd_diagram(
    data_path: str,
    methods: List[str],
    itrs: Union[List[str], str],
    display_names: List[str],
    dist_measure: str,
    savefile_path: str,
    *,
    alpha: float = 0.05,
    subdir: str = "output/classification",
    result_csv_name: str = "classification_results.csv",
    export_csv: bool = False,
    metric_col: str = "acc",   # <-- NEW: choose which column to treat as the score
) -> pd.DataFrame:
    """
    Build and save a Critical Difference (CD) diagram with Friedman-Nemenyi test results.
    Uses fetch_results_sorted to collect results. The selected metric column is mapped to 'accuracy'.
    """
    # Fetch results (maps metric_col -> 'accuracy' for compatibility with draw_cd_diagram)
    results_sorted = fetch_results_sorted(
        data_root=data_path,
        methods=methods,
        itrs=itrs,
        display_names=display_names,
        dist_measure=dist_measure,
        subdir=subdir,
        metric_col=metric_col,   # acc, ri, or other valid metrics
        result_csv_name=result_csv_name

    )

    # Report averages for the chosen metric
    print(f"Average {metric_col} by classifier:")
    print(results_sorted.groupby("classifier_name")["accuracy"].mean())

    # Ensure output directory exists
    save_path = Path(savefile_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Draw and save CD diagram
    draw_cd_diagram(df_perf=results_sorted, name=str(save_path), alpha=alpha)

    # Optional CSV
    if export_csv:
        results_sorted.to_csv(save_path.with_suffix(".csv"), index=False)

    return results_sorted


def create_numpy_dataset(
        name = 'ArrowHead',
        path='data/ucr/Univariate_ts/',
        return_meta_data =False,
        equalize_length=True,
        fill_missing = True,
        resample=False,
        test_size=0.3,
        random_state=0
    ):
    train_dataset_path = os.path.join(path,'{}/{}_{}.ts'.format(name, name,'TRAIN'))
    X_train,y_train,train_meta_data = load_from_tsfile(train_dataset_path,return_meta_data=True)

    test_dataset_path = os.path.join(path,'{}/{}_{}.ts'.format(name, name,'TEST'))
    test_dataset_path = path +'/{}/{}_{}.ts'.format(name, name,'TEST')
    X_test,y_test,test_meta_data = load_from_tsfile(test_dataset_path,return_meta_data=True)
        
    if (not (train_meta_data['equallength'] and test_meta_data['equallength'])) and equalize_length:
        max_len = max([X_train.shape[2],X_test.shape[2]])

        extend_tmp = []
        if(X_train.shape[2] < X_test.shape[2]):
            X_train = np.pad(X_train, [(0,0),(0,0),(0,max_len - X_train.shape[2])],mode = 'constant',constant_values=np.nan)
        elif(X_train.shape[2] > X_test.shape[2]):
            X_test = np.pad(X_test, [(0,0),(0,0),(0,max_len - X_test.shape[2])],mode = 'constant',constant_values=np.nan)

        X_train = resample_length(X_train,max_len)
        X_test = resample_length(X_test,max_len)
    if (train_meta_data['missing'] or test_meta_data['missing']) and fill_missing:
        X_train = fill_nontrailing_missing(X_train)

        X_test = fill_nontrailing_missing(X_test)

    if resample:
        X_full = np.concatenate([X_train,X_test],axis=0)
        y_full = np.concatenate([y_train,y_test],axis=0)

        X_train,X_test,y_train,y_test = train_test_split(X_full,y_full,test_size=test_size)
    
    return X_train,y_train,X_test,y_test   



def compare_methods_scatter(
    *,
    methods: List[str],
    itrs: Union[str, List[str]],
    display_names: List[str],
    dist_measures: Union[str, List[str]],
    result_dir: str,
    output_dir: str,
    save_filename: str,
    metric: str = "acc",
    datasets: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 8),
    font_size: int = 35,
    marker_size: int = 65,
    missing_sentinel: float = -1.0,
    result_csv_name: str = "classification_results.csv"
) -> pd.DataFrame:
    """
    Compare two methods across datasets on a given metric and plot M0 vs M1 scatter with Win/Tie/Loss table.

    Parameters
    ----------
    methods : list[str]
        Method directory names (length 2).
    itrs : str | list[str]
        Iteration tags per method; a single string will be broadcast to both.
    display_names : list[str]
        Two names to show in the figure (length 2).
    dist_measures : str | list[str]
        Distance/metric selector in classification_results.csv; a single string broadcasts.
    result_dir : str
        Base directory that contains: <result_dir>/<method>/<dataset>/itr_<itr>/classification_results.csv
    output_dir : str
        Directory to save the figure.
    save_filename : str
        Output image filename (e.g., '1NN_comparison_plot_A_vs_B.png').
    metric : str
        Column to compare (e.g., 'acc', 'ri', 'precision', 'recall', 'f1', ...).
    datasets : list[str] | None
        Which datasets to scan. Defaults to tools.univariate if available; otherwise required.
    figsize : (w, h)
        Matplotlib figure size.
    font_size : int
        Axis label font size.
    marker_size : int
        Scatter marker size.
    missing_sentinel : float
        Value used when a dataset result is missing.

    Returns
    -------
    pd.DataFrame
        Results table with columns ['method','dataset',<metric>, 'precision','recall','f1'] (where present),
        filtered to datasets that have non-missing values for BOTH methods.
    """
    assert len(methods) == 2, "This wrapper currently compares exactly two methods."
    assert len(display_names) == 2, "display_names must have length 2."

    # Broadcast itrs / dist_measures if singletons are provided
    if isinstance(itrs, str):
        itrs = [itrs, itrs]
    if isinstance(dist_measures, str):
        dist_measures = [dist_measures, dist_measures]

    assert len(itrs) == 2, "itrs must be length 2 (or a single string to broadcast)."
    assert len(dist_measures) == 2, "dist_measures must be length 2 (or a single string to broadcast)."

    # Datasets
    if datasets is None:
        if DEFAULT_DATASETS is None:
            raise ValueError("No datasets provided and tools.univariate unavailable. Pass datasets=[...].")
        datasets = DEFAULT_DATASETS

    # Collect results per method/dataset
    rows = []
    for i, method in enumerate(methods):
        itr = itrs[i]
        disp = display_names[i]
        dist_measure = dist_measures[i]

        for dataset in datasets:
            run_dir = Path(result_dir) / method / dataset / f"itr_{itr}"
            result_csv = run_dir / result_csv_name

            if not result_csv.exists():
                rows.append({"method": disp, "dataset": dataset, metric: missing_sentinel})
                continue

            df = pd.read_csv(result_csv)
            # Keep last matching row for the chosen metric selector

            # df = df[df["metric"] == dist_measure]
            if df.empty:
                rows.append({"method": disp, "dataset": dataset, metric: missing_sentinel})
                continue

            row = df.iloc[-1, :]

            # Build a safe row dict
            out = {"method": disp, "dataset": dataset}
            # Include the requested metric (or missing)
            out[metric] = float(row[metric]) if metric in row.index else missing_sentinel
            # Optionally include other known columns if present (helps inspection)
            for extra in ("precision", "recall", "f1"):
                if extra in row.index:
                    try:
                        out[extra] = float(row[extra])
                    except Exception:
                        pass

            rows.append(out)

    results = pd.DataFrame(rows)

    # Keep only datasets where BOTH methods have non-missing values for the chosen metric
    m0, m1 = display_names
    results_pivot = results.pivot(index="dataset", columns="method", values=metric)
    mask_valid = (results_pivot[m0] != missing_sentinel) & (results_pivot[m1] != missing_sentinel)
    completed_datasets = results_pivot.index[mask_valid].tolist()

    results_nz = results[results["dataset"].isin(completed_datasets)].copy()
    # Sort inside each method to align pairs visually if needed later
    res_m0 = results_nz[results_nz["method"] == m0].sort_values(by=["dataset"])
    res_m1 = results_nz[results_nz["method"] == m1].sort_values(by=["dataset"])

    v0 = res_m0[metric].to_numpy(dtype=float)
    v1 = res_m1[metric].to_numpy(dtype=float)

    # Win/Tie/Loss (larger is better)
    wins = v1 > v0
    ties = v1 == v0
    losses = v1 < v0

    win_count, tie_count, loss_count = int(wins.sum()), int(ties.sum()), int(losses.sum())
    denom = max(1, len(completed_datasets))  # avoid division by zero

    print(f"{m0} vs {m1} on '{metric}'")
    print(f"Win %:  {win_count / denom:.3f}")
    print(f"Tie %:  {tie_count / denom:.3f}")
    print(f"Loss %: {loss_count / denom:.3f}")

    # --- Plot ---
    plt.rcParams.update({'font.size': 30})
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # Light green upper triangle region (m0 >= m1)
    t1 = plt.Polygon([[0, 0], [0, 1], [1, 1]], color="green", alpha=0.25)
    ax.add_patch(t1)

    # Scatter
    ax.scatter(x=v0, y=v1, c="darkblue", s=marker_size)

    ax.set_xlabel(m0, fontsize=font_size)
    ax.set_ylabel(m1, fontsize=font_size)

    # Diagonal y=x
    ax.axline((0, 0), (1, 1), color="tomato", linewidth=3)

    # If metric is bounded in [0,1], use that; otherwise auto-scale
    if np.all((v0 >= 0) & (v0 <= 1)) and np.all((v1 >= 0) & (v1 <= 1)):
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Win/Tie/Loss table
    rows_lbl = ["Win", "Tie", "Loss"]
    columns_lbl = ["Statistics"]
    cell_text = [[str(win_count)], [str(tie_count)], [str(loss_count)]]
    tab = plt.table(
        cellText=cell_text,
        rowLabels=rows_lbl,
        colLabels=columns_lbl,
        colWidths=[0.15] * 3,
        loc="lower right",
    )
    tab.set_fontsize(30)
    tab.scale(3.1, 3.1)

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / save_filename
    plt.savefig(out_path.as_posix(), dpi=500, bbox_inches="tight")
    plt.close(fig)

    return results_nz


def plot_tlb_3d_for_dataset(
    dataset_name: str,
    csv_path: str,
    draw_version: str,
    save_dir: str,
    save_name: str,
    *,
    a_fixed: int = 10,
    w_fixed: int = 10,
    cmap = cm.YlGnBu,
    alpha: float = 0.5,
    figsize = (10, 6),
    zlim = (0.0, 1.19),
    view_elev: float = 30,
    view_azim: float = -60,
) -> None:
    # --- load & normalize (unchanged) ---
    p = Path(csv_path)
    csv_file = p / f"{dataset_name}_tlb_results.csv" if p.is_dir() else p
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")

    df = pd.read_csv(csv_file)
    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str) == str(dataset_name)].copy()
        if df.empty:
            raise ValueError(f"No rows for dataset='{dataset_name}' in {csv_file}")

    def _norm_method(row):
        m = str(row.get("method", "")).lower().strip()
        param = str(row.get("param", "")).lower().strip()
        if m == "spartan":
            if param in ("wodaa", "none"):
                return "SPARTAN_woDAA"
            if param.startswith("daa"):
                return "SPARTAN"
            return "SPARTAN"
        if m == "sfa":
            return "SFA"
        if m == "sax":
            return "SAX"
        return m.upper()

    for col in ("a", "w", "tlb"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_file}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["method"] = df.apply(_norm_method, axis=1)

    if draw_version.lower() == "ablation":
        desired_methods = ["SPARTAN", "SPARTAN_woDAA", "SFA", "SAX"]
    else:
        desired_methods = ["SPARTAN", "SFA", "SAX"]

    present_methods = [m for m in desired_methods if (df["method"] == m).any()]
    if not present_methods:
        raise ValueError("No matching methods found in the data to plot.")

    def choose_fixed(fixed: str, target_val: int) -> int:
        avail = np.sort(df[fixed].dropna().unique())
        if len(avail) == 0:
            raise ValueError(f"No values available for '{fixed}' in the data.")
        return int(target_val if target_val in avail else avail[np.argmin(np.abs(avail - target_val))])

    def available_values(vary: str, fixed: str, fixed_val: int) -> np.ndarray:
        vals = np.sort(df[df[fixed] == fixed_val][vary].dropna().unique())
        return vals

    w_use = choose_fixed("w", w_fixed)
    a_use = choose_fixed("a", a_fixed)

    a_values = available_values("a", "w", w_use)
    if a_values.size == 0:
        a_values = np.sort(df["a"].dropna().unique())

    w_values = available_values("w", "a", a_use)
    if w_values.size == 0:
        w_values = np.sort(df["w"].dropna().unique())

    def collect_column(mname: str, fixed: str, fixed_val: int, vary: str, vary_vals: np.ndarray) -> np.ndarray:
        out = []
        for vv in vary_vals:
            sub = df[(df["method"] == mname) & (df[fixed] == fixed_val) & (df[vary] == vv)]
            out.append(float(sub.iloc[-1]["tlb"]) if not sub.empty else np.nan)
        return np.array(out, dtype=float)

    data_top = None
    if a_values.size > 0:
        top_cols = [collect_column(m, "w", w_use, "a", a_values) for m in present_methods]
        data_top = np.nan_to_num(np.column_stack(top_cols) if top_cols else np.empty((0, 0)), nan=0.0)

    data_bottom = None
    if w_values.size > 0:
        bot_cols = [collect_column(m, "a", a_use, "w", w_values) for m in present_methods]
        data_bottom = np.nan_to_num(np.column_stack(bot_cols) if bot_cols else np.empty((0, 0)), nan=0.0)

    # --- COLORS (legacy palette) ---
   
    legacy_positions = np.array([2.0, 1.5, 0.8, 0.0], dtype=float)
    legacy_norm = Normalize(vmin=0, vmax=3)  # fixed, independent of method count
    row_colors_full = cmap(legacy_norm(legacy_positions))
    row_colors = row_colors_full[:len(present_methods)]  # take as many as needed (3 or 4)

    # --- Plot ---
    fig = plt.figure(figsize=figsize)
    x_scale, dx, dy = 0.4, 0.3, 0.8

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    if (data_top is not None) and (data_top.size > 0) and (a_values.size > 0):
        for i in range(data_top.shape[1]):
            xpos, ypos = np.meshgrid(x_scale * i, a_values)
            ax.bar3d(xpos.flatten(), ypos.flatten(), np.zeros_like(xpos).flatten(),
                     dx, dy, data_top[:, i].flatten(),
                     shade=True, alpha=alpha, color=row_colors[i])
        ax.set_xticks(np.arange(len(present_methods)) * x_scale + dx / 2)
        ax.set_xticklabels(present_methods, rotation=10, ha="right")
        ax.set_yticks(a_values); ax.set_ylabel("Alphabet Size")
        ax.set_zlabel("TLB"); ax.set_zlim(*zlim)
        ax.view_init(elev=view_elev, azim=view_azim)
    else:
        ax.text(0.5, 0.5, 0.5, "No data for (fix w, vary a)", ha="center", va="center")
        ax.set_axis_off()

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    if (data_bottom is not None) and (data_bottom.size > 0) and (w_values.size > 0):
        for i in range(data_bottom.shape[1]):
            xpos, ypos = np.meshgrid(x_scale * i, w_values)
            ax.bar3d(xpos.flatten(), ypos.flatten(), np.zeros_like(xpos).flatten(),
                     dx, dy, data_bottom[:, i].flatten(),
                     shade=True, alpha=alpha, color=row_colors[i])
        ax.set_xticks(np.arange(len(present_methods)) * x_scale + dx / 2)
        ax.set_xticklabels(present_methods, rotation=10, ha="right")
        ax.set_yticks(w_values); ax.set_ylabel("Word Length")
        ax.set_zlabel("TLB"); ax.set_zlim(*zlim)
        ax.view_init(elev=view_elev, azim=view_azim)
    else:
        ax.text(0.5, 0.5, 0.5, "No data for (fix a, vary w)", ha="center", va="center")
        ax.set_axis_off()

    plt.tight_layout()
    plt.suptitle(f"TLB on {dataset_name}")

    out_dir = Path(save_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base, ext = os.path.splitext(save_name)
    if draw_version.lower() == "ablation" and "_ablation" not in base:
        save_name = f"{base}_ablation{ext or '.jpg'}"
    plt.savefig((out_dir / save_name).as_posix(), dpi=500, bbox_inches="tight")
    plt.close()


def draw_tlb_cd_from_csv(
    *,
    csv_path: str,
    a: int,
    w: int,
    save_dir: str,
    save_file_prefix: str = "tlb_CD_plot",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Build a CD plot from a pre-aggregated TLB CSV.

    Expected columns in csv:
      ['dataset','method','param','a','w','tlb', ...]
    The function:
      - filters rows by (a, w)
      - normalizes method names (SPARTAN/woDAA/SFA/SAX; others uppercased)
      - renames to the schema expected by draw_cd_diagram
      - saves <prefix>_{Ndatasets}_a{a}w{w}.pdf and .csv to save_dir
      - returns the DataFrame used for the CD plot
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # basic checks & numeric coercion
    for col in ("a", "w", "tlb"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # select the (a, w) slice
    df = df[(df["a"] == a) & (df["w"] == w)].copy()
    if df.empty:
        raise ValueError(f"No rows for a={a}, w={w} in {csv_path}")
    

    # normalize method labels
    def _normalize_method(row):
        m = str(row.get("method", "")).lower().strip()
        param = str(row.get("param", "")).lower().strip()
        if m == "spartan":
            if param in ("wodaa", "none"):
                return "SPARTAN_woDAA"
            if param.startswith("daa") or param.startswith("dp"):
                return "SPARTAN"
            return "SPARTAN"
        if m == "sfa" and param == "none":
            return "SFA"
        if m == "sax":
            return "SAX"
        return m.upper()

    df.loc[:, "method"] = df.apply(_normalize_method, axis=1)

    # rename for draw_cd_diagram
    plot_df = df.rename(
        columns={
            "dataset": "dataset_name",
            "method": "classifier_name",
            "tlb": "accuracy",
        }
    )[["dataset_name", "classifier_name", "accuracy"]].copy()

    # save outputs
    n_datasets = plot_df["dataset_name"].nunique()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    base = f"{save_file_prefix}_{n_datasets}dataset_{df['method'].nunique()}models_a{a}w{w}"
    pdf_path = save_dir / f"{base}.pdf"
    print("save CD results to: ", pdf_path)
    csv_out = save_dir / f"{base}.csv"

    draw_cd_diagram(df_perf=plot_df, name=str(pdf_path), alpha=alpha)
    plot_df.to_csv(csv_out, index=False)

    # optional: quick summary
    avg = plot_df.groupby("classifier_name")["accuracy"].mean()
    print("Average accuracy (TLB) by classifier:")
    print(avg)

    return plot_df



def draw_tlb_pairwise_plot(
    csv_path: str,
    method1: str,
    method2: str,
    a: int,
    w: int,
    output_dir: str,
    save_filename: str,
):
    """
    Draws a pairwise TLB scatter plot comparing two methods.

    Args:
        csv_path: Path to the aggregated CSV file containing TLB results.
        method1: Name of the first method (e.g., 'SPARTAN').
        method2: Name of the second method (e.g., 'SPARTAN_woDAA').
        a: Alphabet size to filter by.
        w: Word length to filter by.
        output_dir: Directory where the plot will be saved.
        save_filename: Output image file name (e.g., 'comparison_plot_SPARTAN_SPARTAN_woDAA_a4w4.png').
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[(df['a'] == a) & (df['w'] == w)].copy()

    def normalize_method(row):
        m = row['method'].lower()
        p = str(row.get('param', '')).lower()
        if m == 'spartan':
            return 'SPARTAN_woDAA' if p == 'wodaa' else 'SPARTAN'
 
        return row['method'].upper()

    df['method'] = df.apply(normalize_method, axis=1)
    df = df.rename(columns={'tlb': 'accuracy'})

    # Draw pairwise plot
    draw_pairwise_plot(
        df,
        [method2, method1],
        output_dir=output_dir,
        output_filename=save_filename,
    )

    print(f"Pairwise plot saved to: {os.path.join(output_dir, save_filename)}")