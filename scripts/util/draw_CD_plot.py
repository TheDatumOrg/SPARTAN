import os
from pathlib import Path
import pandas as pd

from .Friedman_Nemenyi_test import draw_cd_diagram
from .tools import univariate


# def build_cd_diagram(
#     data_path: str,
#     methods: list,
#     itrs: list,
#     display_names: list,
#     dist_measure: str,
#     savefile_path: str,
#     *,
#     alpha: float = 0.05,
#     subdir: str = "output/classification",
#     export_csv: bool = False
# ) -> None:
#     """
#     Build and save a Critical Difference (CD) diagram with Friedman-Nemenyi test results.

#     Parameters
#     ----------
#     data_path : str
#         Root directory of experiment results.
#         Each result is expected at:
#         <data_path>/<subdir>/<method>/<dataset>/itr_<itr>/classification_results.csv
#     methods : list[str]
#         List of method names (used for directory names and metric filtering).
#     itrs : list[str]
#         Iteration identifiers. If only one element is given, it will be applied to all methods.
#     display_names : list[str]
#         List of method names to show on the diagram.
#     dist_measure : str
#         Metric column filter (e.g., 'symbol_l1').
#     savefile_path : str
#         Full path to save the CD diagram PDF.
#         The CSV summary will also be saved if export_csv=True.
#     alpha: float
#         Alpha value for statistical test. Default is 0.05.
#     subdir : str, optional
#         Subdirectory under data_path. Default is "output/classification".
#     export_csv : bool, optional
#         If True, also saves the result table as CSV.

#     Returns
#     -------
#     None
#     """
#     # Handle single iteration case
#     if len(itrs) == 1:
#         itrs = [itrs[0] for _ in range(len(methods))]

#     assert len(methods) == len(itrs), "methods and itrs must have the same length"
#     assert len(methods) == len(display_names), "methods and display_names must have the same length"

#     root = Path(data_path) / subdir
#     results = pd.DataFrame()

#     def _metric_filter(df: pd.DataFrame, method_name: str, dist_measure: str) -> pd.DataFrame:
#         """Filter results by method-specific metric rules."""
#         if method_name in ['euclidean']:
#             return df[df['metric'] == 'euclidean']
#         else:
#             return df[df['metric'] == dist_measure]

#     # Collect results across datasets
#     for i, method in enumerate(methods):
#         itr_tag = itrs[i]
#         disp_name = display_names[i]

#         for dataset in univariate:
#             run_dir = root / method / dataset / f"itr_{itr_tag}"
#             result_csv = run_dir / "classification_results.csv"

#             if not run_dir.exists():
#                 print("Missing directory:", str(run_dir))
#                 dataset_result = {'classifier_name': disp_name, 'dataset_name': dataset, 'accuracy': -1.0}
#             elif len(list(run_dir.iterdir())) == 0:
#                 print("Empty directory:", str(run_dir))
#                 dataset_result = {'classifier_name': disp_name, 'dataset_name': dataset, 'accuracy': -1.0}
#             else:
#                 if not result_csv.exists():
#                     print("Missing file:", str(result_csv))
#                     dataset_result = {'classifier_name': disp_name, 'dataset_name': dataset, 'accuracy': -1.0}
#                 else:
#                     df = pd.read_csv(result_csv)
#                     df = _metric_filter(df, method, dist_measure)

#                     if df.empty:
#                         print("No matching metric:", str(run_dir))
#                         dataset_result = {'classifier_name': disp_name, 'dataset_name': dataset, 'accuracy': -1.0}
#                     else:
#                         acc = float(df['acc'].values[0])  # Take first row if multiple experiments
#                         dataset_result = {'classifier_name': disp_name, 'dataset_name': dataset, 'accuracy': acc}

#             results = pd.concat([results, pd.DataFrame([dataset_result])], ignore_index=True)

#     # Sort and report average accuracy
#     results_sorted = results.sort_values(by=['dataset_name', 'classifier_name'])
#     avg_acc = results_sorted.groupby('classifier_name')['accuracy'].mean()
#     print("Average accuracy by classifier:")
#     print(avg_acc)

#     # Ensure save directory exists
#     savefile_path = Path(savefile_path)
#     savefile_path.parent.mkdir(parents=True, exist_ok=True)

#     # Draw and save CD diagram
#     draw_cd_diagram(df_perf=results_sorted, name=str(savefile_path), alpha=alpha)

#     # Optionally export CSV
#     if export_csv:
#         csv_path = savefile_path.with_suffix(".csv")
#         results_sorted.to_csv(csv_path, index=False)


