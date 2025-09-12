import os
import pandas as pd
from pathlib import Path
from util.Friedman_Nemenyi_test import draw_cd_diagram

def find_all_results(base_dir: str, methods: list, itrs: list) -> dict:
    """
    Collect all result CSVs for each method and itr.
    Returns:
        A dictionary: { method_name: { dataset_name: csv_path } }
    """
    results = {}
    for method, itr in zip(methods, itrs):
        method_results = {}
        search_root = Path(base_dir) / method
        for dataset_dir in search_root.glob(f"*/itr_{itr}/anomaly_results.csv"):
            dataset_name = dataset_dir.parent.parent.name  # e.g., KDD21-001_...
            method_results[dataset_name] = dataset_dir
        results[method] = method_results
    return results

def intersect_datasets(results_dict: dict) -> list:
    """Find common datasets across all methods"""
    dataset_sets = [set(method_data.keys()) for method_data in results_dict.values()]
    return sorted(set.intersection(*dataset_sets))

def load_metrics(results_dict: dict, datasets: list, metric: str) -> pd.DataFrame:
    """Load selected metric from all matching CSVs"""
    rows = []
    for method, dataset_csvs in results_dict.items():
        for dataset in datasets:
            csv_path = dataset_csvs[dataset]
            df = pd.read_csv(csv_path)
            val = float(df[metric].values[0]) if metric in df.columns else -1.0
            rows.append({
                "classifier_name": method,
                "dataset_name": dataset,
                "accuracy": val
            })
    return pd.DataFrame(rows).sort_values(by=["dataset_name", "classifier_name"])

def run_cd_diagram_pipeline(
    base_dir: str,
    methods: list,
    itrs: list,
    metric: str,
    alpha: float,
    save_path: str
):
    from pathlib import Path
    import os

    results_dict = find_all_results(base_dir, methods, itrs)
    common_datasets = intersect_datasets(results_dict)
    print(f"Using {len(common_datasets)} common datasets")

    results_df = load_metrics(results_dict, common_datasets, metric)
    os.makedirs(Path(save_path).parent, exist_ok=True)
    draw_cd_diagram(df_perf=results_df, name=save_path, alpha=alpha)
    results_df.to_csv(Path(save_path).with_suffix(".csv"), index=False)

if __name__ == "__main__":
    
    run_cd_diagram_pipeline(
        base_dir="./output/anomaly/result",
        methods=['sax', 'sfa', 'spartan', 'euclidean'],
        itrs=['a16_w16_win100', 'a16_w16_win100', 'a16_w16_win100', 'euc_win100'],
        metric="VUS-PR",
        alpha=0.05,
        save_path="./result/Section5_8/CD_anomaly_vuspr_euclidean.pdf"
    )

    run_cd_diagram_pipeline(
        base_dir="./output/anomaly/result",
        methods=['sax', 'sfa', 'spartan', 'euclidean'],
        itrs=['a16_w16_win100', 'a16_w16_win100', 'a16_w16_win100', 'euc_win100'],
        metric="VUS-ROC",
        alpha=0.05,
        save_path="./result/Section5_8/CD_anomaly_vusroc_euclidean.pdf"
    )