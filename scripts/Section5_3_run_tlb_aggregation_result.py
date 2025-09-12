import os
import sys
import pandas as pd


def find_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def aggregate_csv_files(directory):
    csv_files = find_csv_files(directory)
    df_list = [pd.read_csv(file) for file in csv_files]
    aggregated_df = pd.concat(df_list, ignore_index=True)
    return aggregated_df

# # Example usage
directory_path = 'output/tlb'  # Replace with your directory path
aggregated_df = aggregate_csv_files(directory_path)

# Display the first few rows of the aggregated DataFrame
print(aggregated_df.head())
print(aggregated_df.shape)

save_path = "result/Section5_3"
os.makedirs(save_path, exist_ok=True)
aggregated_df.to_csv(os.path.join(save_path, 'tlb_results_finished.csv'), index=False)
