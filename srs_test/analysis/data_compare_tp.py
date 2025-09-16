import json
import os
import glob
import pandas as pd
import re

TARGET_MAP = {
    # 'tp_copy_distance1': 'custom all reduce in the distance1',
    # 'tp_copy_distance2': 'custom all reduce in the distance2',
    # 'tp_global_distance1': 'NCCL all reduce in the distance1',
    'simple_alpaca': 'custom all reduce without TP',
    'tp_alpaca_distance1': 'custom all reduce with Alpaca in the distance1',
    'tp_alpaca_distance2': 'custom all reduce with Alpaca in the distance2'
}

def extract_tp_metrics(json_file, dataset_name):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract qps from filename (pattern: ...-X.Xqps-...)
        match = re.search(r'(\d+\.\d+)qps', os.path.basename(json_file))
        request_rate = float(match.group(1)) if match else None

        metrics = {
            'filename': os.path.basename(json_file),
            'request_rate': request_rate,
            'throughput': data.get('request_throughput'),
            'mean_ttft_ms': data.get('mean_ttft_ms'),
            'p99_ttft_ms': data.get('p99_ttft_ms'),
            'mean_tpot_ms': data.get('mean_tpot_ms'),
            'p99_tpot_ms': data.get('p99_tpot_ms'),
            'mean_itl_ms': data.get('mean_itl_ms'),
            'method': TARGET_MAP.get(dataset_name, dataset_name)
        }
        
        return metrics
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return None

def process_directory(dir_name):
    path_pattern = os.path.join("../result_backup", dir_name, "*.json")
    files = glob.glob(path_pattern)
    
    if not files:
        print(f"No JSON files found in ../result_backup/{dir_name}")
        return pd.DataFrame()
    
    all_metrics = []
    for f in files:
        m = extract_tp_metrics(f, dir_name)
        if m:
            all_metrics.append(m)
    
    df = pd.DataFrame(all_metrics)
    df = df.sort_values(by='request_rate', ascending=True).reset_index(drop=True)
    num_cols = ['request_rate', 'throughput', 'mean_ttft_ms', 'p99_ttft_ms',
                'mean_tpot_ms', 'p99_tpot_ms', 'mean_itl_ms']
    df[num_cols] = df[num_cols].round(2)
    return df

import matplotlib.pyplot as plt

def plot_metrics(df):
    copy_df = df[df['method'].isin([
        TARGET_MAP['tp_copy_distance1'],
        TARGET_MAP['tp_copy_distance2']
    ])]
    alpaca_df = df[df['method'].isin([
        TARGET_MAP['tp_alpaca_distance1'],
        TARGET_MAP['tp_alpaca_distance2']
    ])]

    # Figure 1: Copy distance1 vs distance2 - mean_ttft_ms
    plt.figure()
    for method in copy_df['method'].unique():
        sub = copy_df[copy_df['method'] == method]
        plt.plot(sub['request_rate'], sub['mean_ttft_ms'] / 1000, marker='o', label=method)
    plt.xlabel("Request Rate (qps)")
    plt.ylabel("Mean TTFT (s)")
    plt.title("NarrativeQA Distance1 vs Distance2 - TTFT")
    plt.legend()
    plt.grid(True)
    plt.savefig("copy_ttft.png", dpi=300)

    # Figure 2: Copy distance1 vs distance2 - mean_tpot_ms
    plt.figure()
    for method in copy_df['method'].unique():
        sub = copy_df[copy_df['method'] == method]
        plt.plot(sub['request_rate'], sub['mean_tpot_ms'] / 1000, marker='o', label=method)
    plt.xlabel("Request Rate (qps)")
    plt.ylabel("Mean TPOT (s)")
    plt.title("NarrativeQA Distance1 vs Distance2 - TPOT")
    plt.legend()
    plt.grid(True)
    plt.savefig("copy_tpot.png", dpi=300)

    # Figure 3: Alpaca distance1 vs distance2 - mean_ttft_ms (log x-axis)
    plt.figure()
    for method in alpaca_df['method'].unique():
        sub = alpaca_df[alpaca_df['method'] == method]
        plt.semilogx(sub['request_rate'], sub['mean_ttft_ms'], marker='o', label=method)
    plt.xlabel("Request Rate (qps, log scale)")
    plt.ylabel("Mean TTFT (ms)")
    plt.title("Alpaca Distance1 vs Distance2 - TTFT")
    plt.legend()
    plt.grid(True, which="both")
    plt.ylim(bottom=0)
    plt.xticks(alpaca_df['request_rate'].unique(), labels=[str(r) for r in alpaca_df['request_rate'].unique()])
    plt.savefig("alpaca_ttft.png", dpi=300)

    # Figure 4: Alpaca distance1 vs distance2 - mean_tpot_ms (log x-axis)
    plt.figure()
    for method in alpaca_df['method'].unique():
        sub = alpaca_df[alpaca_df['method'] == method]
        plt.semilogx(sub['request_rate'], sub['mean_tpot_ms'], marker='o', label=method)
    plt.xlabel("Request Rate (qps, log scale)")
    plt.ylabel("Mean TPOT (ms)")
    plt.title("Alpaca Distance1 vs Distance2 - TPOT")
    plt.legend()
    plt.grid(True, which="both")
    plt.ylim(bottom=0)
    plt.xticks(alpaca_df['request_rate'].unique(), labels=[str(r) for r in alpaca_df['request_rate'].unique()])
    plt.savefig("alpaca_tpot.png", dpi=300)

    plt.show()


def main():
    all_data = []
    for dir_name in TARGET_MAP.keys():
        df = process_directory(dir_name)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        print("No valid data collected.")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print("=== Combined Metrics ===")
    combined_df = combined_df.sort_values(by=['method', 'request_rate']).reset_index(drop=True)
    
    for method in combined_df['method'].unique():
        method_df = combined_df[combined_df['method'] == method]
        print(f"--- {method} ---")
        print(method_df[['request_rate', 'throughput', 'mean_ttft_ms', 'p99_ttft_ms', 
                         'mean_tpot_ms', 'p99_tpot_ms', 'mean_itl_ms']].to_string(index=False))
        print("-" * 80)

    # plot_metrics(combined_df)

if __name__ == "__main__":
    main()
