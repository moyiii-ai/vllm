import json
import os
import glob
import pandas as pd
import re

TARGET_MAP = {
    'tp_copy': 'custom all reduce',
    'tp_global': 'NCCL all reduce',
    # 'simple': 'NCCL all reduce without TP'
    'tp_small': 'custom all reduce with small dataset'
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

if __name__ == "__main__":
    main()
