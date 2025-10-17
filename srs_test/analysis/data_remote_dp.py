import json
import os
import glob
import pandas as pd
import re
import matplotlib.pyplot as plt

TARGET_MAP = {
    'remote_dp': 'Remote DP',
    'remote_dp_balance': 'Remote DP (Balance)'
}

def extract_tp_metrics(json_file, dataset_name):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        basename = os.path.basename(json_file)
        match_qps = re.search(r'(\d+\.\d+)qps', basename)
        request_rate = float(match_qps.group(1)) if match_qps else None

        if basename.endswith("8000.json"):
            gpu = "GPU1"
        elif basename.endswith("8001.json"):
            gpu = "GPU2"
        else:
            gpu = "Unknown"

        balance_flag = "balance" if "balance" in dataset_name else "unbalance"
        gpu_label = f"{gpu}_{balance_flag}"

        metrics = {
            'filename': basename,
            'request_rate': request_rate,
            'throughput': data.get('request_throughput'),
            'mean_ttft_ms': data.get('mean_ttft_ms'),
            'p99_ttft_ms': data.get('p99_ttft_ms'),
            'mean_tpot_ms': data.get('mean_tpot_ms'),
            'p99_tpot_ms': data.get('p99_tpot_ms'),
            'gpu': gpu,
            'gpu_label': gpu_label,
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
    df = df.sort_values(by=['request_rate', 'gpu_label']).reset_index(drop=True)
    num_cols = ['request_rate', 'throughput', 'mean_ttft_ms', 'p99_ttft_ms',
                'mean_tpot_ms', 'p99_tpot_ms']
    df[num_cols] = df[num_cols].round(2)
    return df

def plot_individual_metrics(df):
    metrics_info = [
        ('throughput', 'Throughput (req/s)', False),
        ('mean_ttft_ms', 'Mean TTFT (s)', True),
        ('p99_ttft_ms', 'P99 TTFT (s)', True),
        ('mean_tpot_ms', 'Mean TPOT (s)', True),
        ('p99_tpot_ms', 'P99 TPOT (s)', True)
    ]
    
    fixed_order = ["GPU2_unbalance", "GPU1_unbalance", "GPU2_balance", "GPU1_balance"]
    
    for metric, ylabel, is_time_metric in metrics_info:
        plt.figure(figsize=(8,5))
        
        for gpu_label in fixed_order:
            if gpu_label not in df['gpu_label'].unique():
                continue
            sub = df[df['gpu_label'] == gpu_label]
            y_data = sub[metric] / 1000.0 if is_time_metric else sub[metric]
            plt.plot(sub['request_rate'], y_data, marker='o', label=gpu_label)
        
        plt.xlabel("Request Rate (qps)")
        plt.ylabel(ylabel)
        plt.title(f"Remote DP - {ylabel} GPU Comparison (Balance vs Unbalance)")
        plt.grid(True)
        plt.legend()

        plt.xticks(df['request_rate'].unique(), labels=[str(r) for r in df['request_rate'].unique()])

        if is_time_metric:
            max_val = (df[metric].max() / 1000.0)
            plt.ylim(bottom=-5, top=max_val * 1.1)
        else:
            max_val = df[metric].max()
            plt.ylim(bottom=0, top=max_val * 1.1)

        plt.tight_layout()
        plt.savefig(f"remote_dp_{metric}_balance_comparison.png", dpi=300)
        plt.show()

def main():
    df_unbalance = process_directory('remote_dp')
    df_balance = process_directory('remote_dp_balance')
    
    df_all = pd.concat([df_unbalance, df_balance], ignore_index=True)
    if df_all.empty:
        return
    
    print("=== Remote DP Metrics (Balance vs Unbalance) ===")
    print(df_all[['request_rate', 'gpu_label', 'throughput', 'mean_ttft_ms', 'p99_ttft_ms',
                  'mean_tpot_ms', 'p99_tpot_ms']].to_string(index=False))
    
    plot_individual_metrics(df_all)

if __name__ == "__main__":
    main()
