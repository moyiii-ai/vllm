import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_vllm_metrics(json_file):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        metrics = {
            'filename': os.path.basename(json_file),
            'mean_ttft_ms': data.get('mean_ttft_ms'),
            'mean_tpot_ms': data.get('mean_tpot_ms'),
            'request_rate': data.get('request_rate'),
            'request_throughput': data.get('request_throughput')
        }
        
        # Validate request_rate for proper sorting
        if metrics['request_rate'] is None or not isinstance(metrics['request_rate'], (int, float)):
            print(f"Warning: File {json_file} contains invalid request_rate, skipping.")
            return None
        return metrics
    except Exception as e:
        print(f"Error processing file {json_file}: {str(e)}")
        return None

def process_all_vllm_files(target_dir):
    """Process all JSON files in ../result_backup/[target_dir] and return sorted metrics"""
    json_files = glob.glob(f"../result_backup/{target_dir}/*.json")
    
    if not json_files:
        print(f"No JSON files found in ../result_backup/{target_dir} directory.")
        return None
    
    # Group files by their base name (without the -port suffix)
    file_groups = defaultdict(list)
    for file in json_files:
        filename = os.path.basename(file)
        # Split on last occurrence of "-" to handle port numbers (e.g., -8000, -8001)
        if "-" in filename:
            base_name = filename.rsplit("-", 1)[0]
            file_groups[base_name].append(file)
        else:
            # Handle files without port suffix as single entries
            file_groups[filename].append(file)
    
    all_metrics = []
    # Process each group of files
    for base_name, files in file_groups.items():
        group_data = []
        for file in files:
            file_metrics = extract_vllm_metrics(file)
            if file_metrics:
                group_data.append(file_metrics)
        
        if not group_data:
            continue
        
        # Calculate aggregated metrics for the group
        aggregated = {
            'filename': f"{base_name} (combined)",
            'request_rate': sum(item['request_rate'] for item in group_data),
            'request_throughput': sum(item['request_throughput'] for item in group_data),
            'mean_ttft_ms': sum(item['mean_ttft_ms'] for item in group_data if item['mean_ttft_ms'] is not None) / len(group_data),
            'mean_tpot_ms': sum(item['mean_tpot_ms'] for item in group_data if item['mean_tpot_ms'] is not None) / len(group_data)
        }
        all_metrics.append(aggregated)
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df_sorted = metrics_df.sort_values(by='request_rate', ascending=True).reset_index(drop=True)
    return metrics_df_sorted, target_dir

def plot_latency_vs_throughput(df, latency_type, output_file_prefix):
    plt.figure(figsize=(10, 6))
    
    # Configure plot based on latency type
    if latency_type == 'mean_ttft_ms':
        color = 'blue'
        label = 'Mean TTFT (ms)'
        title = 'vLLM: Request Throughput vs Mean TTFT'
        output_filename = f"{output_file_prefix}_ttft.png"
    else:
        color = 'red'
        label = 'Mean TPOT (ms)'
        title = 'vLLM: Request Throughput vs Mean TPOT'
        output_filename = f"{output_file_prefix}_tpot.png"
    
    plt.scatter(
        df['request_throughput'],
        df[latency_type],
        color=color,
        s=80,
        label=label
    )
    
    plt.xlabel('Request Throughput (req/s)', fontsize=12)
    plt.ylabel(label, fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=300)
    print(f"Generated plot: {output_filename}")

if __name__ == "__main__":
    # Configure target directory here
    target_directory = "dp_lmcache_g4"
    
    vllm_metrics_df, dir_name = process_all_vllm_files(target_directory)
    
    if vllm_metrics_df is not None:
        print(f"Extracted vLLM Benchmark Metrics (Sorted by Request Rate, Directory: {dir_name}):")
        print("-----------------------------------------------------------------------------")
        print(vllm_metrics_df.to_string(index=False))
        
        # Save metrics to CSV
        output_csv = f"{dir_name}_summary.csv"
        vllm_metrics_df.to_csv(output_csv, index=False)
        print(f"\nSorted metrics saved to: {output_csv}")
        
        # Generate visualizations
        valid_df = vllm_metrics_df.dropna(subset=['mean_ttft_ms', 'mean_tpot_ms', 'request_throughput'])
        
        if not valid_df.empty:
            plot_latency_vs_throughput(valid_df, 'mean_ttft_ms', dir_name)
            plot_latency_vs_throughput(valid_df, 'mean_tpot_ms', dir_name)
        else:
            print("\nWarning: Insufficient valid data for generating plots.")