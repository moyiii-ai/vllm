import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_vllm_metrics(json_file):
    """Extract vLLM metrics from a single JSON file"""
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

def process_standard_files(target_dir):
    """Process standard files (without grouping) - suitable for lmcache"""
    json_files = glob.glob(f"../result_backup/{target_dir}/*.json")
    
    if not json_files:
        print(f"No JSON files found in ../result_backup/{target_dir} directory.")
        return None
    
    all_metrics = []
    for file in json_files:
        file_metrics = extract_vllm_metrics(file)
        if file_metrics:
            all_metrics.append(file_metrics)
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df_sorted = metrics_df.sort_values(by='request_rate', ascending=True).reset_index(drop=True)
    # Add a column to identify the dataset
    metrics_df_sorted['dataset'] = target_dir
    return metrics_df_sorted

def process_grouped_files(target_dir):
    """Process files requiring group aggregation - suitable for dp_lmcache_g4 and dp_lmcache_g5"""
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
            'mean_tpot_ms': sum(item['mean_tpot_ms'] for item in group_data if item['mean_tpot_ms'] is not None) / len(group_data),
            'dataset': target_dir  # Add dataset identifier
        }
        all_metrics.append(aggregated)
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df_sorted = metrics_df.sort_values(by='request_rate', ascending=True).reset_index(drop=True)
    return metrics_df_sorted

def plot_comparative_latency(df, latency_type, output_file):
    """Generate comparative plot with lines for each dataset"""
    plt.figure(figsize=(12, 7))
    
    # Configure plot based on latency type
    if latency_type == 'mean_ttft_ms':
        y_label = 'Mean TTFT (ms)'
        title = 'vLLM: Request Throughput vs Mean TTFT (Comparison)'
    else:
        y_label = 'Mean TPOT (ms)'
        title = 'vLLM: Request Throughput vs Mean TPOT (Comparison)'
    
    # Define styles for each dataset
    datasets = {
        'lmcache': {'color': 'blue', 'marker': 'o', 'label': 'lmcache'},
        'dp_lmcache_g4': {'color': 'green', 'marker': 's', 'label': 'dp_lmcache_g4'},
        'dp_lmcache_g5': {'color': 'red', 'marker': '^', 'label': 'dp_lmcache_g5'}
    }
    
    # Plot each dataset
    for dataset, style in datasets.items():
        dataset_df = df[df['dataset'] == dataset].sort_values(by='request_throughput')
        if not dataset_df.empty:
            plt.plot(
                dataset_df['request_throughput'],
                dataset_df[latency_type],
                color=style['color'],
                marker=style['marker'],
                label=style['label'],
                linewidth=2,
                markersize=8
            )
    
    plt.xlabel('Request Throughput (req/s)', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300)
    print(f"Generated comparative plot: {output_file}")

def generate_grouped_output(combined_df):
    """Group by request rate and output ttft, tpot, and request throughput for the 3 directories"""
    # Create output directory
    output_dir = "grouped_by_request_rate"
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by request rate
    grouped = combined_df.groupby('request_rate')
    
    # Prepare output data
    result_data = []
    for rate, group in grouped:
        # Extract metrics for each dataset
        row = {'request_rate': rate}
        for dataset in ['lmcache', 'dp_lmcache_g4', 'dp_lmcache_g5']:
            dataset_data = group[group['dataset'] == dataset]
            if not dataset_data.empty:
                row[f'{dataset}_ttft'] = round(dataset_data.iloc[0]['mean_ttft_ms'], 2)
                row[f'{dataset}_tpot'] = round(dataset_data.iloc[0]['mean_tpot_ms'], 2)
                row[f'{dataset}_throughput'] = round(dataset_data.iloc[0]['request_throughput'], 2)
            else:
                row[f'{dataset}_ttft'] = None
                row[f'{dataset}_tpot'] = None
                row[f'{dataset}_throughput'] = None
        
        result_data.append(row)
    
    # Create result DataFrame and sort
    result_df = pd.DataFrame(result_data)
    result_df = result_df.sort_values(by='request_rate').reset_index(drop=True)
    
    # Save as CSV file
    output_file = os.path.join(output_dir, 'grouped_metrics_by_request_rate.csv')
    result_df.to_csv(output_file, index=False)
    print(f"\nGrouped results by request rate saved to: {output_file}")
    
    # Display in requested format
    print("\n===== Performance Metrics Grouped by Request Rate =====")
    # Print header
    print(f"{'Request Rate':<12} {'Dataset':<15} {'Throughput':<10} {'TTFT':<10} {'TPOT':<10}")
    print("-" * 60)
    
    # Print each request rate group
    for _, row in result_df.iterrows():
        rate = row['request_rate']
        # Print lmcache data
        print(f"{rate:<12} {'lmcache':<15} {row['lmcache_throughput'] if row['lmcache_throughput'] is not None else '-':<10} {row['lmcache_ttft'] if row['lmcache_ttft'] is not None else '-':<10} {row['lmcache_tpot'] if row['lmcache_tpot'] is not None else '-':<10}")
        # Print dp_lmcache_g4 data
        print(f"{'':<12} {'dp_lmcache_g4':<15} {row['dp_lmcache_g4_throughput'] if row['dp_lmcache_g4_throughput'] is not None else '-':<10} {row['dp_lmcache_g4_ttft'] if row['dp_lmcache_g4_ttft'] is not None else '-':<10} {row['dp_lmcache_g4_tpot'] if row['dp_lmcache_g4_tpot'] is not None else '-':<10}")
        # Print dp_lmcache_g5 data
        print(f"{'':<12} {'dp_lmcache_g5':<15} {row['dp_lmcache_g5_throughput'] if row['dp_lmcache_g5_throughput'] is not None else '-':<10} {row['dp_lmcache_g5_ttft'] if row['dp_lmcache_g5_ttft'] is not None else '-':<10} {row['dp_lmcache_g5_tpot'] if row['dp_lmcache_g5_tpot'] is not None else '-':<10}")
        print("-" * 60)
    
    return result_df

def main():
    # Process all three directories
    dirs = ['lmcache', 'dp_lmcache_g4', 'dp_lmcache_g5']
    all_data = []
    
    for dir_name in dirs:
        print(f"Processing {dir_name}...")
        if dir_name == 'lmcache':
            df = process_standard_files(dir_name)
        else:
            df = process_grouped_files(dir_name)
        
        if df is not None:
            all_data.append(df)
            # Save individual dataset CSV
            df.to_csv(f"{dir_name}_summary.csv", index=False)
            print(f"Saved {dir_name} metrics to {dir_name}_summary.csv")
    
    if not all_data:
        print("No valid data processed from any directory.")
        return
    
    # Combine all data for comparison
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv("combined_summary.csv", index=False)
    print("\nSaved combined metrics to combined_summary.csv")
    
    # Group by request rate and output results
    generate_grouped_output(combined_df)
    
    # Generate comparative plots
    valid_df = combined_df.dropna(subset=['mean_ttft_ms', 'mean_tpot_ms', 'request_throughput'])
    
    if not valid_df.empty:
        plot_comparative_latency(valid_df, 'mean_ttft_ms', 'comparison_ttft.png')
        plot_comparative_latency(valid_df, 'mean_tpot_ms', 'comparison_tpot.png')
    else:
        print("Warning: Insufficient valid data for generating comparative plots.")

if __name__ == "__main__":
    main()
    