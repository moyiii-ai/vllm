import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def calculate_percentile(data, percentile):
    """Calculate percentile for a list of numerical values
    Args:
        data (list): List of numerical values to process (cannot be empty)
        percentile (float): Percentile to calculate (e.g., 99 for P99)
    Returns:
        float/None: Calculated percentile value or None if input data is empty
    """
    if not data or not isinstance(data, list):
        return None
    sorted_data = sorted(data)
    n = len(sorted_data)
    index = (n - 1) * (percentile / 100)
    floor_index = int(np.floor(index))
    fractional = index - floor_index
    
    if floor_index + 1 < n:
        return sorted_data[floor_index] + fractional * (sorted_data[floor_index + 1] - sorted_data[floor_index])
    else:
        return sorted_data[floor_index]

def extract_vllm_metrics(json_file):
    """Extract raw metrics (TTFT, TPOT, ITLS) and basic stats from a single JSON file
    Collects raw values to enable grouped P99 calculation for multi-file groups
    Args:
        json_file (str): Full path to the JSON metrics file
    Returns:
        dict/None: Dictionary of extracted metrics or None if processing fails
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract and convert raw TTFT values to ms (filter invalid values)
        raw_ttfts = [t * 1000 for t in data.get('ttfts', []) if isinstance(t, (int, float))]
        # Extract and convert raw TPOT values to ms
        raw_tpots = [t * 1000 for t in data.get('tpots', []) if isinstance(t, (int, float))]
        # Extract, flatten and convert raw ITLS values to ms
        raw_itls = []
        for item in data.get('itls', []):
            if isinstance(item, list):
                raw_itls.extend([i * 1000 for i in item if isinstance(i, (int, float))])
            elif isinstance(item, (int, float)):
                raw_itls.append(item * 1000)
        
        metrics = {
            'filename': os.path.basename(json_file),
            'mean_ttft_ms': data.get('mean_ttft_ms'),
            'mean_tpot_ms': data.get('mean_tpot_ms'),
            'request_rate': data.get('request_rate'),
            'request_throughput': data.get('request_throughput'),
            'raw_ttfts': raw_ttfts,
            'raw_tpots': raw_tpots,
            'raw_itls': raw_itls,
            'raw_mean_itls': data.get('mean_itls')
        }
        
        # Validate request_rate to avoid sorting errors
        if metrics['request_rate'] is None or not isinstance(metrics['request_rate'], (int, float)):
            print(f"Warning: Invalid request_rate in {json_file} - skipping")
            return None
        
        return metrics
    except Exception as e:
        print(f"Error processing {json_file}: {str(e)}")
        return None

def process_standard_files(target_dir):
    """Process files without grouping (for 'lmcache' directory)
    Calculates P99 metrics from raw values and returns sorted DataFrame
    Args:
        target_dir (str): Name of directory under ../result_backup/
    Returns:
        pd.DataFrame/None: Sorted metrics or None if no valid files
    """
    json_path = os.path.join("../result_backup", target_dir, "*.json")
    json_files = glob.glob(json_path)
    
    if not json_files:
        print(f"No JSON files found in ../result_backup/{target_dir}")
        return None
    
    all_metrics = []
    for file in json_files:
        file_metrics = extract_vllm_metrics(file)
        if file_metrics:
            # Calculate P99 for TTFT/TPOT/ITLS from raw data
            file_metrics['p99_ttft_ms'] = calculate_percentile(file_metrics['raw_ttfts'], 99)
            file_metrics['p99_tpot_ms'] = calculate_percentile(file_metrics['raw_tpots'], 99)
            file_metrics['p99_itl_ms'] = calculate_percentile(file_metrics['raw_itls'], 99)
            
            # Remove raw lists to clean output
            for key in ['raw_ttfts', 'raw_tpots', 'raw_itls']:
                del file_metrics[key]
            
            all_metrics.append(file_metrics)
    
    # Sort by request_rate and add dataset label
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df_sorted = metrics_df.sort_values(by='request_rate', ascending=True).reset_index(drop=True)
    metrics_df_sorted['dataset'] = target_dir
    
    return metrics_df_sorted

def process_grouped_files(target_dir):
    """Process files requiring grouping (for 'dp_lmcache_g4'/'dp_lmcache_g5')
    Merges raw values from grouped files to calculate aggregated stats + P99
    Args:
        target_dir (str): Name of directory under ../result_backup/
    Returns:
        pd.DataFrame/None: Aggregated metrics or None if no valid files
    """
    json_path = os.path.join("../result_backup", target_dir, "*.json")
    json_files = glob.glob(json_path)
    
    if not json_files:
        print(f"No JSON files found in ../result_backup/{target_dir}")
        return None
    
    # Group files by base name (ignore port suffix like "-8000")
    file_groups = defaultdict(list)
    for file in json_files:
        filename = os.path.basename(file)
        base_name = filename.rsplit("-", 1)[0] if "-" in filename else filename
        file_groups[base_name].append(file)
    
    all_aggregated = []
    for base_name, group_files in file_groups.items():
        group_raw_ttfts = []
        group_raw_tpots = []
        group_raw_itls = []
        group_basic = []
        
        # Collect raw data and basic metrics from group files
        for file in group_files:
            file_metrics = extract_vllm_metrics(file)
            if file_metrics:
                group_basic.append(file_metrics)
                group_raw_ttfts.extend(file_metrics['raw_ttfts'])
                group_raw_tpots.extend(file_metrics['raw_tpots'])
                group_raw_itls.extend(file_metrics['raw_itls'])
        
        if not group_basic:
            print(f"Warning: No valid data in group {base_name} - skipping")
            continue
        
        # Calculate aggregated mean metrics (handle None values)
        valid_mean_ttft = [m['mean_ttft_ms'] for m in group_basic if m['mean_ttft_ms'] is not None]
        valid_mean_tpot = [m['mean_tpot_ms'] for m in group_basic if m['mean_tpot_ms'] is not None]
        
        total_throughput = sum(m['request_throughput'] for m in group_basic)
        avg_throughput = total_throughput / len(group_basic) if group_basic else None

        aggregated = {
            'filename': f"{base_name} (combined)",
            'request_rate': group_basic[0]['request_rate'],
            # 'request_rate': sum(m['request_rate'] for m in group_basic),    
            'request_throughput': total_throughput,
            # 'request_throughput': avg_throughput,
            'mean_ttft_ms': sum(valid_mean_ttft) / len(valid_mean_ttft) if valid_mean_ttft else None,
            'mean_tpot_ms': sum(valid_mean_tpot) / len(valid_mean_tpot) if valid_mean_tpot else None,
            'p99_ttft_ms': calculate_percentile(group_raw_ttfts, 99),
            'p99_tpot_ms': calculate_percentile(group_raw_tpots, 99),
            'p99_itl_ms': calculate_percentile(group_raw_itls, 99),
            'dataset': target_dir
        }
        
        all_aggregated.append(aggregated)
    
    # Sort by request_rate
    metrics_df = pd.DataFrame(all_aggregated)
    metrics_df_sorted = metrics_df.sort_values(by='request_rate', ascending=True).reset_index(drop=True)
    
    return metrics_df_sorted

def plot_comparative_metrics_throughput(df, metric_type, output_file, min_rate=None, max_rate=None):
    """Generate comparative plot (only for request rates with full dataset data)
    Args:
        df (pd.DataFrame): Combined metrics from all datasets
        metric_type (str): Metric to plot (e.g., 'p99_ttft_ms', 'mean_tpot_ms')
        output_file (str): Path to save plot
    """
    plt.figure(figsize=(12, 7))

    if min_rate is not None:
        df = df[df['request_rate'] >= min_rate]
    if max_rate is not None:
        df = df[df['request_rate'] <= max_rate]
    
    # Plot configuration for each metric
    metric_config = {
        'mean_ttft_ms': ('Mean TTFT (ms)', 'Average Request Throughput vs Mean TTFT'),
        'mean_tpot_ms': ('Mean TPOT (ms)', 'Request Throughput vs Mean TPOT'),
        'p99_ttft_ms': ('P99 TTFT (ms)', 'Average Request Throughput vs P99 TTFT'),
        'p99_tpot_ms': ('P99 TPOT (ms)', 'Request Throughput vs P99 TPOT'),
        'p99_itl_ms': ('P99 Inter-token Latency (ms)', 'Request Throughput vs P99 ITL')
    }
    y_label, title = metric_config.get(metric_type, (metric_type, metric_type))
    
    # Filter valid request rates (full data from all 3 datasets)
    all_rates = df['request_rate'].unique()
    required_datasets = {'lmcache', 'dp_lmcache_g4', 'dp_lmcache_g5'}
    valid_rates = []
    
    for rate in all_rates:
        rate_data = df[df['request_rate'] == rate]
        present_datasets = set(rate_data['dataset'].unique())
        # Check if all datasets exist and have non-None metric values
        if present_datasets == required_datasets:
            metric_values = rate_data[metric_type].dropna()
            if len(metric_values) == 3:
                valid_rates.append(rate)
    
    df_valid = df[df['request_rate'].isin(valid_rates)]
    print(f"Plotting {metric_type}: {len(valid_rates)}/{len(all_rates)} valid request rates")
    
    # Dataset style configuration
    dataset_styles = {
        'lmcache': {'color': '#1f77b4', 'marker': 'o', 'label': 'lmcache'},
        'dp_lmcache_g4': {'color': '#2ca02c', 'marker': 's', 'label': 'dp_lmcache_g4'},
        'dp_lmcache_g5': {'color': '#d62728', 'marker': '^', 'label': 'dp_lmcache_g5'}
    }
    
    # Plot each dataset
    for dataset, style in dataset_styles.items():
        dataset_data = df_valid[df_valid['dataset'] == dataset].sort_values(by='request_throughput')
        if not dataset_data.empty:
            plt.plot(
                dataset_data['request_throughput'],
                dataset_data[metric_type],
                color=style['color'],
                marker=style['marker'],
                label=style['label'],
                linewidth=2,
                markersize=8,
                alpha=0.8
            )
    
    # Plot formatting
    ticks = np.arange(0.4, max_rate + 0.1, 0.4)
    plt.xticks(ticks)
    plt.xlabel('Average Request Throughput (req/s)', fontsize=12, fontweight='bold')
    # plt.xlabel('Request Throughput (req/s)', fontsize=12, fontweight='bold')
    plt.ylabel(y_label, fontsize=12, fontweight='bold')
    plt.title(f'{title}', fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}\n")

def plot_comparative_metrics_rate(df, metric_type, output_file, min_rate=None, max_rate=None):
    """Generate comparative plot (only for request rates with full dataset data)
    Args:
        df (pd.DataFrame): Combined metrics from all datasets
        metric_type (str): Metric to plot (e.g., 'p99_ttft_ms', 'mean_tpot_ms')
        output_file (str): Path to save plot
    """
    plt.figure(figsize=(12, 7))

    if min_rate is not None:
        df = df[df['request_rate'] >= min_rate]
    if max_rate is not None:
        df = df[df['request_rate'] <= max_rate]
    
    # Plot configuration for each metric
    metric_config = {
        'mean_ttft_ms': ('Mean TTFT (ms)', 'Input Request Rate vs Mean TTFT'),
        'mean_tpot_ms': ('Mean TPOT (ms)', 'Input Request Rate vs Mean TPOT'),
        'p99_ttft_ms': ('P99 TTFT (ms)', 'Input Request Rate vs P99 TTFT'),
        'p99_tpot_ms': ('P99 TPOT (ms)', 'Input Request Rate vs P99 TPOT'),
        'p99_itl_ms': ('P99 Inter-token Latency (ms)', 'Input Request Rate vs P99 ITL'),
        'request_throughput': ('Total Throughput (req/s)', 'Input Request Rate vs Total Throughput')
    }
    y_label, title = metric_config.get(metric_type, (metric_type, metric_type))
    
    # Filter valid request rates (full data from all 3 datasets)
    all_rates = df['request_rate'].unique()
    required_datasets = {'lmcache', 'dp_lmcache_g4', 'dp_lmcache_g5'}
    valid_rates = []
    
    # for rate in all_rates:
    #     rate_data = df[df['request_rate'] == rate]
    #     present_datasets = set(rate_data['dataset'].unique())
    #     # Check if all datasets exist and have non-None metric values
    #     if present_datasets == required_datasets:
    #         metric_values = rate_data[metric_type].dropna()
    #         if len(metric_values) == 3:
    #             valid_rates.append(rate)
    
    # df_valid = df[df['request_rate'].isin(valid_rates)]
    df_valid = df
    valid_rates = all_rates.tolist()
    print(f"Plotting {metric_type}: {len(valid_rates)}/{len(all_rates)} valid request rates")
    
    # Dataset style configuration
    dataset_styles = {
        'lmcache': {'color': '#1f77b4', 'marker': 'o', 'label': 'lmcache'},
        'dp_lmcache_g4': {'color': '#2ca02c', 'marker': 's', 'label': 'dp_lmcache_g4'},
        'dp_lmcache_g5': {'color': '#d62728', 'marker': '^', 'label': 'dp_lmcache_g5'}
    }
    
    # Plot each dataset
    for dataset, style in dataset_styles.items():
        dataset_data = df_valid[df_valid['dataset'] == dataset].sort_values(by='request_rate')
        if not dataset_data.empty:
            plt.plot(
                dataset_data['request_rate'],
                dataset_data[metric_type],
                color=style['color'],
                marker=style['marker'],
                label=style['label'],
                linewidth=2,
                markersize=8,
                alpha=0.8
            )

    if metric_type == 'request_throughput':
        # Get lmcache data
        lmcache_data = df_valid[df_valid['dataset'] == 'lmcache'].sort_values(by='request_rate')
        if not lmcache_data.empty:
            # Calculate twice the throughput
            doubled_throughput = lmcache_data[metric_type] * 2
            # Plot the yellow line
            plt.plot(
                lmcache_data['request_rate'],
                doubled_throughput,
                color='#ffcc00',  # Yellow color
                linestyle='--',   # Dashed line to distinguish
                marker='x',       # Different marker
                label='2x lmcache',
                linewidth=2,
                markersize=8,
                alpha=0.8
            )
    
    # Plot formatting
    ticks = np.arange(0.4, max_rate + 0.1, 0.4)
    plt.xticks(ticks)
    plt.xlabel('Input Request Rate (req/s)', fontsize=12, fontweight='bold')
    # plt.xlabel('Request Throughput (req/s)', fontsize=12, fontweight='bold')
    plt.ylabel(y_label, fontsize=12, fontweight='bold')
    plt.title(f'{title}', fontsize=14, fontweight='bold', pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}\n")

def generate_grouped_output(combined_df):
    """Generate grouped output (CSV + console print) by request rate
    Args:
        combined_df (pd.DataFrame): Combined metrics from all datasets
    Returns:
        pd.DataFrame: Grouped metrics DataFrame
    """
    # Create output directory
    output_dir = "grouped_by_request_rate"
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by request rate and process each group
    grouped = combined_df.groupby('request_rate')
    result_data = []
    
    for rate, group in grouped:
        row = {'request_rate': round(rate, 2)}
        
        # Extract metrics for each dataset
        for dataset in ['lmcache', 'dp_lmcache_g4', 'dp_lmcache_g5']:
            dataset_data = group[group['dataset'] == dataset]
            
            if not dataset_data.empty:
                row[f'{dataset}_throughput'] = round(dataset_data.iloc[0]['request_throughput'], 2)
                row[f'{dataset}_mean_ttft'] = round(dataset_data.iloc[0]['mean_ttft_ms'], 2) if dataset_data.iloc[0]['mean_ttft_ms'] is not None else None
                row[f'{dataset}_p99_ttft'] = round(dataset_data.iloc[0]['p99_ttft_ms'], 2) if dataset_data.iloc[0]['p99_ttft_ms'] is not None else None
                row[f'{dataset}_mean_tpot'] = round(dataset_data.iloc[0]['mean_tpot_ms'], 2) if dataset_data.iloc[0]['mean_tpot_ms'] is not None else None
                row[f'{dataset}_p99_tpot'] = round(dataset_data.iloc[0]['p99_tpot_ms'], 2) if dataset_data.iloc[0]['p99_tpot_ms'] is not None else None
                row[f'{dataset}_p99_itl'] = round(dataset_data.iloc[0]['p99_itl_ms'], 2) if dataset_data.iloc[0]['p99_itl_ms'] is not None else None
            else:
                row[f'{dataset}_throughput'] = None
                row[f'{dataset}_mean_ttft'] = None
                row[f'{dataset}_p99_ttft'] = None
                row[f'{dataset}_mean_tpot'] = None
                row[f'{dataset}_p99_tpot'] = None
                row[f'{dataset}_p99_itl'] = None
        
        result_data.append(row)
    
    # Convert to DataFrame and sort
    result_df = pd.DataFrame(result_data)
    result_df = result_df.sort_values(by='request_rate').reset_index(drop=True)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'grouped_metrics_by_request_rate.csv')
    result_df.to_csv(csv_path, index=False)
    print(f"Grouped CSV saved to: {csv_path}\n")
    
    # Print to console
    print("===== Grouped Performance Metrics by Request Rate =====")
    print(f"{'Request Rate':<12} {'Dataset':<15} {'Throughput':<12} {'Mean TTFT':<12} {'P99 TTFT':<12} {'Mean TPOT':<12} {'P99 TPOT':<12} {'P99 ITL':<12}")
    print("-" * 100)
    
    for _, row in result_df.iterrows():
        rate = row['request_rate']
        # Print each dataset's data for current request rate
        for dataset in ['lmcache', 'dp_lmcache_g4', 'dp_lmcache_g5']:
            throughput = row[f'{dataset}_throughput'] if row[f'{dataset}_throughput'] is not None else '-'
            mean_ttft = row[f'{dataset}_mean_ttft'] if row[f'{dataset}_mean_ttft'] is not None else '-'
            p99_ttft = row[f'{dataset}_p99_ttft'] if row[f'{dataset}_p99_ttft'] is not None else '-'
            mean_tpot = row[f'{dataset}_mean_tpot'] if row[f'{dataset}_mean_tpot'] is not None else '-'
            p99_tpot = row[f'{dataset}_p99_tpot'] if row[f'{dataset}_p99_tpot'] is not None else '-'
            p99_itl = row[f'{dataset}_p99_itl'] if row[f'{dataset}_p99_itl'] is not None else '-'
            
            print(f"{rate:<12} {dataset:<15} {throughput:<12} {mean_ttft:<12} {p99_ttft:<12} {mean_tpot:<12} {p99_tpot:<12} {p99_itl:<12}")
        print("-" * 100)
    
    return result_df

def main():
    """Main function to coordinate data processing, output, and plotting"""
    # Directories to process (lmcache = standard; others = grouped)
    target_dirs = ['lmcache', 'dp_lmcache_g4', 'dp_lmcache_g5']
    all_data = []
    
    # Step 1: Process each directory
    for dir_name in target_dirs:
        print(f"=== Processing directory: {dir_name} ===")
        if dir_name == 'lmcache':
            # Process lmcache with standard logic (no grouping)
            dir_df = process_standard_files(dir_name)
        else:
            # Process dp_lmcache groups (g4/g5) with grouping logic
            dir_df = process_grouped_files(dir_name)
        
        if dir_df is not None:
            all_data.append(dir_df)
            # Save individual directory CSV (keep original feature)
            dir_csv_path = f"{dir_name}_summary.csv"
            dir_df.to_csv(dir_csv_path, index=False)
            print(f"Saved individual summary for {dir_name} to: {dir_csv_path}\n")
    
    # Exit if no valid data was collected
    if not all_data:
        print("Error: No valid metrics data collected from any directory.")
        return
    
    # Step 2: Combine data from all directories
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_csv_path = "combined_summary.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"=== Combined Data ===")
    print(f"Saved combined metrics to: {combined_csv_path}\n")
    
    # Step 3: Generate grouped output (CSV + console print)
    generate_grouped_output(combined_df)
    
    # Step 4: Generate comparative plots (only for valid data)
    # Filter out rows with missing key metrics to avoid plot errors
    valid_plot_df = combined_df.dropna(subset=['request_throughput', 'mean_ttft_ms', 'mean_tpot_ms', 'p99_ttft_ms', 'p99_tpot_ms', 'p99_itl_ms'])
    
    if not valid_plot_df.empty:
        print("=== Generating Comparative Plots ===")
        my_max_rate = 4.0
        # plot_comparative_metrics_throughput(valid_plot_df, 'mean_ttft_ms', 'throughput_mean_ttft.png', max_rate=my_max_rate)
        # plot_comparative_metrics_throughput(valid_plot_df, 'mean_tpot_ms', 'throughput_mean_tpot.png', max_rate=my_max_rate)
        # plot_comparative_metrics_throughput(valid_plot_df, 'p99_ttft_ms', 'throughput_p99_ttft.png', max_rate=my_max_rate)
        # plot_comparative_metrics_throughput(valid_plot_df, 'p99_tpot_ms', 'throughput_p99_tpot.png', max_rate=my_max_rate)
        # plot_comparative_metrics_throughput(valid_plot_df, 'p99_itl_ms', 'throughput_p99_itl.png', max_rate=my_max_rate)
        
        plot_comparative_metrics_rate(valid_plot_df, 'mean_ttft_ms', 'rate_mean_ttft.png', max_rate=my_max_rate)
        plot_comparative_metrics_rate(valid_plot_df, 'mean_tpot_ms', 'rate_mean_tpot.png', max_rate=my_max_rate)
        plot_comparative_metrics_rate(valid_plot_df, 'p99_ttft_ms', 'rate_p99_ttft.png', max_rate=my_max_rate)
        plot_comparative_metrics_rate(valid_plot_df, 'p99_tpot_ms', 'rate_p99_tpot.png', max_rate=my_max_rate)
        plot_comparative_metrics_rate(valid_plot_df, 'p99_itl_ms', 'rate_p99_itl.png', max_rate=my_max_rate)
        plot_comparative_metrics_rate(valid_plot_df, 'request_throughput', 'rate_throughput.png', max_rate=my_max_rate)
    else:
        print("Warning: Insufficient valid data to generate comparative plots.")

if __name__ == "__main__":
    main()