import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

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

def process_directory(target_dir):
    """Process all JSON files in a directory and return metrics"""
    json_files = glob.glob(f"../result_backup/{target_dir}/*.json")
    
    if not json_files:
        print(f"No JSON files found in ../result_backup/{target_dir} directory.")
        return None
    
    all_metrics = []
    for file in json_files:
        file_metrics = extract_vllm_metrics(file)
        if file_metrics:
            file_metrics['directory'] = target_dir  # Add directory identifier
            all_metrics.append(file_metrics)
    
    if not all_metrics:
        print(f"No valid metrics found in {target_dir}")
        return None
        
    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df.sort_values(by='request_rate', ascending=True).reset_index(drop=True)

def plot_comparison(all_dfs, latency_type, output_file):
    """Generate comparison plot with lines for each directory"""
    plt.figure(figsize=(12, 7))
    
    # Configure plot based on latency type
    if latency_type == 'mean_ttft_ms':
        label = 'Mean TTFT (ms)'
        title = 'vLLM: Request Throughput vs Mean TTFT'
    else:
        label = 'Mean TPOT (ms)'
        title = 'vLLM: Request Throughput vs Mean TPOT'
    
    # Plot styles for each directory
    styles = {
        'dp_lmcache_g5': {'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        'dp_lmcache_g4': {'color': 'green', 'marker': 's', 'linestyle': '--'},
        'lmcache': {'color': 'red', 'marker': '^', 'linestyle': '-.'}
    }
    
    # Plot data for each directory
    for dir_name, df in all_dfs.items():
        style = styles.get(dir_name, {'color': 'black', 'marker': 'o'})
        plt.plot(
            df['request_throughput'],
            df[latency_type],
            label=dir_name,
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            markersize=8,
            linewidth=2
        )
    
    plt.xlabel('Request Throughput (req/s)', fontsize=12)
    plt.ylabel(label, fontsize=12)
    plt.title(title, fontsize=14, pad=15)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300)
    print(f"Generated comparison plot: {output_file}")

def print_formatted_summary(combined_df, directories):
    """Print summary formatted by request rate groups"""
    print("\n=== Formatted Metrics Summary (Grouped by Request Rate) ===")
    
    # Get unique request rates sorted in ascending order
    unique_rates = sorted(combined_df['request_rate'].unique())
    
    # Define columns to display
    display_columns = ['directory', 'mean_ttft_ms', 'mean_tpot_ms', 'request_throughput']
    
    # Print header
    header = f"{'Request Rate':<15} | {'Directory':<15} | {'TTFT (ms)':<12} | {'TPOT (ms)':<12} | {'Throughput (req/s)'}"
    print("\n" + header)
    print("-" * len(header))
    
    # Print each rate group
    for i, rate in enumerate(unique_rates):
        # Get all entries for this request rate
        rate_group = combined_df[combined_df['request_rate'] == rate]
        
        # Print rate header
        print(f"\n{rate:.1f} qps")
        print("-" * 20)
        
        # Print each directory's data for this rate
        for dir_name in directories:
            dir_data = rate_group[rate_group['directory'] == dir_name]
            if not dir_data.empty:
                row = dir_data.iloc[0]
                print(f"{'':<15} | {dir_name:<15} | {row['mean_ttft_ms']:<12.2f} | {row['mean_tpot_ms']:<12.2f} | {row['request_throughput']:.2f}")
            else:
                print(f"{'':<15} | {dir_name:<15} | {'N/A':<12} | {'N/A':<12} | {'N/A'}")
        
        # Add empty line between rate groups (except after last group)
        if i < len(unique_rates) - 1:
            print()

if __name__ == "__main__":
    # List of directories to process
    directories = ["dp_lmcache_g5", "dp_lmcache_g4", "lmcache"]
    
    # Process all directories and collect data
    all_data = {}
    all_metrics = []
    
    for dir_name in directories:
        print(f"Processing directory: {dir_name}")
        df = process_directory(dir_name)
        if df is not None:
            all_data[dir_name] = df
            all_metrics.append(df)
    
    if not all_data:
        print("No valid data found in any directory. Exiting.")
        exit(1)
    
    # Create and save combined metrics CSV
    combined_df = pd.concat(all_metrics, ignore_index=True)
    combined_df = combined_df.sort_values(by=['request_rate', 'directory']).reset_index(drop=True)
    combined_csv = "combined_metrics_summary.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"\nCombined metrics saved to: {combined_csv}")
    
    # Print formatted summary grouped by request rate
    print_formatted_summary(combined_df, directories)
    
    # Prepare valid data for plotting
    valid_data = {}
    columns_to_check = ['request_throughput', 'mean_ttft_ms', 'mean_tpot_ms']
    for dir_name, df in all_data.items():
        valid_df = df.dropna(subset=columns_to_check)
        if not valid_df.empty:
            valid_data[dir_name] = valid_df
        else:
            print(f"Warning: No valid data for plotting in {dir_name}")
    
    if valid_data:
        # Generate comparison plots
        plot_comparison(valid_data, 'mean_ttft_ms', 'ttft_comparison.png')
        plot_comparison(valid_data, 'mean_tpot_ms', 'tpot_comparison.png')
    else:
        print("Warning: Insufficient valid data for generating plots.")
    