import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def extract_vllm_metrics(json_file):
    """Extract target metrics from a single vLLM benchmark JSON file"""
    try:
        # Open and load JSON data from the file
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract required fields; return None if a field is missing
        metrics = {
            'filename': os.path.basename(json_file),  # Record source filename for traceability
            'mean_ttft_ms': data.get('mean_ttft_ms'),  # Mean Time to First Token (ms)
            'mean_tpot_ms': data.get('mean_tpot_ms'),  # Mean Time per Output Token (ms, excl. 1st)
            'request_rate': data.get('request_rate'),  # Configured request rate (req/s)
            'request_throughput': data.get('request_throughput')  # Actual throughput (req/s)
        }
        return metrics
    except Exception as e:
        # Print error message if file processing fails (e.g., corrupt JSON, missing file)
        print(f"Error processing file {json_file}: {str(e)}")
        return None

def process_all_vllm_files(file_pattern="vllm-*qps.json"):
    """Batch process all JSON files matching the vllm-xxqps.json pattern"""
    # Find all files in current directory that match the pattern
    json_files = glob.glob(file_pattern)
    
    # Exit early if no matching files are found
    if not json_files:
        print(f"No files matching pattern '{file_pattern}' found.")
        return None
    
    # Collect metrics from all valid files
    all_metrics = []
    for file in json_files:
        file_metrics = extract_vllm_metrics(file)
        if file_metrics:  # Skip None results from failed file processing
            all_metrics.append(file_metrics)
    
    # Convert list of metrics to DataFrame for easy analysis/plotting
    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df

if __name__ == "__main__":
    # Step 1: Batch extract metrics from all vLLM JSON files
    vllm_metrics_df = process_all_vllm_files()
    
    if vllm_metrics_df is not None:
        # Step 2: Print raw extracted data for quick verification
        print("Extracted vLLM Benchmark Metrics:")
        print("---------------------------------")
        print(vllm_metrics_df.to_string(index=False))  # Print without index for readability
        
        # Step 3: Save metrics to CSV for long-term storage/analysis
        output_csv = "vllm_benchmark_metrics_summary.csv"
        vllm_metrics_df.to_csv(output_csv, index=False)
        print(f"\nMetrics saved to CSV file: {output_csv}")
        
        # Step 4: Generate basic visualization (Throughput vs Latency)
        plt.figure(figsize=(10, 6))  # Set plot size (width, height) in inches
        
        # Plot Mean TTFT (blue) and Mean TPOT (red) against Request Throughput
        plt.scatter(
            vllm_metrics_df['request_throughput'], 
            vllm_metrics_df['mean_ttft_ms'], 
            color='blue', 
            s=80,  # Marker size
            label='Mean TTFT (ms)'
        )
        plt.scatter(
            vllm_metrics_df['request_throughput'], 
            vllm_metrics_df['mean_tpot_ms'], 
            color='red', 
            s=80,
            label='Mean TPOT (ms)'
        )
        
        # Add plot labels, title, legend and grid
        plt.xlabel('Request Throughput (req/s)', fontsize=12)
        plt.ylabel('Latency (ms)', fontsize=12)
        plt.title('vLLM: Request Throughput vs Latency (TTFT/TPOT)', fontsize=14, pad=15)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)  # Light grid for better readability
        
        # Save plot to file (PNG format for high compatibility)
        plot_output = "vllm_throughput_vs_latency.png"
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        plt.savefig(plot_output, dpi=300)  # Save with 300 DPI for clear image
        print(f"Throughput-Latency plot saved to: {plot_output}")