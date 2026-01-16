#!/usr/bin/env python3
"""
Plot combined results from read_write_result.txt and benchmark_results_combined.csv
Generates a comparison plot with 5 lines:
- 1-way read copy kernel
- 1-way write copy kernel
- 1-way read cudaMemcpyPeer
- all gather
- all reduce
"""

import re
import csv
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_size_to_mb(size_str):
    """Convert size string (e.g., '4KB', '1MB', '1GB') to MB (float)."""
    size_str = size_str.strip().upper()
    
    # Extract number and unit
    match = re.match(r'([\d.]+)\s*([KMGT]?B)', size_str)
    if not match:
        return None
    
    value = float(match.group(1))
    unit = match.group(2)
    
    # Convert to MB
    if unit == 'B':
        return value / (1024 * 1024)
    elif unit == 'KB':
        return value / 1024
    elif unit == 'MB':
        return value
    elif unit == 'GB':
        return value * 1024
    else:
        return None

def parse_read_write_file(filename):
    """Parse read_write_result.txt and extract data."""
    data = {
        'sizes': [],
        'read_copy_kernel': [],
        'write_copy_kernel': [],
        'read_cudamemcpypeer': []
    }
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Pattern to match data size lines
    size_pattern = r'Data Size:\s*([\d.]+\s*[KMGT]?B)'
    # Pattern to match 1-way Read line
    read_pattern = r'1-way Read\s*:\s*copy kernel\s*([\d.]+)\s*GB/s.*?cudaMemcpyPeer\s*([\d.]+)\s*GB/s'
    # Pattern to match 1-way Write line
    write_pattern = r'1-way Write:\s*copy kernel\s*([\d.]+)\s*GB/s'
    
    # Find all data size sections
    sections = re.split(r'-{50,}', content)
    
    for section in sections:
        # Extract size
        size_match = re.search(size_pattern, section)
        if not size_match:
            continue
        
        size_str = size_match.group(1)
        size_mb = parse_size_to_mb(size_str)
        if size_mb is None:
            continue
        
        # Extract 1-way Read data
        read_match = re.search(read_pattern, section)
        if read_match:
            read_copy = float(read_match.group(1))
            read_cuda = float(read_match.group(2))
            data['sizes'].append(size_mb)
            data['read_copy_kernel'].append(read_copy)
            data['read_cudamemcpypeer'].append(read_cuda)
        else:
            continue
        
        # Extract 1-way Write data
        write_match = re.search(write_pattern, section)
        if write_match:
            write_copy = float(write_match.group(1))
            data['write_copy_kernel'].append(write_copy)
        else:
            # If write not found, skip this section
            if len(data['sizes']) > len(data['write_copy_kernel']):
                data['sizes'].pop()
                data['read_copy_kernel'].pop()
                data['read_cudamemcpypeer'].pop()
            continue
    
    return data

def parse_benchmark_csv(filename):
    """Parse benchmark_results_combined.csv and extract allgather and allreduce data.
    Note: count_per_rank is the number of elements, need to convert to actual data size.
    Each element is float (4 bytes), so data_size = count_per_rank * 4 bytes.
    """
    data = {
        'allgather': {'sizes': [], 'throughput': []},
        'allreduce': {'sizes': [], 'throughput': []}
    }
    
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            operation = row['operation'].strip().lower()
            count_per_rank = int(row['count_per_rank'])
            throughput = float(row['avg_throughput_gbps'])
            
            # Convert count_per_rank to actual data size in MB
            # count_per_rank elements * 4 bytes per float = data_size_bytes
            data_size_bytes = count_per_rank * 4
            size_mb = data_size_bytes / (1024 * 1024)
            
            if operation == 'allgather':
                data['allgather']['sizes'].append(size_mb)
                data['allgather']['throughput'].append(throughput)
            elif operation == 'allreduce':
                data['allreduce']['sizes'].append(size_mb)
                data['allreduce']['throughput'].append(throughput)
    
    return data

def format_size_label(size_mb):
    """Format size in MB to human-readable KB/MB/GB format."""
    if size_mb < 1:
        # Convert to KB
        size_kb = size_mb * 1024
        if size_kb == int(size_kb):
            return f"{int(size_kb)} KB"
        else:
            return f"{size_kb:.2f} KB"
    elif size_mb < 1024:
        # Keep as MB
        if size_mb == int(size_mb):
            return f"{int(size_mb)} MB"
        else:
            return f"{size_mb:.2f} MB"
    else:
        # Convert to GB
        size_gb = size_mb / 1024
        if size_gb == int(size_gb):
            return f"{int(size_gb)} GB"
        else:
            return f"{size_gb:.2f} GB"

def plot_combined(read_write_data, benchmark_data, output_file='combined_comparison.png'):
    """Generate combined comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Collect all x values to determine tick positions
    all_sizes = []
    if read_write_data['sizes']:
        all_sizes.extend(read_write_data['sizes'])
    if benchmark_data['allgather']['sizes']:
        all_sizes.extend(benchmark_data['allgather']['sizes'])
    if benchmark_data['allreduce']['sizes']:
        all_sizes.extend(benchmark_data['allreduce']['sizes'])
    
    if not all_sizes:
        print("Error: No data to plot")
        return
    
    # Plot read_write data
    if read_write_data['sizes']:
        ax.plot(read_write_data['sizes'], read_write_data['read_copy_kernel'], 
                'o-', label='1-way Read Copy Kernel', linewidth=2, markersize=6, color='#1f77b4')
        ax.plot(read_write_data['sizes'], read_write_data['write_copy_kernel'], 
                's-', label='1-way Write Copy Kernel', linewidth=2, markersize=6, color='#ff7f0e')
        ax.plot(read_write_data['sizes'], read_write_data['read_cudamemcpypeer'], 
                '^-', label='1-way Read cudaMemcpyPeer', linewidth=2, markersize=6, color='#2ca02c')
    
    # Plot benchmark data
    if benchmark_data['allgather']['sizes']:
        ax.plot(benchmark_data['allgather']['sizes'], benchmark_data['allgather']['throughput'], 
                'd-', label='All-Gather', linewidth=2, markersize=6, color='#d62728')
    
    if benchmark_data['allreduce']['sizes']:
        ax.plot(benchmark_data['allreduce']['sizes'], benchmark_data['allreduce']['throughput'], 
                'v-', label='All-Reduce', linewidth=2, markersize=6, color='#9467bd')
    
    # Set labels and title
    ax.set_xlabel('Data Size', fontsize=12)
    ax.set_ylabel('Throughput (GB/s)', fontsize=12)
    ax.set_title('Performance Comparison: Read/Write vs All-Gather/All-Reduce', fontsize=14, fontweight='bold')
    
    # Set log scale for x-axis
    min_size = min(all_sizes)
    max_size = max(all_sizes)
    
    # Ensure min_size is positive and handle very small values
    if min_size <= 0:
        min_size = 0.001  # 1 KB minimum
    if max_size <= min_size:
        max_size = min_size * 1000
    
    # Set reasonable limits with padding
    xlim_min = max(0.001, min_size * 0.8)  # At least 1 KB
    xlim_max = max_size * 1.2
    
    ax.set_xscale('log', base=10)
    ax.set_xlim(xlim_min, xlim_max)
    
    # Use only actual data points as ticks (no intermediate values)
    unique_sizes = sorted(set(all_sizes))
    
    # Set ticks only at actual data points
    ax.set_xticks(unique_sizes)
    ax.set_xticklabels([format_size_label(size) for size in unique_sizes], rotation=45, ha='right')
    ax.tick_params(axis='x', which='minor', length=0)  # Hide minor ticks
    
    # Add grid
    ax.grid(True, alpha=0.3, which='both')
    
    # Add legend
    ax.legend(fontsize=10, loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Combined comparison plot saved to: {output_file}")
    
    plt.close()

def main():
    read_write_file = 'read_write_result.txt'
    benchmark_file = 'benchmark_results_combined.csv'
    output_file = 'combined_comparison.png'
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        read_write_file = sys.argv[1]
    if len(sys.argv) > 2:
        benchmark_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    try:
        # Parse read_write data
        print(f"Parsing {read_write_file}...")
        read_write_data = parse_read_write_file(read_write_file)
        print(f"  Found {len(read_write_data['sizes'])} data points")
        
        # Parse benchmark data
        print(f"Parsing {benchmark_file}...")
        benchmark_data = parse_benchmark_csv(benchmark_file)
        print(f"  All-Gather: {len(benchmark_data['allgather']['sizes'])} data points")
        print(f"  All-Reduce: {len(benchmark_data['allreduce']['sizes'])} data points")
        
        # Generate plot
        print(f"\nGenerating combined comparison plot...")
        plot_combined(read_write_data, benchmark_data, output_file)
        
        print("\nDone!")
        
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
