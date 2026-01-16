#!/usr/bin/env python3
"""
Parse benchmark results and generate a summary table or CSV.
Usage: python3 parse_results.py [benchmark_results.txt] [output.csv]
"""

import re
import sys
from collections import defaultdict

def format_size(bytes_size):
    """Format size with appropriate unit (B, KB, MB, GB)."""
    if bytes_size < 1024:
        value = bytes_size
        unit = "B"
    elif bytes_size < 1024 * 1024:
        value = bytes_size / 1024
        unit = "KB"
    elif bytes_size < 1024 * 1024 * 1024:
        value = bytes_size / (1024 * 1024)
        unit = "MB"
    else:
        value = bytes_size / (1024 * 1024 * 1024)
        unit = "GB"
    
    # Remove .00 for integer values
    if value == int(value):
        return f"{int(value)} {unit}"
    else:
        return f"{value:.2f} {unit}"

def parse_results(filename):
    """Parse benchmark output file and extract results.
    Supports both all-gather and all-reduce output formats.
    """
    results = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Try to detect operation type by checking for total_elems field
    # All-Gather: [Rank X] count_per_rank=..., total_elems=..., avg_latency=... ms, throughput=... GB/s
    # All-Reduce: [Rank X] count_per_rank=..., avg_latency=... ms, throughput=... GB/s
    has_total_elems = 'total_elems=' in content
    
    if has_total_elems:
        # All-Gather pattern
        pattern = r'\[Rank (\d+)\]\s+count_per_rank=(\d+),\s+total_elems=(\d+),\s+avg_latency=([\d.]+)\s+ms,\s+throughput=([\d.]+)\s+GB/s'
        operation = 'allgather'
    else:
        # All-Reduce pattern
        pattern = r'\[Rank (\d+)\]\s+count_per_rank=(\d+),\s+avg_latency=([\d.]+)\s+ms,\s+throughput=([\d.]+)\s+GB/s'
        operation = 'allreduce'
    
    matches = re.findall(pattern, content)
    
    # Group by count_per_rank
    by_size = defaultdict(dict)
    
    for match in matches:
        if operation == 'allgather':
            rank = int(match[0])
            count_per_rank = int(match[1])
            total_elems = int(match[2])
            latency = float(match[3])
            throughput = float(match[4])
        else:  # allreduce
            rank = int(match[0])
            count_per_rank = int(match[1])
            latency = float(match[2])
            throughput = float(match[3])
            total_elems = count_per_rank  # For all-reduce, total_elems = count_per_rank
        
        if count_per_rank not in by_size:
            by_size[count_per_rank] = {}
        by_size[count_per_rank][rank] = {
            'latency': latency,
            'throughput': throughput,
            'total_elems': total_elems
        }
    
    # Convert to list of results
    for count_per_rank in sorted(by_size.keys()):
        size_data = by_size[count_per_rank]
        if 0 in size_data and 1 in size_data:
            # Average across both ranks
            avg_latency = (size_data[0]['latency'] + size_data[1]['latency']) / 2
            avg_throughput = (size_data[0]['throughput'] + size_data[1]['throughput']) / 2
            size_bytes = count_per_rank * 4  # float32 = 4 bytes
            results.append({
                'operation': operation,
                'count_per_rank': count_per_rank,
                'size_bytes': size_bytes,
                'size_mb': size_bytes / 1024 / 1024,  # For CSV compatibility
                'size_formatted': format_size(size_bytes),  # Formatted string with unit
                'rank0_latency': size_data[0]['latency'],
                'rank1_latency': size_data[1]['latency'],
                'avg_latency': avg_latency,
                'rank0_throughput': size_data[0]['throughput'],
                'rank1_throughput': size_data[1]['throughput'],
                'avg_throughput': avg_throughput,
            })
    
    return results

def print_table(results):
    """Print results as a formatted table."""
    if not results:
        return
    
    # Detect operation type from first result
    operation = results[0].get('operation', 'unknown')
    op_name = 'All-Gather' if operation == 'allgather' else 'All-Reduce' if operation == 'allreduce' else 'Unknown'
    
    print("\n" + "="*100)
    print(f"Benchmark Results Summary - {op_name}")
    print("="*100)
    print(f"{'Size':<15} {'Count/Rank':<15} {'Latency (ms)':<20} {'Throughput (GB/s)':<25}")
    print(f"{'':15} {'':15} {'Rank0':<10} {'Rank1':<10} {'Rank0':<12} {'Rank1':<12} {'Avg':<12}")
    print("-"*100)
    
    for r in results:
        print(f"{r['size_formatted']:<15} {r['count_per_rank']:<15} "
              f"{r['rank0_latency']:<10.3f} {r['rank1_latency']:<10.3f} "
              f"{r['rank0_throughput']:<12.3f} {r['rank1_throughput']:<12.3f} {r['avg_throughput']:<12.3f}")
    
    print("="*100)
    print("\nNote: Throughput is based on input data size (count_per_rank) for fair comparison.")

def save_csv(results, filename):
    """Save results to CSV file."""
    import csv
    
    if not results:
        return
    
    operation = results[0].get('operation', 'unknown')
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'operation', 'count_per_rank', 'size_mb',
            'rank0_latency_ms', 'rank1_latency_ms', 'avg_latency_ms',
            'rank0_throughput_gbps', 'rank1_throughput_gbps', 'avg_throughput_gbps'
        ])
        
        for r in results:
            writer.writerow([
                r.get('operation', operation),
                r['count_per_rank'], f"{r['size_mb']:.2f}",
                f"{r['rank0_latency']:.3f}", f"{r['rank1_latency']:.3f}", f"{r['avg_latency']:.3f}",
                f"{r['rank0_throughput']:.3f}", f"{r['rank1_throughput']:.3f}", f"{r['avg_throughput']:.3f}"
            ])
    
    print(f"\nCSV saved to: {filename}")

def plot_comparison(allgather_results, allreduce_results, output_dir='.'):
    """Generate comparison plots for latency and throughput."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not available. Skipping plot generation.")
        print("Install with: pip install matplotlib numpy")
        return
    
    if not allgather_results or not allreduce_results:
        print("Warning: Missing results for plotting. Skipping plot generation.")
        return
    
    # Extract data
    ag_sizes = [r['size_mb'] for r in allgather_results]
    ar_sizes = [r['size_mb'] for r in allreduce_results]
    ag_latency = [r['avg_latency'] for r in allgather_results]
    ar_latency = [r['avg_latency'] for r in allreduce_results]
    ag_throughput = [r['avg_throughput'] for r in allgather_results]
    ar_throughput = [r['avg_throughput'] for r in allreduce_results]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Latency comparison
    ax1.plot(ag_sizes, ag_latency, 'o-', label='All-Gather', linewidth=2, markersize=8)
    ax1.plot(ar_sizes, ar_latency, 's-', label='All-Reduce', linewidth=2, markersize=8)
    ax1.set_xlabel('Buffer Size (MB)', fontsize=12)
    ax1.set_ylabel('Average Latency (ms)', fontsize=12)
    ax1.set_title('Latency Comparison: All-Gather vs All-Reduce', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xlim(left=min(min(ag_sizes), min(ar_sizes)) * 0.8, 
                  right=max(max(ag_sizes), max(ar_sizes)) * 1.2)
    
    # Plot 2: Throughput comparison
    ax2.plot(ag_sizes, ag_throughput, 'o-', label='All-Gather', linewidth=2, markersize=8)
    ax2.plot(ar_sizes, ar_throughput, 's-', label='All-Reduce', linewidth=2, markersize=8)
    ax2.set_xlabel('Buffer Size (MB)', fontsize=12)
    ax2.set_ylabel('Average Throughput (GB/s)', fontsize=12)
    ax2.set_title('Throughput Comparison: All-Gather vs All-Reduce', fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xlim(left=min(min(ag_sizes), min(ar_sizes)) * 0.8, 
                  right=max(max(ag_sizes), max(ar_sizes)) * 1.2)
    
    plt.tight_layout()
    
    # Save figure
    output_file = f"{output_dir}/benchmark_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nComparison plots saved to: {output_file}")
    
    # Also save separate figures
    fig1, ax1_sep = plt.subplots(figsize=(10, 6))
    ax1_sep.plot(ag_sizes, ag_latency, 'o-', label='All-Gather', linewidth=2, markersize=8, color='#1f77b4')
    ax1_sep.plot(ar_sizes, ar_latency, 's-', label='All-Reduce', linewidth=2, markersize=8, color='#ff7f0e')
    ax1_sep.set_xlabel('Buffer Size (MB)', fontsize=12)
    ax1_sep.set_ylabel('Average Latency (ms)', fontsize=12)
    ax1_sep.set_title('Latency Comparison: All-Gather vs All-Reduce', fontsize=14, fontweight='bold')
    ax1_sep.set_xscale('log', base=2)
    ax1_sep.grid(True, alpha=0.3)
    ax1_sep.legend(fontsize=11)
    ax1_sep.set_xlim(left=min(min(ag_sizes), min(ar_sizes)) * 0.8, 
                      right=max(max(ag_sizes), max(ar_sizes)) * 1.2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/latency_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    fig2, ax2_sep = plt.subplots(figsize=(10, 6))
    ax2_sep.plot(ag_sizes, ag_throughput, 'o-', label='All-Gather', linewidth=2, markersize=8, color='#1f77b4')
    ax2_sep.plot(ar_sizes, ar_throughput, 's-', label='All-Reduce', linewidth=2, markersize=8, color='#ff7f0e')
    ax2_sep.set_xlabel('Buffer Size (MB)', fontsize=12)
    ax2_sep.set_ylabel('Average Throughput (GB/s)', fontsize=12)
    ax2_sep.set_title('Throughput Comparison: All-Gather vs All-Reduce', fontsize=14, fontweight='bold')
    ax2_sep.set_xscale('log', base=2)
    ax2_sep.grid(True, alpha=0.3)
    ax2_sep.legend(fontsize=11)
    ax2_sep.set_xlim(left=min(min(ag_sizes), min(ar_sizes)) * 0.8, 
                      right=max(max(ag_sizes), max(ar_sizes)) * 1.2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print(f"  - latency_comparison.png")
    print(f"  - throughput_comparison.png")
    print(f"  - benchmark_comparison.png (combined)")

def main():
    # Support parsing one or two files
    if len(sys.argv) < 2:
        print("Usage: python3 parse_results.py <allgather_file> [allreduce_file] [output_csv]")
        print("  If only one file is provided, it will be parsed as a single benchmark.")
        print("  If two files are provided, both will be parsed and compared.")
        sys.exit(1)
    
    allgather_file = sys.argv[1]
    allreduce_file = sys.argv[2] if len(sys.argv) > 2 else None
    output_csv = sys.argv[3] if len(sys.argv) > 3 else 'benchmark_results.csv'
    
    try:
        # Parse all-gather results
        allgather_results = parse_results(allgather_file)
        
        if not allgather_results:
            print(f"No results found in {allgather_file}")
            return
        
        # Print all-gather table
        print_table(allgather_results)
        
        # Parse all-reduce results if provided
        allreduce_results = None
        if allreduce_file:
            print("\n" + "="*120)
            allreduce_results = parse_results(allreduce_file)
            
            if not allreduce_results:
                print(f"No results found in {allreduce_file}")
            else:
                # Print all-reduce table
                print_table(allreduce_results)
                
                # Generate comparison plots
                print("\n" + "="*120)
                print("Generating comparison plots...")
                plot_comparison(allgather_results, allreduce_results)
        
        # Save CSV (combine both if available)
        if allreduce_results:
            # Save combined CSV
            import csv
            combined_csv = output_csv.replace('.csv', '_combined.csv')
            with open(combined_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'operation', 'count_per_rank', 'size_mb',
                    'rank0_latency_ms', 'rank1_latency_ms', 'avg_latency_ms',
                    'rank0_throughput_gbps', 'rank1_throughput_gbps', 'avg_throughput_gbps'
                ])
                
                for r in allgather_results + allreduce_results:
                    writer.writerow([
                        r.get('operation', 'allgather'),
                        r['count_per_rank'], f"{r['size_mb']:.2f}",
                        f"{r['rank0_latency']:.3f}", f"{r['rank1_latency']:.3f}", f"{r['avg_latency']:.3f}",
                        f"{r['rank0_throughput']:.3f}", f"{r['rank1_throughput']:.3f}", f"{r['avg_throughput']:.3f}"
                    ])
            print(f"\nCombined CSV saved to: {combined_csv}")
        else:
            save_csv(allgather_results, output_csv)
        
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
