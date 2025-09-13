#!/usr/bin/env python3
import os
import re
from collections import defaultdict
import matplotlib.pyplot as plt

# -----------------------------
# Parse .dat files
# -----------------------------
DAT_FILES = [f for f in os.listdir('.') if f.endswith('.dat')]

# Nested dict: results[data_size][kernel][mode][op] = throughput
results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

# Regex to match benchmark lines
line_regex = re.compile(
    r"DataSize:\s+([0-9]+ [KMGT]?B),\s+([12]-way),.*?(?:Operation:\s*(read|write))?.*?([\d.]+) GB/s"
)

for file in DAT_FILES:
    fname = file.lower()
    if 'copy' in fname:
        kernel_type = 'copy kernel'
    elif 'global' in fname:
        kernel_type = 'ld_st'
    elif 'cuda' in fname:
        kernel_type = 'cudaMemcpyPeerAsync'
    else:
        kernel_type = 'unknown'

    with open(file, 'r') as f:
        for line in f:
            match = line_regex.search(line)
            if match:
                data_size, mode, op, tp = match.groups()
                tp_val = float(tp)
                # Determine op if missing (2-way, CUDA style)
                if op is None:
                    op = 'read' if 'read' in file.lower() else 'write'
                # For 2-way, only take first throughput
                results[data_size][kernel_type][mode][op] = tp_val

# -----------------------------
# Print table
# -----------------------------
def size_key(s):
    num, unit = s.split()
    num = int(num)
    unit_m = {'B':1, 'KB':1024, 'MB':1024**2, 'GB':1024**3, 'TB':1024**4}
    return num * unit_m[unit]

data_sizes_sorted = sorted(results.keys(), key=size_key)

for ds in data_sizes_sorted:
    print(f"Data Size: {ds}")
    for mode in ['1-way', '2-way']:
        for op in ['read','write']:
            line = f"{mode} {op.capitalize():<5}:"
            for kernel in ['copy kernel', 'ld_st', 'cudaMemcpyPeerAsync']:
                tp = results[ds].get(kernel, {}).get(mode, {}).get(op, None)
                if tp is not None:
                    line += f" {kernel} {tp:.2f} GB/s   |"
                else:
                    line += f" {kernel} N/A   |"
            # Remove last pipe
            line = line.rstrip('|')
            print(line)
    print('-'*80)

# -----------------------------
# Plotting
# -----------------------------
kernels = ['copy kernel', 'ld_st', 'cudaMemcpyPeerAsync']
modes = ['1-way', '2-way']
ops = ['read', 'write']

x_labels = data_sizes_sorted

for kernel in kernels:
    if kernel not in results[data_sizes_sorted[0]]:
        continue
    fig, ax = plt.subplots(figsize=(12,6))

    n = len(data_sizes_sorted)
    width = 0.18
    x = range(n)
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    labels = ['1-way read', '1-way write', '2-way read', '2-way write']

    for i, (mode, op) in enumerate([(m,o) for m in modes for o in ops]):
        y = [results[ds][kernel].get(mode, {}).get(op, 0) for ds in data_sizes_sorted]
        ax.bar([xi + offsets[i] for xi in x], y, width=width, color=colors[i], label=labels[i])

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Data Size')
    ax.set_ylabel('Throughput (GB/s)')
    ax.set_title(f'Throughput Benchmark - {kernel}')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    fig_name = f'{kernel.replace(" ", "_")}_throughput.png'
    plt.savefig(fig_name)
    print(f"Saved {fig_name}")
    plt.close()
