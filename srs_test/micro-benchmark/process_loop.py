import os
import re
from collections import defaultdict

# Pattern: extract throughput number before "GB/s"
THROUGHPUT_RE = re.compile(r"Throughput:\s*([\d.]+)\s*GB/s")

# Mapping from filename keyword → readable label
KERNEL_LABELS = {
    "copy": "copy kernel",
    "global": "ld_st",
    "cuda": "cudaMemcpyPeer"
}

# Data structure: results[mode][direction][kernel] = throughput
# e.g. results["1-way"]["Read"]["copy kernel"] = 272.59
results = defaultdict(lambda: defaultdict(dict))

# Traverse all .dat files in current directory
for fname in sorted(f for f in os.listdir(".") if f.endswith(".dat")):
    with open(fname, "r") as f:
        content = f.read()

    # extract throughput values
    matches = THROUGHPUT_RE.findall(content)
    if not matches:
        continue
    throughput = float(matches[0])  # for 2-way, only take first line

    # infer direction (read/write), kernel type, and mode (1-way or 2-way)
    base = fname.lower()

    # kernel type
    kernel_type = next((KERNEL_LABELS[k] for k in KERNEL_LABELS if k in base), None)
    if not kernel_type:
        continue

    # read/write
    direction = "Read" if "read" in base else "Write"

    # 1-way or 2-way
    if "two" in base or "2" in base:
        mode = "2-way"
    else:
        mode = "1-way"

    results[mode][direction][kernel_type] = throughput

# Pretty print results
def fmt_line(mode, direction):
    items = []
    for label in ["copy kernel", "ld_st", "cudaMemcpyPeer"]:
        val = results[mode][direction].get(label)
        if val is not None:
            items.append(f"{label} {val:.2f} GB/s")
    return f"{mode} {direction}: " + "   |   ".join(items)

print(fmt_line("1-way", "Read"))
print(fmt_line("1-way", "Write"))
print(fmt_line("2-way", "Read"))
print(fmt_line("2-way", "Write"))
