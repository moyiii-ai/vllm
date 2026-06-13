import re
import matplotlib.pyplot as plt

file = "distance2.log"

data = {}

with open(file, "r") as f:
    content = f.read()

blocks = content.split("--------------------------------------------------------------------------------")

for block in blocks:
    block = block.strip()
    if not block:
        continue

    m = re.search(r"Data Size:\s*([0-9]+(?:\s*[KMG]B)?)", block)
    if not m:
        continue
    data_size = m.group(1).replace(" ", "")

    lines = block.split("\n")[1:]
    for line in lines:
        line = line.strip()
        m = re.match(r"(1-way|2-way)\s+(Read|Write)\s*:\s*(.*)", line)
        if not m:
            continue
        way, direction, rest = m.groups()

        methods = re.findall(r"(copy kernel|ld_st|cudaMemcpyPeerAsync)\s+([0-9.]+)\s*GB/s", rest)
        for method, val in methods:
            key = (data_size, direction, way, method)
            data[key] = float(val)

def size_key(x):
    num = int(re.match(r"(\d+)", x).group(1))
    unit = "B"
    if "KB" in x: unit = "KB"
    elif "MB" in x: unit = "MB"
    elif "GB" in x: unit = "GB"
    scale = {"B":1, "KB":1024, "MB":1024**2, "GB":1024**3}
    return num * scale[unit]

sorted_sizes = sorted(set(k[0] for k in data.keys()), key=size_key)

color_map = {
    "copy kernel": "#1f77b4",
    "ld_st": "#2ca02c",
    "cudaMemcpyPeerAsync": "#d62728"
}

scenes = [
    ("1-way", "Read"),
    ("1-way", "Write"),
    ("2-way", "Read"),
    ("2-way", "Write"),
]

for way, direction in scenes:
    fig, ax = plt.subplots(figsize=(10,6))
    for method in color_map.keys():
        ys = []
        for size in sorted_sizes:
            key = (size, direction, way, method)
            ys.append(data.get(key, None))

        ax.plot(
            sorted_sizes, ys,
            marker="o", linewidth=2,
            label=method,
            color=color_map[method]
        )

    ax.set_xlabel("Data Size")
    ax.set_ylabel("Throughput (GB/s)")
    ax.set_title(f"{way} {direction} Throughput vs Data Size")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{way}_{direction}.png")
    plt.close()

fig, ax = plt.subplots(figsize=(12,8))
for way, direction in scenes:
    for method in color_map.keys():
        ys = []
        for size in sorted_sizes:
            key = (size, direction, way, method)
            ys.append(data.get(key, None))

        ax.plot(
            sorted_sizes, ys,
            marker="o", linewidth=1.5,
            label=f"{method} {way} {direction}"
        )

ax.set_xlabel("Data Size")
ax.set_ylabel("Throughput (GB/s)")
ax.set_title("Throughput vs Data Size (All)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.subplots_adjust(right=0.75)
plt.savefig("ALL.png")
plt.close()
