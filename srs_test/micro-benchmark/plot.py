import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

files = ["distance1.log", "distance2.log", "distance3.log"]
distances = ["distance1", "distance2", "distance3"]

data = {}

for fname, dist in zip(files, distances):
    with open(fname, "r") as f:
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
            if not line:
                continue
            m = re.match(r"(1-way|2-way)\s+(Read|Write)\s*:\s*(.*)", line)
            if not m:
                continue
            way, direction, rest = m.groups()

            methods = re.findall(r"(copy kernel|ld_st|cudaMemcpyPeerAsync)\s+([0-9.]+)\s*GB/s", rest)
            for method, val in methods:
                key = (data_size, method, direction, way)
                if key not in data:
                    data[key] = {}
                data[key][dist] = float(val)

color_map = {
    "copy kernel": {"Read": "#6baed6", "Write": "#08519c"},
    # "ld_st": {"Read": "#74c476", "Write": "#006d2c"},
    "cudaMemcpyPeerAsync": {"Read": "#fb6a4a", "Write": "#a50f15"}
}
way_styles = {
    "1-way": "--",
    "2-way": "-"
}

def size_key(x):
    num = int(re.match(r"(\d+)", x).group(1))
    unit = "B"
    if "KB" in x: unit = "KB"
    elif "MB" in x: unit = "MB"
    elif "GB" in x: unit = "GB"
    scale = {"B":1, "KB":1024, "MB":1024**2, "GB":1024**3}
    return num * scale[unit]

for data_size in sorted(set(k[0] for k in data.keys()), key=size_key):
    fig, ax = plt.subplots(figsize=(10,6))
    for method in ["copy kernel", "cudaMemcpyPeerAsync"]:
        for direction in ["Read", "Write"]:
            for way in ["1-way", "2-way"]:
                key = (data_size, method, direction, way)
                if key not in data:
                    continue
                vals = [data[key].get(dist, None) for dist in distances]
                ax.plot(
                    distances, vals,
                    color=color_map[method][direction],
                    linestyle=way_styles[way],
                    marker="o",
                    linewidth=2
                )

    legend1_elements = [
        Line2D([0], [0], color=color_map["copy kernel"]["Read"], lw=2, label="copy kernel Read"),
        Line2D([0], [0], color=color_map["copy kernel"]["Write"], lw=2, label="copy kernel Write"),
        # Line2D([0], [0], color=color_map["ld_st"]["Read"], lw=2, label="ld_st Read"),
        # Line2D([0], [0], color=color_map["ld_st"]["Write"], lw=2, label="ld_st Write"),
        Line2D([0], [0], color=color_map["cudaMemcpyPeerAsync"]["Read"], lw=2, label="cudaMemcpy Read"),
        Line2D([0], [0], color=color_map["cudaMemcpyPeerAsync"]["Write"], lw=2, label="cudaMemcpy Write"),
    ]
    legend1 = ax.legend(handles=legend1_elements, loc="upper left", bbox_to_anchor=(1.05, 1), title="Method + Read/Write")
    ax.add_artist(legend1)

    legend2_elements = [
        Line2D([0], [0], color="black", linestyle="-", lw=2, label="2-way"),
        Line2D([0], [0], color="black", linestyle="--", lw=2, label="1-way"),
    ]
    ax.legend(handles=legend2_elements, loc="lower left", bbox_to_anchor=(1.05, 0), title="Transfer Mode")

    ax.set_title(f"Throughput vs Distance (Data Size: {data_size})")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Throughput (GB/s)")

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.savefig(f"{data_size}.png")
    plt.savefig(f"{data_size}.pdf")
    plt.close()
