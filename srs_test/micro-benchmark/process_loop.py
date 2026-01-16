import re
import os
import glob
import matplotlib.pyplot as plt

def parse_size(s):
    s = s.upper()
    if s.endswith("KB"):
        return (s, int(float(s[:-2])*1024))
    if s.endswith("MB"):
        return (s, int(float(s[:-2])*1024**2))
    if s.endswith("GB"):
        return (s, int(float(s[:-2])*1024**3))
    return (s, int(s))

throughput_re = re.compile(r"Throughput:\s*([\d.]+)\s*GB/s")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "result_loop")
files = glob.glob(os.path.join(data_dir, "*.dat"))

data = {}

for f in files:
    fname = os.path.basename(f)
    m = re.match(r"(\d)_([a-z]+)_([a-z]+)_(.+)\.dat", fname)
    if not m:
        continue
    way, op, kernel, size_str = m.groups()
    way = int(way)
    op = op.lower()
    kernel = kernel.lower()
    size_label, size_val = parse_size(size_str)

    with open(f, "r") as fin:
        content = fin.read()
    m_tp = throughput_re.search(content)
    if not m_tp:
        continue
    tp = float(m_tp.group(1))

    if kernel == "copy":
        kernel_type = "copy kernel"
    elif kernel == "global":
        kernel_type = "ld_st"
    elif kernel == "cuda":
        kernel_type = "cudaMemcpyPeer"
    else:
        kernel_type = kernel

    data.setdefault(size_val, {"label": size_label})
    data[size_val].setdefault(way, {}).setdefault(op, {})[kernel_type] = tp

sizes = sorted(data.keys())

def get_val(sz, way, op, kt):
    return data[sz].get(way, {}).get(op, {}).get(kt, None)

# Print formatted output
for sz in sizes:
    label = data[sz]["label"]
    print(f"Data Size: {label}")
    for way in [1,2]:
        for op in ["read","write"]:
            v1 = get_val(sz, way, op, "copy kernel")
            v2 = get_val(sz, way, op, "ld_st")
            v3 = get_val(sz, way, op, "cudaMemcpyPeer")
            print(f"{way}-way {op.capitalize():5}: "
                  f"copy kernel {v1:.2f} GB/s   | "
                  f"ld_st {v2:.2f} GB/s   | "
                  f"cudaMemcpyPeer {v3:.2f} GB/s")
    print("-"*80)

# Plot
for way in [1,2]:
    for op in ["read","write"]:
        plt.figure(figsize=(8,5))
        x = [sz/1024 for sz in sizes]  # KB
        for kt in ["copy kernel","ld_st","cudaMemcpyPeer"]:
            y = [get_val(sz, way, op, kt) for sz in sizes]
            plt.plot(x, y, marker="o", label=kt)
        plt.xlabel("Data Size (KB)")
        plt.ylabel("Throughput (GB/s)")
        plt.title(f"{way}-way {op.capitalize()}")
        plt.xscale("log")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{way}way_{op}.png")
        plt.savefig(f"{way}way_{op}.pdf")
        plt.close()

print("Done. Check formatted output above and plot files: 1way_read.png ...")

kernel_list = ["copy kernel", "ld_st", "cudaMemcpyPeer"]

for kt in kernel_list:
    plt.figure(figsize=(8,5))
    x = [sz/1024 for sz in sizes]  # KB
    for way in [1,2]:
        for op in ["read","write"]:
            y = [get_val(sz, way, op, kt) for sz in sizes]
            label = f"{way}-way {op}"
            plt.plot(x, y, marker="o", label=label)

    plt.xlabel("Data Size (KB)")
    plt.ylabel("Throughput (GB/s)")
    plt.title(f"{kt}")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    kt_fname = kt.lower().replace(" ", "_")
    plt.legend()
    plt.savefig(f"{kt_fname}.png")
    plt.savefig(f"{kt_fname}.pdf")
    plt.close()

print("Additional figures saved: copy_kernel.png, ld_st.png, cudamemcpypeer.png")