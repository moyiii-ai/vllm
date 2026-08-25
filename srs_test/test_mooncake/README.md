# Dual vLLM + shared Mooncake master (embedded DRAM segments)

Yes — multiple vLLM instances may share one `mooncake_master`. Each uses
`MooncakeStoreConnector` in **embedded** mode and contributes its own
`global_segment_size` of **local** CPU DRAM to the pool.

## Topology

`CUDA_DEVICE_ORDER=PCI_BUS_ID` (set in `common.env`):

| nvidia-smi | Device | NUMA | PIX NIC (BDF) | Role |
|------------|--------|------|----------------|------|
| GPU0 | A100 `30:00.0` | 0 | `2f:00.0` | **local** vLLM |
| GPU1 | L40S `41:00.0` | 0 | — | unused in `a100-l40s` |
| GPU2 | L40S `C3:00.0` | 1 | `c1:00.0` | **remote** vLLM |

RDMA **names** (`mlx5_*`) can renumber after reboot; scripts resolve by PCI BDF
(`H1_RDMA_BDF=0000:2f:00.0`, `H2_RDMA_BDF=0000:c1:00.0`) and verify at start.

| Role | netns | NIC BDF | IP | NUMA / GPU | Port |
|------|-------|---------|----|------------|------|
| master + metadata | h1 | — | 10.1.1.1 | CPU0 / NUMA0 | 50051 / 8080 |
| local vLLM | h1 | `2f:00.0` | 10.1.1.1 | NUMA0 / GPU0 (A100) | 8000 |
| remote vLLM | h2 | `c1:00.0` | 10.1.1.2 | NUMA1 / GPU2 (L40S) | 8001 |

Warmup writes prompts into each instance's embedded DRAM segment.
By default Put is **pinned** to each role's own segment
(`10.1.1.1:19001` local / `10.1.1.2:19002` remote via `MOONCAKE_PREFERRED_SEGMENT`).
Pass `--no-prefer-segment` (or `PREFER_SEGMENT=0`) to let Mooncake allocate freely
across the pool. Get is unchanged: local vLLM can still fetch remote-segment keys
over RDMA. After pin + re-warmup, `run_smoke.sh local` should be NUMA0-only (memcpy);
`remote` / `both` still exercise NUMA1-via-RDMA (mlx5_0 GDR).

## Prerequisites

```bash
conda activate vllm-0.22.1
# After each reboot, run host setup once (netns / NIC / peermem / optional h1 internet):
#   ~/intrahost-app-workloads/test_mooncake/setup.sh
# Default model: RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic (public, not gated).
# First start_server downloads ~8GB into ~/.cache/huggingface. Override with MODEL=...
PYTHONHASHSEED=0 python split_narrativeqa_no_overlap.py
python build_measure_roundrobin.py   # L/R interleaved dataset for ./run_measure.sh
```

## Run (recommended: 3 terminals for logs)

```bash
# Terminal A — master on CPU0
./start_store.sh

# Terminal B — local vLLM (GPU0 A100, contributes NUMA0 DRAM)
./start_server.sh local

# Terminal C — remote vLLM (GPU2 L40S, contributes NUMA1 DRAM)
./start_server.sh remote

# Terminal D — clients
./smoke_warmup.sh    # curl only the smoke lines onto each side
./run_smoke.sh       # 1 local + 1 remote curl concurrently on A100
# or: ./warmup.sh    # full split halves in parallel
./run_measure.sh     # round-robin local/remote halves on A100 (build_measure_roundrobin.py)
# ./run_measure.sh --legacy   # old: full narrativeqa.jsonl order
./stop_all.sh
```

Or detach everything:

```bash
./start_all.sh
./smoke_warmup.sh && ./run_smoke.sh
# or: ./warmup.sh && ./run_measure.sh
./stop_all.sh
```

## Knobs

| Env / flag | Default | Meaning |
|------------|---------|---------|
| `MODEL` | `RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic` | FP8-dynamic weights; KV still BF16 with default `kv_cache_dtype=auto`. Public HF repo (no Meta gate). |
| `GPU_PAIR` / `--l40s-l40s` | `a100-l40s` | `a100-l40s`: local GPU0 A100 + remote GPU2 L40S; `l40s-l40s`: local GPU1 L40S + remote GPU2 L40S |
| `GLOBAL_SEGMENT_SIZE` | `128GB` | DRAM each vLLM mounts into the shared pool (total ≈ 2×) |
| `MAX_MODEL_LEN` | `65536` | raise if prompts truncate |
| `OUT_LEN` / `REQUEST_RATE` | `16` / `1` | client decode length / QPS |
| `MC_STORE_MEMCPY` | `1` | local DRAM↔GPU via cudaMemcpy (vs RNIC loopback) |
| `PREFER_SEGMENT` / `--no-prefer-segment` | `1` (on) | pin Put to each role's `H*_SEGMENT`; pass `--no-prefer-segment` or `PREFER_SEGMENT=0` to disable. Restart + re-warmup after change |
| `H1_SEGMENT` / `H2_SEGMENT` | `10.1.1.1:19001` / `10.1.1.2:19002` | TE hostname (+ Put pin target when prefer is on) |
| `ENABLE_CROSS_LAYERS_BLOCKS` | `1` | pack layers into one GPU segment per block (`num_segments=1`); set `0` for per-layer scatter. Restart + re-warmup after toggle |

### L40S+L40S local memcpy check

```bash
# 1) Force Gen4 on L40S GPU1 path (often idles at Gen1!)
sudo ~/intrahost-app-workloads/test_mooncake/setup.sh   # now includes 3e:02.0

# 2) Restart stack on L40S local
./stop_all.sh
./start_all.sh --l40s-l40s
./smoke_warmup.sh --l40s-l40s

# 3) Capture CPU→GPU on switch 3e (SES 44:00.0): ports 112=gpu, 128=cpu
cd ~/intrahost-app-workloads/pcie-switch-telemetry/pex-mon
sudo ./pex-mon -d 44:00.0 -p 96,112,128 -g 0 -t 30 -o data/smoke_local_l40s.csv

# 4) In another terminal, run local-only smoke
./run_smoke.sh --l40s-l40s local

# 5) Plot (Gb/s)
cd data-analysis
python plot-cpu-rnic-throughput.py ../data/smoke_local_l40s.csv --series cpu \
  -p 96,112,128 --port-mapping 96:nic,112:gpu,128:cpu \
  --title "CPU→GPU L40S local (enable_cross_layers_blocks=true)" \
  -o ../data/smoke_local_l40s.png
```

Note: A100 local path is on switch `2b` (no MPT SES) — pex-mon `-d 44:00.0` only sees the **NUMA0 L40S** switch.

## What “good” looks like

After warmup, `logs/master.log` should show **Mem Storage > 0** and **Keys > 0**,
with **two clients**. During `run_measure.sh` / `run_smoke.sh`, Get rates rise;
RNIC counters on `mlx5_0` should show RDMA for remote-segment hits into A100,
while local hits come from NUMA0 DRAM via memcpy.
