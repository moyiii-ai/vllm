# MultiConnector Flow — Offloading (A) + Mooncake (B)

Node A (local) and Node B (remote) share one Mooncake master, but **only B
contributes Mooncake DRAM**. A's CPU budget goes to `OffloadingConnector`.

## Topology

```
Node A (local / h1 / GPU LOCAL):
  MultiConnector
  ├── OffloadingConnector       kv_role=kv_both   (128 GiB CPU)
  └── MooncakeStoreConnector    kv_role=kv_consumer
                                mode=standalone-store, global_segment_size=0

Node B (remote / h2 / GPU REMOTE):
  MooncakeStoreConnector        kv_role=kv_both
                                mode=embedded, global_segment_size=128GB
                                MOONCAKE_PREFERRED_SEGMENT=H2_SEGMENT
```

Same host / netns / RDMA layout as `../test_mooncake` (default `GPU_PAIR=a100-l40s`).

## Why the measure guard

MultiConnector **saves to all** connectors. Without a guard, the first time A
fetches a B-produced prefix over GDRDMA, Offloading would re-cache it locally —
poisoning later remote loads into H2D hits.

Measure therefore sends every request with:

```json
"kv_transfer_params": {"max_offload_tokens": 0}
```

(Requires the cherry-picked `#39983` selective-offload support in this tree.)

Warmup does **not** set this param, so A's local half still populates Offloading.

## Prerequisites

```bash
conda activate vllm-0.22.1
# After reboot, once:
#   ~/intrahost-app-workloads/test_mooncake/setup.sh

# Datasets (reuse test_mooncake; do not rebuild here unless missing):
cd ../test_mooncake
PYTHONHASHSEED=0 python split_narrativeqa_no_overlap.py
python build_measure_roundrobin.py
cd ../test_multiconnector
```

Confirm `max_offload_tokens` exists:

```bash
grep -n max_offload_tokens \
  ../../vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
```

## Run

```bash
./start_all.sh
./warmup.sh          # local→A Offloading, remote→B Mooncake (parallel)
./run_measure.sh     # RR dataset → A only, max_offload_tokens=0
./stop_all.sh
```

Or stepwise:

```bash
./start_store.sh                 # terminal / BACKGROUND=1
./start_server.sh local          # Node A
./start_server.sh remote         # Node B (Put pin ON by default)
./warmup.sh
./run_measure.sh
./stop_all.sh
```

## Knobs

| Env / flag | Default | Meaning |
|------------|---------|---------|
| `OFFLOADING_CPU_BYTES` | `128 GiB` | Node A OffloadingConnector host memory |
| `OFFLOADING_BLOCK_SIZE` | `128` | Offloaded block size (tokens; multiple of `--block-size`) |
| `GLOBAL_SEGMENT_SIZE` | `128GB` | Node B Mooncake embedded DRAM |
| `LOCAL_BUFFER_SIZE` | `1GB` | Per-rank Mooncake transfer buffer (A and B) |
| `PREFER_SEGMENT` / `--no-prefer-segment` | `1` | Node B Put pin to `H2_SEGMENT` |
| `GPU_PAIR` / `--l40s-l40s` | `a100-l40s` | Local/remote GPU pair |
| `OUT_LEN` / `REQUEST_RATE` | `16` / warmup `2`, measure `3` | Client decode / QPS |
| `ENABLE_CROSS_LAYERS_BLOCKS` | `1` | Mooncake cross-layer packing |
| `MC_STORE_MEMCPY` | `1` | Local DRAM↔GPU via cudaMemcpy when applicable |

## Expected measure paths

| Prompt class | Warmup target | Measure on A |
|--------------|---------------|--------------|
| local half | A Offloading | Offloading hit → H2D |
| remote half | B Mooncake | Offloading miss → Mooncake Get (GDRDMA from B) |

With `max_offload_tokens=0`, remote Gets must **not** land in A's Offloading tier.
