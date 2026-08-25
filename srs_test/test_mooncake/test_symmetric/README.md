# Symmetric MultiConnector — Offloading + Mooncake on both GPUs

Both vLLM servers use the same `MultiConnector` stack and share one
`mooncake_master`. Unlike `../test_multiconnector` (unidirectional: only the
remote contributes Mooncake DRAM, measure hits local only), this experiment
is **symmetric**: each side stores its warmup half locally, then both serve
the full round-robin dataset so own-half KV comes from Offloading (H2D) and
other-half KV comes from Mooncake GDRDMA.

## Topology

Default `GPU_PAIR=l40s-l40s` (`CUDA_DEVICE_ORDER=PCI_BUS_ID`):

```
Node local  (h1 / GPU1 L40S / NUMA0):
  MultiConnector
  ├── OffloadingConnector       kv_role=kv_both   (64 GiB CPU)
  └── MooncakeStoreConnector    kv_role=kv_both
                                mode=embedded, global_segment_size=64GB
                                MOONCAKE_PREFERRED_SEGMENT=H1_SEGMENT (10.1.1.1:19001)

Node remote (h2 / GPU2 L40S / NUMA1):
  MultiConnector
  ├── OffloadingConnector       kv_role=kv_both   (64 GiB CPU)
  └── MooncakeStoreConnector    kv_role=kv_both
                                mode=embedded, global_segment_size=64GB
                                MOONCAKE_PREFERRED_SEGMENT=H2_SEGMENT (10.1.1.2:19002)
```

Same host / netns / RDMA layout as `../` (`h1`=`2f:00.0`, `h2`=`c1:00.0`).
Pass `--a100-l40s` to use GPU0 A100 + GPU2 L40S instead.

`64 GiB + 64 GB` per NUMA is intentional: each node has ~258 GiB; `128+128`
would OOM. NarrativeQA unique KV is ~73 GiB total, so each local half (~37 GiB)
fits either tier. Override with `OFFLOADING_CPU_BYTES` / `GLOBAL_SEGMENT_SIZE`.

## Why the measure guard

MultiConnector **saves to all** connectors. Without a guard, the first GDRDMA
Get of the other GPU's prefix would be re-cached into local Offloading —
later other-half requests would become H2D instead of GDR.

Measure therefore sends every request with:

```json
"kv_transfer_params": {"max_offload_tokens": 0}
```

(Requires the cherry-picked `#39983` selective-offload support in this tree.)

Warmup does **not** set this param, so each GPU's local half populates both
Offloading and Mooncake (Put-pinned to that GPU's own DRAM segment).

## Prerequisites

```bash
conda activate vllm-0.22.1
# After reboot, once (also forces L40S GPU1 PCIe Gen4 — it often idles at Gen1):
#   ~/intrahost-app-workloads/test_mooncake/setup.sh

# Datasets (reuse parent test_mooncake; do not rebuild here unless missing):
cd ..
PYTHONHASHSEED=0 python split_narrativeqa_no_overlap.py
python build_measure_roundrobin.py
cd test_symmetric
```

Confirm `max_offload_tokens` exists:

```bash
grep -n max_offload_tokens \
  ../../../vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
```

## Run

```bash
./start_all.sh          # default: GPU1 + GPU2 L40S, Put pin ON both sides
./warmup.sh             # local half → GPU1, remote half → GPU2 (parallel)
./run_measure.sh        # full RR dataset → BOTH servers (parallel)
./stop_all.sh
```

### Smoke (path check)

`run_measure.sh` sends the round-robin mix to both servers. KV Transfer
metrics are 10 s aggregates with no dataset id, so you cannot tell whether
a given dump was a local-half H2D or a remote-half GDR. Use one prompt per
path instead:

```bash
./start_all.sh                 # same deploy as the full experiment
./warmup_smoke.sh              # 1 local-line → local, 1 remote-line → remote
./run_measure_smoke.sh         # sequential, both to the local server:
                               #   (1) local-half  → Offloading CPU→GPU
                               #   (2) remote-half → Mooncake load_get,
                               #       Offloading GPU→CPU ≈ 0
```

`run_measure_smoke.sh` waits for the next `KV Transfer metrics:` line after
each curl and prints a PASS/FAIL verdict. Same `max_offload_tokens=0` guard
as measure. Line indices default to 1; pass `3 5` (or `LOCAL_IDX` /
`REMOTE_IDX`) to match a specific warmup pair.

Or stepwise:

```bash
./start_store.sh                 # terminal / BACKGROUND=1
./start_server.sh local          # GPU1, pin H1_SEGMENT
./start_server.sh remote         # GPU2, pin H2_SEGMENT
./warmup.sh
./run_measure.sh
./stop_all.sh
```

GPU / pin overrides:

```bash
./start_all.sh --a100-l40s
./start_all.sh --no-prefer-segment
GPU_PAIR=a100-l40s ./start_all.sh
```

## Knobs

| Env / flag | Default | Meaning |
|------------|---------|---------|
| `GPU_PAIR` / `--l40s-l40s` / `--a100-l40s` | `l40s-l40s` | `l40s-l40s`: GPU1+GPU2; `a100-l40s`: GPU0+GPU2 |
| `OFFLOADING_CPU_BYTES` | `64 GiB` | OffloadingConnector host memory **per server** |
| `OFFLOADING_BLOCK_SIZE` | `128` | Offloaded block size (tokens) |
| `GLOBAL_SEGMENT_SIZE` | `64GB` | Mooncake embedded DRAM **per server** |
| `LOCAL_BUFFER_SIZE` | `1GB` | Per-rank Mooncake transfer buffer |
| `PREFER_SEGMENT` / `--no-prefer-segment` | `1` | Put pin to each role's own `H*_SEGMENT` |
| `OUT_LEN` / `REQUEST_RATE` | `16` / warmup `2`, measure `3` | Client decode / QPS **per server** |
| `ENABLE_CROSS_LAYERS_BLOCKS` | `1` | Mooncake cross-layer packing |
| `MC_STORE_MEMCPY` | `1` | Local DRAM↔GPU via cudaMemcpy when applicable |

## Expected paths

| Phase | Traffic | Path |
|-------|---------|------|
| warmup local | local-half → GPU1 | save Offloading (NUMA0) **and** Mooncake `H1_SEGMENT` |
| warmup remote | remote-half → GPU2 | save Offloading (NUMA1) **and** Mooncake `H2_SEGMENT` |
| measure on GPU1 | local-half | Offloading hit → H2D |
| measure on GPU1 | remote-half | Offloading miss → Mooncake Get (GDRDMA from NUMA1) |
| measure on GPU2 | remote-half | Offloading hit → H2D |
| measure on GPU2 | local-half | Offloading miss → Mooncake Get (GDRDMA from NUMA0) |

With `max_offload_tokens=0`, GDR Gets must **not** land in the local Offloading
tier. Cross traffic is bidirectional on the two RNICs.
