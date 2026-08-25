#!/bin/bash
# Measurement: round-robin L/R dataset → Node A only, with
#   kv_transfer_params={"max_offload_tokens": 0}
# so OffloadingConnector does NOT re-cache remotely-fetched (Node B) prefixes.
#
# Expected after prefer-segment warmup:
#   local-half  → OffloadingConnector hit (H2D)
#   remote-half → Offloading miss → MooncakeStore Get over GDRDMA from B
#
# Usage:
#   ./run_measure.sh
#   ./run_measure.sh --legacy
#   ./run_measure.sh path/to.jsonl
#   REQUEST_RATE=10 ./run_measure.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export REQUEST_RATE="${REQUEST_RATE:-3}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.env"
activate_conda

DATASET=""
LEGACY=0
for arg in "$@"; do
  case "$arg" in
    --legacy) LEGACY=1 ;;
    --l40s-l40s|l40s-l40s) export GPU_PAIR=l40s-l40s; apply_gpu_pair ;;
    --a100-l40s|a100-l40s) export GPU_PAIR=a100-l40s; apply_gpu_pair ;;
    -*)
      echo "Unknown flag: $arg"
      echo "Usage: $0 [--legacy] [dataset.jsonl]"
      exit 1
      ;;
    *)
      if [[ -n "$DATASET" ]]; then
        echo "ERROR: multiple dataset paths: $DATASET and $arg"
        exit 1
      fi
      DATASET="$arg"
      ;;
  esac
done

if [[ -z "$DATASET" ]]; then
  if [[ "$LEGACY" == "1" ]]; then
    DATASET="$DATASET_PATH"
  else
    DATASET="$DATASET_MEASURE"
  fi
fi

if [[ ! -f "$DATASET" ]]; then
  echo "Missing dataset: $DATASET"
  if [[ "$DATASET" == "$DATASET_MEASURE" ]]; then
    echo "Build it in test_mooncake:"
    echo "  cd $MOONCAKE_TEST_DIR && python build_measure_roundrobin.py"
  fi
  exit 1
fi

# Layer-1 guard: disable Offloading store for every measure request.
export BENCH_EXTRA_BODY='{"kv_transfer_params":{"max_offload_tokens":0}}'

echo "Requesting sudo for netns ..."
sudo -v

echo "=== measure config (MultiConnector Flow) ==="
echo "  target:  Node A GPU$LOCAL_GPU ($LOCAL_GPU_NAME) :$VLLM_PORT_LOCAL (netns $H1_NS)"
echo "  dataset: $DATASET ($(count_jsonl "$DATASET") prompts)"
if [[ "$DATASET" == "$DATASET_MEASURE" ]]; then
  echo "  mode:    round-robin local/remote (Offloading H2D + Mooncake GDRDMA)"
elif [[ "$LEGACY" == "1" ]]; then
  echo "  mode:    legacy full narrativeqa.jsonl"
fi
echo "  guard:   max_offload_tokens=0 (no Offloading re-cache of remote KV)"
echo "  out_len: $OUT_LEN  rate: $REQUEST_RATE"
echo

wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"
phase_mark start measure
run_bench_on "$H1_NS" "$VLLM_PORT_LOCAL" "$DATASET" "measure_local"
phase_mark end measure

echo
echo "Measure done. Results under $LOG_DIR/measure_local/"
ls -lt "$LOG_DIR/measure_local"/*.json 2>/dev/null | head -n 5 || true
echo "Hint: local-half → Offloading H2D; remote-half → RDMA Get from B ($H2_RDMA)."
