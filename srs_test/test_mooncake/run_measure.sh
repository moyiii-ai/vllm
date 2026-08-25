#!/bin/bash
# Measurement: serve measure prompts on local vLLM (GPU0 / LOCAL_GPU).
# After prefer-segment warmup, local-half KV is on NUMA0 and remote-half on NUMA1;
# the default dataset round-robins the two halves so loads mix DRAM hits + RDMA.
#
# Usage:
#   ./run_measure.sh                       # narrativeqa_measure_rr.jsonl (default)
#   ./run_measure.sh --legacy              # full narrativeqa.jsonl (old order)
#   ./run_measure.sh path/to.jsonl         # custom dataset
#   REQUEST_RATE=10 ./run_measure.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Measure default rate; set before common.env so its :-1 does not win.
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
    echo "Build it with:"
    echo "  python $SCRIPT_DIR/build_measure_roundrobin.py"
    echo "  (needs narrativeqa_local.jsonl + narrativeqa_remote.jsonl from split)"
  fi
  exit 1
fi

echo "Requesting sudo for netns ..."
sudo -v

echo "=== measure config ==="
echo "  target:  local vLLM GPU$LOCAL_GPU ($LOCAL_GPU_NAME) :$VLLM_PORT_LOCAL (netns $H1_NS)"
echo "  dataset: $DATASET ($(count_jsonl "$DATASET") prompts)"
if [[ "$DATASET" == "$DATASET_MEASURE" ]]; then
  echo "  mode:    round-robin local/remote KV sides (expect DRAM + RDMA mix)"
elif [[ "$LEGACY" == "1" ]]; then
  echo "  mode:    legacy full narrativeqa.jsonl"
fi
echo "  out_len: $OUT_LEN  rate: $REQUEST_RATE"
echo "  expect:  KV hits from NUMA0 DRAM + remote NUMA1 via RDMA ($H1_RDMA)"
echo

wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"
phase_mark start measure
run_bench_on "$H1_NS" "$VLLM_PORT_LOCAL" "$DATASET" "measure_local"
phase_mark end measure

echo
echo "Measure done. Results under $LOG_DIR/measure_local/"
ls -lt "$LOG_DIR/measure_local"/*.json 2>/dev/null | head -n 5 || true
echo "Hint: watch RNIC counters on $H1_RDMA / $H2_RDMA and master Get rates while this runs."
