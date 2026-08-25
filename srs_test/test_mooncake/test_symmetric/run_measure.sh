#!/bin/bash
# Measurement: round-robin L/R dataset → BOTH servers in parallel, with
#   kv_transfer_params={"max_offload_tokens": 0}
# so OffloadingConnector does NOT re-cache prefixes fetched over Mooncake GDR.
#
# After prefer-segment warmup (local half on local, remote half on remote):
#   own-half     → OffloadingConnector hit (local CPU DRAM, H2D)
#   other-half   → Offloading miss → MooncakeStore Get over GDRDMA
#
# Usage:
#   ./run_measure.sh                 # both servers, narrativeqa_measure_rr.jsonl
#   ./run_measure.sh both
#   ./run_measure.sh local           # only local (debug)
#   ./run_measure.sh remote
#   ./run_measure.sh --legacy
#   ./run_measure.sh path/to.jsonl
#   REQUEST_RATE=10 ./run_measure.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export REQUEST_RATE="${REQUEST_RATE:-3}"

DATASET=""
LEGACY=0
TARGET="both"
for arg in "$@"; do
  case "$arg" in
    --legacy) LEGACY=1 ;;
    --l40s-l40s|l40s-l40s) export GPU_PAIR=l40s-l40s ;;
    --a100-l40s|a100-l40s) export GPU_PAIR=a100-l40s ;;
    local|remote|both|all) TARGET="$arg" ;;
    -*)
      echo "Unknown flag: $arg"
      echo "Usage: $0 [--legacy] [--l40s-l40s|--a100-l40s] [both|local|remote] [dataset.jsonl]"
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

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.env"
activate_conda

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
# Without this, the first GDR Get would be re-cached in Offloading and later
# "other-half" prompts would become H2D instead of GDRDMA.
export BENCH_EXTRA_BODY='{"kv_transfer_params":{"max_offload_tokens":0}}'

echo "Requesting sudo for netns ..."
sudo -v

echo "=== measure config (symmetric MultiConnector) ==="
echo "  GPU_PAIR=$GPU_PAIR  target=$TARGET"
echo "  local:   GPU$LOCAL_GPU ($LOCAL_GPU_NAME) :$VLLM_PORT_LOCAL ($H1_NS)"
echo "  remote:  GPU$REMOTE_GPU ($REMOTE_GPU_NAME) :$VLLM_PORT_REMOTE ($H2_NS)"
echo "  dataset: $DATASET ($(count_jsonl "$DATASET") prompts)"
if [[ "$DATASET" == "$DATASET_MEASURE" ]]; then
  echo "  mode:    round-robin local/remote (Offloading H2D + Mooncake GDRDMA)"
elif [[ "$LEGACY" == "1" ]]; then
  echo "  mode:    legacy full narrativeqa.jsonl"
fi
echo "  guard:   max_offload_tokens=0 (no Offloading re-cache of GDR KV)"
echo "  out_len: $OUT_LEN  rate: $REQUEST_RATE (per server)"
warn_if_local_gpu_pcie_slow
warn_if_remote_gpu_pcie_slow
echo

measure_local() {
  wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"
  run_bench_on "$H1_NS" "$VLLM_PORT_LOCAL" "$DATASET" "measure_local"
}

measure_remote() {
  wait_vllm_ready "$H2_NS" "$VLLM_PORT_REMOTE" "$LOG_DIR/vllm_remote.log"
  run_bench_on "$H2_NS" "$VLLM_PORT_REMOTE" "$DATASET" "measure_remote"
}

case "$TARGET" in
  local)
    phase_mark start measure
    measure_local
    phase_mark end measure
    ;;
  remote)
    phase_mark start measure
    measure_remote
    phase_mark end measure
    ;;
  both|all)
    wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"
    wait_vllm_ready "$H2_NS" "$VLLM_PORT_REMOTE" "$LOG_DIR/vllm_remote.log"
    phase_mark start measure
    measure_local &
    pid_a=$!
    measure_remote &
    pid_b=$!
    ec=0
    wait "$pid_a" || ec=1
    wait "$pid_b" || ec=1
    phase_mark end measure
    if [[ "$ec" != "0" ]]; then
      echo "ERROR: one measure client failed"
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 [--legacy] [--l40s-l40s|--a100-l40s] [both|local|remote] [dataset.jsonl]"
    exit 1
    ;;
esac

echo
echo "Measure done."
echo "  local results:  $LOG_DIR/measure_local/"
ls -lt "$LOG_DIR/measure_local"/*.json 2>/dev/null | head -n 3 || true
echo "  remote results: $LOG_DIR/measure_remote/"
ls -lt "$LOG_DIR/measure_remote"/*.json 2>/dev/null | head -n 3 || true
echo "Hint: own-half → Offloading H2D; other-half → RDMA Get ($H1_RDMA ↔ $H2_RDMA)."
