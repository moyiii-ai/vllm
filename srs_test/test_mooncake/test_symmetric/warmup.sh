#!/bin/bash
# Warmup for symmetric MultiConnector:
#   local half  -> local server  (Offloading + Mooncake both store to NUMA0 DRAM)
#   remote half -> remote server (Offloading + Mooncake both store to NUMA1 DRAM)
#
# MultiConnector saves to ALL connectors, so each local request populates both
# Offloading (first) and MooncakeStore (second, Put-pinned to own segment).
# Do NOT set max_offload_tokens here.
#
# Usage:
#   ./warmup.sh              # both halves in parallel
#   ./warmup.sh local
#   ./warmup.sh remote
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export REQUEST_RATE="${REQUEST_RATE:-2}"

ROLE="both"
for arg in "$@"; do
  case "$arg" in
    --l40s-l40s|l40s-l40s) export GPU_PAIR=l40s-l40s ;;
    --a100-l40s|a100-l40s) export GPU_PAIR=a100-l40s ;;
    local|remote|both|all) ROLE="$arg" ;;
    *)
      echo "Usage: $0 [--l40s-l40s|--a100-l40s] [both|local|remote]"
      exit 1
      ;;
  esac
done

# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.env"
activate_conda

if [[ ! -f "$DATASET_LOCAL" || ! -f "$DATASET_REMOTE" ]]; then
  echo "Missing split datasets at:"
  echo "  $DATASET_LOCAL"
  echo "  $DATASET_REMOTE"
  echo "Build them in test_mooncake:"
  echo "  cd $MOONCAKE_TEST_DIR && PYTHONHASHSEED=0 python split_narrativeqa_no_overlap.py"
  exit 1
fi

echo "Requesting sudo for netns ..."
sudo -v
echo "=== warmup config (symmetric MultiConnector) ==="
echo "  GPU_PAIR=$GPU_PAIR  local=GPU$LOCAL_GPU ($LOCAL_GPU_NAME)  remote=GPU$REMOTE_GPU ($REMOTE_GPU_NAME)"
echo "  out_len: $OUT_LEN  rate: $REQUEST_RATE"
echo "  local  -> GPU$LOCAL_GPU Offloading + Mooncake pin $H1_SEGMENT"
echo "  remote -> GPU$REMOTE_GPU Offloading + Mooncake pin $H2_SEGMENT"
warn_if_local_gpu_pcie_slow
warn_if_remote_gpu_pcie_slow

warmup_one() {
  local role="$1"
  unset BENCH_EXTRA_BODY || true
  if [[ "$role" == "local" ]]; then
    wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"
    phase_mark start warmup
    run_bench_on "$H1_NS" "$VLLM_PORT_LOCAL" "$DATASET_LOCAL" "warmup_local"
    phase_mark end warmup
  else
    wait_vllm_ready "$H2_NS" "$VLLM_PORT_REMOTE" "$LOG_DIR/vllm_remote.log"
    phase_mark start warmup
    run_bench_on "$H2_NS" "$VLLM_PORT_REMOTE" "$DATASET_REMOTE" "warmup_remote"
    phase_mark end warmup
  fi
}

case "$ROLE" in
  local|remote)
    warmup_one "$ROLE"
    ;;
  both|all)
    echo "=== warmup BOTH (parallel) ==="
    echo "  local : $DATASET_LOCAL  -> :$VLLM_PORT_LOCAL (GPU$LOCAL_GPU $LOCAL_GPU_NAME / NUMA0)"
    echo "  remote: $DATASET_REMOTE -> :$VLLM_PORT_REMOTE (GPU$REMOTE_GPU $REMOTE_GPU_NAME / NUMA1)"
    wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"
    wait_vllm_ready "$H2_NS" "$VLLM_PORT_REMOTE" "$LOG_DIR/vllm_remote.log"
    unset BENCH_EXTRA_BODY || true
    phase_mark start warmup
    run_bench_on "$H1_NS" "$VLLM_PORT_LOCAL" "$DATASET_LOCAL" "warmup_local" &
    pid_a=$!
    run_bench_on "$H2_NS" "$VLLM_PORT_REMOTE" "$DATASET_REMOTE" "warmup_remote" &
    pid_b=$!
    ec=0
    wait "$pid_a" || ec=1
    wait "$pid_b" || ec=1
    phase_mark end warmup
    if [[ "$ec" != "0" ]]; then
      echo "ERROR: one warmup client failed"
      exit 1
    fi
    ;;
  *)
    echo "Usage: $0 [--l40s-l40s|--a100-l40s] [both|local|remote]"
    exit 1
    ;;
esac

echo
echo "Warmup done."
echo "  Each GPU Offloading holds its local-half KV (H2D on measure)."
echo "  Each Mooncake segment holds the same local-half KV (GDRDMA for the other GPU)."
echo "  Check master:  grep -E 'Mem Storage|Keys:' $LOG_DIR/master.log | tail"
