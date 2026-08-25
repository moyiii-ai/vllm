#!/bin/bash
# Warmup: each split half is served once on its owning vLLM so KV lands in that
# instance's embedded DRAM segment (local=NUMA0/GPU0 A100, remote=NUMA1/GPU2 L40S).
#
# For smoke-only priming (one curl per side), use ./smoke_warmup.sh instead.
#
# Usage:
#   ./warmup.sh              # both halves in parallel
#   ./warmup.sh local        # only narrativeqa_local -> GPU0 (A100)
#   ./warmup.sh remote       # only narrativeqa_remote -> GPU2 (NUMA1 L40S)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Warmup default 2 QPS; set before common.env so its :-1 does not win.
# Override: REQUEST_RATE=1 ./warmup.sh
export REQUEST_RATE="${REQUEST_RATE:-2}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.env"
activate_conda

ROLE="${1:-both}"

if [[ ! -f "$DATASET_LOCAL" || ! -f "$DATASET_REMOTE" ]]; then
  echo "Missing split datasets. Run:"
  echo "  PYTHONHASHSEED=0 python split_narrativeqa_no_overlap.py"
  exit 1
fi

echo "Requesting sudo for netns ..."
sudo -v
echo "=== warmup config ==="
echo "  out_len: $OUT_LEN  rate: $REQUEST_RATE"
warmup_one() {
  local role="$1"
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
    echo "  local : $DATASET_LOCAL  -> :$VLLM_PORT_LOCAL (GPU$LOCAL_GPU A100 / NUMA0 DRAM)"
    echo "  remote: $DATASET_REMOTE -> :$VLLM_PORT_REMOTE (GPU$REMOTE_GPU L40S / NUMA1 DRAM)"
    wait_vllm_ready "$H1_NS" "$VLLM_PORT_LOCAL" "$LOG_DIR/vllm_local.log"
    wait_vllm_ready "$H2_NS" "$VLLM_PORT_REMOTE" "$LOG_DIR/vllm_remote.log"
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
    echo "Usage: $0 [both|local|remote]"
    exit 1
    ;;
esac

echo
echo "Warmup done. Check master Mem Storage / Keys in logs/master.log"
echo "  rg 'Mem Storage|Keys:' logs/master.log | tail"
