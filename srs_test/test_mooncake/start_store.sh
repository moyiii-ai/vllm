#!/bin/bash
# Mooncake master + HTTP metadata on CPU0 / NUMA0 / h1.
# Shared by both embedded vLLM instances (no mooncake_client).
#
# Default: foreground log stream (Ctrl+C stops).  BACKGROUND=1 or --bg to detach.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.env"
activate_conda
mkdir -p "$LOG_DIR"

BACKGROUND="${BACKGROUND:-0}"
[[ "${1:-}" == "--bg" ]] && BACKGROUND=1

# Host prep (netns / NIC / peermem / optional h1 internet) is NOT done here.
# After each reboot, run once:
#   ~/intrahost-app-workloads/test_mooncake/setup.sh

# Avoid `lsmod | grep -q` under pipefail: grep -q early-exit SIGPIPEs lsmod (exit 141).
if ! grep -qx nvidia_peermem < <(lsmod | awk '{print $1}'); then
  echo "WARNING: nvidia_peermem not loaded. GPU RDMA register will fail."
  echo "  Fix: sudo modprobe nvidia_peermem  (or re-run test_mooncake/setup.sh)"
fi

MASTER_BIN="$(command -v mooncake_master)"
META_BIN="$(command -v mooncake_http_metadata_server)"

: >"$LOG_DIR/metadata.log"
: >"$LOG_DIR/master.log"

PIDS=()

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "Stopping Mooncake master stack ..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${PIDS[@]:-}"; do
    kill -9 "$pid" 2>/dev/null || true
  done
  if ip netns list 2>/dev/null | awk '{print $1}' | grep -qx "$MASTER_NS"; then
    sudo ip netns pids "$MASTER_NS" 2>/dev/null | while read -r p; do
      cmd="$(ps -o args= -p "$p" 2>/dev/null || true)"
      if echo "$cmd" | grep -Eq 'mooncake_http_metadata_server|mooncake_master'; then
        sudo kill -9 "$p" 2>/dev/null || true
      fi
    done
  fi
  rm -f "$LOG_DIR/metadata.pid" "$LOG_DIR/master.pid" "$LOG_DIR/store_tail.pid"
  echo "Master stack stopped."
}

start_one() {
  local name="$1"
  shift
  netns_run "$MASTER_NS" "$@" >"$LOG_DIR/${name}.log" 2>&1 &
  local pid=$!
  PIDS+=("$pid")
  echo "$pid" >"$LOG_DIR/${name}.pid"
  echo "Started $name (wrapper pid $pid)"
}

echo "=== Mooncake master on CPU${MASTER_NUMA}/NUMA${MASTER_NUMA} (netns $MASTER_NS, $MASTER_IP) ==="
echo "Starting metadata on $MASTER_IP:$METADATA_PORT ..."
start_one metadata numactl --cpunodebind="$MASTER_NUMA" --membind="$MASTER_NUMA" bash -c \
  "exec \"$META_BIN\" --host $MASTER_IP --port $METADATA_PORT"
sleep 1

echo "Starting mooncake_master on :$MASTER_PORT ..."
start_one master numactl --cpunodebind="$MASTER_NUMA" --membind="$MASTER_NUMA" bash -c \
  "exec \"$MASTER_BIN\" --port=$MASTER_PORT --enable_http_metadata_server=false"
sleep 2

for name in metadata master; do
  pid="$(cat "$LOG_DIR/${name}.pid")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: $name exited immediately. Log:"
    sed -n '1,80p' "$LOG_DIR/${name}.log" || true
    exit 1
  fi
done

echo "Master up. Both vLLMs use master_server_address=$MASTER_IP:$MASTER_PORT"
echo "Streaming logs (Ctrl+C or ./stop_all.sh to stop)."

if [[ "$BACKGROUND" == "1" ]]; then
  echo "BACKGROUND=1: detaching. Logs: $LOG_DIR/{metadata,master}.log"
  exit 0
fi

trap cleanup INT TERM EXIT
tail -n +1 -F "$LOG_DIR/metadata.log" "$LOG_DIR/master.log" &
TAIL_PID=$!
PIDS+=("$TAIL_PID")
echo "$TAIL_PID" >"$LOG_DIR/store_tail.pid"
wait "$TAIL_PID"
