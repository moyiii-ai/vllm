#!/bin/bash
# RDMA-only test script
# - Starts RDMA server (CPU-only) then RDMA client (GPU4)
# - Client notifies this script (SIGUSR1) after handshake completes
# - Script signals client (SIGUSR1) to start RDMA writes
# - Verifies handshake + transfer completion via signal, exit code, and logs

set -euo pipefail

# Configuration (override via env or args)
ITERATIONS=${ITERATIONS:-55}
GPU_ID=${GPU_ID:-4}
RDMA_SERVER_PORT=${RDMA_SERVER_PORT:-60000}
SERVER_IP=${SERVER_IP:-10.1.1.2}

# Network namespaces
SERVER_NETNS=${SERVER_NETNS:-h2}
CLIENT_NETNS=${CLIENT_NETNS:-h3}

# NUMA nodes
SERVER_NUMA=${SERVER_NUMA:-0}
CLIENT_NUMA=${CLIENT_NUMA:-1}

# Logs
LOG_DIR=${LOG_DIR:-rdma_results}
RDMA_CLIENT_LOG="$LOG_DIR/rdma_client.log"
RDMA_CLIENT_STDOUT="$LOG_DIR/rdma_client.stdout.log"
RDMA_SERVER_LOG="$LOG_DIR/rdma_server.log"

# Args: [iterations] [server_ip]
if [ $# -gt 0 ]; then
  ITERATIONS=$1
fi
if [ $# -gt 1 ]; then
  SERVER_IP=$2
fi

if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [ "$ITERATIONS" -lt 1 ]; then
  echo "Error: ITERATIONS must be a positive integer"
  echo "Usage: $0 [iterations] [server_ip]"
  exit 1
fi

if [ ! -f "./rdma_server" ] || [ ! -x "./rdma_server" ]; then
  echo "Error: rdma_server not found/executable. Please build with: make"
  exit 1
fi
if [ ! -f "./rdma_client" ] || [ ! -x "./rdma_client" ]; then
  echo "Error: rdma_client not found/executable. Please build with: make"
  exit 1
fi

if ! ip netns list | grep -q "^$SERVER_NETNS"; then
  echo "Error: Network namespace '$SERVER_NETNS' not found"
  ip netns list || true
  exit 1
fi
if ! ip netns list | grep -q "^$CLIENT_NETNS"; then
  echo "Error: Network namespace '$CLIENT_NETNS' not found"
  ip netns list || true
  exit 1
fi

mkdir -p "$LOG_DIR"

RDMA_SERVER_PID=""
RDMA_CLIENT_PID=""

wait_pid_timeout() {
  local pid=$1
  local timeout_sec=$2
  local name=$3

  local ticks=0
  local max_ticks=$((timeout_sec * 10))
  local timed_out=0

  while kill -0 "$pid" 2>/dev/null; do
    if [ "$ticks" -ge "$max_ticks" ]; then
      echo "Timeout waiting for $name (PID: $pid). Sending SIGTERM..."
      kill -SIGTERM "$pid" 2>/dev/null || true
      sleep 1
      if kill -0 "$pid" 2>/dev/null; then
        echo "$name still running. Sending SIGKILL..."
        kill -SIGKILL "$pid" 2>/dev/null || true
      fi
      timed_out=1
      break
    fi
    sleep 0.1
    ticks=$((ticks + 1))
  done
  wait "$pid" 2>/dev/null
  local rc=$?
  if [ "$timed_out" -eq 1 ] && [ "$rc" -eq 0 ]; then
    rc=124
  fi
  return "$rc"
}

cleanup() {
  echo ""
  echo "Cleanup: stopping RDMA processes..."

  if [ -n "${RDMA_CLIENT_PID:-}" ] && kill -0 "$RDMA_CLIENT_PID" 2>/dev/null; then
    echo "Stopping RDMA Client (PID: $RDMA_CLIENT_PID)..."
    kill -SIGINT "$RDMA_CLIENT_PID" 2>/dev/null || true
    wait_pid_timeout "$RDMA_CLIENT_PID" 5 "RDMA Client" || true
  fi

  if [ -n "${RDMA_SERVER_PID:-}" ] && kill -0 "$RDMA_SERVER_PID" 2>/dev/null; then
    echo "Stopping RDMA Server (PID: $RDMA_SERVER_PID)..."
    kill -SIGTERM "$RDMA_SERVER_PID" 2>/dev/null || true
    wait_pid_timeout "$RDMA_SERVER_PID" 5 "RDMA Server" || true
  fi
}
trap cleanup SIGINT SIGTERM EXIT

get_numactl_cmd() {
  local numa_node=$1
  if command -v numactl &>/dev/null; then
    echo "numactl --membind=$numa_node --cpunodebind=$numa_node"
  else
    echo ""
  fi
}

NUMACTL_SERVER=$(get_numactl_cmd "$SERVER_NUMA")
NUMACTL_CLIENT=$(get_numactl_cmd "$CLIENT_NUMA")

echo "================================================================"
echo "RDMA-only test"
echo "================================================================"
echo "  RDMA Server: $SERVER_NETNS netns, NUMA$SERVER_NUMA, CPU-only, $SERVER_IP:$RDMA_SERVER_PORT"
echo "  RDMA Client: $CLIENT_NETNS netns, NUMA$CLIENT_NUMA, GPU$GPU_ID, WRITE, iterations=$ITERATIONS"
echo "  Handshake sync: client -> (SIGUSR1) -> this script"
echo "  Start sync:     this script -> (SIGUSR1) -> client"
echo "Logs:"
echo "  $RDMA_SERVER_LOG"
echo "  $RDMA_CLIENT_STDOUT"
echo "  $RDMA_CLIENT_LOG"
echo "================================================================"
echo ""

echo "[1/3] Starting RDMA Server (CPU-only)..."
sudo ip netns exec "$SERVER_NETNS" $NUMACTL_SERVER ./rdma_server -c >"$RDMA_SERVER_LOG" 2>&1 &
RDMA_SERVER_PID=$!
echo "RDMA Server started (PID: $RDMA_SERVER_PID)"
sleep 1

echo "[2/3] Starting RDMA Client (GPU$GPU_ID, WRITE)..."
RDMA_READY=0
trap 'RDMA_READY=1' SIGUSR1
sudo ip netns exec "$CLIENT_NETNS" $NUMACTL_CLIENT ./rdma_client \
  -a "$SERVER_IP" -p "$RDMA_SERVER_PORT" -w \
  -g "$GPU_ID" -i "$ITERATIONS" \
  -l "$RDMA_CLIENT_LOG" -P "$$" -t 1 \
  >"$RDMA_CLIENT_STDOUT" 2>&1 &
RDMA_CLIENT_PID=$!
echo "RDMA Client started (PID: $RDMA_CLIENT_PID)"

echo "Waiting for RDMA handshake completion (client SIGUSR1 -> script)..."
READY_TIMEOUT_SEC=${READY_TIMEOUT_SEC:-30}
READY_TICKS=$((READY_TIMEOUT_SEC * 10))
for ((i=0; i<READY_TICKS; i++)); do
  if [ "$RDMA_READY" -eq 1 ]; then
    echo "✓ Handshake successful (ready signal received)"
    break
  fi
  if ! kill -0 "$RDMA_CLIENT_PID" 2>/dev/null; then
    echo "Error: RDMA client exited before handshake ready signal."
    echo "Client stdout: $RDMA_CLIENT_STDOUT"
    exit 1
  fi
  sleep 0.1
done
if [ "$RDMA_READY" -ne 1 ]; then
  echo "Error: did not receive handshake ready signal within ${READY_TIMEOUT_SEC}s"
  echo "Client stdout: $RDMA_CLIENT_STDOUT"
  exit 1
fi

echo "[3/3] Signaling client to start RDMA writes (SIGUSR1)..."
kill -SIGUSR1 "$RDMA_CLIENT_PID" 2>/dev/null || true

echo "Waiting for RDMA client to finish transfers..."
CLIENT_TIMEOUT_SEC=${CLIENT_TIMEOUT_SEC:-120}
set +e
wait_pid_timeout "$RDMA_CLIENT_PID" "$CLIENT_TIMEOUT_SEC" "RDMA Client"
CLIENT_RC=$?
set -e

if [ "$CLIENT_RC" -ne 0 ]; then
  echo "Error: RDMA client exited non-zero: $CLIENT_RC"
  echo "Client stdout: $RDMA_CLIENT_STDOUT"
  exit "$CLIENT_RC"
fi

# Basic transfer verification from logs
if grep -q "=== RDMA Client Results ===" "$RDMA_CLIENT_STDOUT" 2>/dev/null; then
  echo "✓ Transfer completed (client printed results)"
else
  echo "Warning: client did not print results marker. Check logs:"
  echo "  $RDMA_CLIENT_STDOUT"
fi

if grep -q "Total iterations: $ITERATIONS" "$RDMA_CLIENT_STDOUT" 2>/dev/null; then
  echo "✓ Iteration count matches ($ITERATIONS)"
else
  echo "Warning: iteration count marker not found/mismatch. Check:"
  echo "  $RDMA_CLIENT_STDOUT"
fi

echo "Stopping RDMA server..."
kill -SIGTERM "$RDMA_SERVER_PID" 2>/dev/null || true
wait_pid_timeout "$RDMA_SERVER_PID" 5 "RDMA Server" || true

echo ""
echo "RDMA-only test finished successfully."

