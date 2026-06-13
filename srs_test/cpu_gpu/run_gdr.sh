#!/bin/bash
# Run GDR test with ib_write_bw (gdr_write_server + gdr_write_client).
#   read:  server = h3, numa1, GPU4; client = h2, numa0, CPU only
#   write: server = h2, numa0, CPU only; client = h3, numa1, GPU4

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Arguments:
#   1: MODE = read | write
#   2: SERVER_IP (for client to connect; default 10.1.1.2 for write, 10.1.1.3 for read)
#   3: MSG_SIZE  (default: 33554432)
#   4: ITERATIONS (default: 1000)

MODE=${1:-write}
if [[ "$MODE" != "read" && "$MODE" != "write" ]]; then
  echo "Usage: $0 read|write [SERVER_IP] [MSG_SIZE] [ITERATIONS]"
  exit 2
fi

MSG_SIZE=${3:-33554432}
ITERATIONS=${4:-1000}

if [[ "$MODE" == "read" ]]; then
  SERVER_IP=${2:-10.1.1.3}
else
  SERVER_IP=${2:-10.1.1.2}
fi

echo "================================================================"
echo "Run GDR $MODE (ib_write_bw)"
echo "================================================================"
echo "Mode: $MODE"
echo "Server IP (client connects to): $SERVER_IP"
echo "Message Size: $MSG_SIZE bytes"
echo "Iterations: $ITERATIONS"
echo "================================================================"

cd "$SCRIPT_DIR"

if [[ "$MODE" == "read" ]]; then
  # read: server h3/numa1/GPU4, client h2/numa0/CPU only (use different TCP port to avoid conflict with write)
  echo "[1/2] Starting write server on h3 (NUMA1, GPU4)..."
  PORT=18516 NETNS=h3 NUMA_NODE=1 USE_GPU=1 GPU_ID=4 IB_DEVICE=mlx5_5 ./gdr_write_server.sh "$MSG_SIZE" "$ITERATIONS" &
  SERVER_PID=$!
  sleep 3
  echo "[2/2] Starting write client on h2 (NUMA0, CPU only)..."
  PORT=18516 NETNS=h2 NUMA_NODE=0 USE_GPU=0 IB_DEVICE=mlx5_0 ./gdr_write_client.sh "$SERVER_IP" "$MSG_SIZE" "$ITERATIONS"
else
  # write: server h2/numa0/CPU only, client h3/numa1/GPU4 (default port 18515)
  echo "[1/2] Starting write server on h2 (NUMA0, CPU only)..."
  ./gdr_write_server.sh "$MSG_SIZE" "$ITERATIONS" &
  SERVER_PID=$!
  sleep 3
  echo "[2/2] Starting write client on h3 (NUMA1, GPU4)..."
  ./gdr_write_client.sh "$SERVER_IP" "$MSG_SIZE" "$ITERATIONS"
fi

echo "Waiting for server (PID $SERVER_PID) to finish..."
wait "$SERVER_PID" 2>/dev/null || true
echo "All done."
