#!/usr/bin/env bash
set -euo pipefail

PIPE0="/tmp/server_dp0_pipe_$$"
PIPE1="/tmp/server_dp1_pipe_$$"
WARMUP_LOG="warmup.txt"
FORMAL_LOG="remote_trace.txt"
DP1_LOG="remote_trace1.txt"

# Create a new process group for clean termination
set -m
trap 'cleanup' EXIT INT TERM

> "$WARMUP_LOG"
> "$FORMAL_LOG"
> "$DP1_LOG"

[[ -p "$PIPE0" ]] && rm -f "$PIPE0"
[[ -p "$PIPE1" ]] && rm -f "$PIPE1"
mkfifo "$PIPE0" "$PIPE1"

MAIN_PID=$$

cleanup() {
  if [[ "$$" -eq "$MAIN_PID" ]]; then
    echo "[Cleanup] Cleaning up..."
    kill -TERM -$$ >/dev/null 2>&1 || true  # kill entire group safely
    rm -f "$PIPE0" "$PIPE1"
  fi
}

# ---- Logger functions ----
logger_dp0() {
  stage="warmup"
  trap 'stage="formal"' USR1
  while IFS= read -r line; do
    if [[ "$stage" == "warmup" ]]; then
      printf '%s\n' "$line" >> "$WARMUP_LOG"
    else
      printf '%s\n' "$line" >> "$FORMAL_LOG"
    fi
  done < "$PIPE0"
}

logger_dp1() {
  trap 'stage="formal"' USR1
  while IFS= read -r line; do
    printf '%s\n' "$line" >> "$DP1_LOG"
  done < "$PIPE1"
}

# ---- Start loggers ----
logger_dp0 &
LOGGER0_PID=$!
logger_dp1 &
LOGGER1_PID=$!

# ---- Start servers ----
./server_dp0.sh 2> "$PIPE0" &
SERVER0_PID=$!
./server_dp1.sh 2> "$PIPE1" &
SERVER1_PID=$!

echo "Servers are starting..."
sleep 3

# ---- Warmup phase ----
echo
echo ">>> Press ENTER to start warmup clients <<<"
read -r _
./client_benchmark_warmup_dp.sh & CLIENT_WARM_PID=$!
wait "$CLIENT_WARM_PID" || true

# ---- Switch to formal phase ----
kill -USR1 "$LOGGER0_PID" "$LOGGER1_PID" 2>/dev/null || true
echo "[Logger] switched to formal phase"

# ---- Formal phase ----
./client_benchmark_dp.sh & CLIENT_FORMAL_PID=$!
wait "$CLIENT_FORMAL_PID" || true

# ---- Normal stop ----
echo "[Main] Benchmark complete. Stopping servers..."
exit 0
