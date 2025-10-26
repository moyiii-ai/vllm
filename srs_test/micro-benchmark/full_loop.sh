#!/bin/bash

sizes=("4KB" "32KB" "128KB" "512KB" "1MB" "128MB" "1GB" "8GB")
impls=("copy" "cuda" "global")
modes=("1" "2")
ops=("read" "write")
timeout=10

for impl in "${impls[@]}"; do
    for mode in "${modes[@]}"; do
        for op in "${ops[@]}"; do
            for size in "${sizes[@]}"; do
                outfile="${mode}_${op}_${impl}_${size}.dat"
                echo "Running: impl=$impl mode=$mode op=$op size=$size timeout=$timeout"

                # Start run_loop.sh in its own process group
                setsid ./run_loop.sh "$impl" "$mode" "$op" -s "$size" > "$outfile" 2>&1 &
                pid=$!

                # Send SIGINT to the entire process group after $timeout seconds
                (
                    sleep $timeout
                    if kill -0 $pid 2>/dev/null; then
                        echo "Sending SIGINT to process group $pid after $timeout s..."
                        kill -SIGINT -$pid
                    fi
                ) &

                # Wait for run_loop.sh to fully exit
                wait $pid
                echo "Finished: $outfile"
            done
        done
    done
done