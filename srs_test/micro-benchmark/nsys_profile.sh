nsys profile \
    --trace=cuda,osrt \
    --cudabacktrace=all \
    --cuda-memory-usage=true \
    --stats=true \
    ./run_benchmark_global.sh read