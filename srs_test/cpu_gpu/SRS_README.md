# infinite_loop_throughput

**Objective:**  
Run data transfer in an infinite loop and measure sustained throughput.

- baseline: 2GPU + CPU read from GPU `./run_concurrent.sh dma read`
- variation: 2GPU + RNIC read from GPU `./run_variation.sh`
- tries that didn't meet our expectation:
    - 2GPU + CPU dma write to GPU: `./run_concurrent.sh dma write`
    - 2GPU + CPU mmio write to GPU: `./run_concurrent.sh mmio`
    - 2RNIC + CPU write to GPU: `./cpu_gdr/run_concurrent.sh`