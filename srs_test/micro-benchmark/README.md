# gpu_transfer_variants

**Objective:**  
Evaluate different GPU-to-GPU data transfer implementations.

**Transfer Methods**
- cuda: cudaMemcpyPeer
- copy: custom copy kernel
- global: ld.global / st.global instructions

**Scripts**
- infinite loop until interrupt: `run_loop.sh`
- repeat a fix number: `run_throughput.sh`