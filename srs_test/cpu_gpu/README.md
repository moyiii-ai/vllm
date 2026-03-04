# CPU-GPU and GPU-GPU DMA Transfer Tools

This directory contains scripts and programs for testing concurrent data transfers:
1. GPU0 -> GPU2 DMA write
2. GPU1 -> GPU2 DMA write
3. CPU <-> GPU2 transfer (DMA or MMIO mode)

## Files

- `gpu_to_gpu_dma.cu`: CUDA program for GPU-to-GPU DMA write operations using `cudaMemcpyPeer`
- `cpu_to_gpu_dma.cu`: CUDA program for GPU->CPU DMA read (initiated from GPU via `cudaMemcpy`)
- `cpu_to_gpu_mmio.cu`: CUDA program for CPU->GPU MMIO write using unified memory
- `run_concurrent.sh`: Shell script to run all operations concurrently
- `Makefile`: Build configuration

## Building

Compile the programs using Make:

```bash
make
```

This will create three executables:
- `gpu_to_gpu_dma`: GPU-to-GPU DMA write program
- `cpu_to_gpu_dma`: GPU->CPU DMA read program (initiated from GPU)
- `cpu_to_gpu_mmio`: CPU->GPU MMIO write program (using unified memory)

## Usage

### Individual Operations

#### GPU-to-GPU DMA Write
```bash
# GPU0 -> GPU2
./gpu_to_gpu_dma 0 2 [size]

# GPU1 -> GPU2
./gpu_to_gpu_dma 1 2 [size]
```

Examples:
```bash
./gpu_to_gpu_dma 0 2 1GB    # Transfer 1GB from GPU0 to GPU2
./gpu_to_gpu_dma 1 2 512MB  # Transfer 512MB from GPU1 to GPU2
```

#### GPU->CPU DMA Read
```bash
./cpu_to_gpu_dma <source_gpu_id> [size]
```

Example:
```bash
./cpu_to_gpu_dma 2 1GB  # DMA read from GPU2 to CPU (initiated from GPU)
```

#### CPU->GPU MMIO Write
```bash
./cpu_to_gpu_mmio <dest_gpu_id> [size]
```

Example:
```bash
./cpu_to_gpu_mmio 2 1GB  # MMIO write from CPU to GPU2 (using unified memory)
```

### Concurrent Execution

The `run_concurrent.sh` script automatically sets `CUDA_VISIBLE_DEVICES=2,3,4`, which maps logical GPU IDs 0,1,2 in the code to physical GPUs 2,3,4.

#### Using Shell Script
```bash
./run_concurrent.sh <dma|mmio> [size] [gpu_to_gpu_bin] [cpu_to_gpu_dma_bin] [cpu_to_gpu_mmio_bin]
```

The script supports two modes:
- **dma mode**: Uses `cpu_to_gpu_dma` - GPU->CPU DMA read initiated from GPU via `cudaMemcpy`
- **mmio mode**: Uses `cpu_to_gpu_mmio` - CPU->GPU MMIO write using unified memory

The script starts processes in this order:
1. CPU <-> GPU2 transfer (DMA or MMIO mode)
2. GPU0 -> GPU2 DMA write
3. GPU1 -> GPU2 DMA write

Examples:
```bash
# Run in DMA mode with default 1GB size
./run_concurrent.sh dma

# Run in MMIO mode with custom size
./run_concurrent.sh mmio 512MB

# Run with custom binary paths
./run_concurrent.sh dma 1GB ./gpu_to_gpu_dma ./cpu_to_gpu_dma ./cpu_to_gpu_mmio
```

**Note**: The script automatically sets `CUDA_VISIBLE_DEVICES=2,3,4`, so:
- Logical GPU 0 → Physical GPU 2
- Logical GPU 1 → Physical GPU 3
- Logical GPU 2 → Physical GPU 4

Output logs will be saved to:
- `cpu_to_gpu2_dma.log` (DMA mode) or `cpu_to_gpu2_mmio.log` (MMIO mode)
- `gpu0_to_gpu2.log`
- `gpu1_to_gpu2.log`

## Requirements

- CUDA Toolkit 13.0+ (for compute capability 12.0 support)
- NVIDIA GPU with peer-to-peer access support
- GCC/G++ compiler

## Notes

- The programs run continuously until interrupted (Ctrl+C)
- Statistics are printed every 100 iterations to stdout
- Final statistics are printed to stderr and will be displayed when Ctrl+C is pressed
- **DMA mode**: GPU->CPU DMA read is initiated from GPU side using `cudaMemcpy(..., cudaMemcpyDeviceToHost)`
- **MMIO mode**: CPU->GPU MMIO write uses unified memory (`cudaMallocManaged`). When CPU writes to unified memory that is resident on GPU, this triggers PCIe MMIO writes
- GPU-to-GPU DMA uses `cudaMemcpyPeer` for peer-to-peer transfers
- Peer-to-peer access is enabled bidirectionally between source and destination GPUs
- For best performance, ensure GPUs support NVLink or have high-bandwidth PCIe connections

## Troubleshooting

1. **"Cannot access peer GPU" error**: 
   - Ensure GPUs support peer-to-peer access
   - Check with `nvidia-smi topo -m` to verify GPU topology

2. **Compilation errors**:
   - Verify CUDA path is correct in Makefile
   - Check compute capability matches your GPUs
   - Update `-arch=sm_XX` in Makefile if needed

3. **Permission errors**:
   - Make script executable: `chmod +x run_concurrent.sh`

4. **GPU ID mapping**:
   - The script uses `CUDA_VISIBLE_DEVICES=2,3,4` to map logical GPUs 0,1,2 to physical GPUs 2,3,4
   - To use different physical GPUs, modify the `CUDA_VISIBLE_DEVICES` setting in `run_concurrent.sh`

5. **MMIO mode not working as expected**:
   - Unified memory behavior depends on CUDA driver and GPU architecture
   - The system may migrate pages between CPU and GPU automatically
   - For consistent MMIO behavior, ensure unified memory pages are resident on GPU
