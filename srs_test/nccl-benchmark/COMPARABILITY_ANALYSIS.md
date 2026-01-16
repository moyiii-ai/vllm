# All-Gather vs All-Reduce 可比性分析

## 1. 当前实现的差异

### 1.1 Kernel 调用参数

**All-Gather:**
```cpp
int threads = 256;
int blocks = static_cast<int>((count_per_rank + threads - 1) / threads);
init_buffer<<<blocks, threads, 0, stream>>>(d_buf, offset, count_per_rank, static_cast<float>(rank));
```

**All-Reduce:**
```cpp
int threads = 256;
int blocks = static_cast<int>((count_per_rank + threads - 1) / threads);
init_buffer<<<blocks, threads, 0, stream>>>(d_buf, count_per_rank, static_cast<float>(rank + 1));
```

**相同点：**
- ✅ `threads = 256` - 相同
- ✅ `blocks` 计算方式相同
- ✅ 启动配置 `<<<blocks, threads, 0, stream>>>` 相同

**不同点：**
- ❌ Kernel 签名不同（All-Gather 有 `offset` 参数）
- ❌ 初始化值不同（`rank` vs `rank + 1`）

### 1.2 Kernel 实现

**All-Gather:**
```cpp
__global__ void init_buffer(float* buf, size_t offset, size_t n, float value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    buf[offset + idx] = value;  // 有 offset
  }
}
```

**All-Reduce:**
```cpp
__global__ void init_buffer(float* buf, size_t n, float value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    buf[idx] = value;  // 没有 offset
  }
}
```

**差异：**
- All-Gather 的 kernel 多一个 `offset` 参数
- 内存访问模式略有不同（`buf[offset + idx]` vs `buf[idx]`）

### 1.3 Buffer 大小

**All-Gather:**
```cpp
size_t total_count = count_per_rank * nranks;  // 2 * count_per_rank
size_t bytes_per_iter = total_count * sizeof(float);
cudaMalloc(&d_buf, bytes_per_iter);
```

**All-Reduce:**
```cpp
size_t bytes_per_iter = count_per_rank * sizeof(float);
cudaMalloc(&d_buf, bytes_per_iter);
```

**差异：**
- All-Gather 分配的内存是 All-Reduce 的 `nranks` 倍（2 倍）
- 这是操作本身的特性，无法避免

### 1.4 初始化值

**All-Gather:**
```cpp
init_buffer<<<...>>>(d_buf, offset, count_per_rank, static_cast<float>(rank));
// Rank 0: 初始化为 0.0
// Rank 1: 初始化为 1.0
```

**All-Reduce:**
```cpp
init_buffer<<<...>>>(d_buf, count_per_rank, static_cast<float>(rank + 1));
// Rank 0: 初始化为 1.0
// Rank 1: 初始化为 2.0
```

**差异：**
- 初始化值不同（`rank` vs `rank + 1`）
- 这可能影响性能（但影响应该很小）

### 1.5 测试参数

**相同点：**
- ✅ `warmup = 5` - 相同
- ✅ `iters` - 相同（默认 100，可通过参数指定）
- ✅ 计时方式 - 都使用 CUDA events
- ✅ Stream 创建和使用 - 相同

## 2. 可比性分析

### 2.1 完全可比的方面

1. **Kernel 启动配置**
   - ✅ 相同的 threads per block (256)
   - ✅ 相同的 blocks 计算方式
   - ✅ 相同的 stream 使用

2. **测试流程**
   - ✅ 相同的 warmup 次数
   - ✅ 相同的迭代次数
   - ✅ 相同的计时方式

3. **NCCL 调用**
   - ✅ 相同的 communicator 初始化
   - ✅ 相同的 stream 使用
   - ✅ 相同的同步方式

### 2.2 部分可比的方面

1. **初始化 Kernel**
   - ⚠️ Kernel 签名不同，但功能相同
   - ⚠️ 初始化值不同，但都是常量值
   - ⚠️ 内存访问模式略有不同（offset vs 无 offset）

2. **内存分配**
   - ⚠️ Buffer 大小不同（操作特性，无法避免）
   - ⚠️ 但初始化的数据量相同（都是 `count_per_rank`）

### 2.3 不可比的方面（操作特性）

1. **NCCL 操作本身**
   - ❌ All-Gather 和 All-Reduce 是不同类型的操作
   - ❌ All-Gather 传输更多数据（输出是输入的 nranks 倍）
   - ❌ All-Reduce 需要额外计算（归约操作）

2. **内存使用**
   - ❌ All-Gather 需要更多内存（`count_per_rank × nranks`）
   - ❌ All-Reduce 需要较少内存（`count_per_rank`）

## 3. 已完成的改进 ✅

### 3.1 统一 Kernel 实现 ✅

**已实现：** 两个程序现在使用相同的 kernel 签名

```cpp
// 统一版本（两个程序都使用）
__global__ void init_buffer(float* buf, size_t offset, size_t n, float value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    buf[offset + idx] = value;
  }
}

// All-Gather 调用
init_buffer<<<blocks, threads, 0, stream>>>(d_buf, offset, count_per_rank, static_cast<float>(rank));

// All-Reduce 调用（offset = 0）
init_buffer<<<blocks, threads, 0, stream>>>(d_buf, 0, count_per_rank, static_cast<float>(rank));
```

**改进效果：**
- ✅ 相同的 kernel 实现
- ✅ 相同的编译优化
- ✅ 相同的内存访问模式

### 3.2 统一初始化值 ✅

**已实现：** 两个程序现在使用相同的初始化值（`rank`）

```cpp
// 都使用 rank
init_buffer<<<...>>>(d_buf, offset, count_per_rank, static_cast<float>(rank));
```

**改进效果：**
- ✅ 相同的初始化值
- ✅ 消除初始化差异对性能的影响

### 3.3 统一内存访问模式 ✅

**已实现：** 两个程序都使用 offset 参数（All-Reduce 的 offset = 0）

**改进效果：**
- ✅ 完全相同的 kernel 实现
- ✅ 完全相同的编译优化
- ✅ 完全相同的内存访问模式

## 4. 当前可比性总结

### ✅ 可以公平比较的方面

1. **给定相同的输入大小**（`count_per_rank`），可以比较：
   - 延迟（latency）
   - 吞吐量（throughput，基于输入数据量）
   - 性能趋势（随 buffer size 的变化）

2. **测试方法相同**：
   - 相同的 warmup 和迭代次数
   - 相同的计时方式
   - 相同的 GPU 绑定

### ⚠️ 需要注意的差异

1. **内存使用不同**：
   - All-Gather 使用更多内存
   - 可能影响缓存性能

2. **初始化细节不同**：
   - Kernel 签名不同
   - 初始化值不同
   - 但这些影响应该很小

3. **操作本质不同**：
   - All-Gather 主要是数据传输
   - All-Reduce 需要额外计算
   - 这是操作本身的特性

### ❌ 不可比的方面（操作特性）

1. **输出数据量**：
   - All-Gather: `count_per_rank × nranks`
   - All-Reduce: `count_per_rank`
   - 这是操作的定义，无法改变

2. **计算复杂度**：
   - All-Gather: 主要是传输
   - All-Reduce: 传输 + 归约计算
   - 这是操作的本质差异

## 5. 结论

**当前实现的可比性（已改进）：**

✅ **高度可比**：
- ✅ Kernel 实现完全相同（已统一）
- ✅ Kernel 启动配置相同
- ✅ 初始化值相同（已统一）
- ✅ 测试流程相同
- ✅ 计时方式相同
- ✅ 给定相同输入大小，可以公平比较性能

⚠️ **部分可比**（操作特性，无法避免）：
- ⚠️ 内存使用不同（All-Gather 需要更多内存）
- ⚠️ 但这是操作本身的特性，不是测试方法的问题

❌ **不可比**（操作特性，无法改变）：
- ❌ 输出数据量不同（All-Gather: `count_per_rank × nranks`, All-Reduce: `count_per_rank`）
- ❌ 计算复杂度不同（All-Gather: 传输，All-Reduce: 传输 + 计算）

**已完成的改进：**
1. ✅ 统一 kernel 实现（使用相同的 kernel 签名）
2. ✅ 统一初始化值（都使用 `rank`）
3. ✅ 统一内存访问模式（都使用 offset 参数）

**结论：** 当前实现**高度可比**。除了操作本身的特性差异（内存使用、输出数据量、计算复杂度），所有测试方法相关的方面都已统一。可以公平地比较两种操作的性能。
