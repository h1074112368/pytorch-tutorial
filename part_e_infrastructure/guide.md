# Part E: 基础设施

## 概述

基础设施层提供了代码生成、设备管理、分布式训练和性能分析等支撑功能。

## 学习路径

| 子模块 | 核心内容 | 关键源码 |
|--------|----------|----------|
| **e1_torchgen** | 代码生成, native_functions.yaml | `torchgen/` |
| **e2_cuda_system** | CUDA 内存/流/Graph/NCCL | `torch/cuda/`, `c10/cuda/` |
| **e3_distributed** | DDP, FSDP, RPC | `torch/distributed/` |
| **e4_profiler** | Kineto, TensorBoard | `torch/profiler/` |

## torchgen — 代码生成

torchgen 从声明式定义自动生成 C++/Python 代码：

```
native_functions.yaml (操作声明)
    │
    ▼ torchgen
    ├── ATen C++ 内核注册代码
    ├── Python 绑定代码
    ├── Autograd 导数代码
    ├── Dispatcher 注册代码
    ├── Meta 内核代码
    ├── Functionalization 代码
    └── Lazy IR 代码
```

## CUDA 系统

```
torch.cuda
    ├── 设备管理: device_count, set_device, current_device
    ├── 内存管理: memory_allocated, empty_cache, max_memory_allocated
    ├── 流管理: Stream, Event, StreamContext
    ├── NCCL: 集合通信 (all_reduce, all_gather)
    ├── CUDA Graphs: CUDAGraph
    ├── AMP: 自动混合精度 (autocast, GradScaler)
    └── 随机数: manual_seed, manual_seed_all
```

## 分布式训练

```
torch.distributed
    ├── ProcessGroup: 进程组通信
    │     ├── ProcessGroupNCCL (GPU)
    │     ├── ProcessGroupGloo (CPU)
    │     └── ProcessGroupMPI
    ├── DDP: DistributedDataParallel
    ├── FSDP: FullyShardedDataParallel
    │     ├── 全分片: 将参数分片到所有 GPU
    │     ├── 混合分片: 部分参数分片
    │     └── 包装策略: auto_wrap, size_based_wrap
    └── RPC: 远程过程调用
```
