# e2: CUDA 系统

## 核心问题

PyTorch 的 CUDA 系统如何管理内存、流和通信？

## CUDA 系统架构

```
torch.cuda
├── 设备管理
│   ├── device_count()        # GPU 数量
│   ├── set_device(device)    # 设置当前设备
│   ├── current_device()      # 当前设备
│   └── is_available()        # CUDA 是否可用
│
├── 内存管理
│   ├── memory_allocated()    # 已分配内存
│   ├── max_memory_allocated()# 峰值内存
│   ├── empty_cache()         # 清空缓存
│   └── memory_stats()        # 内存统计
│
├── 流管理
│   ├── Stream                # CUDA 流
│   ├── Event                 # CUDA 事件
│   └── StreamContext         # 流上下文
│
├── NCCL 通信
│   ├── init_process_group()  # 初始化进程组
│   ├── all_reduce()          # 全归约
│   └── all_gather()          # 全收集
│
├── CUDA Graphs
│   └── CUDAGraph             # CUDA Graph
│
└── AMP (自动混合精度)
    ├── autocast()            # 自动类型转换
    └── GradScaler()          # 梯度缩放
```

## C++ 层

| 目录 | 说明 |
|------|------|
| `c10/cuda/` | CUDA 核心 (Stream, Guard, Allocator) |
| `aten/src/ATen/cuda/` | ATen CUDA 基础设施 |
| `torch/csrc/cuda/` | Python-C++ 绑定 |

## 学习检查点

- [ ] 理解 CUDA 内存管理 (caching allocator)
- [ ] 知道 CUDA Stream 的用途
- [ ] 理解 CUDA Graph 的原理
- [ ] 知道 AMP 的工作机制
