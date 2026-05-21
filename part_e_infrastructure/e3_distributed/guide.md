# e3: 分布式训练

## 核心问题

PyTorch 如何实现分布式训练？DDP 和 FSDP 有什么区别？

## 分布式架构

```
torch.distributed
├── ProcessGroup (进程组通信)
│   ├── ProcessGroupNCCL (GPU 通信)
│   ├── ProcessGroupGloo (CPU 通信)
│   └── ProcessGroupMPI
│
├── DDP (DistributedDataParallel)
│   ├── 每个进程持有完整模型副本
│   ├── 反向传播时 all-reduce 梯度
│   └── 通信与计算重叠 (bucket 机制)
│
├── FSDP (FullyShardedDataParallel)
│   ├── 将参数分片到所有 GPU
│   ├── 前向/反向时 all-gather 需要的参数
│   └── 支持更大的模型 (参数不全在内存中)
│
└── RPC (远程过程调用)
    ├── 分布式模型并行
    └── 异步执行
```

## DDP vs FSDP

| 特性 | DDP | FSDP |
|------|-----|------|
| 模型存储 | 每个进程完整副本 | 参数分片 |
| 内存占用 | O(模型大小) | O(模型大小/进程数) |
| 通信 | all-reduce 梯度 | all-gather 参数 |
| 适用场景 | 小模型 | 大模型 |

## 学习检查点

- [ ] 理解 DDP 的梯度同步机制
- [ ] 知道 FSDP 的参数分片原理
- [ ] 理解 ProcessGroup 的通信原语
