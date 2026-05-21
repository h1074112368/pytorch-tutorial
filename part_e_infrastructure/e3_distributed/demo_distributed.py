"""
e3: 分布式训练 - 可运行示例
运行方式: python demo_distributed.py
"""

import torch


def demo_distributed_overview():
    print("=" * 60)
    print("1. 分布式训练概述")
    print("=" * 60)

    print("  PyTorch 分布式训练方案:")
    print("    1. DDP: 每个进程持有完整模型, all-reduce 梯度")
    print("    2. FSDP: 参数分片到所有 GPU, all-gather 参数")
    print("    3. RPC: 远程过程调用, 分布式模型并行")
    print()


def demo_ddp():
    print("=" * 60)
    print("2. DDP (DistributedDataParallel)")
    print("=" * 60)

    print("  DDP 使用方式:")
    print("""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group("nccl")
    model = MyModel().cuda()
    model = DDP(model, device_ids=[local_rank])

    for data, target in dataloader:
        output = model(data)
        loss = criterion(output, target)
        loss.backward()  # DDP 自动 all-reduce 梯度
        optimizer.step()
    """)
    print()

    print("  DDP 的梯度同步:")
    print("    - 反向传播时, DDP 自动 all-reduce 梯度")
    print("    - 使用 bucket 机制: 将参数分组, 通信与计算重叠")
    print("    - 每个 bucket 的梯度就绪后立即开始通信")
    print()


def demo_fsdp():
    print("=" * 60)
    print("3. FSDP (FullyShardedDataParallel)")
    print("=" * 60)

    print("  FSDP 使用方式:")
    print("""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    model = MyModel()
    model = FSDP(model)

    for data, target in dataloader:
        output = model(data)  # FSDP 自动 all-gather 参数
        loss = criterion(output, target)
        loss.backward()       # FSDP 自动 all-gather 梯度
        optimizer.step()
    """)
    print()

    print("  DDP vs FSDP:")
    print("  ┌──────────────┬──────────────────┬──────────────────┐")
    print("  │ 特性         │ DDP              │ FSDP             │")
    print("  ├──────────────┼──────────────────┼──────────────────┤")
    print("  │ 模型存储     │ 每进程完整副本   │ 参数分片         │")
    print("  │ 内存占用     │ O(模型大小)      │ O(模型/进程数)   │")
    print("  │ 通信         │ all-reduce 梯度  │ all-gather 参数  │")
    print("  │ 适用场景     │ 小模型           │ 大模型           │")
    print("  └──────────────┴──────────────────┴──────────────────┘")
    print()


def demo_launch():
    print("=" * 60)
    print("4. 启动分布式训练")
    print("=" * 60)

    print("  使用 torchrun 启动:")
    print("    torchrun --nproc_per_node=4 train.py")
    print()
    print("  使用 torch.distributed.launch 启动:")
    print("    python -m torch.distributed.launch --nproc_per_node=4 train.py")
    print()


if __name__ == "__main__":
    demo_distributed_overview()
    demo_ddp()
    demo_fsdp()
    demo_launch()
    print("e3: 分布式训练 学习完成!")
