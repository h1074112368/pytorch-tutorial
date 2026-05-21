"""
e2: CUDA 系统 - 可运行示例
运行方式: python demo_cuda.py
"""

import torch


def demo_cuda_device():
    print("=" * 60)
    print("1. CUDA 设备管理")
    print("=" * 60)

    available = torch.cuda.is_available()
    print(f"  CUDA 可用: {available}")
    if available:
        print(f"  GPU 数量: {torch.cuda.device_count()}")
        print(f"  当前设备: {torch.cuda.current_device()}")
        print(f"  设备名称: {torch.cuda.get_device_name(0)}")
    else:
        print("  (CUDA 不可用, 仅展示 API)")
    print()


def demo_cuda_memory():
    print("=" * 60)
    print("2. CUDA 内存管理")
    print("=" * 60)

    print("  关键 API:")
    print("    torch.cuda.memory_allocated()     # 已分配内存")
    print("    torch.cuda.max_memory_allocated() # 峰值内存")
    print("    torch.cuda.empty_cache()          # 清空缓存")
    print()

    print("  CUDA Caching Allocator:")
    print("    - 预分配大块内存, 减少 cudaMalloc 调用")
    print("    - 释放的内存返回缓存, 而非 cudaFree")
    print("    - 避免内存碎片化")
    print()


def demo_cuda_stream():
    print("=" * 60)
    print("3. CUDA 流 (Stream)")
    print("=" * 60)

    print("  CUDA 流允许并行执行多个操作:")
    print("""
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    with torch.cuda.stream(s1):
        y1 = model1(x1)  # 在 s1 上执行

    with torch.cuda.stream(s2):
        y2 = model2(x2)  # 在 s2 上执行 (并行)

    torch.cuda.synchronize()  # 等待所有流完成
    """)
    print()


def demo_amp():
    print("=" * 60)
    print("4. AMP (自动混合精度)")
    print("=" * 60)

    print("  AMP 自动将部分操作转为 float16:")
    print("""
    scaler = torch.amp.GradScaler('cuda')

    with torch.amp.autocast('cuda'):
        output = model(input)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    """)
    print()

    print("  AMP 的好处:")
    print("    - 减少内存占用 (float16 是 float32 的一半)")
    print("    - 加速计算 (Tensor Core)")
    print("    - 自动处理数值稳定性 (GradScaler)")
    print()


if __name__ == "__main__":
    demo_cuda_device()
    demo_cuda_memory()
    demo_cuda_stream()
    demo_amp()
    print("e2: CUDA 系统 学习完成!")
