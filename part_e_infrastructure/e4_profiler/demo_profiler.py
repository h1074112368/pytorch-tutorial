"""
e4: 性能分析 - 可运行示例
运行方式: python demo_profiler.py
"""

import torch
import torch.nn as nn


def demo_profiler_basics():
    print("=" * 60)
    print("1. Profiler 基础")
    print("=" * 60)

    model = nn.Linear(100, 10)
    x = torch.randn(32, 100)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            model(x)

    print("  CPU 操作 Top 5 (按 CPU 时间):")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))
    print()


def demo_profiler_schedule():
    print("=" * 60)
    print("2. Profiler 调度")
    print("=" * 60)

    print("  schedule() 参数:")
    print("    wait=1   # 等待 1 个 step")
    print("    warmup=1 # 预热 1 个 step")
    print("    active=3 # 记录 3 个 step")
    print("    repeat=1 # 重复 1 次")
    print()

    print("  使用方式:")
    print("""
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    ) as prof:
        for step, (x, y) in enumerate(dataloader):
            train_step(x, y)
            prof.step()  # 推进调度
    """)
    print()


def demo_profiler_analysis():
    print("=" * 60)
    print("3. 分析结果")
    print("=" * 60)

    print("  关键指标:")
    print("    cpu_time_total  - CPU 总时间")
    print("    cuda_time_total - CUDA 总时间")
    print("    self_cpu_time   - 自身 CPU 时间 (不含子操作)")
    print("    cpu_memory_usage - CPU 内存使用")
    print("    cuda_memory_usage - CUDA 内存使用")
    print()

    print("  常见性能瓶颈:")
    print("    1. CPU 瓶颈: 数据加载、预处理太慢")
    print("    2. GPU 利用率低: kernel 太小, launch 开销大")
    print("    3. 内存瓶颈: 峰值内存太高, 触发 OOM")
    print("    4. 通信瓶颈: 分布式训练的 all-reduce 太慢")
    print()


if __name__ == "__main__":
    demo_profiler_basics()
    demo_profiler_schedule()
    demo_profiler_analysis()
    print("e4: Profiler 学习完成!")
