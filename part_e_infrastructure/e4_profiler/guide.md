# e4: 性能分析 (Profiler)

## 核心问题

如何使用 PyTorch Profiler 分析训练性能瓶颈？

## Profiler 架构

```
torch.profiler
├── profile()          # 性能分析上下文管理器
├── schedule()         # 分析调度 (等待/预热/记录/采样)
├── ProfilerActivity  # 分析活动类型 (CPU, CUDA)
├── ProfilerAction    # 分析动作 (RECORD, WARMUP, SAMPLE, etc.)
└── tensorboard_trace_handler()  # TensorBoard 导出
```

## 使用示例

```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for i, (x, y) in enumerate(dataloader):
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        prof.step()
```

## 学习检查点

- [ ] 理解 Profiler 的使用方式
- [ ] 知道 schedule 的等待/预热/记录机制
- [ ] 能使用 TensorBoard 可视化分析结果
