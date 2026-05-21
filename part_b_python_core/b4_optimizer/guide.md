# b4: 优化器

## 核心问题

优化器如何使用梯度更新参数？学习率调度器如何工作？

## 关键源码

| 文件 | 说明 |
|------|------|
| `torch/optim/optimizer.py` | Optimizer 基类 |
| `torch/optim/sgd.py` | SGD |
| `torch/optim/adam.py` | Adam |
| `torch/optim/adamw.py` | AdamW |
| `torch/optim/lr_scheduler.py` | 学习率调度器 |

## Optimizer 核心机制

### 1. 基类

```python
class Optimizer:
    param_groups: list[dict]  # 参数组 (每组可有不同 lr)
    state: dict               # 优化器状态 (如 Adam 的 m, v)

    def zero_grad(self): ...  # 清零梯度
    def step(self): ...       # 更新参数
    def add_param_group(self): ...  # 添加参数组
```

### 2. 常用优化器

| 优化器 | 更新规则 | 适用场景 |
|--------|----------|----------|
| SGD | w -= lr * grad | 简单, 配合 momentum |
| SGD+momentum | v = μ*v + grad; w -= lr*v | CV 训练常用 |
| Adam | m=β1*m+g; v=β2*v+g²; w-=lr*m/(√v+ε) | 通用, NLP 常用 |
| AdamW | Adam + 解耦权重衰减 | Transformer 训练 |

### 3. 学习率调度器

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
for epoch in range(100):
    train(...)
    scheduler.step()  # 更新学习率
```

## 学习检查点

- [ ] 理解 Optimizer 的 param_groups 和 state
- [ ] 知道 SGD+momentum vs Adam vs AdamW 的区别
- [ ] 理解学习率调度器的使用方式
