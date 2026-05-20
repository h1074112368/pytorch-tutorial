# b2: 自动微分 (Autograd)

## 核心问题

PyTorch 如何在前向传播时构建计算图？反向传播如何工作？

## Autograd 架构

```
前向传播:
  x (requires_grad=True)
    │
    ▼
  y = x * 2  → 创建 MulBackward0 节点, y.grad_fn = MulBackward0
    │
    ▼
  z = y.sum() → 创建 SumBackward0 节点, z.grad_fn = SumBackward0

反向传播:
  z.backward()
    │
    ▼
  Engine 按拓扑排序遍历:
    1. SumBackward0.apply(grad) → grad_y
    2. MulBackward0.apply(grad_y) → grad_x
```

## 关键源码

### `torch/autograd/function.py` — Function 基类

```python
class Function:
    """自定义 autograd 函数的基类"""

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """前向传播 - 必须实现"""

    @staticmethod
    def backward(ctx, *grad_outputs):
        """反向传播 - 必须实现"""

    @classmethod
    def apply(cls, *args, **kwargs):
        """调用入口 - 不要直接调用 forward"""
```

### `torch/autograd/grad_mode.py` — 梯度模式

```python
class no_grad:       # 禁用梯度计算 (推理模式)
class enable_grad:   # 启用梯度计算
class inference_mode: # 推理模式 (更高效, 禁用版本计数和自动微分)
```

### `torch/autograd/graph.py` — 计算图节点

```python
class Node:
    """计算图中的节点"""
    next_functions  # 指向父节点的边
    name()          # 节点名称
    metadata()      # 元数据
```

## 自定义 Function 示例

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)  # 保存用于反向传播
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors     # 取出保存的值
        grad_input = grad_output.clone()
        grad_input[x < 0] = 0     # ReLU 的梯度
        return grad_input

# 使用
my_relu = MyReLU.apply
```

## 学习检查点

- [ ] 理解 define-by-run 的计算图构建方式
- [ ] 知道 grad_fn 如何记录计算图
- [ ] 能实现自定义 Function (forward + backward)
- [ ] 理解 no_grad / enable_grad / inference_mode 的区别
- [ ] 知道 save_for_backward 的作用和内存安全
