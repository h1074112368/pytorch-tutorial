# d2: Prims + Refs

## 核心问题

Prims 和 Refs 如何定义算子的层次体系？它们在编译栈中扮演什么角色？

## 三层算子抽象

```
torch.add (高层 API)
    │  [refs: 参考实现, 基于 prims 构建]
    ▼
torch._refs.add
    │  [prims: 原子操作, 不可再分]
    ▼
torch._prims.add
    │  [impl_aten: 委托到 ATen C++ 内核]
    ▼
aten::add (C++ ATen 内核)
```

## Prims — 原语操作

约 100+ 个不可再分的原子操作：

| 类别 | 操作 |
|------|------|
| 逐元素一元 | abs, cos, exp, log, sqrt, sin, tanh, neg |
| 逐元素二元 | add, mul, div, pow, eq, lt, gt, ne |
| 视图操作 | as_strided, broadcast_in_dim, squeeze, transpose |
| 形状操作 | cat, reshape, rev, collapse |
| 归约操作 | sum, amax, amin, prod, var |
| 类型转换 | convert_element_type, device_put, clone |
| 张量创建 | empty_strided, iota, scalar_tensor, normal |

## Refs — 参考实现

PyTorch 高层 API 的纯 Python 实现，完全基于 prims：

```python
# torch._refs.softmax 的实现 (基于 prims)
def softmax(x, dim):
    x_max = prims.amax(x, dim, keepdim=True)
    exp_x = prims.exp(x - x_max)
    return exp_x / prims.sum(exp_x, dim, keepdim=True)
```

## TorchRefsMode

```python
# 将 torch.* 调用重定向到 refs
with TorchRefsMode():
    y = torch.add(x, y)  # 实际调用 torch._refs.add
```

## 学习检查点

- [ ] 理解 prims → refs → ATen 的三层关系
- [ ] 知道 prims 的操作类别
- [ ] 理解 TorchRefsMode 的作用
