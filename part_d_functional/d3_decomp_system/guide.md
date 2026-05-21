# d3: 分解系统

## 核心问题

分解系统的注册机制如何工作？pre-autograd 和 post-autograd 分解有什么区别？

## 分解表层次

```python
global_decomposition_table = defaultdict(dict)

# 三种分解表:
decomposition_table["post_autograd"]  # autograd 之后的分解 (Inductor 使用)
decomposition_table["pre_autograd"]   # autograd 之前的分解 (Dynamo 使用)
decomposition_table["meta"]           # 元数据分解 (形状推理)
```

## 注册机制

```python
@register_decomposition(torch.ops.aten.clamp_min)
def clamp_min(x, min):
    return torch.clamp(x, min=min)

# 等价于:
decomposition_table["post_autograd"][aten.clamp_min] = clamp_min
```

## Core ATen 分解

`core_aten_decompositions()` 返回约 200+ 个分解规则，将 ATen 算子分解到 Core ATen 算子集。

## 学习检查点

- [ ] 理解三种分解表的区别
- [ ] 知道 register_decomposition 的使用方式
- [ ] 理解 Core ATen 分解的作用
