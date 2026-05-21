# c4: 算子分解

## 核心问题

算子分解如何将高层算子转换为底层算子？Core ATen 算子集是什么？

## 关键源码

| 文件 | 说明 |
|------|------|
| `torch/_decomp/__init__.py` | 分解表和注册机制 |
| `torch/_decomp/decompositions.py` | 具体分解实现 |

## 分解层次

```
Post-autograd 分解 (AOTAutograd 之后)
    │  将 ATen 算子分解到 Core ATen 算子集
    │  约 200+ 个算子需要分解
    │
    ▼
Core ATen 算子集
    │  约 170 个基本算子
    │  Inductor 只需处理这些基本算子
    │
    ▼
Inductor Lowering
    │  Core ATen → Inductor IR
```

## 分解示例

```python
@register_decomposition(aten.clamp_min)
def clamp_min(x, min):
    return torch.clamp(x, min=min)

@register_decomposition(aten.addmm)
def addmm(bias, x, weight):
    mm = torch.mm(x, weight)
    return torch.add(bias, mm)
```

## 学习检查点

- [ ] 理解分解的作用和动机
- [ ] 知道 Core ATen 算子集的概念
- [ ] 理解 post-autograd 分解的时机
