# c3: AOTAutograd

## 核心问题

AOTAutograd 如何分离前向/反向图？functionalize 如何消除变异操作？

## AOTAutograd 架构

```
FX Graph (Dynamo 输出)
    │
    ▼
AOTAutograd
    ├── 1. functionalize: 消除变异操作 (in-place → out-of-place)
    ├── 2. vjp: 计算反向图 (使用 functorch)
    ├── 3. 分解: 高层算子 → Core ATen
    │
    ▼
前向图 + 反向图 (两个 FX GraphModule)
    │
    ▼
Inductor 分别编译前向和反向图
```

## 关键源码

| 文件 | 说明 |
|------|------|
| `torch/_functorch/aot_autograd.py` | AOTAutograd 主逻辑 |
| `torch/_functorch/functionalize.py` | 函数化变换 |

## 核心概念

### functionalize

将变异操作转换为非变异操作：

```python
# 原始 (变异)
def forward(self, x):
    x.add_(1)  # in-place
    return x

# 函数化后 (非变异)
def forward(self, x):
    x = x.add(1)  # out-of-place
    return x
```

### vjp (向量-雅可比积)

```python
# 前向图
def forward(x):
    y = x * 2
    z = y.sum()
    return z

# 反向图 (由 vjp 自动生成)
def backward(grad_z):
    grad_y = grad_z.expand(y_shape)  # sum 的反向
    grad_x = grad_y * 2              # mul 的反向
    return grad_x
```

## 学习检查点

- [ ] 理解 AOTAutograd 的三步流程
- [ ] 知道 functionalize 的作用
- [ ] 理解 vjp 如何生成反向图
- [ ] 知道前向/反向图分别编译的好处
