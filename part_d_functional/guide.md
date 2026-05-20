# Part D: 函数变换层

## 概述

函数变换层提供了 JAX 风格的函数式编程能力，是 PT2 编译栈的重要基础设施。

## 学习路径

| 子模块 | 核心内容 | 关键源码 |
|--------|----------|----------|
| **d1_functorch** | vmap, grad, vjp, jvp, functionalize | `torch/_functorch/` |
| **d2_prims_refs** | PrimTorch 原语, 参考实现 | `torch/_prims/`, `torch/_refs/` |
| **d3_decomp_system** | 分解表, Core ATen, 分解注册 | `torch/_decomp/` |

## 三层算子抽象

```
torch.add (高层API)
    │
    │ [refs: 参考实现, 基于 prims 构建]
    ▼
torch._refs.add
    │
    │ [prims: 原子操作, 不可再分]
    ▼
torch._prims.add
    │
    │ [impl_aten: 委托到 ATen C++ 内核]
    ▼
aten::add (C++ ATen 内核)
```

## functorch 变换

| 变换 | 说明 | 示例 |
|------|------|------|
| `vmap` | 向量化映射 | `vmap(fn)(batch_x)` |
| `grad` | 标量梯度 | `grad(fn)(x)` |
| `vjp` | 向量-雅可比积 (反向AD) | `vjp(fn)(x)` |
| `jvp` | 雅可比-向量积 (前向AD) | `jvp(fn)(x)` |
| `jacrev` | 反向模式雅可比 | `jacrev(fn)(x)` = `vmap(vjp(fn))` |
| `jacfwd` | 前向模式雅可比 | `jacfwd(fn)(x)` |
| `hessian` | 海森矩阵 | `hessian(fn)(x)` = `jacfwd(jacrev(fn))` |
| `functionalize` | 函数化 | `functionalize(fn)(x)` |

## 在编译栈中的角色

```
Dynamo → FX Graph → AOTAutograd (使用 functorch)
                          │
                          ├── vjp: 计算反向图
                          ├── functionalize: 消除变异操作
                          └── 分解: 高层算子 → Core ATen/Prims
```
