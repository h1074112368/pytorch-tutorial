# d1: functorch

## 核心问题

functorch 如何实现 JAX 风格的函数变换？vmap 和 grad 的底层机制是什么？

## functorch 变换 API

| 变换 | 说明 | 数学表示 |
|------|------|----------|
| `vmap(fn)` | 向量化映射 | `fn: (d) → (e)` → `vmap(fn): (B,d) → (B,e)` |
| `grad(fn)` | 标量梯度 | `fn: (d) → R` → `grad(fn): (d) → (d)` |
| `vjp(fn)` | 向量-雅可比积 | `fn: (d) → (e)` → `vjp(fn): (d, e) → (d)` |
| `jvp(fn)` | 雅可比-向量积 | `fn: (d) → (e)` → `jvp(fn): (d, d) → (e)` |
| `jacrev(fn)` | 反向模式雅可比 | `jacrev = vmap(vjp(fn))` |
| `jacfwd(fn)` | 前向模式雅可比 | `jacfwd(fn): (d) → (e,d)` |
| `hessian(fn)` | 海森矩阵 | `hessian = jacrev(jacrev(fn))` |
| `functionalize(fn)` | 函数化 | 消除变异操作 |

## 变换层栈机制

functorch 使用动态层栈管理嵌套变换：

```python
# 嵌套变换
vmap(grad(fn))(x)
    │
    ├── 进入 vmap: _vmap_increment_nesting()
    │     └── 进入 grad: _grad_increment_nesting()
    │           └── 执行 fn(x)
    │           └── 退出 grad: _grad_decrement_nesting()
    └── 退出 vmap: _vmap_decrement_nesting()
```

## 在编译栈中的角色

```
AOTAutograd 使用 functorch 变换:
    1. functionalize: 消除变异操作 (in-place → out-of-place)
    2. vjp: 计算反向图 (前向图 → 反向图)
    3. 分解: 高层算子 → Core ATen
```

## 学习检查点

- [ ] 理解 vmap 的向量化映射机制
- [ ] 知道 grad 和 vjp 的区别
- [ ] 理解 jacrev = vmap(vjp) 的组合
- [ ] 知道变换层栈如何管理嵌套变换
- [ ] 理解 functionalize 在编译栈中的作用
