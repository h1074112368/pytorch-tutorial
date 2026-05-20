# a3: Dispatcher 调度器

## 核心问题

PyTorch 的 Dispatcher 如何将一个操作路由到正确的后端实现？

## Dispatcher 架构

```
用户调用: torch.add(x, y)
    │
    ▼
Python 绑定 → Dispatcher::call(aten::add, ...)
    │
    ▼
构建 DispatchKeySet (从 Tensor 的 dispatch_key_set_ 收集)
    │
    ▼
按优先级遍历 DispatchKey:
    │
    ├── 1. Functionalize      → 函数化变换 (如果活跃)
    ├── 2. Python             → Python 回退 (如果有 Python 实现)
    ├── 3. CompositeExplicitAutograd → 复合自动微分
    ├── 4. Autograd           → 自动微分 (记录计算图)
    ├── 5. Sparse             → 稀疏张量
    ├── 6. Backend (CPU/CUDA/XPU/MPS/PrivateUse1) → 实际计算
    └── 7. Meta               → 形状推理 (FakeTensor)
```

## DispatchKey 优先级

Dispatcher 按固定优先级遍历 DispatchKey：

```
优先级 (高 → 低):
  Functionalize > Python > CompositeExplicitAutograd > Autograd
  > SparseCPU/SparseCUDA > CPU/CUDA/XPU/MPS > Meta > CompositeImplicitAutograd
```

**关键设计**: Autograd 在 Backend 之前，确保即使有 CPU/CUDA kernel，也会先经过 Autograd 层记录计算图。

## 内核注册

```cpp
// 注册 CPU 内核
TORCH_LIBRARY_IMPL(aten, CPU, m) {
    m.impl("add.Tensor", add_cpu_kernel);
}

// 注册 CUDA 内核
TORCH_LIBRARY_IMPL(aten, CUDA, m) {
    m.impl("add.Tensor", add_cuda_kernel);
}

// 注册 Autograd 内核
TORCH_LIBRARY_IMPL(aten, Autograd, m) {
    m.impl("add.Tensor", add_autograd_kernel);
}
```

## 自定义后端注册 (PrivateUse1)

```cpp
// 注册自定义后端 (如 NPU)
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("add.Tensor", npu_add_kernel);
    m.impl("mm.default", npu_mm_kernel);
    // ... 更多操作
}
```

## Dispatcher 在编译栈中的角色

```
Dynamo 捕获 → FX Graph → AOTAutograd → Decomposition → Inductor
                                                    │
                                                    ▼
                                          Inductor 生成代码调用
                                          Dispatcher::call(aten::op, ...)
                                                    │
                                                    ▼
                                          路由到正确的后端 kernel
```

## 学习检查点

- [ ] 理解 Dispatcher 的调度流程
- [ ] 知道 DispatchKey 的优先级
- [ ] 理解 Autograd 在 Backend 之前的原因
- [ ] 知道如何注册自定义后端 (PrivateUse1)
- [ ] 理解 Dispatcher 在编译栈中的角色

## 下一步

完成 Part A 后，进入 [Part B: Python 核心层](../../part_b_python_core/guide.md)
