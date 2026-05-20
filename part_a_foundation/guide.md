# Part A: C++ 底层基础

## 概述

PyTorch 的 C++ 层是整个框架的基石，采用分层架构：

```
c10 (最底层，核心类型和工具)
  │
  ▼
ATen (张量操作库，依赖 c10)
  │
  ▼
torch/csrc (Python-C++ 绑定层，依赖 ATen 和 c10)
```

## 为什么从 C++ 底层开始学？

1. **理解本质**: Python 层的 `torch.Tensor` 本质上是 C++ `TensorImpl` 的薄包装
2. **调度机制**: Dispatcher 是 PyTorch 所有操作的核心路由，理解它才能理解多后端支持
3. **性能关键**: 所有性能敏感的操作都在 C++ 层实现
4. **扩展基础**: 自定义 C++ 扩展需要理解底层类型系统

## 学习路径

| 子模块 | 核心内容 | 关键源码 |
|--------|----------|----------|
| **a1_c10_core** | TensorImpl, Storage, DispatchKey, SymInt, Device | `c10/core/`, `c10/util/` |
| **a2_aten_tensor** | Tensor, TensorBase, native ops, autograd C++ | `aten/src/ATen/` |
| **a3_dispatcher** | Dispatcher, DispatchKey, kernel registration | `aten/src/ATen/core/dispatch/` |

## 核心概念速查

### c10 核心类型

| 类型 | 文件 | 说明 |
|------|------|------|
| `TensorImpl` | `c10/core/TensorImpl.h` | 张量实现的核心类 |
| `StorageImpl` | `c10/core/StorageImpl.h` | 存储实现 |
| `DispatchKey` | `c10/core/DispatchKey.h` | 调度键 (CPU, CUDA, Autograd...) |
| `SymInt` | `c10/core/SymInt.h` | 符号整数 (动态形状) |
| `Device` | `c10/core/Device.h` | 设备抽象 |
| `ScalarType` | `c10/core/ScalarType.h` | 数据类型枚举 |
| `intrusive_ptr` | `c10/util/intrusive_ptr.h` | 侵入式智能指针 |
| `Half` / `BFloat16` | `c10/util/Half.h` | FP16 / BF16 类型 |

### ATen 核心类型

| 类型 | 文件 | 说明 |
|------|------|------|
| `Tensor` | `aten/src/ATen/core/Tensor.h` | 用户可见的张量类 |
| `TensorBase` | `aten/src/ATen/core/TensorBase.h` | Tensor 的基类 |
| `Dispatcher` | `aten/src/ATen/core/dispatch/Dispatcher.h` | 中央调度器 |
| `NativeFunction` | `torchgen/model.py` | 原生函数定义 |

### Dispatcher 调度键

| DispatchKey | 说明 |
|-------------|------|
| `CPU` | CPU 后端 |
| `CUDA` | CUDA 后端 |
| `XPU` | Intel XPU 后端 |
| `MPS` | Apple Metal 后端 |
| `Autograd` | 自动微分 |
| `Meta` | 元数据/形状推理 |
| `Functionalize` | 函数化变换 |
| `Python` | Python 回退 |
| `CompositeExplicitAutograd` | 复合显式自动微分 |
| `PrivateUse1` | 自定义后端 (NPU 等) |

## 与其他 Part 的关系

- Part A 是 Part B (Python 核心) 的 C++ 基础
- Dispatcher 是 Part C (编译栈) 的底层调度机制
- Part E (torchgen) 生成 c10/ATen 的 C++ 代码
