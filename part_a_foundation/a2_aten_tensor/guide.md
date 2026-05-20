# a2: ATen 张量库

## 核心问题

ATen 如何实现所有张量操作？native functions 的组织方式是什么？

## ATen 架构

```
aten/src/ATen/
├── core/                    # 核心抽象
│   ├── Tensor.h             # 用户可见的 Tensor 类
│   ├── TensorBase.h         # Tensor 基类 (持有 intrusive_ptr<TensorImpl>)
│   ├── TensorImpl.h         # Tensor 实现 (继承 c10::TensorImpl)
│   ├── Scalar.h             # 标量值
│   ├── ScalarType.h         # 标量类型
│   ├── TensorOptions.h      # 张量选项 (dtype, device, layout)
│   ├── Generator.h          # 随机数生成器
│   ├── ivalue.h             # IValue (JIT 动态类型)
│   ├── function_schema.h    # 函数模式
│   └── dispatch/            # 调度器
│       ├── Dispatcher.h     # 中央调度器
│       ├── DispatchHandler.h
│       └── KernelFunction.h # 内核函数
│
├── native/                  # 原生操作实现
│   ├── BinaryOps.cpp        # 二元操作 (add, sub, mul, div...)
│   ├── UnaryOps.cpp         # 一元操作 (abs, exp, log, sin...)
│   ├── ReduceOps.cpp        # 归约操作 (sum, max, mean...)
│   ├── Convolution.cpp      # 卷积操作
│   ├── Linear.cpp           # 线性操作
│   ├── Loss.cpp             # 损失函数
│   ├── Normalization.cpp    # 归一化 (BatchNorm, LayerNorm)
│   ├── TensorFactories.cpp  # 张量工厂 (zeros, ones, randn...)
│   ├── cpu/                 # CPU 特定内核 (向量化)
│   ├── cuda/                # CUDA 特定内核
│   ├── mkldnn/              # oneDNN 后端
│   ├── cudnn/               # cuDNN 后端
│   └── metal/               # Metal 后端
│
├── cuda/                    # CUDA 基础设施
│   ├── CUDAContext.h        # CUDA 上下文
│   └── CUDAGeneratorImpl.h  # CUDA 随机数生成器
│
└── functorch/               # functorch C++ 层
    ├── DynamicLayer.h       # 动态层
    └── TensorWrapper.h      # 张量包装器
```

## 核心类详解

### 1. Tensor — 用户可见的张量类

```cpp
// aten/src/ATen/core/Tensor.h
class Tensor: public TensorBase {
    // Tensor 是 TensorBase 的子类
    // TensorBase 持有 intrusive_ptr<TensorImpl>
    // TensorImpl 持有 Storage, sizes, strides, dispatch_key_set 等

    // 关键方法:
    const ScalarType scalar_type() const;  // 数据类型
    Device device() const;                  // 设备
    IntArrayRef sizes() const;              // 形状
    IntArrayRef strides() const;            // 步幅
    bool is_contiguous() const;             // 是否连续
    Tensor to(...) const;                   // 类型/设备转换
    Tensor reshape(...) const;              // 重塑
    // ... 数百个操作方法
};
```

### 2. native_functions.yaml — 操作声明

所有 ATen 操作都在 `aten/src/ATen/native/native_functions.yaml` 中声明：

```yaml
# 示例: add 操作的声明
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  dispatch:
    CPU: add_tensor_cpu
    CUDA: add_tensor_cuda
  structured: True
  structured_delegate: add.out
```

### 3. 操作实现层次

```
用户调用: torch.add(x, y)
    │
    ▼
Python 绑定 (torch/csrc/aten/...)
    │
    ▼
Dispatcher::call(aten::add, dispatch_key_set, x, y)
    │
    ├── DispatchKey::Autograd → add_autograd (记录计算图)
    ├── DispatchKey::CPU      → add_cpu_kernel (CPU 实现)
    ├── DispatchKey::CUDA     → add_cuda_kernel (CUDA 实现)
    └── DispatchKey::Meta     → add_meta_kernel (形状推理)
```

## 学习检查点

- [ ] 理解 Tensor → TensorBase → TensorImpl 的类层次
- [ ] 知道 native_functions.yaml 的作用
- [ ] 理解 CPU/CUDA 内核的组织方式
- [ ] 知道 ATen 操作如何通过 Dispatcher 路由

## 下一步

完成本模块后，进入 [a3: Dispatcher 调度器](../a3_dispatcher/guide.md)
