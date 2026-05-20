# a1: c10 核心库

## 核心问题

PyTorch 最底层的 C++ 库提供了哪些基础类型？Tensor 的内存模型是怎样的？

## c10 架构

```
c10/
├── core/           # 核心类型
│   ├── TensorImpl.h      # 张量实现 (最核心的类)
│   ├── StorageImpl.h     # 存储实现
│   ├── DispatchKey.h     # 调度键
│   ├── SymInt.h          # 符号整数 (动态形状)
│   ├── Device.h          # 设备抽象
│   ├── ScalarType.h      # 数据类型
│   ├── Layout.h          # 张量布局
│   ├── Allocator.h       # 内存分配器
│   └── impl/             # 实现细节
│       ├── SizesAndStrides.h  # 大小和步幅
│       ├── COW.h              # 写时复制
│       └── LocalDispatchKeySet.h  # 本地调度键集
├── cuda/           # CUDA 核心
│   ├── CUDAStream.h      # CUDA 流
│   └── CUDAGuard.h       # CUDA 设备守卫
├── util/           # 通用工具
│   ├── ArrayRef.h        # 数组引用
│   ├── Half.h            # FP16
│   ├── BFloat16.h        # BF16
│   ├── intrusive_ptr.h   # 侵入式智能指针
│   └── Exception.h       # 异常
└── xpu/            # Intel XPU
```

## 核心类详解

### 1. TensorImpl — 张量实现的核心

源码: `c10/core/TensorImpl.h`

```cpp
class C10_API TensorImpl : public c10::intrusive_ptr_target {
    // 核心数据成员:
    c10::Storage storage_;              // 底层存储
    c10::SymInt sizes_[kMaxDims];       // 各维度大小 (支持符号)
    c10::SymInt strides_[kMaxDims];     // 各维度步幅 (支持符号)
    int64_t storage_offset_;            // 存储偏移
    DispatchKeySet dispatch_key_set_;   // 调度键集合
    ScalarType scalar_type_;            // 数据类型
    bool is_contiguous_;                // 是否连续
    // ...
};
```

**Tensor 的内存模型**:
```
TensorImpl
  ├── Storage (持有 data_ptr + nbytes)
  │     └── StorageImpl (持有 void* data, size_t nbytes, Allocator*)
  │           └── 实际数据内存 (1D 连续字节数组)
  │
  ├── sizes_ = [2, 3, 4]    # 逻辑形状
  ├── strides_ = [12, 4, 1] # 步幅 (元素数)
  └── storage_offset_ = 0   # 偏移量

  访问元素 [i, j, k] = data_ptr[(i * strides_[0] + j * strides_[1] + k * strides_[2]) * itemsize + storage_offset_ * itemsize]
```

### 2. StorageImpl — 存储实现

源码: `c10/core/StorageImpl.h`

```cpp
class C10_API StorageImpl : public c10::intrusive_ptr_target {
    c10::DataPtr data_ptr_;     // 数据指针 (带自定义 deleter)
    int64_t nbytes_;            // 字节数
    Allocator* allocator_;      // 内存分配器
    // ...
};
```

### 3. DispatchKey — 调度键

源码: `c10/core/DispatchKey.h`

调度键决定了操作被路由到哪个后端实现：

```cpp
enum class DispatchKey {
    CPU,                // CPU 后端
    CUDA,               // CUDA 后端
    XPU,                // Intel XPU
    MPS,                // Apple Metal
    Meta,               // 元数据/形状推理
    Autograd,           // 自动微分
    Functionalize,      // 函数化变换
    Python,             // Python 回退
    CompositeExplicitAutograd,  // 复合显式自动微分
    PrivateUse1,        // 自定义后端 (NPU 等)
    // ... 更多
};
```

### 4. SymInt — 符号整数 (动态形状)

源码: `c10/core/SymInt.h`

```cpp
class SymInt {
    // 可以是具体值: SymInt(64)
    // 也可以是符号: SymInt("s0")  (来自 ShapeEnv)
    // 支持算术运算: s0 * 2 + 1
    // 用于 PT2 编译栈的动态形状支持
};
```

### 5. intrusive_ptr — 侵入式智能指针

源码: `c10/util/intrusive_ptr.h`

```cpp
template <T, Destructor>
class intrusive_ptr {
    // PyTorch 的核心智能指针
    // 比 std::shared_ptr 更高效:
    //   - 引用计数存储在对象内部 (侵入式)
    //   - 避免额外的内存分配
    //   - 支持自定义析构函数
    // Tensor, Storage, Generator 等都使用 intrusive_ptr 管理
};
```

## 学习检查点

- [ ] 理解 TensorImpl 的内存模型 (sizes, strides, storage_offset)
- [ ] 知道 Storage 和 StorageImpl 的关系
- [ ] 理解 DispatchKey 的作用和常见键值
- [ ] 知道 SymInt 如何支持动态形状
- [ ] 理解 intrusive_ptr 的设计动机

## 下一步

完成本模块后，进入 [a2: ATen 张量库](../a2_aten_tensor/guide.md)
