"""
a2: ATen 张量库 - 可运行示例
运行方式: python demo_aten.py
"""

import torch


def demo_tensor_class_hierarchy():
    """演示 Tensor 类层次"""
    print("=" * 60)
    print("1. Tensor 类层次 (C++ → Python)")
    print("=" * 60)

    x = torch.randn(2, 3)
    print(f"  type(x) = {type(x)}")
    print()

    print("  C++ 类层次:")
    print("""
    intrusive_ptr_target          # 引用计数基类
      └── c10::TensorImpl        # 核心张量实现
            └── ATen::TensorImpl  # ATen 扩展

    TensorBase                    # 持有 intrusive_ptr<TensorImpl>
      └── Tensor                  # 用户可见的张量类 (添加操作方法)

    Python 层:
    torch.Tensor                  # Python 包装, 持有 C++ Tensor
    """)
    print()


def demo_aten_operations():
    """演示 ATen 操作"""
    print("=" * 60)
    print("2. ATen 操作 (aten::*)")
    print("=" * 60)

    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    ops = [
        ("aten::add", lambda: x + y),
        ("aten::mul", lambda: x * y),
        ("aten::matmul", lambda: x @ y.T),
        ("aten::relu", lambda: torch.relu(x)),
        ("aten::sum", lambda: x.sum()),
        ("aten::exp", lambda: x.exp()),
    ]

    for name, op in ops:
        result = op()
        print(f"  {name:20s} → shape={result.shape}")

    print()
    print("  所有 ATen 操作都在 native_functions.yaml 中声明")
    print("  位置: aten/src/ATen/native/native_functions.yaml")
    print()


def demo_aten_dispatch():
    """演示 ATen 操作的调度"""
    print("=" * 60)
    print("3. ATen 操作调度流程")
    print("=" * 60)

    print("  调用 torch.add(x, y) 的完整路径:")
    print("""
    Python: torch.add(x, y)
        │
        ▼
    Python 绑定: torch/csrc/aten/...
        │
        ▼
    Dispatcher::call(aten::add, dispatch_key_set, x, y)
        │
        ├── Autograd → 记录计算图 (训练模式)
        ├── CPU      → add_cpu_kernel (x 在 CPU 上)
        ├── CUDA     → add_cuda_kernel (x 在 CUDA 上)
        └── Meta     → add_meta_kernel (形状推理)
    """)
    print()


def demo_native_implementations():
    """演示原生实现"""
    print("=" * 60)
    print("4. 原生操作实现 (native/)")
    print("=" * 60)

    print("  CPU 实现: aten/src/ATen/native/")
    print("    BinaryOps.cpp    # add, sub, mul, div")
    print("    UnaryOps.cpp     # abs, exp, log, sin")
    print("    ReduceOps.cpp    # sum, max, mean, argmax")
    print("    Convolution.cpp  # conv1d, conv2d, conv3d")
    print("    Linear.cpp       # mm, addmm, bmm")
    print("    Normalization.cpp # batch_norm, layer_norm")
    print()

    print("  CPU 向量化: aten/src/ATen/native/cpu/")
    print("    使用 SIMD 指令 (SSE, AVX, AVX2, AVX512)")
    print("    通过 VEC macro 实现统一向量化代码")
    print()

    print("  CUDA 实现: aten/src/ATen/native/cuda/")
    print("    使用 CUDA kernel 实现")
    print("    部分操作使用 cuBLAS, cuDNN")
    print()


def demo_tensor_options():
    """演示 TensorOptions"""
    print("=" * 60)
    print("5. TensorOptions (张量选项)")
    print("=" * 60)

    x = torch.randn(2, 3, dtype=torch.float32, device="cpu")
    print(f"  dtype:   {x.dtype}     # C++: ScalarType")
    print(f"  device:  {x.device}    # C++: Device")
    print(f"  layout:  {x.layout}    # C++: Layout (strided/sparse)")
    print()

    print("  C++ TensorOptions:")
    print("    struct TensorOptions {")
    print("        ScalarType dtype_;")
    print("        Device device_;")
    print("        Layout layout_;")
    print("        bool requires_grad_;")
    print("    };")
    print()


if __name__ == "__main__":
    demo_tensor_class_hierarchy()
    demo_aten_operations()
    demo_aten_dispatch()
    demo_native_implementations()
    demo_tensor_options()

    print("=" * 60)
    print("a2: ATen 张量库 学习完成!")
    print("下一步: 阅读 a3_dispatcher/guide.md 学习 Dispatcher")
    print("=" * 60)
