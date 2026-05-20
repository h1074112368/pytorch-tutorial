"""
a1: c10 核心库 - 可运行示例

演示 PyTorch 底层 C++ 类型在 Python 层的映射
运行方式: python demo_c10.py
"""

import torch


def demo_tensor_impl():
    """演示 TensorImpl 在 Python 层的映射"""
    print("=" * 60)
    print("1. TensorImpl → torch.Tensor")
    print("=" * 60)

    x = torch.randn(2, 3, 4)
    print(f"  shape:    {x.shape}        # 对应 C++ sizes_")
    print(f"  stride:   {x.stride()}    # 对应 C++ strides_")
    print(f"  dtype:    {x.dtype}        # 对应 C++ scalar_type_")
    print(f"  device:   {x.device}       # 对应 C++ device_")
    print(f"  nbytes:   {x.nbytes}       # 底层存储字节数")
    print(f"  element_size: {x.element_size()}  # 单个元素字节数")
    print(f"  is_contiguous: {x.is_contiguous()}  # 是否连续")
    print()

    print("  Tensor 内存模型:")
    print("""
    TensorImpl (C++)
      ├── Storage (持有 data_ptr + nbytes)
      │     └── StorageImpl (void* data, size_t nbytes, Allocator*)
      │           └── 实际数据内存 (1D 连续字节数组)
      │
      ├── sizes_ = [2, 3, 4]    # 逻辑形状
      ├── strides_ = [12, 4, 1] # 步幅 (元素数)
      └── storage_offset_ = 0   # 偏移量

    访问元素 [i, j, k] = data_ptr[(i*12 + j*4 + k*1) * itemsize]
    """)
    print()


def demo_storage():
    """演示 Storage"""
    print("=" * 60)
    print("2. Storage → 底层存储")
    print("=" * 60)

    x = torch.randn(2, 3)
    print(f"  storage:       {x.storage()}")
    print(f"  storage_offset: {x.storage_offset()}")
    print(f"  data_ptr:      {x.data_ptr()}")
    print()

    y = x[1, :]
    print(f"  切片后:")
    print(f"  storage_offset: {y.storage_offset()}  # 偏移量变了")
    print(f"  data_ptr:      {y.data_ptr()}  # 但共享同一块存储")
    print()

    print("  Storage 的核心:")
    print("    - 多个 Tensor 可以共享同一个 Storage (视图)")
    print("    - 切片/reshape 只改变 sizes/strides/offset, 不复制数据")
    print("    - 对应 C++ 的 StorageImpl")
    print()


def demo_dispatch_key():
    """演示 DispatchKey"""
    print("=" * 60)
    print("3. DispatchKey → 操作调度")
    print("=" * 60)

    x_cpu = torch.randn(3)
    x_cuda = torch.randn(3, device="cuda") if torch.cuda.is_available() else None

    print("  DispatchKey 决定了操作被路由到哪个后端:")
    print("    CPU Tensor  → DispatchKey::CPU  → CPU kernel")
    if x_cuda is not None:
        print("    CUDA Tensor → DispatchKey::CUDA → CUDA kernel")
    print()

    print("  常见 DispatchKey:")
    keys = [
        ("CPU", "CPU 后端"),
        ("CUDA", "CUDA 后端"),
        ("XPU", "Intel XPU 后端"),
        ("MPS", "Apple Metal 后端"),
        ("Autograd", "自动微分"),
        ("Meta", "元数据/形状推理"),
        ("Functionalize", "函数化变换"),
        ("PrivateUse1", "自定义后端 (NPU 等)"),
    ]
    for key, desc in keys:
        print(f"    {key:25s} {desc}")
    print()

    print("  调度流程:")
    print("    torch.add(x, y)")
    print("    → Dispatcher::call(aten::add, x.dispatch_key_set(), ...)")
    print("    → 根据 DispatchKey 选择 kernel:")
    print("      CPU:  aten::add_cpu")
    print("      CUDA: aten::add_cuda")
    print("      Autograd: add_autograd (反向传播)")
    print()


def demo_symint():
    """演示 SymInt (动态形状)"""
    print("=" * 60)
    print("4. SymInt → 符号整数 (动态形状)")
    print("=" * 60)

    print("  SymInt 可以是具体值或符号:")
    print("    SymInt(64)     # 具体值: 64")
    print("    SymInt('s0')   # 符号: 运行时确定")
    print()

    print("  在 torch.compile 中的使用:")
    print("""
    @torch.compile
    def forward(x):
        # x.shape = [s0, 64]  其中 s0 是动态维度
        # Inductor IR 使用 SymInt 表示 s0
        return x + 1
    """)
    print()

    print("  SymInt 的运算:")
    print("    s0 * 2       # 符号乘法")
    print("    s0 + 1       # 符号加法")
    print("    s0 * s1      # 两个符号相乘")
    print()

    x = torch.randn(3, 4)
    print(f"  静态形状: {x.shape}  # 编译时已知")
    print(f"  动态形状: [s0, 4]    # s0 在编译时未知, 运行时确定")
    print()


def demo_intrusive_ptr():
    """演示 intrusive_ptr"""
    print("=" * 60)
    print("5. intrusive_ptr → 侵入式智能指针")
    print("=" * 60)

    print("  PyTorch 使用 intrusive_ptr 而非 shared_ptr:")
    print()
    print("  ┌──────────────────┬───────────────────┬──────────────────┐")
    print("  │ 特性             │ intrusive_ptr     │ shared_ptr       │")
    print("  ├──────────────────┼───────────────────┼──────────────────┤")
    print("  │ 引用计数位置     │ 对象内部 (侵入式) │ 额外分配 (非侵入) │")
    print("  │ 内存分配次数     │ 1次               │ 2次              │")
    print("  │ 性能             │ 更高              │ 较低             │")
    print("  │ 自定义析构       │ 支持              │ 有限             │")
    print("  └──────────────────┴───────────────────┴──────────────────┘")
    print()

    print("  使用 intrusive_ptr 的类:")
    print("    - TensorImpl   (张量实现)")
    print("    - StorageImpl  (存储实现)")
    print("    - GeneratorImpl (随机数生成器)")
    print("    - Python 对象槽")
    print()


def demo_data_types():
    """演示数据类型"""
    print("=" * 60)
    print("6. 数据类型 (ScalarType)")
    print("=" * 60)

    types = [
        (torch.float32, "FP32", "4字节, 最常用"),
        (torch.float16, "FP16", "2字节, GPU 推理"),
        (torch.bfloat16, "BF16", "2字节, 训练常用"),
        (torch.int64, "INT64", "8字节, 索引"),
        (torch.int32, "INT32", "4字节"),
        (torch.bool, "BOOL", "1字节"),
        (torch.uint8, "UINT8", "1字节, 量化"),
        (torch.int8, "INT8", "1字节, 量化"),
    ]

    for dtype, name, desc in types:
        print(f"  {str(dtype):20s} {name:6s} {desc}")
    print()

    print("  对应 C++ 类型:")
    print("    float32  → float (4字节)")
    print("    float16  → c10::Half (2字节)")
    print("    bfloat16 → c10::BFloat16 (2字节)")
    print("    int64    → int64_t (8字节)")
    print()


if __name__ == "__main__":
    demo_tensor_impl()
    demo_storage()
    demo_dispatch_key()
    demo_symint()
    demo_intrusive_ptr()
    demo_data_types()

    print("=" * 60)
    print("a1: c10 核心库 学习完成!")
    print("下一步: 阅读 a2_aten_tensor/guide.md 学习 ATen 张量库")
    print("=" * 60)
