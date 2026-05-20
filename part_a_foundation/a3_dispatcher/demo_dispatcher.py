"""
a3: Dispatcher 调度器 - 可运行示例
运行方式: python demo_dispatcher.py
"""

import torch


def demo_dispatch_flow():
    """演示调度流程"""
    print("=" * 60)
    print("1. Dispatcher 调度流程")
    print("=" * 60)

    print("  调用 torch.add(x, y) 的调度路径:")
    print("""
    Python: torch.add(x, y)
        │
        ▼
    Dispatcher::call(aten::add, dispatch_key_set, x, y)
        │
        ├── 1. Functionalize  → 函数化变换 (如果活跃)
        ├── 2. Python         → Python 回退
        ├── 3. Autograd       → 自动微分 (记录计算图)
        ├── 4. Backend        → CPU/CUDA/XPU/MPS (实际计算)
        └── 5. Meta           → 形状推理 (FakeTensor)
    """)
    print()


def demo_dispatch_key_priority():
    """演示 DispatchKey 优先级"""
    print("=" * 60)
    print("2. DispatchKey 优先级")
    print("=" * 60)

    keys = [
        ("Functionalize", "函数化变换", "最高"),
        ("Python", "Python 回退", ""),
        ("CompositeExplicitAutograd", "复合显式自动微分", ""),
        ("Autograd", "自动微分", "关键: 在 Backend 之前"),
        ("SparseCPU/SparseCUDA", "稀疏张量", ""),
        ("CPU/CUDA/XPU/MPS", "后端实现", "实际计算"),
        ("PrivateUse1", "自定义后端 (NPU)", "与 CPU/CUDA 同级"),
        ("Meta", "形状推理", ""),
        ("CompositeImplicitAutograd", "复合隐式自动微分", "最低"),
    ]

    for key, desc, note in keys:
        note_str = f"  ← {note}" if note else ""
        print(f"  {key:35s} {desc:20s}{note_str}")
    print()

    print("  关键设计: Autograd 在 Backend 之前!")
    print("  → 确保即使有 CPU/CUDA kernel, 也会先经过 Autograd 层记录计算图")
    print()


def demo_backend_dispatch():
    """演示后端调度"""
    print("=" * 60)
    print("3. 后端调度 (CPU vs CUDA)")
    print("=" * 60)

    x_cpu = torch.randn(3)
    y_cpu = torch.randn(3)
    z_cpu = x_cpu + y_cpu
    print(f"  CPU:  x.device={x_cpu.device}, z.device={z_cpu.device}")
    print(f"        调度到: aten::add_cpu_kernel")
    print()

    if torch.cuda.is_available():
        x_cuda = torch.randn(3, device="cuda")
        y_cuda = torch.randn(3, device="cuda")
        z_cuda = x_cuda + y_cuda
        print(f"  CUDA: x.device={x_cuda.device}, z.device={z_cuda.device}")
        print(f"        调度到: aten::add_cuda_kernel")
    else:
        print("  CUDA: (不可用)")
    print()


def demo_autograd_dispatch():
    """演示 Autograd 调度"""
    print("=" * 60)
    print("4. Autograd 调度 (训练 vs 推理)")
    print("=" * 60)

    x = torch.randn(3, requires_grad=True)
    y = x * 2
    print(f"  训练模式 (requires_grad=True):")
    print(f"    y.grad_fn = {y.grad_fn}")
    print(f"    → Autograd key 活跃, 记录计算图")
    print()

    with torch.no_grad():
        x2 = torch.randn(3, requires_grad=True)
        y2 = x2 * 2
        print(f"  推理模式 (no_grad):")
        print(f"    y2.grad_fn = {y2.grad_fn}")
        print(f"    → Autograd key 跳过, 直接调用 Backend kernel")
    print()


def demo_meta_dispatch():
    """演示 Meta 调度 (形状推理)"""
    print("=" * 60)
    print("5. Meta 调度 (形状推理 / FakeTensor)")
    print("=" * 60)

    print("  Meta dispatch 用于形状推理, 不执行实际计算:")
    print("""
    with torch._subclasses.fake_tensor.FakeTensorMode():
        x = torch.randn(2, 3)  # FakeTensor, 不分配内存
        y = x + 1              # 只推理形状, 不计算
        # y.shape = (2, 3), y.dtype = float32
    """)
    print()

    print("  Meta dispatch 在编译栈中的角色:")
    print("    Dynamo: 使用 FakeTensor 推理形状, 不执行计算")
    print("    AOTAutograd: 使用 Meta dispatch 分离前向/反向图")
    print("    Inductor: 使用 Meta dispatch 进行形状推理")
    print()


def demo_custom_backend():
    """演示自定义后端注册"""
    print("=" * 60)
    print("6. 自定义后端注册 (PrivateUse1)")
    print("=" * 60)

    print("  C++ 注册:")
    print("""
    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("add.Tensor", npu_add_kernel);
        m.impl("mm.default", npu_mm_kernel);
    }
    """)

    print("  Python 注册 (torch.library):")
    print("""
    import torch.library

    my_lib = torch.library.Library("aten", "IMPL", "PrivateUse1")
    my_lib.impl("add.Tensor", npu_add)
    my_lib.impl("mm.default", npu_mm)
    """)
    print()

    print("  NPU 使用 PrivateUse1 注册自定义后端:")
    print("    torch_npu 将 'npu' 映射到 'PrivateUse1' DispatchKey")
    print("    所有 aten 操作可以被路由到 NPU kernel")
    print()


if __name__ == "__main__":
    demo_dispatch_flow()
    demo_dispatch_key_priority()
    demo_backend_dispatch()
    demo_autograd_dispatch()
    demo_meta_dispatch()
    demo_custom_backend()

    print("=" * 60)
    print("a3: Dispatcher 调度器 学习完成!")
    print("Part A 全部完成! 下一步: 阅读 part_b_python_core/guide.md")
    print("=" * 60)
