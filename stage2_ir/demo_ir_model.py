"""
阶段2: IR 中间表示 - 可运行示例

演示 Inductor IR 的分层 Box 模型、视图变换、融合机制
运行方式: python demo_ir_model.py
"""

import torch
import torch.fx
import sympy


def demo_ir_box_model():
    """演示 IR 分层 Box 模型"""
    print("=" * 60)
    print("1. IR 分层 Box 模型")
    print("=" * 60)

    print("""
  TensorBox (顶层, 对应 torch.Tensor)
    │  职责: 处理视图关系 (reshape, expand, permute...)
    │  持有: data -> StorageBox
    │
    ├── View 层 (零拷贝视图变换)
    │     ├── ExpandView   (广播扩展)
    │     ├── PermuteView  (维度重排)
    │     ├── SqueezeView  (压缩维度)
    │     ├── SliceView    (切片)
    │     └── DtypeView    (类型转换)
    │
    └── StorageBox (存储层)
       │  职责: 处理布局优化 (stride order)
       │  持有: data -> Buffer
       │
       └── Buffer (1D 内存分配)
             ├── InputBuffer       (图输入)
             ├── ComputedBuffer    (计算产生)
             ├── ConstantBuffer    (常量)
             └── TemplateBuffer    (模板内核)
    """)
    print()


def demo_inner_fn():
    """演示 inner_fn 闭包机制"""
    print("=" * 60)
    print("2. inner_fn 闭包机制 (延迟求值)")
    print("=" * 60)

    print("  未融合: 两个独立的 Pointwise")
    print("""
    # add: z = x + y
    add_ir = Pointwise(
        inner_fn=lambda idx: ops.add(x.load(idx), y.load(idx)),
        ranges=[M, N],
    )

    # relu: w = relu(z)
    relu_ir = Pointwise(
        inner_fn=lambda idx: ops.relu(z.load(idx)),
        ranges=[M, N],
    )
    """)

    print("  融合后: 一个 Pointwise")
    print("""
    # fused: w = relu(x + y)
    fused_ir = Pointwise(
        inner_fn=lambda idx: ops.relu(ops.add(x.load(idx), y.load(idx))),
        ranges=[M, N],
    )
    """)

    print("  inner_fn 的设计精髓:")
    print("    - 闭包捕获计算逻辑，延迟到代码生成时才展开")
    print("    - 多个 Pointwise 的 inner_fn 可以嵌套组合 → 融合")
    print("    - 避免中间结果的内存分配和读写")
    print()


def demo_view_nodes():
    """演示视图节点"""
    print("=" * 60)
    print("3. 视图节点 (零拷贝)")
    print("=" * 60)

    x = torch.randn(2, 3, 4)
    print(f"  原始张量: shape={x.shape}, stride={x.stride()}")

    # ExpandView
    y = x.expand(5, 3, 4)
    print(f"  expand(5,3,4): shape={y.shape}, stride={y.stride()}")
    print(f"    → stride[0]=0, 广播维度步长为零, 不复制数据")

    # PermuteView
    z = x.permute(2, 1, 0)
    print(f"  permute(2,1,0): shape={z.shape}, stride={z.stride()}")
    print(f"    → 只交换 stride, 不复制数据")

    # SliceView
    w = x[:, 1:3, :]
    print(f"  [:,1:3,:]: shape={w.shape}, stride={w.stride()}")
    print(f"    → 只调整 offset 和 size, 不复制数据")

    # View (reshape)
    v = x.view(6, 4)
    print(f"  view(6,4): shape={v.shape}, stride={v.stride()}")
    print(f"    → 重新计算 stride, 不复制数据")

    print()
    print("  IR 中的视图节点对应关系:")
    print("    x.expand(...)    → ExpandView")
    print("    x.permute(...)   → PermuteView")
    print("    x[:, 1:3, :]    → SliceView")
    print("    x.view(...)      → View")
    print("    x.squeeze(...)   → SqueezeView")
    print("    x.to(torch.bf16) → DtypeView")
    print()


def demo_compute_nodes():
    """演示计算节点"""
    print("=" * 60)
    print("4. 计算节点类型")
    print("=" * 60)

    print("  Pointwise (逐元素操作):")
    print("    add, sub, mul, relu, sigmoid, tanh, exp, log...")
    print("    特征: 输出每个元素独立计算, 无数据依赖")
    print()

    print("  Reduction (归约操作):")
    print("    sum, max, min, mean, argmax, argmin...")
    print("    特征: 两层循环 - 外层遍历非归约维度, 内层遍历归约维度")
    print()

    print("  Scatter (散射写入):")
    print("    scatter, index_put, scatter_reduce...")
    print("    特征: 继承 Pointwise, 但写入模式不同")
    print()

    print("  Scan (前缀扫描):")
    print("    cumsum, cumprod...")
    print("    特征: 输出依赖前序元素的计算结果")
    print()

    print("  外部内核:")
    print("    ExternKernel - 无法融合的 ATen 操作 (如 conv, bmm)")
    print("    FallbackKernel - 回退到 eager 模式")
    print()


def demo_layout():
    """演示布局节点"""
    print("=" * 60)
    print("5. 布局节点 (Layout)")
    print("=" * 60)

    x = torch.randn(32, 64)
    print(f"  连续布局 (contiguous):")
    print(f"    shape={x.shape}, stride={x.stride()}")

    x_cl = x.to(memory_format=torch.channels_last)
    print(f"  channels-last 布局:")
    print(f"    shape={x_cl.shape}, stride={x_cl.stride()}")

    print()
    print("  IR 布局类型:")
    print("    FixedLayout     - stride 不可变 (输入/输出)")
    print("    FlexibleLayout  - stride 可优化 (中间计算)")
    print()
    print("  FlexibleLayout 的优化:")
    print("    Scheduler 可以根据后续操作选择最优 stride order")
    print("    例如: 如果后续是 permute(1,0), 可以直接生成转置布局")
    print()


def demo_ir_construction_example():
    """演示 z = relu(x + y) 的 IR 构建过程"""
    print("=" * 60)
    print("6. IR 构建示例: z = relu(x + y)")
    print("=" * 60)

    print("""
  Step 1: 创建输入 TensorBox
    x = TensorBox(StorageBox(InputBuffer("x", FixedLayout(...))))
    y = TensorBox(StorageBox(InputBuffer("y", FixedLayout(...))))

  Step 2: Lowering aten.add → Pointwise IR
    add_ir = Pointwise(
        device="cpu",
        dtype=torch.float32,
        inner_fn=lambda idx: ops.add(x.load(idx), y.load(idx)),
        ranges=[32, 64],
    )
    add_buffer = ComputedBuffer("add_0", FixedLayout(...), add_ir)
    add_tensorbox = TensorBox(StorageBox(add_buffer))

  Step 3: Lowering aten.relu → Pointwise IR (融合后)
    relu_ir = Pointwise(
        device="cpu",
        dtype=torch.float32,
        inner_fn=lambda idx: ops.relu(ops.add(x.load(idx), y.load(idx))),
        ranges=[32, 64],
    )
    relu_buffer = ComputedBuffer("relu_0", FixedLayout(...), relu_ir)
    z = TensorBox(StorageBox(relu_buffer))
    """)

    print("  关键观察:")
    print("    - 融合后 relu 的 inner_fn 内嵌了 add 的计算逻辑")
    print("    - 中间结果 add_0 不需要实际分配内存")
    print("    - 最终代码生成时, 只生成一个 Triton kernel")
    print()


def demo_ir_to_scheduler():
    """演示 IR 到 Scheduler 的桥梁"""
    print("=" * 60)
    print("7. IR → Scheduler 桥梁")
    print("=" * 60)

    print("""
  每个 IR Buffer 都有 make_scheduler() 方法:

    ComputedBuffer.make_scheduler()
        → SchedulerNode (可参与融合和调度)

    TemplateBuffer.make_scheduler()
        → ExternKernelSchedulerNode (不参与通用融合)

    FallbackKernel.make_scheduler()
        → NopKernelSchedulerNode (无操作)
    """)

    print("  SchedulerNode 包含:")
    print("    - node: ir.Buffer (对应的 IR 节点)")
    print("    - users: list[NodeUser] (使用者列表)")
    print("    - dependencies: 读/写依赖关系")
    print()


if __name__ == "__main__":
    demo_ir_box_model()
    demo_inner_fn()
    demo_view_nodes()
    demo_compute_nodes()
    demo_layout()
    demo_ir_construction_example()
    demo_ir_to_scheduler()

    print("=" * 60)
    print("阶段2 学习完成!")
    print("下一步: 阅读 stage3_lowering/guide.md 学习 Lowering")
    print("=" * 60)
