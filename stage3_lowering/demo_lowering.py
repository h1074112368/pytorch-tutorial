"""
阶段3: Lowering - FX Graph → IR - 可运行示例

演示算子 lowering 注册、GraphLowering 工作流程、fallback 机制
运行方式: python demo_lowering.py
"""

import torch
import torch.fx


def demo_fx_graph_structure():
    """演示 FX Graph 的结构 (Lowering 的输入)"""
    print("=" * 60)
    print("1. FX Graph 结构 (GraphLowering 的输入)")
    print("=" * 60)

    class SimpleModel(torch.nn.Module):
        def forward(self, x, y):
            z = x + y
            w = torch.relu(z)
            return w

    model = SimpleModel()
    gm = torch.fx.symbolic_trace(model)

    print("  FX Graph 节点:")
    for node in gm.graph.nodes:
        print(f"    {node.op:15s} {node.name:10s} target={node.target}")

    print()
    print("  GraphLowering.run() 会遍历这些节点:")
    print("    placeholder x  → InputBuffer('x') → TensorBox")
    print("    placeholder y  → InputBuffer('y') → TensorBox")
    print("    call_function add  → lowerings[aten.add](x, y) → Pointwise")
    print("    call_function relu → lowerings[aten.relu](z) → Pointwise (融合)")
    print("    output         → 收集 graph_outputs")
    print()


def demo_lowering_registration():
    """演示 lowering 注册模式"""
    print("=" * 60)
    print("2. Lowering 注册模式")
    print("=" * 60)

    print("  全局字典: lowerings[aten_op] = lowering_fn")
    print()

    print("  逐元素操作 (Pointwise):")
    print("""
    @register_lowering(aten.add)
    def add(x, y):
        x, y = promote_types(x, y)
        return Pointwise(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=lambda index: ops.add(x.load(index), y.load(index)),
            ranges=x.get_size(),
        )
    """)

    print("  归约操作 (Reduction):")
    print("""
    @register_lowering(aten.sum)
    def sum_(x, dim=None, keepdim=False):
        return Reduction(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=lambda index: x.load(index),
            ranges=non_reduction_size,
            reduction_ranges=reduction_size,
            reduction_type="sum",
        )
    """)

    print("  模板操作 (Template):")
    print("""
    @register_lowering(aten.mm)
    def mm(x, y):
        # mm 使用 GEMM 模板, 不走 Pointwise 融合
        return mm_template(x, y)
    """)

    print("  Fallback 操作:")
    print("""
    make_fallback(aten.some_unsupported_op)
    # → lowerings[aten.some_unsupported_op] = fallback_fn
    # → fallback_fn 创建 FallbackKernel, 运行时调用 eager
    """)


def demo_lowering_flow():
    """演示 Lowering 的完整流程"""
    print("=" * 60)
    print("3. Lowering 完整流程")
    print("=" * 60)

    print("""
  FX Node: call_function aten.add, args=(x, y)
      │
      ▼
  GraphLowering.run_node(n)
      │
      ├── super().run_node(n)  ←── torch.fx.Interpreter
      │     │
      │     ├── 查找 lowerings[aten.add]
      │     ├── 调用 add(x, y)
      │     │     ├── promote_types(x, y)  # 类型提升
      │     │     └── Pointwise(
      │     │           inner_fn=lambda idx: ops.add(x.load(idx), y.load(idx)),
      │     │           ranges=x.get_size(),
      │     │         )
      │     │
      │     └── 返回 TensorBox(StorageBox(ComputedBuffer("add_0", ...)))
      │
      ├── 后处理:
      │     ├── 检查是否 fallback
      │     ├── 处理布局约束 (Triton kernel wrapper)
      │     ├── 处理 magic method (SymInt 运算)
      │     ├── 强制 stride 对齐
      │     └── 优化 channels-last 布局
      │
      └── 注册到 graph:
            register_buffer(ComputedBuffer("add_0", ...))
            register_operation(Pointwise(...))
    """)
    print()


def demo_decomposition_vs_lowering():
    """演示算子分解与 Lowering 的关系"""
    print("=" * 60)
    print("4. 算子分解 vs Lowering")
    print("=" * 60)

    print("  算子分解 (在 Lowering 之前执行):")
    print("""
    aten.addmm(x, y, z)
        │  [分解]
        ▼
    t = aten.mm(y, z)
    result = aten.add(x, t)
    """)

    print("  分解的好处:")
    print("    - 让 Inductor 有更多融合机会")
    print("    - add + mm 可以被 Scheduler 重新融合为 addmm 模板")
    print("    - 减少 lowering 注册的数量")
    print()

    print("  分解的执行时机:")
    print("    1. Dynamo 捕获 FX Graph (可能包含 addmm)")
    print("    2. AOTAutograd 应用分解 (addmm → mm + add)")
    print("    3. Inductor Lowering (mm → TemplateBuffer, add → Pointwise)")
    print("    4. Scheduler 融合 (mm + add → addmm 模板)")
    print()


def demo_type_promotion():
    """演示类型提升"""
    print("=" * 60)
    print("5. 类型提升 (Type Promotion)")
    print("=" * 60)

    print("  Lowering 中的类型提升:")
    print("""
    @register_lowering(aten.add, type_promotion_kind=DEFAULT)
    def add(x, y):
        x, y = promote_types(x, y)  # 自动类型提升
        ...
    """)

    x = torch.randint(0, 10, (3,))
    y = torch.randn(3)
    z = x + y
    print(f"  int32 + float32 → {z.dtype}")
    print(f"  类型提升: int32 被提升为 float32")
    print()

    print("  提升策略:")
    print("    DEFAULT         - 标准提升 (int + float → float)")
    print("    INT_TO_FLOAT    - 整数提升为浮点")
    print("    ALWAYS_BOOL     - 结果总是布尔")
    print("    COMPLEX_TO_FLOAT - 复数提升为浮点")
    print()


def demo_fallback_mechanism():
    """演示 Fallback 机制"""
    print("=" * 60)
    print("6. Fallback 机制")
    print("=" * 60)

    print("  Fallback 的含义:")
    print("    当某个算子没有注册 lowering 时, 回退到 eager 模式执行")
    print()

    print("  Fallback 的代价:")
    print("    - 无法参与融合 (单独执行)")
    print("    - 需要额外的内存分配")
    print("    - 运行时调用 eager 实现 (性能较低)")
    print()

    print("  Fallback 的注册:")
    print("""
    make_fallback(aten.some_unsupported_op)

    # 等价于:
    def fallback_fn(*args):
        return FallbackKernel(aten.some_unsupported_op, *args)
    lowerings[aten.some_unsupported_op] = fallback_fn
    """)

    print("  Fallback 的适用场景:")
    print("    - 算子使用频率低, 不值得优化")
    print("    - 算子语义复杂, 难以用 IR 表示")
    print("    - 新增的算子, 尚未实现 lowering")
    print()


def demo_placeholder_handling():
    """演示输入处理"""
    print("=" * 60)
    print("7. 输入处理 (placeholder)")
    print("=" * 60)

    print("  GraphLowering.placeholder() 的处理:")
    print("""
    def placeholder(self, target, args, kwargs):
        # 创建 InputBuffer 并包装为 TensorBox
        buffer = InputBuffer(target, FixedLayout(device, dtype, size, stride))
        return TensorBox(StorageBox(buffer))
    """)

    print("  InputBuffer 的关键信息:")
    print("    - name: 输入名称 (如 'x', 'y')")
    print("    - layout: FixedLayout(device, dtype, size, stride)")
    print("    - 输入的 stride 是固定的, 不能优化")
    print()

    print("  动态形状支持:")
    print("    - size 和 stride 可以包含 SymInt (符号整数)")
    print("    - 例如: size=[s0, 64], 其中 s0 是运行时确定的维度")
    print()


if __name__ == "__main__":
    demo_fx_graph_structure()
    demo_lowering_registration()
    demo_lowering_flow()
    demo_decomposition_vs_lowering()
    demo_type_promotion()
    demo_fallback_mechanism()
    demo_placeholder_handling()

    print("=" * 60)
    print("阶段3 学习完成!")
    print("下一步: 阅读 stage4_scheduler/guide.md 学习 Scheduler")
    print("=" * 60)
