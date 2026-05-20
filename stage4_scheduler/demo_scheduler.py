"""
阶段4: Scheduler - 调度与融合 - 可运行示例

演示算子融合、依赖分析、内存优化
运行方式: python demo_scheduler.py
"""

import torch


def demo_fusion_basics():
    """演示融合的基本概念"""
    print("=" * 60)
    print("1. 算子融合基本概念")
    print("=" * 60)

    print("  未融合: 每个 op 是一个独立的 kernel launch")
    print("""
    x → [kernel: add] → tmp → [kernel: relu] → output
         读取: x, y        读取: tmp
         写入: tmp          写入: output

    代价:
      - 2 次 kernel launch
      - tmp 需要分配内存
      - tmp 需要写入 + 读取 (2 次内存访问)
    """)

    print("  融合后: 一个 kernel 完成两个 op")
    print("""
    x → [kernel: add_relu] → output
         读取: x, y
         写入: output

    收益:
      - 1 次 kernel launch
      - 不需要 tmp 的内存分配
      - 中间结果留在寄存器/缓存中 (0 次额外内存访问)
    """)
    print()


def demo_fusion_algorithm():
    """演示融合算法"""
    print("=" * 60)
    print("2. 融合算法 (fuse_nodes)")
    print("=" * 60)

    print("  算法流程:")
    print("""
    for _ in range(10):  # 最多 10 轮迭代
        changed = fuse_nodes_once()
        if not changed:
            break  # 收敛

    fuse_nodes_once():
        for node1, node2 in get_possible_fusions(nodes):
            if not can_fuse(node1, node2):        continue
            if will_fusion_create_cycle(node1, node2): continue
            speedup = speedup_by_fusion(node1, node2)
            if not speedup:                        continue
            fuse(node1, node2)  # → FusedSchedulerNode
    """)
    print()


def demo_can_fuse():
    """演示 can_fuse 判断逻辑"""
    print("=" * 60)
    print("3. can_fuse() 判断逻辑")
    print("=" * 60)

    print("  可以融合的情况:")
    print("    ✓ Pointwise + Pointwise  (最常见)")
    print("    ✓ Reduction + Pointwise  (特定条件)")
    print("    ✓ Pointwise + Reduction  (特定条件)")
    print()

    print("  不可以融合的情况:")
    print("    ✗ 不同设备的节点")
    print("    ✗ ExternKernel 参与的融合")
    print("    ✗ TemplateBuffer 通常不参与融合")
    print("    ✗ 有中间读写依赖的节点")
    print("    ✗ 融合后 kernel 太大")
    print()


def demo_fusion_example():
    """演示融合示例"""
    print("=" * 60)
    print("4. 融合示例: z = relu(x + y * 2)")
    print("=" * 60)

    print("  融合前 (3 个独立 kernel):")
    print("""
    [SchedulerNode: mul]    读取: y, constant(2)
    [SchedulerNode: add]    读取: x, mul
    [SchedulerNode: relu]   读取: add
    """)
    print("  融合后 (1 个 kernel):")
    print("""
    [FusedSchedulerNode: mul_add_relu]
      读取: x, y, constant(2)
      inner_fn: lambda idx: ops.relu(ops.add(x.load(idx), ops.mul(y.load(idx), 2)))
    """)
    print("  收益: 3 次 kernel launch → 1 次, 消除 2 个中间缓冲区")
    print()


def demo_dependency_analysis():
    """演示依赖分析"""
    print("=" * 60)
    print("5. 依赖分析")
    print("=" * 60)

    print("  依赖类型:")
    print("""
    MemoryDep - 内存依赖 (读/写同一块内存)
      例如: node_B 读取 node_A 的输出

    StarDep   - 星型依赖 (依赖所有元素)
      例如: Reduction 操作依赖输入的所有元素

    WeakDep   - 弱依赖 (不阻止融合)
      例如: 仅用于调试/追踪的依赖
    """)

    print("  依赖分析的关键:")
    print("    - 读-写依赖: B 读取 A 的输出 → A 必须在 B 之前")
    print("    - 写-写依赖: 两个节点写同一块内存 → 不能重排")
    print("    - 别名依赖: 两个张量共享底层存储 → 需要追踪")
    print("    - 变异依赖: 节点修改了输入 → 需要特殊处理")
    print()


def demo_cycle_detection():
    """演示循环检测"""
    print("=" * 60)
    print("6. 循环检测 (will_fusion_create_cycle)")
    print("=" * 60)

    print("  场景: 融合 node1 和 node2")
    print("""
    融合前:
      A → node1 → node2 → C
                ↘     ↗
                  B

    融合后 (node1 + node2 = fused):
      A → fused → C
            ↑  ↓
            B

    如果 B 同时依赖 node1 的输出和 node2 的输出,
    融合后 B 和 fused 之间可能形成循环依赖!
    """)

    print("  检测方法: DFS/BFS 检查融合后的依赖图是否有环")
    print()


def demo_memory_optimization():
    """演示内存优化"""
    print("=" * 60)
    print("7. 内存优化")
    print("=" * 60)

    print("  峰值内存优化 (reorder_for_peak_memory):")
    print("""
    原始顺序:
      A → B → C → D
      峰值内存: A + B + C + D (所有缓冲区同时存活)

    优化后:
      A → B → (释放A) → C → (释放B) → D → (释放C)
      峰值内存: B + C + D (更小)
    """)

    print("  compute_last_usage():")
    print("    计算每个缓冲区最后一次被使用的位置")
    print("    在该位置之后可以释放缓冲区")
    print()

    print("  内存池复用:")
    print("    不再使用的缓冲区内存可以被新缓冲区复用")
    print("    通过 memory_planning 实现高效的内存分配")
    print()


def demo_scheduling_backends():
    """演示调度后端"""
    print("=" * 60)
    print("8. 调度后端 (BaseScheduling)")
    print("=" * 60)

    print("  调度后端决定如何为特定设备生成代码:")
    print()
    print("  CPU:")
    print("    CppScheduling → 生成 C++ kernel (使用 SIMD 指令)")
    print("    HalideScheduling → 生成 Halide kernel")
    print("    TritonScheduling → 生成 Triton kernel (CPU Triton)")
    print()
    print("  CUDA:")
    print("    TritonScheduling → 生成 Triton kernel (最常用)")
    print("    CUDACppScheduling → 生成 CUDA C++ kernel (CUTLASS)")
    print("    CUDACombinedScheduling → 组合调度 (委托给 Triton 或 C++)")
    print()
    print("  MPS:")
    print("    MetalScheduling → 生成 Metal kernel")
    print()
    print("  NPU:")
    print("    NPUCombinedScheduling → 组合调度 (线性/非线性/CATLASS)")
    print()


def demo_scheduler_init_steps():
    """演示 Scheduler 初始化步骤"""
    print("=" * 60)
    print("9. Scheduler._init() 完整步骤")
    print("=" * 60)

    steps = [
        ("1. create_scheduler_node()", "IR → SchedulerNode"),
        ("2. decide_global_ordering_of_comms()", "通信全局排序"),
        ("3. compute_dependencies()", "构建依赖图"),
        ("4. topological_sort_schedule()", "拓扑排序"),
        ("5. dead_node_elimination()", "死节点消除"),
        ("6. create_foreach_nodes()", "批量操作合并"),
        ("7. fuse_nodes()", "⭐ 算子融合 (核心)"),
        ("8. merge_loops()", "循环合并"),
        ("9. finalize_multi_template_buffers()", "多模板缓冲区定稿"),
        ("10. reorder_for_peak_memory()", "峰值内存优化"),
        ("11. reorder_compute_and_comm_for_overlap()", "计算通信重叠"),
        ("12. compute_last_usage()", "计算最后使用 (内存回收)"),
    ]

    for step, desc in steps:
        marker = "⭐" if "融合" in desc else "  "
        print(f"  {marker} {step:45s} {desc}")
    print()


if __name__ == "__main__":
    demo_fusion_basics()
    demo_fusion_algorithm()
    demo_can_fuse()
    demo_fusion_example()
    demo_dependency_analysis()
    demo_cycle_detection()
    demo_memory_optimization()
    demo_scheduling_backends()
    demo_scheduler_init_steps()

    print("=" * 60)
    print("阶段4 学习完成!")
    print("下一步: 阅读 stage5_codegen/guide.md 学习代码生成")
    print("=" * 60)
