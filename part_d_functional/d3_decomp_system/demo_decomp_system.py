"""
d3: 分解系统 - 可运行示例
运行方式: python demo_decomp_system.py
"""

import torch


def demo_decomp_tables():
    print("=" * 60)
    print("1. 分解表层次")
    print("=" * 60)

    print("  三种分解表:")
    print("    post_autograd  - autograd 之后的分解 (Inductor 使用)")
    print("    pre_autograd   - autograd 之前的分解 (Dynamo 使用)")
    print("    meta           - 元数据分解 (形状推理)")
    print()

    print("  分解表的注册:")
    print("""
    @register_decomposition(aten.clamp_min)
    def clamp_min(x, min):
        return torch.clamp(x, min=min)
    """)
    print()


def demo_core_aten_decomp():
    print("=" * 60)
    print("2. Core ATen 分解")
    print("=" * 60)

    from torch._decomp import core_aten_decompositions
    decomp = core_aten_decompositions()
    print(f"  Core ATen 分解规则数量: {len(decomp)}")
    print()

    print("  示例分解规则:")
    count = 0
    for op in list(decomp.keys())[:10]:
        print(f"    {op}")
        count += 1
    print(f"    ... 共 {len(decomp)} 个")
    print()


def demo_decomp_in_compile():
    print("=" * 60)
    print("3. 分解在编译流程中的位置")
    print("=" * 60)

    print("  编译流程中的分解:")
    print("""
    Dynamo 捕获 → FX Graph
        │
        ▼
    AOTAutograd
        ├── pre_autograd 分解 (可选)
        ├── vjp (计算反向图)
        └── post_autograd 分解 (必需)
              │
              ▼
        Core ATen 算子图
              │
              ▼
        Inductor Lowering (只需处理 Core ATen)
    """)
    print()

    print("  为什么需要分解?")
    print("    - Inductor 只需实现 Core ATen 的 lowering")
    print("    - 分解后的算子有更多融合机会")
    print("    - 确保不同后端的一致性")
    print()


if __name__ == "__main__":
    demo_decomp_tables()
    demo_core_aten_decomp()
    demo_decomp_in_compile()
    print("d3: 分解系统 学习完成!")
