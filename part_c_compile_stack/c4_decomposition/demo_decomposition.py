"""
c4: 算子分解 - 可运行示例
运行方式: python demo_decomposition.py
"""

import torch


def demo_decomposition_basics():
    print("=" * 60)
    print("1. 算子分解基础")
    print("=" * 60)

    print("  分解: 将一个算子表达为更基础算子的组合")
    print()
    print("  示例:")
    print("    aten.addmm(bias, x, w) → aten.mm(x, w) + aten.add(bias, mm)")
    print("    aten.clamp_min(x, min) → aten.clamp(x, min=min)")
    print("    aten.softplus(x) → log(1 + exp(x)) (数值稳定版)")
    print()

    print("  分解的好处:")
    print("    - Inductor 只需处理基本算子, 减少实现工作量")
    print("    - 分解后的算子有更多融合机会")
    print("    - 确保编译器后端的一致性")
    print()


def demo_core_aten():
    print("=" * 60)
    print("2. Core ATen 算子集")
    print("=" * 60)

    print("  Core ATen: 约 170 个基本算子")
    print("  所有 ATen 算子都可以分解到 Core ATen")
    print()
    print("  Core ATen 的类别:")
    print("    - 逐元素操作: add, mul, div, exp, log, sin, cos...")
    print("    - 归约操作: sum, max, min, mean, argmax...")
    print("    - 形状操作: view, reshape, permute, slice, cat...")
    print("    - 线性代数: mm, bmm, svd, solve...")
    print("    - 卷积: conv1d, conv2d, conv3d...")
    print("    - 归一化: batch_norm, layer_norm...")
    print()


def demo_decomposition_in_compile():
    print("=" * 60)
    print("3. 分解在编译栈中的位置")
    print("=" * 60)

    print("  编译流程中的分解时机:")
    print("""
    Dynamo → FX Graph
        │
        ▼
    AOTAutograd
        ├── functionalize
        ├── vjp (计算反向图)
        └── Post-autograd 分解  ←── 这里执行分解
            │
            ▼
        Core ATen 算子图
            │
            ▼
        Inductor Lowering
    """)
    print()


if __name__ == "__main__":
    demo_decomposition_basics()
    demo_core_aten()
    demo_decomposition_in_compile()
    print("c4: 算子分解 学习完成!")
