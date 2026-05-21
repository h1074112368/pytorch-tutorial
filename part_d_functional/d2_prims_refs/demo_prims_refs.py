"""
d2: Prims + Refs - 可运行示例
运行方式: python demo_prims_refs.py
"""

import torch


def demo_prims():
    print("=" * 60)
    print("1. Prims (原语操作)")
    print("=" * 60)

    print("  Prims 是不可再分的原子操作, 约 100+ 个:")
    print()
    print("  逐元素一元: abs, cos, exp, log, sqrt, sin, tanh")
    print("  逐元素二元: add, mul, div, pow, eq, lt, gt")
    print("  视图操作:   as_strided, broadcast_in_dim, squeeze, transpose")
    print("  形状操作:   cat, reshape, rev, collapse")
    print("  归约操作:   sum, amax, amin, prod, var")
    print("  类型转换:   convert_element_type, device_put, clone")
    print("  张量创建:   empty_strided, iota, scalar_tensor, normal")
    print()

    print("  Prims 的注册:")
    print("""
    # 每个 prim 通过 _make_prim() 注册
    _make_prim(
        schema="add(Tensor self, Tensor other) -> Tensor",
        return_type=RETURN_TYPE.NEW,
        meta=add_meta,          # 形状推理
        impl_aten=add_impl_aten, # ATen 实现
    )
    """)
    print()


def demo_refs():
    print("=" * 60)
    print("2. Refs (参考实现)")
    print("=" * 60)

    print("  Refs 是高层 API 的纯 Python 实现, 基于 prims:")
    print()
    print("  示例: softmax 的 ref 实现")
    print("""
    def softmax(x, dim):
        x_max = prims.amax(x, dim, keepdim=True)
        exp_x = prims.exp(x - x_max)
        return exp_x / prims.sum(exp_x, dim, keepdim=True)
    """)
    print()

    print("  Refs 的作用:")
    print("    - 提供算子语义的明确定义")
    print("    - 作为分解的目标 (高层 → refs → prims)")
    print("    - 支持在 TorchRefsMode 下重定向 torch.* 到 refs")
    print()


def demo_three_layers():
    print("=" * 60)
    print("3. 三层算子抽象")
    print("=" * 60)

    print("  ┌──────────────────────────────────────────────────┐")
    print("  │ torch.add (高层 API)                             │")
    print("  │   ↓ [refs: 参考实现, 基于 prims]                 │")
    print("  │ torch._refs.add (Python 实现)                    │")
    print("  │   ↓ [prims: 原子操作]                            │")
    print("  │ torch._prims.add (最底层原子操作)                │")
    print("  │   ↓ [impl_aten: 委托到 C++]                     │")
    print("  │ aten::add (C++ ATen 内核)                        │")
    print("  └──────────────────────────────────────────────────┘")
    print()

    print("  在编译栈中的角色:")
    print("    Dynamo → FX Graph → AOTAutograd → 分解")
    print("                                         │")
    print("                                         ▼")
    print("                              Core ATen / Prims 算子图")
    print("                                         │")
    print("                                         ▼")
    print("                              Inductor Lowering")
    print()


if __name__ == "__main__":
    demo_prims()
    demo_refs()
    demo_three_layers()
    print("d2: Prims + Refs 学习完成!")
