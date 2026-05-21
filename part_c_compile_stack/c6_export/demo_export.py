"""
c6: torch.export - 可运行示例
运行方式: python demo_export.py
"""

import torch


def demo_export_basics():
    print("=" * 60)
    print("1. torch.export 基础")
    print("=" * 60)

    model = torch.nn.Linear(10, 5)
    x = torch.randn(2, 10)

    exported = torch.export.export(model, (x,))

    print(f"  ExportedProgram 类型: {type(exported)}")
    print(f"  图节点数: {len(list(exported.graph_module.graph.nodes))}")
    print()

    with torch.no_grad():
        y = exported.module()(x)
    print(f"  导出模型运行: 输入 {x.shape}, 输出 {y.shape}")
    print()


def demo_export_vs_jit():
    print("=" * 60)
    print("2. torch.export vs torch.jit")
    print("=" * 60)

    print("  ┌──────────────────┬──────────────────┬──────────────────┐")
    print("  │ 特性             │ torch.jit        │ torch.export     │")
    print("  ├──────────────────┼──────────────────┼──────────────────┤")
    print("  │ 追踪方式         │ AST/trace        │ Dynamo 字节码    │")
    print("  │ 控制流           │ 受限             │ 支持             │")
    print("  │ 动态形状         │ 不支持           │ 支持 (SymInt)    │")
    print("  │ Guard 系统       │ 无               │ 完整             │")
    print("  │ Python 特性      │ 严重受限         │ 广泛支持         │")
    print("  │ AOTInductor      │ 不支持           │ 直接对接         │")
    print("  └──────────────────┴──────────────────┴──────────────────┘")
    print()


def demo_dynamic_shapes():
    print("=" * 60)
    print("3. 动态形状支持")
    print("=" * 60)

    print("  torch.export 支持动态形状:")
    print("""
    from torch.export import Dim
    dim0 = Dim("dim0", min=1, max=1024)

    model = torch.nn.Linear(10, 5)
    x = torch.randn(2, 10)

    exported = torch.export.export(
        model,
        (x,),
        dynamic_shapes={"x": {0: dim0}},
    )
    # → 编译后的模型可以接受 x.shape = (任意, 10)
    """)
    print()


def demo_aot_compile():
    print("=" * 60)
    print("4. AOTInductor 编译")
    print("=" * 60)

    print("  AOTInductor 将 ExportedProgram 编译为共享库:")
    print("""
    exported = torch.export.export(model, (x,))

    # 编译为 .so/.dll 共享库
    so_path = torch._inductor.aot_compile(
        exported,
        (x,),
    )
    # → 生成可以直接加载执行的本地代码
    """)
    print()

    print("  AOTInductor 的优势:")
    print("    - 编译结果可以部署到没有 Python 的环境")
    print("    - 消除 Python 解释器开销")
    print("    - 支持模型版本管理")
    print()


if __name__ == "__main__":
    demo_export_basics()
    demo_export_vs_jit()
    demo_dynamic_shapes()
    demo_aot_compile()
    print("c6: torch.export 学习完成!")
