"""
阶段1: 编译入口与配置 - 可运行示例

演示 torch.compile() 的基本使用、配置项、缓存机制
运行方式: python demo_compile_entry.py
"""

import torch
import torch._inductor.config as config
import os


def demo_basic_compile():
    """基本编译示例"""
    print("=" * 60)
    print("1. 基本 torch.compile() 使用")
    print("=" * 60)

    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
    )
    model.eval()

    x = torch.randn(32, 64)

    compiled_model = torch.compile(model)

    with torch.no_grad():
        result = compiled_model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {result.shape}")
    print(f"编译模式: default")
    print()


def demo_compile_modes():
    """不同编译模式"""
    print("=" * 60)
    print("2. 不同编译模式")
    print("=" * 60)

    model = torch.nn.Linear(64, 10)
    model.eval()
    x = torch.randn(32, 64)

    modes = {
        "default": {},
        "reduce-overhead": {"triton.cudagraphs": True},
        "max-autotune-no-cudagraphs": {
            "max_autotune": True,
            "coordinate_descent_tuning": True,
        },
    }

    for mode_name, options in modes.items():
        print(f"  模式: {mode_name}")
        print(f"  选项: {options}")

    print()
    print("  torch.compile(model)                              # default")
    print("  torch.compile(model, mode='reduce-overhead')      # 启用 cudagraphs")
    print("  torch.compile(model, mode='max-autotune')         # 最大 autotune + cudagraphs")
    print()


def demo_config_options():
    """配置项演示"""
    print("=" * 60)
    print("3. Inductor 配置项")
    print("=" * 60)

    print("  关键配置项:")
    print(f"    debug              = {config.debug}")
    print(f"    fx_graph_cache     = {config.fx_graph_cache}")
    print(f"    cpp_wrapper        = {config.cpp_wrapper}")
    print(f"    triton.cudagraphs  = {config.triton.cudagraphs}")
    print(f"    max_autotune       = {config.max_autotune}")
    print()

    print("  环境变量控制:")
    print("    TORCHINDUCTOR_DEBUG=1           # 启用调试")
    print("    TORCHINDUCTOR_FX_GRAPH_CACHE=0  # 禁用缓存")
    print("    TORCHINDUCTOR_CPP_WRAPPER=1     # 使用 C++ wrapper")
    print()

    print("  代码方式控制:")
    print("    torch.compile(model, options={'triton.cudagraphs': True})")
    print()


def demo_custom_backend():
    """自定义后端演示 - 理解 Dynamo → Inductor 的接口"""
    print("=" * 60)
    print("4. Dynamo 后端接口 (理解 compile_fx 的输入)")
    print("=" * 60)

    def custom_backend(gm: torch.fx.GraphModule, example_inputs):
        """自定义后端: 打印 FX Graph 结构"""
        print("  收到的 GraphModule 节点:")
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                print(f"    [placeholder] {node.name} - 输入占位符")
            elif node.op == "call_module":
                print(f"    [call_module] {node.name} - 调用 nn.Module: {node.target}")
            elif node.op == "call_function":
                print(f"    [call_function] {node.name} - 调用函数: {node.target}")
            elif node.op == "output":
                print(f"    [output] {node.name} - 输出节点")
            else:
                print(f"    [{node.op}] {node.name} - target: {node.target}")

        return gm.forward

    model = torch.nn.Sequential(
        torch.nn.ReLU(),
    )
    model.eval()

    x = torch.randn(2, 3)

    compiled = torch.compile(model, backend=custom_backend)
    with torch.no_grad():
        _ = compiled(x)

    print()
    print("  这就是 Inductor compile_fx() 收到的输入:")
    print("    gm: torch.fx.GraphModule - Dynamo 捕获的计算图")
    print("    example_inputs: 示例输入张量列表")
    print()


def demo_cache_mechanism():
    """缓存机制演示"""
    print("=" * 60)
    print("5. 编译缓存机制")
    print("=" * 60)

    print("  缓存查找流程:")
    print("    1. prepare_key(gm, inputs) → 计算 cache_key")
    print("       ├── graph_hash: 基于 FX Graph 结构")
    print("       └── input_hash: 基于输入形状/类型/设备")
    print("    2. load_with_key(cache_key)")
    print("       ├── [命中] → 反序列化 CompiledFxGraph → 直接返回")
    print("       └── [未命中] → 执行编译 → 存储到缓存")
    print()

    print("  缓存目录:")
    from torch._inductor.runtime.runtime_utils import cache_dir
    print(f"    {cache_dir()}")
    print()

    print("  禁用缓存:")
    print("    TORCHINDUCTOR_FX_GRAPH_CACHE=0 python your_script.py")
    print()


def demo_call_chain():
    """打印完整调用链"""
    print("=" * 60)
    print("6. 完整调用链 (从 torch.compile 到 GraphLowering)")
    print("=" * 60)

    print("""
  torch.compile(model, backend="inductor")
      │
      ▼
  torch._dynamo.optimize("inductor")
      │
      ▼
  torch._inductor.compile_fx.compile_fx(gm, example_inputs)
      │
      ▼
  compile_fx_inner(gm, example_inputs)
      │
      ├── [缓存命中] FxGraphCache → 返回 CompiledFxGraph
      │
      ▼ [缓存未命中]
  _compile_fx_inner(...)
      │
      ▼
  fx_codegen_and_compile(...)
      │
      ▼
  _InProcessFxCompile.codegen_and_compile(...)
      ├── Step 1: view_to_reshape(gm)         # 预处理
      ├── Step 2: FakeTensorProp(gm)          # FakeTensor 传播
      ├── Step 3: record_original_output_strides  # 记录 stride
      ├── Step 4: _recursive_post_grad_passes # Post-grad 优化
      ├── Step 5: GraphLowering.run()         # ← 核心: FX → IR
      ├── Step 6: graph.compile_to_module()   # ← 核心: IR → 代码
      └── Step 7: 构造 CompiledFxGraph        # 编译产物
    """)
    print()


if __name__ == "__main__":
    demo_basic_compile()
    demo_compile_modes()
    demo_config_options()
    demo_custom_backend()
    demo_cache_mechanism()
    demo_call_chain()

    print("=" * 60)
    print("阶段1 学习完成!")
    print("下一步: 阅读 stage2_ir/guide.md 学习 IR 中间表示")
    print("=" * 60)
