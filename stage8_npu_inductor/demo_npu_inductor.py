"""
阶段8: NPU Inductor 扩展 - 可运行示例

演示 NPU Inductor 的架构、patch 机制、v1/v2 对比
运行方式: python demo_npu_inductor.py
"""

import os
import sys


def demo_npu_architecture():
    """演示 NPU Inductor 架构"""
    print("=" * 60)
    print("1. NPU Inductor 架构全景")
    print("=" * 60)

    print("""
  torch.compile(model, backend="inductor")
      │
      ▼
  PyTorch Inductor (torch._inductor)
      │
      ├── [Monkey-Patch 层]
      │   ├── 设备识别: patch_is_gpu, patch_has_triton
      │   ├── 后端注册: register_backend_for_device('npu')
      │   │     └── NPUCombinedScheduling + NPUWrapperCodeGen + CppWrapperNpu
      │   ├── 算子 Lowering: NPU lowering + CATLASS 模板
      │   ├── IR 扩展: ConcatKernel, IndexputTemplate, ScatterTemplate
      │   ├── 代码生成: NPUWrapperCodeGen, CppWrapperNpu
      │   └── 运行时: NPUCachingAutotuner, GridExprNpu
      │
      ├── [调度层]
      │   └── NPUCombinedScheduling
      │       ├── NPUTritonScheduling (线性模式)
      │       ├── NPUNoLinearTritonScheduling (非线性模式)
      │       └── CATLASSScheduling (GEMM 模板)
      │
      ├── [代码生成层]
      │   ├── Python Wrapper: NPUWrapperCodeGen
      │   ├── C++ Wrapper: CppWrapperNpu (rtKernelLaunch)
      │   ├── Triton Kernel: NPUIndexTritonKernel
      │   └── CATLASS Template: CATLASS1xGemmTemplate
      │
      └── [运行时层]
          ├── NPUCachingAutotuner
          ├── GridExprNpu
          └── NPU Profiler Benchmark
    """)
    print()


def demo_patch_classification():
    """演示 Patch A-E 分类"""
    print("=" * 60)
    print("2. Patch A-E 分类框架")
    print("=" * 60)

    categories = [
        ("A-class", "可用原生扩展 API 替代", "register_backend_for_device('npu', ...) ✅"),
        ("B-class", "可用子类 + 方法覆盖替代", "NPUWrapperCodeGen 继承 PythonWrapperCodegen"),
        ("C-class", "需要 NPU 特有子类体系", "CppWrapperNpu, NPUCachingAutotuner"),
        ("D-class", "需要上游 PyTorch PR", "部分 scheduler patch"),
        ("E-class", "包装/注册 patch", "patch_device_to_aten"),
    ]

    for cat, desc, example in categories:
        print(f"  {cat:10s} {desc:30s} 示例: {example}")
    print()

    print("  重构优先级: A > B > E > C > D")
    print("  A-class: 尽量使用原生 API, 消除 patch")
    print("  B-class: 用子类替代 monkey-patch")
    print("  C-class: 保留为 NPU 业务模块")
    print("  D-class: 记录技术债, 最小化范围")
    print("  E-class: 内联注册逻辑")
    print()


def demo_npu_init_flow():
    """演示 NPU 初始化流程"""
    print("=" * 60)
    print("3. NPU 初始化流程 (v1)")
    print("=" * 60)

    steps = [
        ("1", "AsyncCompile.warm_pool()", "预热线程池"),
        ("2", "register_device_op_overrides_npu()", "注册设备操作覆盖"),
        ("3", "register_backend_for_device('npu', ...)", "注册 NPU 后端"),
        ("4", "_register_npu_inductor_decompositons()", "注册算子分解"),
        ("5", "_register_npu_inductor_fallbacks()", "注册 fallback 算子"),
        ("6", "_register_npu_inductor_mm/addmm/bmm", "注册 GEMM 模板"),
        ("7", "patch_scheduler()", "修改融合策略"),
        ("8", "patch_triton_scheduling()", "强制 int32 索引"),
        ("9", "patch_is_gpu() / patch_has_triton()", "设备识别"),
        ("10", "disable_foreach()", "禁用 foreach"),
        ("11", "其他 patch (约 20+ 个)", "各种适配"),
    ]

    for num, code, desc in steps:
        print(f"  Step {num:2s}: {code:50s} # {desc}")
    print()


def demo_npu_scheduling():
    """演示 NPU 调度策略"""
    print("=" * 60)
    print("4. NPU 调度策略 (NPUCombinedScheduling)")
    print("=" * 60)

    print("  三种调度策略:")
    print()
    print("  1. NPUTritonScheduling (线性模式, 默认)")
    print("     - 使用 NPUIndexTritonKernel")
    print("     - 基于索引的 Triton kernel")
    print("     - 适合大多数 Pointwise 和 Reduction 操作")
    print()
    print("  2. NPUNoLinearTritonScheduling (非线性模式)")
    print("     - 使用 NPUTritonKernel / NPUTritonKernelWithLoop")
    print("     - 非线性 Triton kernel")
    print("     - 用于包含间接内存操作的 kernel (cat_store, gather 等)")
    print()
    print("  3. CATLASSScheduling (GEMM 模板)")
    print("     - 使用 CATLASS1xGemmTemplate")
    print("     - 华为自研 GEMM 库")
    print("     - 用于 mm, addmm, bmm 等矩阵乘法")
    print()

    print("  调度选择逻辑:")
    print("""
    choose_node_backend(node):
        if isinstance(node, TemplateBuffer) and use_catlass_template(node):
            return "catlass"
        else:
            return "triton"
    """)
    print()


def demo_npu_cpp_wrapper():
    """演示 CppWrapperNpu"""
    print("=" * 60)
    print("5. CppWrapperNpu (NPU C++ 代码生成)")
    print("=" * 60)

    print("  CppWrapperNpu vs CppWrapperGpu 的关键差异:")
    print()
    print("  ┌──────────────────────┬─────────────────────┬──────────────────────┐")
    print("  │ 特性                 │ CppWrapperGpu       │ CppWrapperNpu        │")
    print("  ├──────────────────────┼─────────────────────┼──────────────────────┤")
    print("  │ Kernel 启动          │ cuLaunchKernel      │ rtKernelLaunch       │")
    print("  │ 参数结构体           │ 标准 CUDA 参数      │ + FFTS/sync/workspace│")
    print("  │ DTYPE 映射           │ 标准                │ bool→int32, fp16→fp  │")
    print("  │ 运行时头文件         │ CUDA runtime        │ torch_npu aoti       │")
    print("  └──────────────────────┴─────────────────────┴──────────────────────┘")
    print()

    print("  NPU 特有参数字段:")
    print("    - ffts_addr: FFTS 地址 (NPU 硬件特性)")
    print("    - sync_block_lock: 同步块锁")
    print("    - workspace_addr: 工作空间地址")
    print()


def demo_npu_autotuner():
    """演示 NPUCachingAutotuner"""
    print("=" * 60)
    print("6. NPUCachingAutotuner (NPU Autotune)")
    print("=" * 60)

    print("  NPUCachingAutotuner 继承 CachingAutotuner, 添加 NPU 特有功能:")
    print()

    print("  NPU 特有编译选项:")
    print("    - compile_mode: 编译模式")
    print("    - multibuffer: 多缓冲支持")
    print("    - enable_vf_fusion: 向量融合")
    print("    - simt_stack_limit: SIMT 栈限制")
    print()

    print("  NPU 特有元数据:")
    print("    - mix_mode: SIMD/SIMT 混合模式")
    print("    - parallel_mode: 并行模式")
    print("    - force_simt_only: 强制 SIMT 模式")
    print()

    print("  NPU 特有功能:")
    print("    - debug 模式: 数据 dump")
    print("    - 精度校验: msprobe 工具")
    print("    - 多线程并行预编译")
    print()


def demo_npu_ir_nodes():
    """演示 NPU 自定义 IR 节点"""
    print("=" * 60)
    print("7. NPU 自定义 IR 节点")
    print("=" * 60)

    nodes = [
        ("ConcatKernel", "NopKernel", "高效 concat (split tiling 模式)"),
        ("IndexputTemplate", "Scatter", "index_put 间接写入"),
        ("ScatterTemplate", "Scatter", "scatter 间接散射"),
    ]

    for name, parent, desc in nodes:
        print(f"  {name:20s} (继承 {parent:10s}) - {desc}")
    print()

    print("  NPU 间接内存操作 (硬件加速):")
    print("    - index_select:   间接读取")
    print("    - gather_template: 间接收集")
    print("    - cat_insert_slice / cat_store: 间接写入 (concat)")
    print("    - indexput_template: 间接写入 (index_put)")
    print("    - scatter_template: 间接散射")
    print()


def demo_v1_vs_v2():
    """演示 v1 vs v2 架构对比"""
    print("=" * 60)
    print("8. v1 vs v2 架构对比")
    print("=" * 60)

    print("  ┌─────────────────┬──────────────────────────┬──────────────────────────┐")
    print("  │ 特性            │ v1                       │ v2                       │")
    print("  ├─────────────────┼──────────────────────────┼──────────────────────────┤")
    print("  │ 初始化方式      │ 过程式 patch 调用        │ 结构化函数组织           │")
    print("  │ 目录结构        │ 扁平                     │ 模块化 (backend/ir/...)  │")
    print("  │ Autotuner 注入  │ 替换类方法               │ autotuner_cls 参数       │")
    print("  │ BackendFeature  │ 未声明                   │ 明确声明                 │")
    print("  │ FallbackKernel  │ 内联在 patch 中          │ 独立 NPUFallbackKernel   │")
    print("  │ GraphLowering   │ 内联在 patch 中          │ 独立 NPUGraphLowering    │")
    print("  │ 环境变量        │ TORCHINDUCTOR_NPU_VERSION=v1 (默认)                   │")
    print("  └─────────────────┴──────────────────────────┴──────────────────────────┘")
    print()

    print("  v2 的改进方向:")
    print("    - 更结构化的初始化流程")
    print("    - 模块化目录结构")
    print("    - 更优雅的 autotuner 注入方式")
    print("    - 明确声明 BackendFeature")
    print("    - 独立的 NPUFallbackKernel 和 NPUGraphLowering 类")
    print()


def demo_native_extension_api():
    """演示 PyTorch Inductor 原生扩展 API"""
    print("=" * 60)
    print("9. PyTorch Inductor 原生扩展 API (推荐使用)")
    print("=" * 60)

    print("  1. 注册设备后端:")
    print("""
    from torch._inductor.codegen.common import register_backend_for_device
    register_backend_for_device(
        device="my_device",
        scheduling_cls=MyScheduling,
        wrapper_codegen_cls=MyWrapperCodeGen,
        cpp_wrapper_cls=MyCppWrapper,
    )
    """)

    print("  2. 注册设备操作覆盖:")
    print("""
    from torch._inductor.codegen.common import register_device_op_overrides
    register_device_op_overrides("my_device", MyDeviceOpOverrides)
    """)

    print("  3. 声明后端特性:")
    print("""
    from torch._inductor.codegen.common import BackendFeature
    class MyScheduling(BaseScheduling):
        @classmethod
        def get_backend_features(cls, device):
            return {BackendFeature.INPLACE_BUFFERS, BackendFeature.TRITON_TEMPLATES}
    """)

    print("  4. 注册算子 lowering:")
    print("""
    from torch._inductor.lowering import register_lowering
    @register_lowering(aten.my_op)
    def my_op_lowering(*args):
        ...
    """)

    print("  5. 注册算子分解:")
    print("""
    from torch._inductor.decomposition import register_decomposition
    @register_decomposition([aten.my_op])
    def my_op_decomp(x):
        ...
    """)
    print()


if __name__ == "__main__":
    demo_npu_architecture()
    demo_patch_classification()
    demo_npu_init_flow()
    demo_npu_scheduling()
    demo_npu_cpp_wrapper()
    demo_npu_autotuner()
    demo_npu_ir_nodes()
    demo_v1_vs_v2()
    demo_native_extension_api()

    print("=" * 60)
    print("全部 8 个阶段学习完成!")
    print()
    print("推荐复习路径:")
    print("  1. 回顾 README.md 中的编译流水线总览")
    print("  2. 重新阅读 stage2_ir (IR) 和 stage4_scheduler (Scheduler)")
    print("  3. 对照 NPU Inductor 源码理解每个 patch 的作用")
    print("  4. 尝试使用原生扩展 API 重构一个 A-class patch")
    print("=" * 60)
