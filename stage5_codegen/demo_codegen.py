"""
阶段5: 代码生成 - 可运行示例

演示后端注册、Triton kernel 生成、wrapper 代码生成
运行方式: python demo_codegen.py
"""

import torch


def demo_backend_registration():
    """演示后端注册"""
    print("=" * 60)
    print("1. 后端注册 (register_backend_for_device)")
    print("=" * 60)

    print("  register_backend_for_device(device, scheduling, wrapper, cpp_wrapper)")
    print()

    print("  默认后端注册:")
    print("  ┌─────────┬───────────────────────────┬──────────────────────┬───────────────┐")
    print("  │ 设备    │ 调度策略                   │ Python Wrapper       │ C++ Wrapper   │")
    print("  ├─────────┼───────────────────────────┼──────────────────────┼───────────────┤")
    print("  │ cpu     │ CppScheduling             │ PythonWrapperCodegen │ CppWrapperCpu │")
    print("  │ cuda    │ CUDACombinedScheduling    │ PythonWrapperCodegen │ CppWrapperGpu │")
    print("  │ xpu     │ TritonScheduling          │ PythonWrapperCodegen │ CppWrapperGpu │")
    print("  │ mps     │ MetalScheduling           │ PythonWrapperCodegen │ CppWrapperGpu │")
    print("  │ npu     │ NPUCombinedScheduling     │ NPUWrapperCodeGen    │ CppWrapperNpu │")
    print("  └─────────┴───────────────────────────┴──────────────────────┴───────────────┘")
    print()

    print("  NPU 后端注册代码:")
    print("""
    from torch._inductor.codegen.common import register_backend_for_device
    register_backend_for_device(
        'npu',
        NPUCombinedScheduling,   # 调度策略
        NPUWrapperCodeGen,       # Python wrapper
        CppWrapperNpu,           # C++ wrapper
    )
    """)
    print()


def demo_backend_features():
    """演示 BackendFeature"""
    print("=" * 60)
    print("2. BackendFeature 枚举")
    print("=" * 60)

    features = [
        ("FOREACH", "支持 foreach 批量操作"),
        ("INPLACE_BUFFERS", "支持原地缓冲区"),
        ("TRITON_TEMPLATES", "支持 Triton 模板 kernel"),
        ("TUPLE_REDUCTION", "支持元组归约"),
        ("BUCKETIZE", "支持 bucketize 操作"),
        ("MASKED_SCATTER_WITH_INDEX", "支持带索引的 masked_scatter"),
        ("REDUCE_TO_SINGLE_ELEMENT", "支持归约到单个元素"),
    ]

    for name, desc in features:
        print(f"  BackendFeature.{name:35s} {desc}")
    print()

    print("  使用方式:")
    print("""
    from torch._inductor.codegen.common import get_backend_features
    features = get_backend_features("cuda")
    if BackendFeature.FOREACH in features:
        # 启用 foreach 优化
        ...
    """)
    print()


def demo_triton_kernel_generation():
    """演示 Triton kernel 生成"""
    print("=" * 60)
    print("3. Triton Kernel 代码生成")
    print("=" * 60)

    print("  Inductor 生成的 Triton kernel (简化版):")
    print("""
    @triton.jit
    def kernel_add_relu(
        x_ptr, y_ptr, output_ptr,
        XBLOCK: tl.constexpr,
        n_elements,
    ):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)
        xmask = xindex < n_elements

        x = tl.load(x_ptr + xindex, mask=xmask)
        y = tl.load(y_ptr + xindex, mask=xmask)

        # 融合的计算: relu(x + y)
        tmp = x + y
        output = tl.where(tmp > 0, tmp, 0.0)

        tl.store(output_ptr + xindex, output, mask=xmask)
    """)
    print()

    print("  代码生成流程:")
    print("    1. TritonScheduling.codegen_fused_node()")
    print("    2. 生成 kernel 函数签名 (指针参数 + tiling 常量)")
    print("    3. 生成 tiling 配置 (XBLOCK, YBLOCK)")
    print("    4. 展开 inner_fn → kernel body")
    print("    5. 生成 grid 函数 (计算 program 数量)")
    print("    6. 生成 kernel 调用代码")
    print()


def demo_python_wrapper():
    """演示 Python wrapper 生成"""
    print("=" * 60)
    print("4. Python Wrapper 代码生成")
    print("=" * 60)

    print("  Inductor 生成的 Python wrapper (简化版):")
    print("""
    import torch
    from torch._inductor.runtime.triton_heuristics import ...

    async_compile = AsyncCompile()

    kernel_add_relu = async_compile.triton('kernel_add_relu', '''
    @triton.jit
    def kernel_add_relu(...):
        ...
    ''')

    def call(args):
        # 输入
        arg_x_1 = args[0]
        arg_y_1 = args[1]

        # 内存分配
        buf_output = torch.empty(...)

        # Kernel 调用
        kernel_add_relu.run(
            arg_x_1, arg_y_1, buf_output,
            XBLOCK=1024,
            n_elements=arg_x_1.numel(),
        )

        return (buf_output,)
    """)
    print()


def demo_cpp_wrapper():
    """演示 C++ wrapper 生成"""
    print("=" * 60)
    print("5. C++ Wrapper 代码生成")
    print("=" * 60)

    print("  Python Wrapper vs C++ Wrapper:")
    print("  ┌──────────────┬─────────────────────┬──────────────────┐")
    print("  │ 特性         │ Python Wrapper      │ C++ Wrapper      │")
    print("  ├──────────────┼─────────────────────┼──────────────────┤")
    print("  │ 启动开销     │ 较高 (解释器)       │ 低 (直接调用)    │")
    print("  │ 灵活性       │ 高 (动态类型)       │ 低 (静态类型)    │")
    print("  │ 调试         │ 容易                │ 困难             │")
    print("  │ 适用场景     │ 通用                │ 高性能推理       │")
    print("  └──────────────┴─────────────────────┴──────────────────┘")
    print()

    print("  启用 C++ Wrapper:")
    print("    TORCHINDUCTOR_CPP_WRAPPER=1 python your_script.py")
    print("    或: torch.compile(model, options={'cpp_wrapper': True})")
    print()


def demo_codegen_flow():
    """演示代码生成流程"""
    print("=" * 60)
    print("6. 代码生成完整流程")
    print("=" * 60)

    print("""
  GraphLowering.codegen()
      │
      ├── 1. init_wrapper_code()       # 创建 PythonWrapperCodegen
      ├── 2. _update_scheduler()       # 创建 Scheduler
      ├── 3. push_codegened_graph()    # 推入代码生成上下文
      │
      ├── 4. scheduler.codegen()       # 按设备生成 kernel 代码
      │     │
      │     ├── CPU: CppScheduling.codegen()
      │     │     └── 生成 C++ kernel (SIMD 指令)
      │     │
      │     └── CUDA: TritonScheduling.codegen()
      │           └── 生成 Triton kernel
      │
      └── 5. wrapper_code.generate()   # 生成 Python wrapper
            │
            ▼
  compile_to_module()
      │
      ├── PyCodeCache.write()          # 写入磁盘
      └── PyCodeCache.load_by_key_path() # 动态加载模块
            │
            ▼
  CompiledFxGraph
      ├── current_callable  # 编译后的可调用函数
      ├── source_code       # 生成的源代码
      └── cudagraph_info    # CUDA Graph 信息
    """)
    print()


def demo_device_op_overrides():
    """演示 DeviceOpOverrides"""
    print("=" * 60)
    print("7. DeviceOpOverrides (设备操作覆盖)")
    print("=" * 60)

    print("  DeviceOpOverrides 定义设备特定的操作:")
    print("""
    class DeviceOpOverrides:
        def stream(self, stream): ...      # 获取/设置当前 stream
        def sync(self): ...                # 设备同步
        def guard(self, lock): ...         # 设备锁
        def set_device(self, device): ...  # 设置当前设备
    """)

    print("  CUDA 的 DeviceOpOverrides:")
    print("""
    class CUDADeviceOpOverrides(DeviceOpOverrides):
        def stream(self, stream):
            return f"torch.cuda.Stream(device=torch.device('cuda:{stream}'))"
        def sync(self):
            return "torch.cuda.synchronize()"
    """)

    print("  NPU 的 DeviceOpOverrides:")
    print("""
    class NPUDeviceOpOverrides(DeviceOpOverrides):
        def stream(self, stream):
            return f"torch.npu.Stream(device=torch.device('npu:{stream}'))"
        def sync(self):
            return "torch.npu.synchronize()"
    """)
    print()


if __name__ == "__main__":
    demo_backend_registration()
    demo_backend_features()
    demo_triton_kernel_generation()
    demo_python_wrapper()
    demo_cpp_wrapper()
    demo_codegen_flow()
    demo_device_op_overrides()

    print("=" * 60)
    print("阶段5 学习完成!")
    print("下一步: 阅读 stage6_runtime/guide.md 学习运行时与 Autotune")
    print("=" * 60)
