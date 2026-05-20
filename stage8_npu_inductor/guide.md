# 阶段8: NPU Inductor 扩展（实战重点）

## 核心问题

NPU Inductor 如何通过 monkey-patch 和原生扩展 API 适配 PyTorch Inductor？v1 和 v2 架构有何区别？

## NPU Inductor 架构全景

```
torch.compile(model, backend="inductor")
    │
    ▼
PyTorch Inductor (torch._inductor)
    │
    ├── [Monkey-Patch 层] ─────────────────────────
    │   ├── 设备识别: patch_is_gpu, patch_has_triton
    │   ├── 后端注册: register_backend_for_device('npu')
    │   │     └── NPUCombinedScheduling + NPUWrapperCodeGen + CppWrapperNpu
    │   ├── 算子 Lowering: NPU lowering 注册 + CATLASS 模板
    │   ├── IR 扩展: ConcatKernel, IndexputTemplate, ScatterTemplate
    │   ├── 代码生成: NPUWrapperCodeGen, CppWrapperNpu
    │   ├── 运行时: NPUCachingAutotuner, GridExprNpu
    │   └── Autotune: NPU tiling, profiling benchmark
    │
    ├── [调度层] ────────────────────────────────────
    │   └── NPUCombinedScheduling
    │       ├── NPUTritonScheduling (线性模式, NPUIndexTritonKernel)
    │       ├── NPUNoLinearTritonScheduling (非线性模式)
    │       └── CATLASSScheduling (GEMM 模板)
    │
    ├── [代码生成层] ────────────────────────────────
    │   ├── Python Wrapper: NPUWrapperCodeGen
    │   ├── C++ Wrapper: CppWrapperNpu (rtKernelLaunch)
    │   ├── Triton Kernel: NPUIndexTritonKernel / NPUTritonKernel
    │   └── CATLASS Template: CATLASS1xGemmTemplate
    │
    └── [运行时层] ──────────────────────────────────
        ├── NPUCachingAutotuner (autotune + precompile)
        ├── GridExprNpu (NPU grid 计算)
        └── NPU Profiler Benchmark
```

## 关键源码文件

### 1. `__init__.py` — NPU 初始化入口

源码位置: `c:\inductor\npu-pytorch\torch_npu\_inductor\__init__.py`

#### v1 初始化流程

```python
# v1: 通过环境变量 TORCHINDUCTOR_NPU_VERSION=v1 (默认)

# Step 1: 预热线程池
AsyncCompile.warm_pool()

# Step 2: 注册设备操作覆盖
register_device_op_overrides_npu()

# Step 3: 注册 NPU 后端
register_backend_for_device('npu', NPUCombinedScheduling, NPUWrapperCodeGen, CppWrapperNpu)

# Step 4: 注册算子 lowering
_register_npu_inductor_decompositons()
_register_npu_inductor_fallbacks()
_register_npu_inductor_mm()
_register_npu_inductor_addmm()
_register_npu_inductor_bmm()
_register_npu_inductor_grouped_mm()
_register_npu_inductor_flex_attention()

# Step 5: 应用所有 patch
patch_codegen_with_cpp_wrapper()
patch_get_cpp_torch_device_options()
patch_device_to_aten()
patch_fallback_kernel_codegen()
patch_aot_code_compiler_compile()
patch_scheduler()
patch_triton_scheduling()
patch_is_gpu()
patch_has_triton()
disable_foreach()
# ... 更多 patch
```

#### v2 初始化流程

```python
# v2: 通过环境变量 TORCHINDUCTOR_NPU_VERSION=v2

def init_npu_inductor_v2():
    # 更结构化的初始化
    _warm_pool()
    _disable_comprehensive_padding()
    _disable_foreach()
    _register_npu_backend()           # 使用 NPUCombinedSchedulingV2
    _register_npu_device_op_overrides()
    _patch_is_gpu()
    _patch_has_triton()
    _register_npu_lowerings()
    _register_npu_decompositions()
    _register_npu_kernels()
    # ... 各种 patch
```

### 2. Patch 分类（A-E 框架）

| 类别 | 说明 | 重构策略 | 示例 |
|------|------|----------|------|
| **A-class** | 可用原生扩展 API 替代 | 用 `register_backend_for_device`, `BackendFeature` | `register_backend_for_device('npu', ...)` ✅ |
| **B-class** | 可用子类 + 方法覆盖替代 | 创建子类，覆盖方法 | `NPUWrapperCodeGen` 继承 `PythonWrapperCodegen` |
| **C-class** | 需要 NPU 特有子类体系 | 保留为 NPU 业务模块 | `CppWrapperNpu`, `NPUCachingAutotuner` |
| **D-class** | 需要上游 PyTorch PR | 记录为技术债 | 部分 scheduler patch |
| **E-class** | 包装/注册 patch | 内联注册逻辑 | `patch_device_to_aten` |

### 3. 完整 Patch 清单

#### 代码生成层 Patch

| Patch 函数 | 所在文件 | Patch 目标 | 功能 |
|---|---|---|---|
| `patch_simplify` | codegen/_sizevars.py | `SizeVarAllocator.simplify` | NPU 符号简化逻辑 |
| `patch_loop_body` | codegen/ir.py | `LoopBody.__call__` | NPU subprocess 上下文管理 |
| `patch_indexing` | codegen/ir.py | `CaptureIndexing`/`SimplifyIndexing` | 添加 NPU 间接内存操作 |
| `patch_gen_common_triton_ext_imports` | codegen/triton.py | `gen_common_triton_ext_imports` | 注入 NPU import |
| `patch_triton_scheduling` | codegen/triton.py | `TritonScheduling.select_index_dtype` | 强制 tl.int32 索引 |
| `patch_device_to_aten` | codegen/cpp_utils.py | `DEVICE_TO_ATEN` | 注册 "npu"→"at::kPrivateUse1" |

#### IR 层 Patch

| Patch 函数 | 所在文件 | Patch 目标 | 功能 |
|---|---|---|---|
| `patch_fallback_kernel_codegen` | ir.py | `FallbackKernel.codegen` | cpp_wrapper 下用 proxy executor |
| `patch_num_splits` | ir.py | `Reduction.num_splits` | NPU reduction 分裂策略 |

#### 运行时层 Patch

| Patch 函数 | 所在文件 | Patch 目标 | 功能 |
|---|---|---|---|
| `patch_create_device_properties` | runtime/__init__.py | `DeviceProperties.create` | NPU 设备属性 |
| `patch_load_cached_autotuning` | runtime/__init__.py | `load_cached_autotuning` | 包装 autotune 缓存 |
| `patch_triton_heuristics_cached_autotune` | runtime/triton_heuristics.py | `cached_autotune` | 替换为 NPUCachingAutotuner |

#### 工具/基础设施层 Patch

| Patch 函数 | 所在文件 | Patch 目标 | 功能 |
|---|---|---|---|
| `patch_is_gpu` | utils.py | `GPU_TYPES` | 添加 "npu" 到 GPU 类型 |
| `patch_has_triton` | utils.py | `has_triton()` | NPU 通过 triton 检查 |
| `disable_foreach` | utils.py | `Scheduler.create_foreach_nodes` | 禁用 foreach |
| `patch_scheduler` | scheduler.py | `Scheduler` 多个方法 | 融合距离/策略调整 |
| `patch_count_bytes` | graph.py | `GraphLowering.count_bytes` | 字节计数逻辑 |
| `patch_codegen_with_cpp_wrapper` | graph.py | `GraphLowering.codegen_with_cpp_wrapper` | cpp_wrapper 入口 |
| `patch_run_node` | graph.py | `GraphLowering.run_node` | 节点执行逻辑 |
| `patch_aot_code_compiler_compile` | codecache.py | `AotCodeCompiler.compile` | AOT 编译适配 |
| `patch_cache_base_get_system` | codecache.py | `CacheBase.get_system` | 缓存 key 加 NPU 信息 |
| `patch_get_cpp_torch_device_options` | cpp_builder.py | `get_cpp_torch_device_options` | NPU 库链接 |
| `patch_get_optimization_cflags` | cpp_builder.py | `_get_optimization_cflags` | C++ 编译优化级别 |
| `patch_async_compile` | async_compile.py | `AsyncCompile` | 添加 catlass 异步编译 |
| `patch_tuning_process` | autotune_process.py | `CUDA_VISIBLE_DEVICES` | 替换为 ASCEND_RT_VISIBLE_DEVICES |
| `patch_algorithm_selector` | select_algorithm.py | `AlgorithmSelectorCache` | 支持 CATLASS benchmark |

### 4. `codegen/npu_combined_scheduling.py` — NPU 调度核心

源码位置: `c:\inductor\npu-pytorch\torch_npu\_inductor\codegen\npu_combined_scheduling.py`

```python
class NPUCombinedScheduling:
    """NPU 组合调度器 - 三种调度策略"""

    def __init__(self, scheduler):
        # 三种子调度器
        self.triton_scheduling = NPUTritonScheduling(scheduler)        # 线性模式
        self.nolinear_scheduling = NPUNoLinearTritonScheduling(scheduler)  # 非线性模式
        self.catlass_scheduling = CATLASSScheduling(scheduler)         # CATLASS GEMM

    def choose_node_backend(self, node):
        """判断节点使用 CATLASS 还是 Triton"""
        if isinstance(node.node, ir.TemplateBuffer):
            if use_catlass_template(node):
                return "catlass"
        return "triton"

    def can_fuse_vertical(self, node1, node2):
        """垂直融合判断 (生产者-消费者)"""
        # 委托给对应的子调度器

    def can_fuse_horizontal(self, node1, node2):
        """水平融合判断 (兄弟节点)"""
        # 委托给对应的子调度器

    def codegen_node(self, node):
        """代码生成"""
        # Ascend 950: 尝试线性 codegen, 失败则回退到非线性
```

### 5. `codegen/cpp_wrapper.py` — CppWrapperNpu

源码位置: `c:\inductor\npu-pytorch\torch_npu\_inductor\codegen\cpp_wrapper.py`

```python
class CppWrapperNpu(CppWrapperGpu):
    """NPU C++ wrapper 代码生成器"""

    # 关键差异 (vs CppWrapperGpu):
    # 1. 使用 rtKernelLaunch / rtKernelLaunchWithFlagV2 启动 kernel
    # 2. 参数结构体包含 NPU 特有字段 (FFTS 地址, sync_block_lock, workspace)
    # 3. DTYPE 映射: bool→int32_t, float16→float, bfloat16→float
    # 4. 使用 torch_npu/csrc/inductor/aoti_runtime/ 下的 NPU 运行时头文件

    class DeferredNpuTritonCallWrapper:
        """NPU Triton kernel 延迟加载和启动"""
        def generate_grid(self):
            # 使用 GridExprNpu 生成 NPU 特有的 grid 计算
        def generate_load_kernel(self):
            # 加载 NPU cubin (包含 mix_mode 参数)
        def generate_launch_kernel(self):
            # 使用 rtKernelLaunch 启动 NPU kernel
```

### 6. `runtime/triton_heuristics.py` — NPUCachingAutotuner

源码位置: `c:\inductor\npu-pytorch\torch_npu\_inductor\runtime\triton_heuristics.py`

```python
class NPUCachingAutotuner(CachingAutotuner):
    """NPU 的 autotuner 实现"""

    def _precompile_config(self, config, ...):
        """编译 Triton kernel, 添加 NPU 特有编译选项"""
        # NPU 特有选项:
        #   compile_mode: 编译模式
        #   multibuffer: 多缓冲
        #   enable_vf_fusion: 向量融合
        #   simt_stack_limit: SIMT 栈限制

    def save_npu_kernel(self, ...):
        """保存编译后的 NPU kernel (npubin 格式)"""
        # NPU 特有元数据:
        #   mix_mode: SIMD/SIMT 混合模式
        #   parallel_mode: 并行模式
        #   force_simt_only: 强制 SIMT 模式

    def run(self, *args, **kwargs):
        """运行 kernel"""
        # 支持 debug 模式 (数据 dump, 精度校验)
        # 支持 msprobe 精度校验
```

### 7. `ir.py` — NPU 自定义 IR 节点

源码位置: `c:\inductor\npu-pytorch\torch_npu\_inductor\ir.py`

```python
class ConcatKernel(NopKernel):
    """高效的 concat 实现"""
    # 两种模式:
    #   1. reindex: 逐元素 Pointwise
    #   2. split tiling: cat_insert_slice / cat_store

class IndexputTemplate(Scatter):
    """支持 indexput_template 间接写入操作"""

class ScatterTemplate(Scatter):
    """支持 scatter_template 间接写入操作"""
```

### 8. `lowering.py` — NPU 算子 Lowering

源码位置: `c:\inductor\npu-pytorch\torch_npu\_inductor\lowering.py`

```python
# NPU 特有 lowering:
#   aten.cat        → ConcatKernel (split tiling 模式)
#   aten.gather     → gather_template
#   aten.index_put  → IndexputTemplate
#   aten.scatter_reduce_ → ScatterTemplate
#   aten.index      → index_select 模板
#   aten.embedding  → index_select 模板
#   aten.pow        → 优化整数幂和特殊幂值
```

### 9. `kernel/mm.py` — CATLASS GEMM 模板

源码位置: `c:\inductor\npu-pytorch\torch_npu\_inductor\kernel\mm.py`

```python
def _register_npu_inductor_mm():
    """注册 aten.mm lowering, 加入 CATLASS 模板选择"""
    @register_lowering(aten.mm)
    def mm(x, y):
        choices = [
            TritonMMTemplate(...),       # Triton mm kernel
            CATLASS1xGemmTemplate(...),  # CATLASS GEMM 模板
        ]
        return autotune_select_algorithm(choices, [x, y])
```

## v1 vs v2 架构对比

| 特性 | v1 | v2 |
|------|----|----|
| 初始化方式 | 过程式 patch 调用 | 结构化函数组织 |
| 目录结构 | 扁平 | 模块化 (backend/, ir/, graph/, lowering/, scheduler/) |
| Autotuner 注入 | 替换类方法 | 使用 `autotuner_cls` 参数 |
| BackendFeature | 未声明 | 明确声明 (INPLACE_BUFFERS, TRITON_TEMPLATES 等) |
| NPUFallbackKernel | 内联在 patch 中 | 独立类 |
| NPUGraphLowering | 内联在 patch 中 | 独立类 |

## NPU 特有概念

### NPU Kernel Type

```python
class NPUKernelType(Enum):
    SIMD = "simd"              # SIMD 模式 (向量计算)
    SIMT_ONLY = "simt_only"    # 仅 SIMT 模式 (标量计算)
    SIMT_TEMPLATE = "simt_template"  # SIMT 模板
    SIMD_SIMT_MIX = "simd_simt_mix"  # SIMD + SIMT 混合模式
```

### NPU 间接内存操作

NPU 硬件支持特殊的间接内存访问操作：

```python
# index_select: 间接读取
# gather_template: 间接收集
# cat_insert_slice / cat_store: 间接写入 (concat 优化)
# indexput_template: 间接写入 (index_put 优化)
# scatter_template: 间接散射
```

### CATLASS GEMM 库

CATLASS 是华为自研的 GEMM 模板库，作为 Triton mm kernel 的替代选择：

```
mm 算子 → autotune 选择
            ├── Triton mm kernel
            └── CATLASS1xGemmTemplate  ← 华为自研
```

## 学习检查点

- [ ] 能画出 NPU Inductor 的架构全景图
- [ ] 理解 v1 和 v2 架构的区别
- [ ] 能对每个 patch 进行 A-E 分类
- [ ] 理解 NPUCombinedScheduling 的三种调度策略
- [ ] 知道 CppWrapperNpu 与 CppWrapperGpu 的关键差异
- [ ] 理解 NPUCachingAutotuner 的 NPU 特有编译选项
- [ ] 知道 NPU 间接内存操作的作用
- [ ] 理解 CATLASS GEMM 模板的选择机制

## PyTorch Inductor 原生扩展 API 参考

为自定义硬件适配 Inductor 时，优先使用这些原生 API：

```python
# 1. 注册设备后端
from torch._inductor.codegen.common import register_backend_for_device
register_backend_for_device(
    device="my_device",
    scheduling_cls=MyScheduling,
    wrapper_codegen_cls=MyWrapperCodeGen,
    cpp_wrapper_cls=MyCppWrapper,
)

# 2. 注册设备操作覆盖
from torch._inductor.codegen.common import register_device_op_overrides
register_device_op_overrides("my_device", MyDeviceOpOverrides)

# 3. 声明后端特性
from torch._inductor.codegen.common import BackendFeature, get_backend_features
# 在 Scheduling 类中实现 get_backend_features()

# 4. 注册算子 lowering
from torch._inductor.lowering import register_lowering
@register_lowering(aten.my_op)
def my_op_lowering(*args):
    ...

# 5. 注册算子分解
from torch._inductor.decomposition import register_decomposition
@register_decomposition([aten.my_op])
def my_op_decomp(x):
    ...
```
