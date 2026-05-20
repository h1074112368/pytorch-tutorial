# 阶段5: 代码生成

## 核心问题

IR 节点如何被翻译为可执行的 Triton kernel 和 Python/C++ wrapper 代码？

## 代码生成总览

```
Scheduler.codegen()
    │
    ▼
按设备分组节点
    │
    ├── CPU 设备 → CppScheduling.codegen()
    │     ├── codegen_nodes()    → 生成 C++/Triton kernel
    │     └── codegen_template() → 生成模板 kernel (GEMM, Conv)
    │
    └── CUDA 设备 → CUDACombinedScheduling.codegen()
          ├── TritonScheduling.codegen()
          │     ├── Pointwise → Triton kernel (逐元素)
          │     ├── Reduction → Triton kernel (归约)
          │     └── Template   → Triton 模板 kernel
          └── CUDACppScheduling.codegen()
                └── CUTLASS 模板 kernel
    │
    ▼
PythonWrapperCodegen / CppWrapperGpu
    ├── 内存分配 (alloc)
    ├── Kernel 调用 (call)
    ├── 输入/输出处理
    └── CUDA Graph 管理
    │
    ▼
PyCodeCache.write() → 写入磁盘
PyCodeCache.load_by_key_path() → 动态加载模块
```

## 关键源码文件

### 1. `codegen/common.py` — 后端注册与特性

源码位置: `c:\inductor\pytorch\torch\_inductor\codegen\common.py`

#### `register_backend_for_device()` (约第319行)

```python
def register_backend_for_device(
    device: str,
    device_scheduling: SchedulingConstructor,
    device_wrapper_codegen: WrapperConstructor,
    device_cpp_wrapper_codegen: Optional[WrapperConstructor] = None,
) -> None:
    """注册设备后端"""
    # 将设备类型映射到三个组件:
    #   1. device_scheduling: 调度策略 (如何调度和融合)
    #   2. device_wrapper_codegen: Python wrapper 代码生成器
    #   3. device_cpp_wrapper_codegen: C++ wrapper 代码生成器
    device_codegens[device] = DeviceCodegen(
        scheduling=device_scheduling,
        wrapper_codegen=device_wrapper_codegen,
        cpp_wrapper_codegen=device_cpp_wrapper_codegen,
    )
```

**默认后端注册** (`init_backend_registration()`):

| 设备 | 调度策略 | Python Wrapper | C++ Wrapper |
|------|----------|---------------|-------------|
| cpu | `CppScheduling` | `PythonWrapperCodegen` | `CppWrapperCpu` |
| cuda | `CUDACombinedScheduling` | `PythonWrapperCodegen` | `CppWrapperGpu` |
| xpu | `TritonScheduling` | `PythonWrapperCodegen` | `CppWrapperGpu` |
| mps | `MetalScheduling` | `PythonWrapperCodegen` | `CppWrapperGpu` |

#### `BackendFeature` 枚举

```python
class BackendFeature(Enum):
    """后端支持的特性"""
    FOREACH = auto()              # 支持 foreach 操作
    INPLACE_BUFFERS = auto()      # 支持原地缓冲区
    TRITON_TEMPLATES = auto()     # 支持 Triton 模板
    TUPLE_REDUCTION = auto()      # 支持元组归约
    BUCKETIZE = auto()            # 支持 bucketize
    MASKED_SCATTER_WITH_INDEX = auto()  # 支持 masked_scatter_with_index
    REDUCE_TO_SINGLE_ELEMENT = auto()   # 支持归约到单个元素
```

#### `DeviceOpOverrides` 类

```python
class DeviceOpOverrides:
    """设备操作覆盖"""
    def stream(self, stream): ...      # 获取/设置当前 stream
    def sync(self): ...                # 设备同步
    def guard(self, lock): ...         # 设备锁
    def set_device(self, device): ...  # 设置当前设备
```

### 2. `codegen/triton.py` — Triton Kernel 代码生成

源码位置: `c:\inductor\pytorch\torch\_inductor\codegen\triton.py`

#### TritonScheduling

```python
class TritonScheduling(BaseScheduling):
    """Triton 调度后端"""

    def codegen(self, *nodes):
        """为节点生成 Triton kernel"""
        for node in nodes:
            if isinstance(node, FusedSchedulerNode):
                self.codegen_fused_node(node)
            elif isinstance(node, ExternKernelSchedulerNode):
                self.codegen_extern_kernel(node)

    def codegen_fused_node(self, node):
        """生成融合的 Triton kernel"""
        # 1. 生成 kernel 函数签名
        # 2. 生成 tiling 配置 (XBLOCK, YBLOCK 等)
        # 3. 生成 kernel body (从 inner_fn 展开)
        # 4. 生成 grid 函数
        # 5. 生成 kernel 调用代码
```

#### 生成的 Triton Kernel 示例

```python
# Inductor 生成的 Triton kernel (简化版)
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
```

### 3. `codegen/wrapper.py` — Python Wrapper 代码生成

源码位置: `c:\inductor\pytorch\torch\_inductor\codegen\wrapper.py`

```python
class PythonWrapperCodegen:
    """Python wrapper 代码生成器"""

    def generate(self, is_inference=False):
        """生成完整的 Python wrapper 代码"""
        # 1. 生成 imports
        # 2. 生成 kernel 函数定义
        # 3. 生成 call() 函数:
        #    a. 输入处理 (alignment check, copy)
        #    b. 内存分配 (alloc)
        #    c. Kernel 调用
        #    d. 输出处理
        #    e. CUDA Graph 管理
```

#### 生成的 Python Wrapper 示例

```python
# Inductor 生成的 Python wrapper (简化版)
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
```

### 4. `codegen/cpp.py` — C++ Wrapper 代码生成

源码位置: `c:\inductor\pytorch\torch\_inductor\codegen\cpp.py`

C++ wrapper 用于消除 Python 开销，直接在 C++ 层调用 kernel：

```python
class CppWrapperGpu:
    """C++ wrapper 代码生成器"""

    def generate(self):
        """生成 C++ wrapper 代码"""
        # 1. 生成 #include 头文件
        # 2. 生成 kernel 函数声明
        # 3. 生成 model() 函数:
        #    a. 输入处理
        #    b. 内存分配
        #    c. Kernel 调用 (cuLaunchKernel 等)
        #    d. 输出处理
```

#### C++ Wrapper vs Python Wrapper

| 特性 | Python Wrapper | C++ Wrapper |
|------|---------------|-------------|
| 启动开销 | 较高 (Python 解释器) | 低 (直接 C++ 调用) |
| 灵活性 | 高 (动态类型) | 低 (静态类型) |
| 调试 | 容易 | 困难 |
| 适用场景 | 通用 | 高性能推理 |

### 5. `output_code.py` — 编译产物

源码位置: `c:\inductor\pytorch\torch\_inductor\output_code.py`

```python
class CompiledFxGraph(OutputCode):
    """编译后的 FX Graph"""

    current_callable: Callable     # 编译后的可调用函数
    cache_key: str                 # 缓存键
    source_code: str               # 生成的源代码
    device_types: list[str]        # 涉及的设备
    mutated_inputs: list[str]      # 被变异的输入
    constants: dict[str, Tensor]   # 编译时常量
    output_strides: list           # 输出 stride 表达式
    cudagraph_info: dict           # CUDA Graph 信息

    def __call__(self, inputs):
        """运行编译后的代码"""
        return self.current_callable(inputs)

    def post_compile(self):
        """后处理 (CUDA Graph 化等)"""
```

## 代码生成流程详解

### Step 1: GraphLowering.codegen()

```python
def codegen(self):
    """触发代码生成"""
    # 1. 初始化 wrapper 代码生成器
    self.init_wrapper_code()

    # 2. 创建/更新 Scheduler
    self._update_scheduler()

    # 3. 推入代码生成上下文
    self.wrapper_code.push_codegened_graph(self)

    # 4. 执行调度器代码生成
    self.scheduler.codegen()

    # 5. 生成最终 wrapper 代码
    self.wrapper_code.generate(self.is_inference)
```

### Step 2: compile_to_module()

```python
def compile_to_module(self):
    """将生成的代码编译为 Python 模块"""
    # 1. 获取 wrapper 代码
    wrapper_code = self.codegen()

    # 2. 写入磁盘
    PyCodeCache.write(source_code, key_path)

    # 3. 动态加载
    module = PyCodeCache.load_by_key_path(key_path)

    # 4. 返回模块的 call 函数
    return module.call
```

## 学习检查点

- [ ] 理解 `register_backend_for_device()` 的三个组件
- [ ] 知道 `BackendFeature` 枚举的各项含义
- [ ] 能解释 Triton kernel 代码生成的流程
- [ ] 理解 Python wrapper vs C++ wrapper 的区别
- [ ] 知道 `CompiledFxGraph` 的核心字段
- [ ] 能画出从 `GraphLowering.codegen()` 到 `CompiledFxGraph` 的流程

## 下一步

完成本阶段后，进入 [阶段6: 运行时与 Autotune](../stage6_runtime/guide.md)
