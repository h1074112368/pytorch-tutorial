# 阶段6: 运行时与 Autotune

## 核心问题

编译后的代码如何在运行时执行？Autotune 如何选择最优的 tiling 配置？

## 运行时总览

```
CompiledFxGraph.__call__(inputs)
    │
    ├── 1. 输入对齐检查 (alignment check)
    ├── 2. 输入复制 (misaligned inputs)
    ├── 3. 调用 current_callable(inputs)
    │     │
    │     ▼
    │   Python Wrapper / C++ Wrapper
    │     ├── 内存分配 (torch.empty)
    │     ├── Kernel 调用
    │     │     ├── Triton kernel: kernel.run(...)
    │     │     └── Extern kernel: torch._C.xxx(...)
    │     └── 返回输出
    │
    └── 4. CUDA Graph 管理 (可选)
```

## 关键源码文件

### 1. `runtime/triton_heuristics.py` — 启发式调度与 Autotune

源码位置: `c:\inductor\pytorch\torch\_inductor\runtime\triton_heuristics.py`

#### CachingAutotuner

```python
class CachingAutotuner:
    """带缓存的 Autotuner"""
    # 核心: 在运行时选择最优的 tiling 配置

    def __init__(self, fn, configs, key, ...):
        self.fn = fn              # Triton kernel 函数
        self.configs = configs    # 候选 tiling 配置列表
        self.key = key            # 缓存键函数

    def run(self, *args, **kwargs):
        """运行 kernel"""
        # 1. 计算缓存键
        key = self.key(*args, **kwargs)

        # 2. 查找缓存
        if key in self.cache:
            config = self.cache[key]
        else:
            # 3. 首次运行: benchmark 所有配置
            config = self.benchmark_all_configs(*args, **kwargs)
            self.cache[key] = config

        # 4. 使用最优配置运行
        self.fn.run(*args, **kwargs, **config)
```

#### 启发式 Tiling 配置

```python
def pointwise_heuristic(size):
    """为逐元素操作生成 tiling 配置"""
    # 根据 size 选择合适的 XBLOCK
    if size < 1024:
        return {"XBLOCK": 256}
    elif size < 65536:
        return {"XBLOCK": 512}
    else:
        return {"XBLOCK": 1024}

def reduction_heuristic(size, reduction_size):
    """为归约操作生成 tiling 配置"""
    # 根据 size 和 reduction_size 选择 XBLOCK 和 RBLOCK
    ...
```

#### Autotune 流程

```
Kernel 首次运行
    │
    ├── 1. 计算缓存键 (基于输入形状/类型)
    │
    ├── 2. 缓存未命中 → benchmark 所有配置
    │     │
    │     ├── config1: XBLOCK=256  → 运行 10 次 → 平均耗时 0.5ms
    │     ├── config2: XBLOCK=512  → 运行 10 次 → 平均耗时 0.3ms
    │     ├── config3: XBLOCK=1024 → 运行 10 次 → 平均耗时 0.2ms  ← 最优
    │     └── config4: XBLOCK=2048 → 运行 10 次 → 平均耗时 0.4ms
    │
    ├── 3. 选择最优配置 → 存入缓存
    │
    └── 4. 使用最优配置运行 kernel

后续运行 (相同形状/类型)
    │
    ├── 1. 计算缓存键
    ├── 2. 缓存命中 → 直接使用最优配置
    └── 3. 运行 kernel
```

### 2. `autotune_process.py` — Autotune 子进程

源码位置: `c:\inductor\pytorch\torch\_inductor\autotune_process.py`

```python
class TuningProcess:
    """在子进程中执行 autotune benchmark"""
    # 为什么需要子进程?
    # - 避免 benchmark 受主进程 GC 影响
    # - 可以设置 CUDA_VISIBLE_DEVICES 控制可见 GPU
    # - 隔离 benchmark 环境, 提高结果可靠性

class TuningProcessPool:
    """Autotune 子进程池"""
    # 管理多个 TuningProcess, 并行执行 benchmark
```

### 3. `codecache.py` — 代码缓存

源码位置: `c:\inductor\pytorch\torch\_inductor\codecache.py`

```python
class PyCodeCache:
    """Python 代码缓存"""
    @staticmethod
    def write(source_code, key_path):
        """将生成的 Python 代码写入磁盘"""
        # 写入到 cache_dir()/key_path.py

    @staticmethod
    def load_by_key_path(key_path):
        """从磁盘加载 Python 模块"""
        # 使用 importlib 励态加载

class FxGraphCache:
    """FX Graph 缓存 (编译结果缓存)"""
    @staticmethod
    def prepare_key(gm, inputs, ...):
        """计算缓存键"""
        # 基于 FX Graph 结构 + 输入形状/类型

    @staticmethod
    def load_with_key(key, ...):
        """加载缓存的编译结果"""
        # 反序列化 CompiledFxGraph

    @staticmethod
    def store_with_key(key, compiled_graph, ...):
        """存储编译结果到缓存"""
```

### 4. `select_algorithm.py` — 算法选择

源码位置: `c:\inductor\pytorch\torch\_inductor\select_algorithm.py`

```python
class AlgorithmSelectorCache:
    """算法选择缓存"""
    # 用于模板内核 (GEMM, Conv) 的算法选择
    # 例如: mm 可以选择 Triton kernel 或 CUTLASS 模板

    def __call__(self, choices, inputs, ...):
        """选择最优算法"""
        # 1. 检查缓存
        # 2. 未命中: benchmark 所有候选算法
        # 3. 选择最优算法
        # 4. 缓存结果
```

## CUDA Graph 机制

### 什么是 CUDA Graph？

CUDA Graph 将一系列 GPU 操作录制为一个图，后续可以整体回放，消除 kernel launch 开销。

```
普通执行:
  CPU → launch kernel1 → 等待 → launch kernel2 → 等待 → ...
  (每次 launch 有 CPU 开销)

CUDA Graph:
  CPU → 录制: kernel1 → kernel2 → ...
  CPU → 回放 graph (一次 launch, 消除多次 launch 开销)
```

### CUDA Graph 化流程

```python
def cudagraphify(model, inputs, ...):
    """将编译后的模型 CUDA Graph 化"""
    # 1. 分配静态输入张量 (地址不变)
    # 2. 预热: 运行几次以填充缓存
    # 3. 录制 CUDA Graph
    # 4. 后续运行: 回放 CUDA Graph
```

### CUDA Graph 的限制

- 输入张量的地址不能改变 (需要静态内存)
- 不支持动态形状
- 不支持某些操作 (如条件分支)

## 学习检查点

- [ ] 理解 `CachingAutotuner` 的缓存机制
- [ ] 知道启发式 tiling 配置如何生成
- [ ] 理解 Autotune 的 benchmark 流程
- [ ] 知道为什么 Autotune 需要子进程
- [ ] 理解 CUDA Graph 的原理和限制
- [ ] 知道 `PyCodeCache` 和 `FxGraphCache` 的区别

## 下一步

完成本阶段后，进入 [阶段7: FX Passes](../stage7_fx_passes/guide.md)
