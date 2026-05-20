# Part C: 编译栈 (PT2)

## 概述

PyTorch 2.x 编译栈是整个框架最核心的创新，实现了从 Python 代码到高效本地代码的编译流水线。

## 编译栈全景

```
torch.compile(model)
    │
    ▼
TorchDynamo (字节码分析)
    │  捕获 Python 字节码 → FX Graph
    │  处理 graph break
    │  Guard 机制确保缓存有效性
    │
    ▼
torch.fx (图 IR)
    │  GraphModule / Node / Proxy
    │  图变换 pass
    │
    ▼
AOTAutograd (自动微分分离)
    │  分离前向/反向图
    │  功能化变异操作
    │  使用 functorch 变换
    │
    ▼
Decomposition (算子分解)
    │  高层算子 → Core ATen / Prims
    │  确保编译器只需处理基本操作
    │
    ▼
TorchInductor (代码生成)
    │  Lowering: FX Graph → IR
    │  Scheduler: 融合 + 排序
    │  Codegen: Triton / C++ kernel
    │
    ▼
CompiledFxGraph (运行时执行)
    │  Autotune: 选择最优 tiling
    │  CUDA Graph: 消除 launch 开销
    │
    ▼
torch.export (AOT 导出)
    │  ExportedProgram: 可序列化
    │  AOTInductor: 编译为共享库
    │  支持动态形状
```

## 学习路径

| 子模块 | 核心内容 | 关键源码 |
|--------|----------|----------|
| **c1_dynamo** | 字节码捕获、Graph Break、Guard | `torch/_dynamo/` |
| **c2_fx_graph** | GraphModule、Node、Proxy、Interpreter | `torch/fx/` |
| **c3_aot_autograd** | 前向/反向分离、功能化 | `torch/_functorch/aot_autograd/` |
| **c4_decomposition** | 算子分解表、Core ATen | `torch/_decomp/` |
| **c5_inductor** | 详见 stage1-8 | `torch/_inductor/` |
| **c6_export** | ExportedProgram、AOTInductor | `torch/export/` |

## 关键概念

### Graph Break

Dynamo 在遇到无法追踪的 Python 操作时，会在当前点"断开"图：

```python
@torch.compile
def forward(x):
    y = x * 2      # ← Dynamo 可以追踪
    if y.sum() > 0: # ← 数据依赖的控制流, Graph Break!
        z = y + 1   # ← 新的图
    else:
        z = y - 1
    return z

# 结果: 两个子图 + 中间的 Python 执行
```

### Guard

Guard 确保编译后的代码在相同条件下可以复用：

```python
# Guard 示例:
# - x.shape == (2, 3)
# - x.dtype == torch.float32
# - x.device == "cpu"
# - len(x) == 2
# 如果 Guard 失败 → 重新编译
```

### FakeTensor

FakeTensor 是 Meta dispatch 的产物，不分配内存，只推理形状：

```python
with FakeTensorMode():
    x = torch.randn(2, 3)  # FakeTensor, 不分配内存
    y = x + 1              # 只推理形状
    # y.shape = (2, 3), y.dtype = float32
```
