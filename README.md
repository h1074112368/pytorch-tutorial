# PyTorch 源码完整学习指南

> 基于 PyTorch 源码（`c:\inductor\pytorch`）和 NPU Inductor 源码（`c:\inductor\npu-pytorch`）

## 架构全景图

```
                    ┌─────────────────────────────────────┐
                    │         用户代码 (Python)            │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │         torch.nn / torch.optim       │
                    │    (神经网络模块 & 优化器)            │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼─────────┐  ┌──────▼──────┐  ┌──────────▼──────────┐
    │   torch.autograd   │  │ torch.cuda  │  │  torch.distributed  │
    │   (自动微分)        │  │ (CUDA支持)  │  │  (分布式训练)        │
    └─────────┬─────────┘  └──────┬──────┘  └──────────┬──────────┘
              │                    │                     │
    ┌─────────▼────────────────────▼─────────────────────▼──────────┐
    │                      torch._C (C++ 绑定)                       │
    │                    torch/csrc/ (C++ 扩展层)                    │
    └─────────┬────────────────────┬────────────────────┬──────────┘
              │                    │                     │
    ┌─────────▼─────────┐  ┌──────▼──────┐  ┌──────────▼──────────┐
    │       ATen         │  │    c10      │  │    torchgen          │
    │  (张量操作库)       │  │ (核心库)    │  │  (代码生成)          │
    └───────────────────┘  └─────────────┘  └─────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │                   编译栈 (PT2)                                │
    │                                                              │
    │  torch._dynamo ──► torch.fx ──► torch._decomp ──► torch._inductor │
    │  (字节码分析)    (图IR)    (算子分解)      (代码生成)          │
    │                                                              │
    │  torch.export ──► ExportedProgram ──► AOTInductor            │
    │  (AOT导出)      (可序列化程序)     (AOT编译)                  │
    └──────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │                   函数变换层                                   │
    │                                                              │
    │  torch._functorch / functorch                                │
    │  (vmap, grad, vjp, jvp, functionalize)                       │
    │                                                              │
    │  torch._prims ──► torch._refs                                │
    │  (原语操作)      (参考实现)                                   │
    └──────────────────────────────────────────────────────────────┘
```

## 学习阶段总览

| 阶段 | 模块 | 核心内容 | 预计时间 |
|------|------|----------|----------|
| **Part A** | C++ 底层基础 | c10 核心库 + ATen 张量库 + Dispatcher 调度 | 2周 |
| **Part B** | Python 核心层 | Tensor + Autograd + nn.Module + Optim | 2周 |
| **Part C** | 编译栈 (PT2) | Dynamo + FX + Decomposition + Inductor + Export | 3周 |
| **Part D** | 函数变换层 | functorch + prims + refs + decomposition | 1周 |
| **Part E** | 基础设施 | torchgen + CUDA + Distributed + Profiler | 1周 |
| **Part F** | NPU 扩展 | NPU Inductor 适配与重构 | 1周 |

## 目录结构

```
learning/
├── README.md                              # 本文件
│
├── part_a_foundation/                     # Part A: C++ 底层基础
│   ├── guide.md                           # 总体文档
│   ├── a1_c10_core/                       # c10 核心库
│   │   ├── guide.md
│   │   └── demo_c10.py
│   ├── a2_aten_tensor/                    # ATen 张量库
│   │   ├── guide.md
│   │   └── demo_aten.py
│   └── a3_dispatcher/                     # Dispatcher 调度器
│       ├── guide.md
│       └── demo_dispatcher.py
│
├── part_b_python_core/                    # Part B: Python 核心层
│   ├── guide.md
│   ├── b1_tensor/                         # torch.Tensor
│   │   ├── guide.md
│   │   └── demo_tensor.py
│   ├── b2_autograd/                       # 自动微分
│   │   ├── guide.md
│   │   └── demo_autograd.py
│   ├── b3_nn_module/                      # nn.Module
│   │   ├── guide.md
│   │   └── demo_nn.py
│   └── b4_optimizer/                      # 优化器
│       ├── guide.md
│       └── demo_optimizer.py
│
├── part_c_compile_stack/                  # Part C: 编译栈 (PT2)
│   ├── guide.md
│   ├── c1_dynamo/                         # TorchDynamo
│   │   ├── guide.md
│   │   └── demo_dynamo.py
│   ├── c2_fx_graph/                       # FX Graph
│   │   ├── guide.md
│   │   └── demo_fx.py
│   ├── c3_aot_autograd/                   # AOTAutograd
│   │   ├── guide.md
│   │   └── demo_aot_autograd.py
│   ├── c4_decomposition/                  # 算子分解
│   │   ├── guide.md
│   │   └── demo_decomposition.py
│   ├── c5_inductor/                       # Inductor (已有 stage1-8)
│   │   └── guide.md                       # 指向 stage1-8 的索引
│   └── c6_export/                         # torch.export
│       ├── guide.md
│       └── demo_export.py
│
├── part_d_functional/                     # Part D: 函数变换层
│   ├── guide.md
│   ├── d1_functorch/                      # functorch
│   │   ├── guide.md
│   │   └── demo_functorch.py
│   ├── d2_prims_refs/                     # prims + refs
│   │   ├── guide.md
│   │   └── demo_prims_refs.py
│   └── d3_decomp_system/                  # 分解系统
│       ├── guide.md
│       └── demo_decomp_system.py
│
├── part_e_infrastructure/                 # Part E: 基础设施
│   ├── guide.md
│   ├── e1_torchgen/                       # 代码生成
│   │   ├── guide.md
│   │   └── demo_torchgen.py
│   ├── e2_cuda_system/                    # CUDA 系统
│   │   ├── guide.md
│   │   └── demo_cuda.py
│   ├── e3_distributed/                    # 分布式训练
│   │   ├── guide.md
│   │   └── demo_distributed.py
│   └── e4_profiler/                       # 性能分析
│       ├── guide.md
│       └── demo_profiler.py
│
├── part_f_npu/                            # Part F: NPU 扩展
│   ├── guide.md
│   └── f1_npu_inductor/                   # NPU Inductor (已有 stage8)
│       └── guide.md                       # 指向 stage8 的索引
│
├── stage1_compile_entry/                  # [已有] Inductor 阶段1
├── stage2_ir/                             # [已有] Inductor 阶段2
├── stage3_lowering/                       # [已有] Inductor 阶段3
├── stage4_scheduler/                      # [已有] Inductor 阶段4
├── stage5_codegen/                        # [已有] Inductor 阶段5
├── stage6_runtime/                        # [已有] Inductor 阶段6
├── stage7_fx_passes/                      # [已有] Inductor 阶段7
└── stage8_npu_inductor/                   # [已有] Inductor 阶段8
```

## 推荐学习路径

```
第1-2周: Part A (C++ 底层)
    c10 核心类型 → ATen 张量操作 → Dispatcher 调度机制

第3-4周: Part B (Python 核心)
    Tensor → Autograd → nn.Module → Optimizer

第5-7周: Part C (编译栈)
    Dynamo → FX → AOTAutograd → Decomposition → Inductor (stage1-8) → Export

第8周:   Part D (函数变换)
    functorch → prims/refs → 分解系统

第9周:   Part E (基础设施)
    torchgen → CUDA → Distributed → Profiler

第10周:  Part F (NPU 扩展)
    NPU Inductor (stage8) + 原生扩展 API
```

## 核心源码文件索引

### C++ 层

| 目录 | 核心文件 | 说明 |
|------|----------|------|
| `c10/core/` | `TensorImpl.h`, `StorageImpl.h`, `DispatchKey.h`, `SymInt.h` | 核心类型定义 |
| `c10/util/` | `ArrayRef.h`, `Half.h`, `BFloat16.h`, `intrusive_ptr.h` | 通用工具 |
| `aten/src/ATen/core/` | `Tensor.h`, `TensorBase.h`, `dispatch/Dispatcher.h` | ATen 核心 |
| `aten/src/ATen/native/` | `BinaryOps.cpp`, `ReduceOps.cpp`, `Convolution.cpp` | 原生操作实现 |
| `torch/csrc/` | `Module.cpp`, `autograd/`, `jit/`, `dynamo/` | Python-C++ 桥梁 |

### Python 层

| 目录 | 核心文件 | 说明 |
|------|----------|------|
| `torch/` | `__init__.py`, `_tensor.py`, `_ops.py` | 顶层 API |
| `torch/autograd/` | `function.py`, `graph.py`, `grad_mode.py` | 自动微分 |
| `torch/nn/` | `modules/module.py`, `functional.py`, `parameter.py` | 神经网络 |
| `torch/optim/` | `optimizer.py`, `adam.py`, `lr_scheduler.py` | 优化器 |
| `torch/_dynamo/` | `eval_frame.py`, `symbolic_convert.py`, `guards.py` | Dynamo |
| `torch/fx/` | `graph.py`, `node.py`, `proxy.py`, `interpreter.py` | FX 图 |
| `torch/_inductor/` | `ir.py`, `lowering.py`, `scheduler.py`, `compile_fx.py` | Inductor |
| `torch/_functorch/` | `apis.py`, `vmap.py`, `eager_transforms.py` | 函数变换 |
| `torch/_prims/` | `__init__.py`, `context.py`, `executor.py` | 原语操作 |
| `torch/_refs/` | `__init__.py` | 参考实现 |
| `torch/_decomp/` | `__init__.py` | 算子分解 |
| `torch/export/` | `exported_program.py`, `_trace.py` | 模型导出 |
| `torchgen/` | `model.py`, `gen.py`, `api/cpp.py` | 代码生成 |

## 调试技巧

```bash
# PyTorch 全局调试
TORCH_LOGS="+dynamo" python your_script.py     # Dynamo 日志
TORCH_LOGS="+aot" python your_script.py        # AOTAutograd 日志
TORCH_LOGS="+ir" python your_script.py         # Inductor IR 日志
TORCH_LOGS="+output_code" python your_script.py # 生成的代码
TORCH_LOGS="+fusion" python your_script.py     # 融合决策
TORCH_LOGS="+schedule" python your_script.py   # 调度信息

# Autograd 调试
TORCH_AUTOGRAD_TRACE=1 python your_script.py   # Autograd 追踪

# CUDA 调试
CUDA_LAUNCH_BLOCKING=1 python your_script.py   # 同步 CUDA 执行

# Dynamo 调试
TORCHDYNAMO_VERBOSE=1 python your_script.py    # Dynamo 详细日志
TORCHDYNAMO_REPRO_AFTER="aot" python your_script.py  # 生成 minifier

# Python 代码方式
import torch._logging
torch._logging.set_logs(dynamo=True, aot=True, ir=True)
```
