# Part F: NPU 扩展

## 概述

NPU Inductor 是华为昇腾 NPU 对 PyTorch Inductor 的扩展实现，展示了如何为自定义硬件适配编译栈。

## 学习路径

本部分内容已在 [stage8_npu_inductor](../../stage8_npu_inductor/guide.md) 中详细覆盖，包括：

- NPU Inductor 架构全景
- Patch A-E 分类框架
- v1 vs v2 架构对比
- NPU 调度策略 (线性/非线性/CATLASS)
- CppWrapperNpu 代码生成
- NPUCachingAutotuner 运行时
- PyTorch Inductor 原生扩展 API

## 关键学习目标

1. **理解适配模式**: NPU 通过 monkey-patch + 原生 API 两种方式适配 PyTorch Inductor
2. **掌握原生扩展 API**: `register_backend_for_device`, `BackendFeature`, `register_lowering` 等
3. **理解重构方向**: v1 → v2 的演进，从 patch 到子类化
4. **硬件特性映射**: NPU 间接内存操作、CATLASS GEMM、SIMT/SIMD 模式

## 原生扩展 API 速查

```python
# 1. 注册设备后端
from torch._inductor.codegen.common import register_backend_for_device
register_backend_for_device("npu", NPUCombinedScheduling, NPUWrapperCodeGen, CppWrapperNpu)

# 2. 注册设备操作覆盖
from torch._inductor.codegen.common import register_device_op_overrides
register_device_op_overrides("npu", NPUDeviceOpOverrides)

# 3. 声明后端特性
from torch._inductor.codegen.common import BackendFeature
class NPUCombinedScheduling(BaseScheduling):
    @classmethod
    def get_backend_features(cls, device):
        return {BackendFeature.INPLACE_BUFFERS, BackendFeature.TRITON_TEMPLATES}

# 4. 注册算子 lowering
from torch._inductor.lowering import register_lowering
@register_lowering(aten.my_op)
def my_op_lowering(*args): ...

# 5. 注册算子分解
from torch._inductor.decomposition import register_decomposition
@register_decomposition([aten.my_op])
def my_op_decomp(x): ...
```

## 详见

- [stage8_npu_inductor/guide.md](../../stage8_npu_inductor/guide.md) — 完整的 NPU Inductor 学习文档
- [stage8_npu_inductor/demo_npu_inductor.py](../../stage8_npu_inductor/demo_npu_inductor.py) — 可运行示例
