# e1: torchgen — 代码生成

## 核心问题

torchgen 如何从声明式定义自动生成 C++/Python 代码？

## torchgen 架构

```
native_functions.yaml (操作声明, ~2000+ 个算子)
derivatives.yaml (导数定义)
    │
    ▼ torchgen
    ├── ATen C++ 内核注册代码
    ├── Python 绑定代码 (torch/_C/_torch_docs.py 等)
    ├── Autograd 导数代码
    ├── Dispatcher 注册代码
    ├── Meta 内核代码
    ├── Functionalization 代码
    ├── Lazy IR 代码
    └── AOT Inductor C shim 代码
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `torchgen/model.py` | 数据模型 (NativeFunction, FunctionSchema, DispatchKey) |
| `torchgen/gen.py` | 主代码生成入口 |
| `torchgen/api/cpp.py` | C++ API 生成 |
| `torchgen/api/python.py` | Python API 生成 |
| `torchgen/api/dispatcher.py` | 调度器代码生成 |

## native_functions.yaml 示例

```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  dispatch:
    CPU: add_tensor_cpu
    CUDA: add_tensor_cuda
  structured: True
```

## 学习检查点

- [ ] 理解 torchgen 的输入和输出
- [ ] 知道 native_functions.yaml 的格式
- [ ] 理解代码生成如何保持一致性
