# c1: TorchDynamo

## 核心问题

Dynamo 如何通过字节码分析将 Python 代码转换为 FX Graph？

## Dynamo 架构

```
torch.compile(model)
    │
    ▼
optimize(backend="inductor")
    │
    ▼
OptimizeContext.__enter__()
    │  注册 CPython 帧评估回调 (PEP 523)
    │
    ▼
convert_frame()
    │  将 Python 帧转换为编译后的代码
    │
    ▼
InstructionTranslator
    │  逐条解释字节码指令
    │  将 Tensor 操作记录到 OutputGraph
    │  遇到不支持的操作 → Graph Break
    │
    ▼
OutputGraph → FX Graph (GraphModule)
    │
    ▼
backend(gm, example_inputs)  → Inductor 编译
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `eval_frame.py` | 入口: `optimize()`, `OptimizeContext` |
| `convert_frame.py` | 帧转换: `ConvertFrameAssert` |
| `symbolic_convert.py` | 符号转换: `InstructionTranslator` |
| `output_graph.py` | FX Graph 构建: `OutputGraph` |
| `guards.py` | Guard 机制: `GuardManager` |
| `bytecode_transformation.py` | 字节码重写 |
| `trace_rules.py` | 追踪规则 |
| `config.py` | 配置选项 |

## Graph Break 机制

```python
# 会导致 Graph Break 的操作:
# 1. 数据依赖的控制流
if x.sum() > 0:  # x 的值在运行时才知道

# 2. 不支持的 Python 特性
import pdb; pdb.set_trace()

# 3. 非 Tensor 操作的副作用
print(x)  # 带有副作用的操作

# 4. 动态类型变化
x = x if flag else x.numpy()
```

## 学习检查点

- [ ] 理解 Dynamo 通过 PEP 523 钩入 Python 解释器
- [ ] 知道 InstructionTranslator 如何将字节码转换为 FX Graph
- [ ] 理解 Graph Break 的触发条件和处理方式
- [ ] 知道 Guard 机制如何确保缓存有效性
- [ ] 理解 FakeTensor 在 Dynamo 中的作用
