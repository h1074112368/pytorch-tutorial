# 阶段3: Lowering — FX Graph → IR

## 核心问题

每个 PyTorch 算子（`aten.*`）是如何被翻译为 Inductor IR 节点的？

## Lowering 总览

```
FX Graph (GraphModule)
    │
    │  GraphLowering.run()  ←── torch.fx.Interpreter
    │
    ├── placeholder 节点 → InputBuffer → TensorBox
    │
    ├── call_function 节点 → lowerings[aten_op](args) → IR 节点
    │     │
    │     ├── aten.add     → Pointwise(inner_fn=add_fn)
    │     ├── aten.relu    → Pointwise(inner_fn=relu_fn)
    │     ├── aten.sum     → Reduction(inner_fn=..., reduction_type="sum")
    │     ├── aten.mm      → TemplateBuffer (GEMM 模板)
    │     └── 不支持的算子  → FallbackKernel (回退 eager)
    │
    └── output 节点 → 收集 graph_outputs
```

## 关键源码文件

### 1. `graph.py` — GraphLowering 类

源码位置: `c:\inductor\pytorch\torch\_inductor\graph.py`

#### GraphLowering 核心架构

```python
class GraphLowering(torch.fx.Interpreter):
    """FX Graph → IR 的转换器"""

    # 核心数据结构
    buffers: list[ir.Buffer]           # 所有 IR 缓冲区
    operations: list[ir.Operation]     # 所有 IR 操作
    graph_inputs: dict[str, TensorBox] # 图输入
    graph_outputs: list[ir.IRNode]     # 图输出
    constants: dict[str, torch.Tensor] # 编译时常量
    name_to_buffer: dict[str, ir.Buffer] # 名称→缓冲区映射
```

#### `run()` 方法 (约第875行)

```python
def run(self, *args):
    """重写 torch.fx.Interpreter.run()"""
    # 遍历 FX Graph 的每个节点
    for node in self.graph.nodes:
        result = self.run_node(node)  # 逐节点处理
    # 收集输出
    return self.graph_outputs
```

#### `run_node()` 方法 (约第1452行)

```python
def run_node(self, n):
    """对每个 FX Node 进行 lowering"""
    if n.op == "placeholder":
        return self.placeholder(n)          # 创建 InputBuffer
    elif n.op == "call_function":
        result = super().run_node(n)        # 调用 lowerings[op]
        # 后处理:
        # - 处理 fallback
        # - 处理 Triton kernel wrapper 布局约束
        # - 处理 magic method (SymInt 运算)
        # - 强制 stride 对齐
        # - 优化 channels-last 布局
        return result
    elif n.op == "output":
        return self.output(n)               # 收集输出
```

#### `placeholder()` 方法 (约第1016行)

```python
def placeholder(self, target, args, kwargs):
    """处理 FX Graph 的输入占位符"""
    # 创建 InputBuffer 并包装为 TensorBox
    buffer = InputBuffer(target, FixedLayout(device, dtype, size, stride))
    return TensorBox(StorageBox(buffer))
```

#### `register_operation()` / `register_buffer()` (约第879/888行)

```python
def register_operation(self, op: ir.Operation):
    """将 IR 操作注册到图中"""
    self.operations.append(op)

def register_buffer(self, buffer: ir.Buffer):
    """将 IR 缓冲区注册到图中"""
    self.buffers.append(buffer)
    self.name_to_buffer[buffer.get_name()] = buffer
```

### 2. `lowering.py` — 算子 Lowering 注册

源码位置: `c:\inductor\pytorch\torch\_inductor\lowering.py`

#### `lowerings` 字典

```python
# 全局字典: aten 算子 → lowering 函数
lowerings: dict[Callable, Callable] = {}

def register_lowering(aten_op, type_promotion_kind=...):
    """装饰器: 注册 aten 算子的 lowering"""
    def decorator(fn):
        lowerings[aten_op] = fn
        return fn
    return decorator
```

#### 典型 Lowering 示例

**逐元素操作 (Pointwise)**:
```python
@register_lowering(aten.add)
def add(x, y):
    # 类型提升
    x, y = promote_types(x, y)
    return Pointwise(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=lambda index: ops.add(x.load(index), y.load(index)),
        ranges=x.get_size(),
    )
```

**归约操作 (Reduction)**:
```python
@register_lowering(aten.sum)
def sum_(x, dim=None, keepdim=False):
    if dim is None:
        # 全局归约
        return Reduction(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=lambda index: x.load(index),
            ranges=[],  # 标量输出
            reduction_ranges=x.get_size(),
            reduction_type="sum",
        )
    else:
        # 沿指定维度归约
        ...
```

**模板操作 (Template)**:
```python
@register_lowering(aten.mm)
def mm(x, y):
    # mm 使用模板内核, 不走 Pointwise 融合
    return TemplateBuffer(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=...,  # GEMM 模板
    )
```

**Fallback 操作**:
```python
# 不支持的算子回退到 eager 模式
make_fallback(aten.some_unsupported_op)
# → 创建 FallbackKernel, 运行时调用 eager 实现
```

### 3. `decomposition.py` — 算子分解

源码位置: `c:\inductor\pytorch\torch\_inductor\decomposition.py`

算子分解在 Lowering **之前**执行，将复杂算子拆分为更基本的操作：

```
aten.addmm(x, y, z)
    │  [分解]
    ▼
aten.mm(y, z) → t
aten.add(x, t)
```

**分解的好处**:
- 让 Inductor 有更多融合机会（`add + mm` 可以融合为 `addmm` 模板）
- 减少 lowering 注册的数量（只需注册基本操作）

## Lowering 的完整流程

```
FX Node: call_function aten.relu
    │
    ▼
GraphLowering.run_node(n)
    │
    ├── super().run_node(n)  ←── torch.fx.Interpreter
    │     │
    │     ├── 查找 lowerings[aten.relu]
    │     ├── 调用 lowering 函数
    │     │     └── relu(x) → Pointwise(inner_fn=relu_fn)
    │     │
    │     └── 返回 TensorBox(StorageBox(ComputedBuffer(...)))
    │
    ├── 后处理:
    │     ├── 检查是否 fallback
    │     ├── 处理布局约束
    │     └── 优化 stride 对齐
    │
    └── 注册到 graph: register_buffer(), register_operation()
```

## 类型提升 (Type Promotion)

Lowering 中一个重要的辅助机制是类型提升：

```python
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND

@register_lowering(aten.add, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def add(x, y):
    x, y = promote_types(x, y)  # 自动类型提升
    ...
```

| 提升策略 | 说明 |
|----------|------|
| `DEFAULT` | 标准类型提升 (int + float → float) |
| `INT_TO_FLOAT` | 整数提升为浮点 |
| `ALWAYS_BOOL` | 结果总是布尔 |
| `COMPLEX_TO_FLOAT` | 复数提升为浮点 |

## Fallback 机制

当某个算子没有注册 lowering 时，使用 Fallback 机制：

```python
def make_fallback(aten_op):
    """注册一个 fallback 算子"""
    def fallback_fn(*args):
        return FallbackKernel(aten_op, *args)
    lowerings[aten_op] = fallback_fn
```

**Fallback 的代价**:
- 无法参与融合
- 需要额外的内存分配
- 运行时调用 eager 实现

**Fallback 的适用场景**:
- 算子使用频率低，不值得优化
- 算子语义复杂，难以用 IR 表示

## 学习检查点

- [ ] 理解 `GraphLowering` 继承 `torch.fx.Interpreter` 的设计
- [ ] 能解释 `lowerings` 字典如何将 `aten.*` 映射到 IR 构造函数
- [ ] 能写出 `aten.add` 和 `aten.sum` 的 lowering 伪代码
- [ ] 理解 `make_fallback()` 的作用和代价
- [ ] 知道算子分解与 lowering 的关系
- [ ] 理解 `inner_fn` 闭包如何实现延迟求值

## 下一步

完成本阶段后，进入 [阶段4: Scheduler](../stage4_scheduler/guide.md)
