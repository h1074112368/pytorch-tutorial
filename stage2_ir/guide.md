# 阶段2: IR 中间表示（最核心）

## 核心问题

Inductor IR 是如何表示张量计算的？为什么采用分层 Box 模型？

## IR 层次结构 — 分层 Box 模型

这是理解 Inductor 的**最关键概念**：

```
TensorBox (顶层，对应 torch.Tensor)
  │  持有 data: StorageBox
  │  职责: 处理视图关系 (reshape, expand, permute...)
  │
  ├── View 层 (零拷贝视图变换)
  │     ├── ExpandView   (广播扩展)
  │     ├── PermuteView  (维度重排)
  │     ├── SqueezeView  (压缩维度)
  │     ├── SliceView    (切片)
  │     ├── ReinterpretView (重新解释内存)
  │     └── DtypeView    (数据类型转换)
  │
  └── StorageBox (存储层)
     │  持有 data: Buffer
     │  职责: 处理布局优化 (stride order)
     │
     └── Buffer (1D 内存分配)
           │  职责: 实际的内存分配和计算
           │
           ├── InputBuffer       (图输入)
           ├── ComputedBuffer    (计算产生)
           ├── ConstantBuffer    (编译时常量)
           ├── TemplateBuffer    (模板内核)
           ├── TritonTemplateBuffer (Triton 模板)
           └── MultiTemplateBuffer  (多模板选择)
```

### 为什么需要三层？

| 层级 | 解决的问题 | 示例 |
|------|-----------|------|
| **TensorBox** | 视图变换是零拷贝的，不需要重新计算 | `x.view(2,3)` 只创建 View，不复制数据 |
| **StorageBox** | 布局优化（stride order）可以延迟决策 | 编译时可以选择最优的内存布局 |
| **Buffer** | 实际的内存分配和计算逻辑 | `Pointwise`/`Reduction` 定义如何计算 |

### 关键设计: `inner_fn` 闭包

```python
# Pointwise IR 节点的创建
def add_lowering(x, y):
    return Pointwise(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=lambda index: ops.add(x.load(index), y.load(index)),
        #      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #      inner_fn: 闭包捕获计算逻辑，延迟到代码生成时才展开
        ranges=x.get_size(),
    )
```

**`inner_fn` 的设计精髓**:
- 它是一个闭包，捕获了输入 `x` 和 `y`
- 接受 `index` 参数，返回该索引位置的计算结果
- 在代码生成阶段，才会被展开为实际的 Triton/C++ 代码
- 这使得**融合**变得自然：多个 Pointwise 的 `inner_fn` 可以嵌套组合

## 核心 IR 节点详解

源码位置: `c:\inductor\pytorch\torch\_inductor\ir.py` (约 7000+ 行)

### 1. 计算节点

#### Pointwise (约第920行)

```python
class Pointwise(Loops):
    """逐元素操作: add, relu, sigmoid, multiply..."""

    def __init__(self, device, dtype, inner_fn, ranges):
        # inner_fn: (index) -> value  闭包
        # ranges: 输出张量的形状

    def make_scheduler(self, ...):
        # 创建 SchedulerNode，用于调度
```

**融合示例**:
```python
# x + y 的 Pointwise
inner_fn_add = lambda idx: ops.add(x.load(idx), y.load(idx))

# relu(x + y) 的 Pointwise (融合后)
inner_fn_fused = lambda idx: ops.relu(ops.add(x.load(idx), y.load(idx)))
```

#### Reduction (约第1061行)

```python
class Reduction(Loops):
    """归约操作: sum, max, mean, argmax..."""

    def __init__(self, device, dtype, inner_fn, ranges,
                 reduction_ranges, reduction_type):
        # inner_fn: (index) -> value  非归约维度的计算
        # reduction_ranges: 归约维度的范围
        # reduction_type: "sum" / "max" / "argmax" 等
```

**Reduction 的两层循环**:
```
外层循环: 遍历非归约维度 (ranges)
内层循环: 遍历归约维度 (reduction_ranges)
         └── 累加/取最大值等归约操作
```

#### Scatter (约第953行)

```python
class Scatter(Pointwise):
    """散射写入: scatter, index_put..."""
    # 继承 Pointwise，但写入模式不同
```

### 2. 视图节点

#### ExpandView (约第2610行)

```python
class ExpandView(BaseView):
    """广播扩展视图 (零拷贝)"""
    # x.shape = (1, 3) → x.expand(5, 3).shape = (5, 3)
    # 不复制数据，只修改 stride 使其看起来像广播
```

#### PermuteView (约第2689行)

```python
class PermuteView(BaseView):
    """维度重排视图 (零拷贝)"""
    # x.shape = (2, 3, 4) → x.permute(2, 1, 0).shape = (4, 3, 2)
    # 不复制数据，只交换 stride
```

#### SliceView (约第3099行)

```python
class SliceView(BaseView):
    """切片视图 (零拷贝)"""
    # x[1:3] → 不复制数据，只调整 offset 和 size
```

### 3. 缓冲区节点

#### ComputedBuffer (约第4006行)

```python
class ComputedBuffer(Buffer):
    """计算产生的缓冲区"""
    # 包含一个 Operation (Pointwise, Reduction 等)
    # make_scheduler() 创建 SchedulerNode
```

#### TemplateBuffer (约第4334行)

```python
class TemplateBuffer(Buffer):
    """模板内核缓冲区"""
    # 用于 GEMM, Conv 等特殊内核
    # 不参与通用的 Pointwise 融合
```

### 4. 布局节点

#### Layout (约第3261行)

```python
class Layout:
    """内存布局描述"""
    device: torch.device
    dtype: torch.dtype
    size: List[Expr]      # 各维度大小
    stride: List[Expr]    # 各维度步长
    offset: Expr          # 偏移量

class FixedLayout(Layout):
    """固定布局 - stride 不可变"""

class FlexibleLayout(Layout):
    """灵活布局 - stride 可在编译时优化"""
    # FlexibleLayout 允许 Scheduler 选择最优的 stride order
    # 例如: 对于后续的 Pointwise 操作，contiguous 布局可能更优
```

## IR 构建示例

### 示例: `z = relu(x + y)` 的 IR 构建

```python
# Step 1: 创建输入 TensorBox
x = TensorBox(StorageBox(InputBuffer("x", FixedLayout(...))))
y = TensorBox(StorageBox(InputBuffer("y", FixedLayout(...))))

# Step 2: Lowering aten.add → Pointwise IR
add_ir = Pointwise(
    device="cpu",
    dtype=torch.float32,
    inner_fn=lambda idx: ops.add(x.load(idx), y.load(idx)),
    ranges=[sympy.Integer(32), sympy.Integer(64)],
)
add_buffer = ComputedBuffer("add_0", FixedLayout(...), add_ir)
add_tensorbox = TensorBox(StorageBox(add_buffer))

# Step 3: Lowering aten.relu → Pointwise IR (融合后)
relu_ir = Pointwise(
    device="cpu",
    dtype=torch.float32,
    inner_fn=lambda idx: ops.relu(ops.add(x.load(idx), y.load(idx))),
    #                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                          融合: relu 的 inner_fn 内嵌了 add 的 inner_fn
    ranges=[sympy.Integer(32), sympy.Integer(64)],
)
relu_buffer = ComputedBuffer("relu_0", FixedLayout(...), relu_ir)
z = TensorBox(StorageBox(relu_buffer))
```

## IR 到代码生成的桥梁

```python
# 每个 IR 节点都有 make_scheduler() 方法
class ComputedBuffer(Buffer):
    def make_scheduler(self, ...):
        # 将 IR 节点转换为 SchedulerNode
        # SchedulerNode 是 Scheduler 处理的基本单位
        return SchedulerNode(self, ...)
```

## 学习检查点

- [ ] 能画出 TensorBox → StorageBox → Buffer 的三层结构图
- [ ] 理解 View 节点为什么是零拷贝的
- [ ] 能解释 `inner_fn` 闭包的设计动机
- [ ] 理解 Pointwise 和 Reduction 的区别
- [ ] 知道 FlexibleLayout 如何支持布局优化
- [ ] 能手动构建 `z = relu(x + y)` 的 IR

## 下一步

完成本阶段后，进入 [阶段3: Lowering](../stage3_lowering/guide.md)
