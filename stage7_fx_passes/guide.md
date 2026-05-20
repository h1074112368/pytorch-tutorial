# 阶段7: FX Passes 优化

## 核心问题

在 Inductor 编译之前和之后，有哪些图优化 pass？它们的作用是什么？

## FX Passes 总览

```
FX Graph (Dynamo 捕获)
    │
    ▼
Pre-grad Passes (前向图优化)
    │   ├── 融合模式匹配 (mm+mm, attention)
    │   ├── 量化模式匹配
    │   ├── 二元折叠
    │   └── 其他优化
    │
    ▼
AOTAutograd (分离前向/反向)
    │
    ▼
Post-grad Passes (反向图优化)
    │   ├── view → reshape 转换
    │   ├── 融合模式匹配 (addmm, bmm)
    │   ├── 内存绑定优化
    │   └── 死代码消除
    │
    ▼
Joint Graph Passes (联合图优化)
    │   ├── 常量折叠
    │   └── 其他优化
    │
    ▼
Inductor Lowering → IR → Scheduler → Codegen
```

## 关键源码文件

### 1. `fx_passes/pre_grad.py` — 前向图优化

源码位置: `c:\inductor\pytorch\torch\_inductor\fx_passes\pre_grad.py`

```python
def pre_grad_passes(gm: torch.fx.GraphModule):
    """前向图优化 pass"""
    # 1. 融合模式匹配
    #    - mm + mm → bmm (特定条件)
    #    - attention 融合 (Flash Attention)
    #    - 量化模式

    # 2. 二元折叠
    #    - 常量 + 常量 → 常量

    # 3. 其他优化
    #    - 分解内存绑定的 mm
    #    - 高效 conv+bn 评估
```

### 2. `fx_passes/post_grad.py` — 反向图优化

源码位置: `c:\inductor\pytorch\torch\_inductor\fx_passes\post_grad.py`

```python
def post_grad_passes(gm: torch.fx.GraphModule):
    """反向图优化 pass"""
    # 1. view → reshape 转换
    #    - view 在 Inductor 中可能引起布局问题
    #    - reshape 更安全, 允许 Inductor 优化布局

    # 2. 融合模式匹配
    #    - addmm 模式
    #    - bmm 模式

    # 3. 内存绑定优化
    #    - reinplace: 将 out-of-place 操作转为 in-place

    # 4. 死代码消除
```

### 3. `fx_passes/joint_graph.py` — 联合图优化

源码位置: `c:\inductor\pytorch\torch\_inductor\fx_passes\joint_graph.py`

```python
def joint_graph_passes(gm: torch.fx.GraphModule):
    """联合图优化 pass (前向+反向)"""
    # 1. 常量折叠
    #    - 编译时已知的常量计算提前执行
    #    - 减少运行时计算量

    # 2. 其他优化
```

### 4. `fx_passes/fuse_attention.py` — Flash Attention 融合

源码位置: `c:\inductor\pytorch\torch\_inductor\fx_passes\fuse_attention.py`

```python
# Flash Attention 融合模式
# 将 scaled_dot_product_attention (SDPA) 模式匹配并替换为
# 高效的 Flash Attention kernel

# 匹配的模式:
#   Q @ K^T / sqrt(d) → softmax → @ V
# 替换为:
#   torch.nn.functional.scaled_dot_product_attention(Q, K, V)
```

### 5. `fx_passes/serialized_patterns/` — 序列化模式

源码位置: `c:\inductor\pytorch\torch\_inductor\fx_passes\serialized_patterns\`

```
_sfdp_pattern_1.py ~ _sfdp_pattern_19.py  # SDPA 的 19 种模式
addmm_pattern.py                            # addmm 融合模式
bmm_pattern.py                              # bmm 融合模式
mm_pattern.py                               # mm 融合模式
```

## 关键优化 Pass 详解

### view → reshape 转换

```python
# 原始代码
def forward(self, x):
    y = x.view(2, 3)  # view: 要求输入是 contiguous 的
    return y

# 优化后
def forward(self, x):
    y = x.reshape(2, 3)  # reshape: 允许非 contiguous 输入
    return y
```

**为什么需要这个转换？**
- `view` 要求输入是 contiguous 的，否则会报错
- `reshape` 更灵活，允许非 contiguous 输入
- Inductor 可以优化 `reshape` 的布局，而 `view` 会限制布局选择

### 常量折叠

```python
# 原始代码
def forward(self, x):
    y = x + 1.0
    z = y * 2.0
    return z

# 常量折叠后 (1.0 * 2.0 = 2.0)
def forward(self, x):
    y = x + 2.0  # 融合了加法和乘法的常量
    return y
```

### 内存绑定优化 (reinplace)

```python
# 原始代码 (out-of-place)
def forward(self, x):
    y = torch.add(x, 1)  # 分配新内存
    return y

# 优化后 (in-place)
def forward(self, x):
    y = torch.add_(x, 1)  # 原地操作, 节省内存
    return y
```

### Flash Attention 融合

```python
# 原始代码 (多次内存访问)
def forward(self, Q, K, V):
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return output

# 融合后 (Flash Attention, 减少内存访问)
def forward(self, Q, K, V):
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    return output
```

## 学习检查点

- [ ] 理解 Pre-grad / Post-grad / Joint Graph Passes 的区别
- [ ] 知道 view → reshape 转换的原因
- [ ] 理解常量折叠的优化效果
- [ ] 知道 Flash Attention 融合的模式匹配
- [ ] 理解 reinplace 优化的条件

## 下一步

完成本阶段后，进入 [阶段8: NPU Inductor](../stage8_npu_inductor/guide.md)
