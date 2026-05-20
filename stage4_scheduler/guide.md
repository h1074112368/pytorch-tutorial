# 阶段4: Scheduler — 调度与融合（最复杂）

## 核心问题

Inductor 如何决定哪些算子可以融合？融合算法如何工作？如何优化内存使用？

## Scheduler 总览

```
IR 节点 (Buffer/Operation)
    │
    ▼
Scheduler._init()
    ├── 1. create_scheduler_node()     # IR → SchedulerNode
    ├── 2. compute_dependencies()      # 构建依赖图
    ├── 3. topological_sort_schedule() # 拓扑排序
    ├── 4. dead_node_elimination()     # 死节点消除
    ├── 5. create_foreach_nodes()      # 批量操作合并
    ├── 6. fuse_nodes()               # ⭐ 算子融合（核心）
    ├── 7. merge_loops()              # 循环合并
    ├── 8. finalize_multi_template_buffers()
    ├── 9. reorder_for_peak_memory()  # 峰值内存优化
    ├── 10. reorder_compute_and_comm_for_overlap()
    └── 11. compute_last_usage()      # 计算最后使用（内存回收）
```

## 关键源码文件

### `scheduler.py` — 调度核心

源码位置: `c:\inductor\pytorch\torch\_inductor\scheduler.py` (约 4500+ 行)

### 核心数据结构

#### SchedulerNode (约第78行)

```python
@dataclasses.dataclass
class SchedulerBuffer:
    """调度缓冲区"""
    scheduler: Scheduler
    node: ir.Buffer                    # 对应的 IR Buffer
    defining_op: Optional[BaseSchedulerNode]  # 定义该缓冲区的操作
    users: list[NodeUser]             # 使用该缓冲区的操作列表
    mpi_buffer: MemoryPlanningInfoForBuffer  # 内存规划信息
```

#### BaseSchedulerNode (约第200行)

```python
class BaseSchedulerNode:
    """调度节点基类"""
    # 子类:
    #   SchedulerNode          - 普通计算节点 (Pointwise, Reduction)
    #   FusedSchedulerNode     - 融合后的节点
    #   ExternKernelSchedulerNode - 外部内核节点
    #   NopKernelSchedulerNode - 无操作节点
```

### 1. 创建调度节点 — `create_scheduler_node()`

```python
def create_scheduler_node(self, node: ir.Buffer) -> BaseSchedulerNode:
    """将 IR Buffer 转换为调度节点"""
    if isinstance(node, (ir.ComputedBuffer, ir.TemplateBuffer)):
        return SchedulerNode(self, node)        # 可融合的计算节点
    elif isinstance(node, ir.ExternKernel):
        return ExternKernelSchedulerNode(self, node)  # 外部内核
    else:
        return NopKernelSchedulerNode(self, node)     # 无操作
```

### 2. 依赖计算 — `compute_dependencies()` (约第2185行)

```python
def compute_dependencies(self):
    """计算节点间的依赖关系"""
    # 依赖类型:
    #   MemoryDep - 内存依赖 (读/写同一块内存)
    #   StarDep   - 星型依赖 (所有元素都依赖)
    #   WeakDep   - 弱依赖 (不阻止融合)
```

**依赖分析的关键**:
- **读-写依赖**: 节点 B 读取节点 A 的输出 → A 必须在 B 之前
- **写-写依赖**: 两个节点写同一块内存 → 不能重排
- **别名依赖**: 两个张量共享底层存储 → 需要追踪
- **变异依赖**: 节点修改了输入 → 需要特殊处理

### 3. ⭐ 算子融合 — `fuse_nodes()` (约第2555行)

这是 Scheduler 最核心的算法：

```python
def fuse_nodes(self):
    """算子融合主循环"""
    for _ in range(10):  # 最多 10 轮迭代
        changed = self.fuse_nodes_once()
        if not changed:
            break  # 收敛

def fuse_nodes_once(self):
    """单轮融合迭代"""
    for node1, node2 in self.get_possible_fusions(nodes):
        # Step 1: 检查融合合法性
        if not self.can_fuse(node1, node2):
            continue

        # Step 2: 检查是否会创建循环依赖
        if self.will_fusion_create_cycle(node1, node2):
            continue

        # Step 3: 评估融合收益
        speedup = self.speedup_by_fusion(node1, node2)
        if speedup is None or speedup is False or speedup == 0:
            continue

        # Step 4: 执行融合
        self.get_backend(device).fuse(node1, node2)
        # → 创建 FusedSchedulerNode

        changed = True
    return changed
```

#### `can_fuse()` 的判断逻辑

```python
def can_fuse(self, node1, node2):
    """判断两个节点是否可以融合"""
    # 1. 设备检查: 必须在同一设备上
    if node1.get_device() != node2.get_device():
        return False

    # 2. 节点类型检查:
    #    - 两个 Pointwise 可以融合
    #    - Pointwise + Reduction 可以融合 (特定条件)
    #    - TemplateBuffer 通常不参与融合
    #    - ExternKernel 不参与融合

    # 3. 数据类型检查: 某些类型组合不支持融合

    # 4. 依赖检查: 不能有中间的读写依赖

    # 5. 大小检查: 融合后的 kernel 不能太大

    return True
```

#### `speedup_by_fusion()` 的评估

```python
def speedup_by_fusion(self, node1, node2):
    """评估融合收益"""
    # 同步评估: 基于启发式规则快速判断
    #   - 两个 Pointwise: 通常有收益 (减少内存读写)
    #   - Pointwise + Reduction: 可能有收益

    # 异步评估: 启动 benchmark 精确测量
    #   - 返回一个 callable (pending fusion)
    #   - 在后台 benchmark 不同配置
    #   - 下轮迭代使用 benchmark 结果

    return speedup_ratio  # > 0 表示有收益
```

#### `will_fusion_create_cycle()` 的循环检测

```python
def will_fusion_create_cycle(self, node1, node2):
    """检测融合是否会创建循环依赖"""
    # 融合后, node1 和 node2 变为一个节点
    # 如果 node2 的其他依赖指向 node1 的前驱, 就会创建循环
    # 使用 DFS/BFS 检测
```

### 4. 融合示例

#### 示例: `z = relu(x + y)`

```
融合前:
  [SchedulerNode: add]  →  [SchedulerNode: relu]
       读取: x, y            读取: add

融合后:
  [FusedSchedulerNode: add_relu]
       读取: x, y
       inner_fn: lambda idx: ops.relu(ops.add(x.load(idx), y.load(idx)))
```

**融合收益**:
- 减少 1 次 kernel launch
- 消除中间结果 `add` 的内存分配和读写
- 提高缓存局部性

#### 示例: 不可以融合的情况

```
[SchedulerNode: add]  →  [ExternKernel: conv]  →  [SchedulerNode: relu]
       读取: x, y          读取: add                  读取: conv

- add 和 conv 不能融合 (conv 是 ExternKernel)
- conv 和 relu 不能融合 (conv 是 ExternKernel)
```

### 5. 循环合并 — `merge_loops()`

```python
def merge_loops(self):
    """合并相同大小的循环"""
    # 如果多个节点有相同的 iteration space (ranges),
    # 可以合并为一个循环, 减少循环开销
```

### 6. 峰值内存优化 — `reorder_for_peak_memory()`

```python
def reorder_for_peak_memory(self):
    """重排节点顺序以降低峰值内存"""
    # 策略: 尽早释放不再使用的缓冲区
    # 依赖 compute_last_usage() 的结果
```

### 7. 代码生成 — `codegen()` (约第4130行)

```python
def codegen(self):
    """按设备分组, 调用后端代码生成"""
    for device, nodes in self.group_by_device():
        backend = self.get_backend(device)
        backend.codegen(nodes)
```

## 调度后端 — BaseScheduling

```python
class BaseScheduling:
    """调度后端基类"""

    def can_fuse(self, node1, node2):
        """判断是否可以融合 (设备特定)"""

    def fuse(self, node1, node2):
        """执行融合 (设备特定)"""

    def codegen(self, nodes):
        """生成代码 (设备特定)"""

# CPU 后端
class CppScheduling(BaseScheduling):
    """CPU 调度后端 - 生成 C++ kernel"""

# CUDA 后端
class TritonScheduling(BaseScheduling):
    """CUDA Triton 调度后端 - 生成 Triton kernel"""

class CUDACombinedScheduling(BaseScheduling):
    """CUDA 组合调度 - 委托给 Triton 或 CUDA C++"""
```

## 学习检查点

- [ ] 能画出 Scheduler._init() 的 11 个步骤
- [ ] 理解 `can_fuse()` 的判断逻辑
- [ ] 能解释 `speedup_by_fusion()` 的同步/异步评估
- [ ] 理解 `will_fusion_create_cycle()` 的循环检测
- [ ] 知道融合的 10 轮迭代收敛机制
- [ ] 理解 `SchedulerNode` vs `FusedSchedulerNode` 的区别
- [ ] 能解释 `reorder_for_peak_memory()` 的优化策略

## 下一步

完成本阶段后，进入 [阶段5: 代码生成](../stage5_codegen/guide.md)
