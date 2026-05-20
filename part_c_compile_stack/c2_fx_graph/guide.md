# c2: FX Graph

## 核心问题

FX Graph 如何表示 Python 程序？如何进行图变换？

## FX 架构

```
torch.fx
├── _symbolic_trace.py   # 符号追踪: Tracer, symbolic_trace()
├── graph.py             # 图 IR: Graph, CodeGen
├── graph_module.py      # 图模块: GraphModule
├── node.py              # 节点: Node
├── proxy.py             # 代理: Proxy, TracerBase
├── interpreter.py       # 解释器: Interpreter, Transformer
├── subgraph_rewriter.py # 子图重写
└── passes/              # 图变换 pass
    ├── shape_prop.py    # 形状传播
    ├── tools_mod.py     # 工具
    └── ...
```

## 核心类

### Node — 图节点

```python
class Node:
    op: str          # 'placeholder', 'get_attr', 'call_function', 'call_module', 'call_method', 'output'
    target: ...      # 目标 (函数/模块名/方法名)
    args: tuple      # 位置参数
    kwargs: dict     # 关键字参数
    name: str        # 唯一名称
    users: dict      # 使用者映射
    meta: dict       # 元数据 (val, tensor_meta 等)
```

### Graph — 有向无环图

```python
class Graph:
    nodes: list[Node]  # 所有节点 (按拓扑排序)
    # 方法:
    def placeholder(self, name): ...    # 添加输入节点
    def call_function(self, fn, args): ...  # 添加函数调用
    def output(self, result): ...       # 添加输出节点
    def print_tabular(self): ...        # 打印表格
```

### GraphModule — 持有 Graph 的 nn.Module

```python
class GraphModule(nn.Module):
    graph: Graph       # 持有的 FX Graph
    # 可以像普通 nn.Module 一样调用
    def forward(self, *args):
        return self.graph(*args)
    def recompile(self): ...  # 从 Graph 重新生成 forward 代码
```

### Interpreter — 解释执行

```python
class Interpreter:
    """逐节点解释执行 FX Graph"""
    def run(self, *args):
        for node in self.graph.nodes:
            result = self.run_node(node)
        return result

    def placeholder(self, target, args, kwargs): ...
    def call_function(self, target, args, kwargs): ...
    def output(self, target, args, kwargs): ...
```

## 学习检查点

- [ ] 理解 Node 的 6 种操作类型
- [ ] 能使用 symbolic_trace 追踪模型
- [ ] 理解 GraphModule 如何持有 Graph
- [ ] 能使用 Interpreter 自定义执行逻辑
- [ ] 知道如何进行子图替换 (replace_pattern)
