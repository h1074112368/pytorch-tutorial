# c6: torch.export

## 核心问题

torch.export 如何将模型导出为可序列化的 ExportedProgram？

## 关键源码

| 文件 | 说明 |
|------|------|
| `torch/export/__init__.py` | 入口: export(), save(), load() |
| `torch/export/exported_program.py` | ExportedProgram 类 |
| `torch/export/_trace.py` | 追踪实现 |

## ExportedProgram

```python
class ExportedProgram:
    graph_module: GraphModule      # FX Graph
    graph_signature: GraphSignature # 输入/输出签名
    state_dict: dict               # 参数和缓冲区
    range_constraints: dict        # 符号形状约束
```

## export vs torch.jit

| 特性 | torch.jit.trace/script | torch.export |
|------|----------------------|-------------|
| 控制流 | trace 无法捕获; script 受限 | 通过 Dynamo 字节码分析 |
| 动态形状 | 不支持 | 支持 (SymInt) |
| Guard 系统 | 无 | 完整 |
| Python 特性 | 严重受限 | 广泛支持 |
| 后端集成 | 有限 | 直接对接 AOTInductor |

## 使用示例

```python
# 导出
exported = torch.export.export(model, (x,))

# 保存/加载
torch.export.save(exported, "model.pt2")
loaded = torch.export.load("model.pt2")

# AOT 编译
so_path = torch._inductor.aot_compile(exported, (x,))
```

## 学习检查点

- [ ] 理解 ExportedProgram 的核心字段
- [ ] 知道 torch.export vs torch.jit 的区别
- [ ] 理解动态形状 (SymInt) 的支持
- [ ] 知道 AOTInductor 的编译流程
