# 阶段1: 编译入口与配置

## 核心问题

`torch.compile(model)` 调用后，代码是如何进入 Inductor 编译器的？

## 调用链总览

```
torch.compile(model, backend="inductor")
    │
    ▼
torch._dynamo.optimize("inductor")  ←── Dynamo 后端注册
    │
    ▼
torch._dynamo.backends.inductor
    │
    ▼
torch._inductor.compile_fx.compile_fx  ←── Inductor 编译入口
    │
    ▼
compile_fx_inner(gm, example_inputs)
    │
    ├── [缓存命中] FxGraphCache.load_with_key() → 返回 CompiledFxGraph
    │
    ▼ [缓存未命中]
_compile_fx_inner(...)
    │
    ▼
fx_codegen_and_compile(...)
    │
    ▼
_InProcessFxCompile.codegen_and_compile(...)
    ├── view_to_reshape(gm)           # 预处理: view→reshape
    ├── FakeTensorProp(gm)            # FakeTensor 传播
    ├── record_original_output_strides # 记录输出 stride
    ├── _recursive_post_grad_passes   # Post-grad 优化 pass
    │
    ▼
GraphLowering.run(*example_inputs)    # ←── 核心编译入口
    │
    ▼
graph.compile_to_module()             # 代码生成 + 加载
    │
    ▼
CompiledFxGraph                       # 编译产物
```

## 关键源码文件

### 1. `torch/_inductor/__init__.py` — 模块入口

源码位置: `c:\inductor\pytorch\torch\_inductor\__init__.py`

```python
def compile(gm, example_inputs, options=None):
    """编译一个 FX graph"""
    from .compile_fx import compile_fx
    return compile_fx(gm, example_inputs, config_patches=options)

def aot_compile(gm, args, kwargs=None, *, options=None):
    """AOT 编译为共享库"""
    from .compile_fx import _aoti_flatten_inputs, compile_fx_aot
    flat_example_inputs, options = _aoti_flatten_inputs(gm, args, kwargs, options=options)
    return compile_fx_aot(gm, flat_example_inputs, config_patches=options)
```

**要点**:
- `compile()` 用于 JIT 编译（`torch.compile()` 路径）
- `aot_compile()` 用于 AOT 编译（`torch.export()` 路径）
- 两者最终都进入 `compile_fx.py`

### 2. `torch/_inductor/compile_fx.py` — 编译主流程

源码位置: `c:\inductor\pytorch\torch\_inductor\compile_fx.py`

#### `compile_fx_inner()` (约第582行)

```python
def compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: list[InputType],
    ...,
):
    # 1. 环境设置
    with dynamo_timed("compile_fx_inner"), ...:
        # 2. 委托给 _compile_fx_inner
        return wrap_compiler_debug(_compile_fx_inner)(
            gm, example_inputs, ...
        )
```

#### `_compile_fx_inner()` (约第636行)

```python
def _compile_fx_inner(gm, example_inputs, ...):
    # 1. 空图快速返回
    if not gm.graph.nodes:
        return ...

    # 2. 缓存查找
    if fx_graph_cache:
        cache_key = FxGraphCache.prepare_key(gm, example_inputs, ...)
        compiled_graph = FxGraphCache.load_with_key(cache_key, ...)
        if compiled_graph is not None:
            return compiled_graph  # 缓存命中!

    # 3. 实际编译
    compiled_graph = fx_codegen_and_compile(gm, example_inputs, ...)

    # 4. 后处理 (CUDA Graph 化等)
    compiled_graph.post_compile()

    return compiled_graph
```

#### `_InProcessFxCompile.codegen_and_compile()` (约第893行)

```python
class _InProcessFxCompile:
    def codegen_and_compile(self, gm, example_inputs, ...):
        # Step 1: 预处理
        view_to_reshape(gm)

        # Step 2: FakeTensor 传播
        with torch.no_grad():
            FakeTensorProp(gm).propagate(*example_inputs)

        # Step 3: 记录输出 stride
        record_original_output_strides(gm)

        # Step 4: Post-grad 优化 pass
        _recursive_post_grad_passes(gm)

        # Step 5: 创建 GraphLowering 并执行
        graph = GraphLowering(gm, ...)
        graph.run(*example_inputs)         # ←── 核心: FX Graph → IR

        # Step 6: 代码生成
        result = graph.compile_to_module()  # ←── 核心: IR → 代码

        # Step 7: 构造编译产物
        return CompiledFxGraph(result, ...)
```

### 3. `torch/_inductor/config.py` — 配置体系

源码位置: `c:\inductor\pytorch\torch\_inductor\config.py`

**配置优先级**（从高到低）:
1. `torch.compile(options={...})` 传入的 `config_patches`
2. 环境变量 `TORCHINDUCTOR_*`
3. 默认值

**关键配置项**:

| 配置项 | 环境变量 | 默认值 | 说明 |
|--------|----------|--------|------|
| `debug` | `TORCHINDUCTOR_DEBUG` | `False` | 调试模式 |
| `fx_graph_cache` | `TORCHINDUCTOR_FX_GRAPH_CACHE` | `True` | FX Graph 缓存 |
| `triton.cudagraphs` | - | `False` | CUDA Graph 支持 |
| `max_autotune` | - | `False` | 最大 autotune |
| `cpp_wrapper` | `TORCHINDUCTOR_CPP_WRAPPER` | `False` | C++ wrapper 模式 |

**编译模式**:

```python
torch.compile(model)                              # default 模式
torch.compile(model, mode="reduce-overhead")      # 启用 cudagraphs
torch.compile(model, mode="max-autotune")         # 最大 autotune + cudagraphs
torch.compile(model, mode="max-autotune-no-cudagraphs")  # 最大 autotune
```

### 4. `torch/_inductor/decomposition.py` — 算子分解

源码位置: `c:\inductor\pytorch\torch\_inductor\decomposition.py`

算子分解决定了哪些 `aten` 算子在进入 Inductor 之前被拆分为更基本的操作:

```python
# 例如: aten.addmm 可能被分解为 aten.mm + aten.add
# 分解的好处: 让 Inductor 有更多融合机会
decompositions = get_decompositions(select_decomp_table())
```

## 缓存机制

### FxGraphCache

```
编译请求 → prepare_key(gm, inputs, ...)
              │
              ├── 计算 graph hash (基于 FX Graph 结构)
              ├── 计算 input hash (基于输入形状/类型/设备)
              └── 组合为 cache_key
              │
              ▼
         load_with_key(cache_key)
              │
              ├── [命中] 反序列化 CompiledFxGraph → 直接返回
              └── [未命中] 执行编译 → 存储到缓存
```

**缓存存储位置**: `torch._inductor.runtime.runtime_utils.cache_dir()`
- 通常在 `~/.cache/torchinductor/` 下

## 学习检查点

- [ ] 能画出从 `torch.compile()` 到 `GraphLowering.run()` 的完整调用链
- [ ] 理解 `compile_fx_inner` 中缓存查找的逻辑
- [ ] 知道 `_InProcessFxCompile.codegen_and_compile` 的 7 个步骤
- [ ] 理解配置项的三级优先级
- [ ] 能解释算子分解的作用和动机

## 下一步

完成本阶段后，进入 [阶段2: IR 中间表示](../stage2_ir/guide.md)
