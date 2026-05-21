# b3: nn.Module

## 核心问题

nn.Module 如何管理参数和子模块？forward 的执行机制是什么？

## 关键源码

| 文件 | 说明 |
|------|------|
| `torch/nn/modules/module.py` | Module 基类 |
| `torch/nn/parameter.py` | Parameter 和 Buffer |
| `torch/nn/functional.py` | 函数式 API |
| `torch/nn/init.py` | 权重初始化 |

## Module 核心机制

### 1. 参数管理

```python
class Module:
    _parameters: dict[str, Parameter]  # 可学习参数
    _buffers: dict[str, Tensor]        # 不可学习的状态 (如 BN 的 running_mean)
    _modules: dict[str, Module]        # 子模块

    def parameters(self): ...          # 递归返回所有参数
    def named_parameters(self): ...    # 递归返回所有命名参数
    def state_dict(self): ...          # 返回完整状态字典
    def load_state_dict(self): ...     # 加载状态字典
```

### 2. 钩子系统

```python
# 前向钩子
module.register_forward_pre_hook(hook)   # forward 前调用
module.register_forward_hook(hook)       # forward 后调用

# 反向钩子
module.register_full_backward_hook(hook) # backward 后调用
```

### 3. 常用模块

| 模块 | 说明 |
|------|------|
| `nn.Linear` | 全连接层: y = xA^T + b |
| `nn.Conv2d` | 2D 卷积 |
| `nn.BatchNorm2d` | 批归一化 |
| `nn.ReLU` / `nn.GELU` | 激活函数 |
| `nn.Dropout` | Dropout 正则化 |
| `nn.Embedding` | 嵌入层 |
| `nn.TransformerEncoder` | Transformer 编码器 |
| `nn.Sequential` | 顺序容器 |

## 学习检查点

- [ ] 理解 Module 的参数管理机制
- [ ] 知道 Parameter 和 Buffer 的区别
- [ ] 理解 state_dict / load_state_dict
- [ ] 知道钩子系统的用途
