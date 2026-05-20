# Part B: Python 核心层

## 概述

Python 核心层是用户最常接触的层次，包括 Tensor、Autograd、nn.Module 和 Optimizer。

## 学习路径

| 子模块 | 核心内容 | 关键源码 |
|--------|----------|----------|
| **b1_tensor** | torch.Tensor API, 视图/存储/操作 | `torch/_tensor.py`, `torch/_C/_tensor.py` |
| **b2_autograd** | Function, backward, 计算图, grad_mode | `torch/autograd/` |
| **b3_nn_module** | Module, Parameter, functional, hooks | `torch/nn/` |
| **b4_optimizer** | Optimizer, SGD, Adam, lr_scheduler | `torch/optim/` |

## 核心关系图

```
torch.Tensor (持有 grad_fn, requires_grad)
    │
    │  前向传播: 每个 op 创建 Node, 记录到计算图
    ▼
torch.autograd (自动微分引擎)
    │
    │  backward(): 按拓扑排序遍历计算图, 计算梯度
    ▼
torch.nn.Module (持有 Parameter, 定义 forward)
    │
    │  parameters(): 返回所有可学习参数
    ▼
torch.optim.Optimizer (使用梯度更新参数)
```
