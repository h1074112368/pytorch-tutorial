"""
b3: nn.Module - 可运行示例
运行方式: python demo_nn.py
"""

import torch
import torch.nn as nn


def demo_module_basics():
    print("=" * 60)
    print("1. nn.Module 基础")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )

    print(f"  模型结构:\n{model}")
    print()
    print(f"  参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"  命名参数:")
    for name, p in model.named_parameters():
        print(f"    {name}: {p.shape}")
    print()


def demo_parameter_buffer():
    print("=" * 60)
    print("2. Parameter vs Buffer")
    print("=" * 60)

    bn = nn.BatchNorm2d(3)
    print("  Parameter (可学习, requires_grad=True):")
    for name, p in bn.named_parameters():
        print(f"    {name}: {p.shape}, requires_grad={p.requires_grad}")

    print("  Buffer (不可学习, 不参与梯度):")
    for name, b in bn.named_buffers():
        print(f"    {name}: {b.shape}, requires_grad={b.requires_grad}")
    print()


def demo_state_dict():
    print("=" * 60)
    print("3. state_dict / load_state_dict")
    print("=" * 60)

    model = nn.Linear(10, 5)
    sd = model.state_dict()
    print(f"  state_dict 键: {list(sd.keys())}")
    for k, v in sd.items():
        print(f"    {k}: {v.shape}")
    print()

    x = torch.randn(2, 10)
    y1 = model(x)
    model.load_state_dict(sd)
    y2 = model(x)
    print(f"  加载后输出一致: {torch.allclose(y1, y2)}")
    print()


def demo_custom_module():
    print("=" * 60)
    print("4. 自定义 Module")
    print("=" * 60)

    class MyModel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim * 2)
            self.fc2 = nn.Linear(dim * 2, dim)
            self.act = nn.GELU()

        def forward(self, x):
            x = self.act(self.fc1(x))
            return self.fc2(x)

    model = MyModel(10)
    x = torch.randn(2, 10)
    y = model(x)
    print(f"  输入: {x.shape}, 输出: {y.shape}")
    print(f"  参数: {sum(p.numel() for p in model.parameters())}")
    print()


if __name__ == "__main__":
    demo_module_basics()
    demo_parameter_buffer()
    demo_state_dict()
    demo_custom_module()
    print("b3: nn.Module 学习完成!")
