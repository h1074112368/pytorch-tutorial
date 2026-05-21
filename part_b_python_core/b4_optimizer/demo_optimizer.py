"""
b4: 优化器 - 可运行示例
运行方式: python demo_optimizer.py
"""

import torch
import torch.nn as nn


def demo_optimizer_basics():
    print("=" * 60)
    print("1. 优化器基础")
    print("=" * 60)

    model = nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    x = torch.randn(2, 10)
    y = torch.randn(2, 5)

    for i in range(3):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        print(f"  Step {i+1}: loss={loss.item():.4f}")
    print()


def demo_adam_vs_sgd():
    print("=" * 60)
    print("2. Adam vs SGD")
    print("=" * 60)

    print("  SGD+momentum: v = μ*v + grad; w -= lr*v")
    print("  Adam: m=β1*m+g; v=β2*v+g²; w-=lr*m/(√v+ε)")
    print("  AdamW: Adam + 解耦权重衰减")
    print()

    model_sgd = nn.Linear(10, 5)
    model_adam = nn.Linear(10, 5)
    model_adam.load_state_dict(model_sgd.state_dict())

    opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9)
    opt_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)

    x = torch.randn(4, 10)
    y = torch.randn(4, 5)

    for i in range(5):
        opt_sgd.zero_grad()
        loss_s = nn.functional.mse_loss(model_sgd(x), y)
        loss_s.backward()
        opt_sgd.step()

        opt_adam.zero_grad()
        loss_a = nn.functional.mse_loss(model_adam(x), y)
        loss_a.backward()
        opt_adam.step()

        print(f"  Step {i+1}: SGD loss={loss_s.item():.4f}, Adam loss={loss_a.item():.4f}")
    print()


def demo_lr_scheduler():
    print("=" * 60)
    print("3. 学习率调度器")
    print("=" * 60)

    model = nn.Linear(10, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print("  CosineAnnealingLR 学习率变化:")
    for epoch in range(10):
        lr = optimizer.param_groups[0]['lr']
        print(f"    Epoch {epoch+1}: lr={lr:.6f}")
        scheduler.step()
    print()


if __name__ == "__main__":
    demo_optimizer_basics()
    demo_adam_vs_sgd()
    demo_lr_scheduler()
    print("b4: 优化器 学习完成!")
