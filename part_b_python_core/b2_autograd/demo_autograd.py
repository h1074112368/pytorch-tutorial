"""
b2: 自动微分 - 可运行示例
运行方式: python demo_autograd.py
"""

import torch


def demo_computation_graph():
    """演示计算图构建"""
    print("=" * 60)
    print("1. 计算图构建 (define-by-run)")
    print("=" * 60)

    x = torch.randn(2, 3, requires_grad=True)
    y = x * 2
    z = y.sum()

    print(f"  x.requires_grad = {x.requires_grad}")
    print(f"  y.grad_fn = {y.grad_fn}")
    print(f"  z.grad_fn = {z.grad_fn}")
    print()

    print("  计算图:")
    print("    x (requires_grad=True)")
    print("    │")
    print("    ▼ MulBackward0")
    print("    y = x * 2")
    print("    │")
    print("    ▼ SumBackward0")
    print("    z = y.sum()")
    print()


def demo_backward():
    """演示反向传播"""
    print("=" * 60)
    print("2. 反向传播")
    print("=" * 60)

    x = torch.randn(2, 3, requires_grad=True)
    y = x * 2
    z = y.sum()

    z.backward()
    print(f"  x.grad = {x.grad}")
    print(f"  dz/dx = d(sum(x*2))/dx = 2")
    print()

    print("  反向传播流程:")
    print("    1. z.backward() 触发")
    print("    2. Engine 按拓扑排序遍历计算图")
    print("    3. SumBackward0.apply(grad=1.0) → grad_y = 1.0")
    print("    4. MulBackward0.apply(grad_y=1.0) → grad_x = 2.0")
    print()


def demo_custom_function():
    """演示自定义 Function"""
    print("=" * 60)
    print("3. 自定义 Function")
    print("=" * 60)

    class MyReLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.clamp(min=0)

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[x < 0] = 0
            return grad_input

    x = torch.randn(5, requires_grad=True)
    y = MyReLU.apply(x)
    y.sum().backward()

    print(f"  x = {x.data}")
    print(f"  y = MyReLU.apply(x) = {y.data}")
    print(f"  x.grad = {x.grad}")
    print()

    print("  自定义 Function 的规则:")
    print("    - forward() 和 backward() 必须是 @staticmethod")
    print("    - 使用 ctx.save_for_backward() 保存张量")
    print("    - 使用 MyReLU.apply() 调用, 不是 MyReLU.forward()")
    print("    - backward() 返回的梯度数量必须等于 forward() 的输入数量")
    print()


def demo_grad_modes():
    """演示梯度模式"""
    print("=" * 60)
    print("4. 梯度模式 (no_grad / enable_grad / inference_mode)")
    print("=" * 60)

    x = torch.randn(3, requires_grad=True)

    with torch.no_grad():
        y = x * 2
        print(f"  no_grad:       y.grad_fn = {y.grad_fn}  (不记录计算图)")

    with torch.enable_grad():
        y = x * 2
        print(f"  enable_grad:   y.grad_fn = {y.grad_fn}  (记录计算图)")

    with torch.inference_mode():
        y = x * 2
        print(f"  inference_mode: y.grad_fn = {y.grad_fn}  (更高效的推理模式)")
    print()

    print("  三种模式对比:")
    print("  ┌─────────────────┬──────────┬──────────┬──────────────────┐")
    print("  │ 特性            │ no_grad  │ enable   │ inference_mode   │")
    print("  ├─────────────────┼──────────┼──────────┼──────────────────┤")
    print("  │ 记录计算图      │ ✗        │ ✓        │ ✗                │")
    print("  │ 版本计数        │ ✓        │ ✓        │ ✗ (更高效)       │")
    print("  │ 保存 autograd   │ ✓        │ ✓        │ ✗                │")
    print("  │ 适用场景        │ 推理     │ 训练     │ 高性能推理       │")
    print("  └─────────────────┴──────────┴──────────┴──────────────────┘")
    print()


def demo_grad_check():
    """演示梯度检查"""
    print("=" * 60)
    print("5. 梯度检查 (gradcheck)")
    print("=" * 60)

    print("  torch.autograd.gradcheck 用于验证梯度的正确性:")
    print("""
    def my_func(x):
        return x ** 3

    x = torch.randn(3, requires_grad=True, dtype=torch.float64)
    torch.autograd.gradcheck(my_func, (x,))
    # → 使用数值微分验证解析梯度的正确性
    """)
    print()

    print("  注意: gradcheck 需要 float64 精度!")
    print("  float32 的数值误差太大, 无法通过 gradcheck")
    print()


if __name__ == "__main__":
    demo_computation_graph()
    demo_backward()
    demo_custom_function()
    demo_grad_modes()
    demo_grad_check()

    print("=" * 60)
    print("b2: 自动微分 学习完成!")
    print("下一步: 阅读 b3_nn_module/guide.md 学习 nn.Module")
    print("=" * 60)
