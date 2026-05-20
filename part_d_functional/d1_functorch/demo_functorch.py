"""
d1: functorch - 可运行示例
运行方式: python demo_functorch.py
"""

import torch
from torch.func import vmap, grad, vjp, jvp, jacrev, jacfwd


def demo_vmap():
    """演示 vmap"""
    print("=" * 60)
    print("1. vmap (向量化映射)")
    print("=" * 60)

    def fn(x):
        return x * 2 + 1

    x = torch.randn(5, 3)

    # 逐元素 vs vmap
    result_loop = torch.stack([fn(x[i]) for i in range(5)])
    result_vmap = vmap(fn)(x)

    print(f"  逐元素结果: {result_loop.shape}")
    print(f"  vmap 结果:  {result_vmap.shape}")
    print(f"  结果相同:   {torch.allclose(result_loop, result_vmap)}")
    print()

    print("  vmap 的优势:")
    print("    - 代码更简洁")
    print("    - 性能更好 (向量化执行)")
    print("    - 可以嵌套: vmap(vmap(fn))")
    print()


def demo_grad():
    """演示 grad"""
    print("=" * 60)
    print("2. grad (标量梯度)")
    print("=" * 60)

    def fn(x):
        return (x ** 2).sum()

    x = torch.randn(3)
    grad_fn = grad(fn)
    g = grad_fn(x)

    print(f"  fn(x) = sum(x^2)")
    print(f"  grad(fn)(x) = 2*x")
    print(f"  计算结果: {g}")
    print(f"  期望结果: {2 * x}")
    print()

    print("  grad vs torch.autograd.grad:")
    print("    grad: 函数式, 无需 requires_grad")
    print("    autograd.grad: 需要设置 requires_grad=True")
    print()


def demo_jacrev_jacfwd():
    """演示雅可比计算"""
    print("=" * 60)
    print("3. jacrev / jacfwd (雅可比矩阵)")
    print("=" * 60)

    def fn(x):
        return x ** 2

    x = torch.randn(3, dtype=torch.float64)

    J_rev = jacrev(fn)(x)
    J_fwd = jacfwd(fn)(x)

    print(f"  fn(x) = x^2")
    print(f"  雅可比矩阵 = diag(2*x)")
    print(f"  jacrev 结果:\n{J_rev}")
    print(f"  jacfwd 结果:\n{J_fwd}")
    print()

    print("  jacrev vs jacfwd:")
    print("  ┌──────────────┬─────────────────────┬─────────────────────┐")
    print("  │ 特性         │ jacrev (反向模式)    │ jacfwd (前向模式)   │")
    print("  ├──────────────┼─────────────────────┼─────────────────────┤")
    print("  │ 实现         │ vmap(vjp(fn))       │ vmap(jvp(fn))       │")
    print("  │ 适合         │ 输出维度 < 输入维度  │ 输入维度 < 输出维度 │")
    print("  │ 内存         │ 需要保存前向值       │ 不需要保存          │")
    print("  └──────────────┴─────────────────────┴─────────────────────┘")
    print()


def demo_functionalize():
    """演示 functionalize"""
    print("=" * 60)
    print("4. functionalize (函数化)")
    print("=" * 60)

    print("  functionalize 将变异操作转换为非变异操作:")
    print("""
    原始 (变异):
      def fn(x):
          x.add_(1)  # in-place 操作
          return x

    函数化后 (非变异):
      def fn(x):
          y = x.add(1)  # out-of-place 操作
          return y
    """)
    print()

    print("  在编译栈中的角色:")
    print("    AOTAutograd 使用 functionalize 消除变异操作")
    print("    因为 Inductor 不支持 in-place 操作的融合")
    print()


if __name__ == "__main__":
    demo_vmap()
    demo_grad()
    demo_jacrev_jacfwd()
    demo_functionalize()

    print("=" * 60)
    print("d1: functorch 学习完成!")
    print("下一步: 阅读 d2_prims_refs/guide.md 学习 prims/refs")
    print("=" * 60)
