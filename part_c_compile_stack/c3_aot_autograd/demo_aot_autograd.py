"""
c3: AOTAutograd - 可运行示例
运行方式: python demo_aot_autograd.py
"""

import torch


def demo_aot_autograd_flow():
    print("=" * 60)
    print("1. AOTAutograd 流程")
    print("=" * 60)

    print("  AOTAutograd 的三步流程:")
    print("""
    FX Graph (Dynamo 输出)
        │
        ├── 1. functionalize: 消除变异操作
        │     x.add_(1) → x = x.add(1)
        │
        ├── 2. vjp: 计算反向图
        │     forward: y = x * 2; z = y.sum()
        │     backward: grad_y = grad_z.expand(...); grad_x = grad_y * 2
        │
        └── 3. 分解: 高层算子 → Core ATen
              aten.addmm → aten.mm + aten.add
    """)
    print()


def demo_functionalize():
    print("=" * 60)
    print("2. functionalize (函数化)")
    print("=" * 60)

    print("  变异操作 → 非变异操作:")
    print("""
    原始 (变异):
      def forward(self, x):
          x.add_(1)    # in-place, 修改输入
          x.mul_(2)    # in-place
          return x

    函数化后 (非变异):
      def forward(self, x):
          x = x.add(1)  # out-of-place, 创建新张量
          x = x.mul(2)  # out-of-place
          return x
    """)
    print()

    print("  为什么需要 functionalize?")
    print("    - Inductor 不支持 in-place 操作的融合")
    print("    - 非变异操作更容易分析和优化")
    print("    - 确保编译的正确性")
    print()


def demo_vjp():
    print("=" * 60)
    print("3. vjp (向量-雅可比积)")
    print("=" * 60)

    print("  前向图 → 反向图:")
    print("""
    前向图:
      def forward(x, w):
          y = x @ w.T       # mm
          z = y + bias       # add
          loss = z.sum()     # sum
          return loss

    反向图 (由 vjp 自动生成):
      def backward(grad_loss):
          grad_z = grad_loss.expand(z_shape)  # sum 的反向
          grad_y = grad_z                      # add 的反向
          grad_bias = grad_z.sum(0)            # add 的反向
          grad_w = grad_y.T @ x               # mm 的反向
          grad_x = grad_y @ w                 # mm 的反向
          return grad_x, grad_w
    """)
    print()

    print("  AOT 编译的好处:")
    print("    - 前向和反向图分别编译, 各自优化")
    print("    - 反向图不需要在运行时构建")
    print("    - 可以对反向图也做融合优化")
    print()


def demo_compile_with_aot():
    print("=" * 60)
    print("4. torch.compile 中的 AOTAutograd")
    print("=" * 60)

    model = torch.nn.Linear(10, 5)
    x = torch.randn(2, 10)

    compiled = torch.compile(model)
    with torch.no_grad():
        y = compiled(x)

    print(f"  torch.compile 自动执行 AOTAutograd:")
    print(f"    1. Dynamo 捕获 FX Graph")
    print(f"    2. AOTAutograd 分离前向/反向")
    print(f"    3. Inductor 分别编译前向和反向")
    print(f"  输入: {x.shape}, 输出: {y.shape}")
    print()


if __name__ == "__main__":
    demo_aot_autograd_flow()
    demo_functionalize()
    demo_vjp()
    demo_compile_with_aot()
    print("c3: AOTAutograd 学习完成!")
