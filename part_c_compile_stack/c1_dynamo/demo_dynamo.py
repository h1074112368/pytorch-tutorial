"""
c1: TorchDynamo - 可运行示例
运行方式: python demo_dynamo.py
"""

import torch
import torch._dynamo as dynamo


def demo_compile_basic():
    """演示基本编译"""
    print("=" * 60)
    print("1. torch.compile() 基本使用")
    print("=" * 60)

    model = torch.nn.Linear(10, 5)
    x = torch.randn(2, 10)

    compiled = torch.compile(model)
    with torch.no_grad():
        y = compiled(x)

    print(f"  输入: {x.shape}")
    print(f"  输出: {y.shape}")
    print()


def demo_graph_break():
    """演示 Graph Break"""
    print("=" * 60)
    print("2. Graph Break 机制")
    print("=" * 60)

    class GraphBreakModel(torch.nn.Module):
        def forward(self, x):
            y = x * 2          # ← Dynamo 可追踪
            # if y.sum() > 0:  # ← 数据依赖控制流 → Graph Break!
            z = y + 1
            return z

    print("  会触发 Graph Break 的操作:")
    print("    1. 数据依赖的控制流: if x.sum() > 0")
    print("    2. 不支持的 Python 特性: pdb.set_trace()")
    print("    3. 非 Tensor 副作用: print(x)")
    print("    4. 动态类型变化: x.numpy()")
    print()

    print("  Graph Break 的处理:")
    print("""
    原始函数:
      y = x * 2          # 子图1
      if y.sum() > 0:    # Graph Break! 回到 Python
          z = y + 1      # 子图2
      else:
          z = y - 1      # 子图2 (另一个版本)
      return z

    编译结果:
      子图1 → Python执行(if判断) → 子图2
    """)
    print()


def demo_guards():
    """演示 Guard 机制"""
    print("=" * 60)
    print("3. Guard 机制")
    print("=" * 60)

    print("  Guard 确保编译后的代码可以复用:")
    print("""
    首次调用: forward(x)  # x.shape=(2,3), dtype=float32
      → 编译 → 生成 Guard:
        - x.shape == (2, 3)
        - x.dtype == torch.float32
        - x.device == "cpu"

    第二次调用: forward(x)  # x.shape=(2,3), dtype=float32
      → Guard 通过 → 复用编译结果 ✅

    第三次调用: forward(x)  # x.shape=(4,3), dtype=float32
      → Guard 失败 → 重新编译 ❌
    """)
    print()

    print("  Guard 的类型:")
    print("    - 形状 Guard:   x.shape == (2, 3)")
    print("    - 类型 Guard:   x.dtype == torch.float32")
    print("    - 设备 Guard:   x.device == 'cpu'")
    print("    - 函数 Guard:   len(x) == 2")
    print("    - 属性 Guard:   self.training == True")
    print()


def demo_dynamo_features():
    """演示 Dynamo 特性"""
    print("=" * 60)
    print("4. Dynamo 特性")
    print("=" * 60)

    print("  Dynamo vs TorchScript:")
    print("  ┌──────────────────┬──────────────────┬──────────────────┐")
    print("  │ 特性             │ TorchScript      │ Dynamo           │")
    print("  ├──────────────────┼──────────────────┼──────────────────┤")
    print("  │ 追踪方式         │ AST分析/字节码   │ CPython帧评估    │")
    print("  │ Python特性支持   │ 有限             │ 广泛             │")
    print("  │ 控制流           │ 受限             │ 支持(有break)    │")
    print("  │ 调试             │ 困难             │ 较容易           │")
    print("  │ 后端集成         │ JIT              │ 任意后端         │")
    print("  └──────────────────┴──────────────────┴──────────────────┘")
    print()

    print("  Dynamo 配置:")
    print("    TORCHDYNAMO_VERBOSE=1    # 详细日志")
    print("    torch._dynamo.config.cache_size_limit = 64  # 缓存大小限制")
    print("    torch._dynamo.config.print_graph_breaks = True  # 打印 break")
    print()


def demo_explain():
    """演示 Dynamo 解释功能"""
    print("=" * 60)
    print("5. Dynamo 解释功能 (explain)")
    print("=" * 60)

    print("  使用 torch._dynamo.explain() 分析编译行为:")
    print("""
    model = torch.nn.Linear(10, 5)
    x = torch.randn(2, 10)

    explanation = torch._dynamo.explain(model)(x)
    print(explanation)
    # → 显示: 图数量、Graph Break 位置、Guard 信息
    """)
    print()


if __name__ == "__main__":
    demo_compile_basic()
    demo_graph_break()
    demo_guards()
    demo_dynamo_features()
    demo_explain()

    print("=" * 60)
    print("c1: TorchDynamo 学习完成!")
    print("下一步: 阅读 c2_fx_graph/guide.md 学习 FX Graph")
    print("=" * 60)
