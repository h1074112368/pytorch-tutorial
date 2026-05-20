"""
c2: FX Graph - 可运行示例
运行方式: python demo_fx.py
"""

import torch
import torch.fx


def demo_symbolic_trace():
    """演示符号追踪"""
    print("=" * 60)
    print("1. 符号追踪 (symbolic_trace)")
    print("=" * 60)

    class MyModel(torch.nn.Module):
        def forward(self, x, y):
            z = x + y
            w = torch.relu(z)
            return w

    model = MyModel()
    gm = torch.fx.symbolic_trace(model)

    print("  原始模型:")
    print(f"    {model}")
    print()
    print("  追踪后的 GraphModule:")
    print(f"    {gm.graph}")
    print()


def demo_node_types():
    """演示节点类型"""
    print("=" * 60)
    print("2. Node 操作类型")
    print("=" * 60)

    model = torch.nn.Sequential(torch.nn.ReLU())
    gm = torch.fx.symbolic_trace(model)

    for node in gm.graph.nodes:
        print(f"  op={node.op:15s} name={node.name:10s} target={node.target}")
    print()

    print("  6 种操作类型:")
    print("    placeholder    - 输入占位符")
    print("    get_attr       - 获取模块属性")
    print("    call_function  - 调用函数 (如 torch.add)")
    print("    call_module    - 调用 nn.Module (如 nn.ReLU)")
    print("    call_method    - 调用方法 (如 x.view)")
    print("    output         - 输出节点")
    print()


def demo_graph_module():
    """演示 GraphModule"""
    print("=" * 60)
    print("3. GraphModule")
    print("=" * 60)

    model = torch.nn.Linear(10, 5)
    gm = torch.fx.symbolic_trace(model)

    x = torch.randn(2, 10)
    with torch.no_grad():
        y = gm(x)

    print(f"  GraphModule 可以像普通 Module 一样调用")
    print(f"  输入: {x.shape}, 输出: {y.shape}")
    print()

    print("  GraphModule 的特性:")
    print("    - 持有 Graph (图结构)")
    print("    - 持有原始 Module 的参数和缓冲区")
    print("    - forward() 由 Graph 自动生成")
    print("    - 修改 Graph 后调用 recompile() 重新生成代码")
    print()


def demo_interpreter():
    """演示 Interpreter"""
    print("=" * 60)
    print("4. Interpreter (解释执行)")
    print("=" * 60)

    class PrintInterpreter(torch.fx.Interpreter):
        def call_function(self, target, args, kwargs):
            print(f"    执行: {target.__name__}")
            return super().call_function(target, args, kwargs)

    model = torch.nn.Sequential(torch.nn.ReLU())
    gm = torch.fx.symbolic_trace(model)

    x = torch.randn(2, 3)
    print("  逐节点执行:")
    with torch.no_grad():
        interp = PrintInterpreter(gm)
        interp.run(x)
    print()

    print("  Interpreter 的用途:")
    print("    - 自定义执行逻辑 (如打印、验证)")
    print("    - 图变换 (Transformer)")
    print("    - 性能分析 (计时每个节点)")
    print("    - 调试 (检查中间值)")
    print()


def demo_graph_transformation():
    """演示图变换"""
    print("=" * 60)
    print("5. 图变换")
    print("=" * 60)

    model = torch.nn.Sequential(torch.nn.ReLU())
    gm = torch.fx.symbolic_trace(model)

    print("  常见图变换:")
    print("    - 子图替换: gm.graph.replace_pattern(...)")
    print("    - 节点融合: 合并多个节点为一个")
    print("    - 死代码消除: 删除无用的节点")
    print("    - 形状传播: 推理每个节点的输出形状")
    print()

    print("  图变换示例 (替换 ReLU 为 LeakyReLU):")
    print("""
    for node in gm.graph.nodes:
        if node.op == 'call_module' and node.target == '0':
            gm.delete_submodule('0')
            gm.add_module('0', torch.nn.LeakyReLU())
    gm.recompile()
    """)
    print()


if __name__ == "__main__":
    demo_symbolic_trace()
    demo_node_types()
    demo_graph_module()
    demo_interpreter()
    demo_graph_transformation()

    print("=" * 60)
    print("c2: FX Graph 学习完成!")
    print("下一步: 阅读 c3_aot_autograd/guide.md 学习 AOTAutograd")
    print("=" * 60)
