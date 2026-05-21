"""
e1: torchgen - 可运行示例
运行方式: python demo_torchgen.py
"""

import torch


def demo_torchgen_overview():
    print("=" * 60)
    print("1. torchgen 概述")
    print("=" * 60)

    print("  torchgen 从声明式定义自动生成代码:")
    print()
    print("  输入:")
    print("    native_functions.yaml  (~2000+ 个算子声明)")
    print("    derivatives.yaml       (导数定义)")
    print()
    print("  输出:")
    print("    ATen C++ 内核注册代码")
    print("    Python 绑定代码")
    print("    Autograd 导数代码")
    print("    Dispatcher 注册代码")
    print("    Meta 内核代码")
    print("    Functionalization 代码")
    print()


def demo_native_functions_yaml():
    print("=" * 60)
    print("2. native_functions.yaml 格式")
    print("=" * 60)

    print("  示例声明:")
    print("""
  - func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
    dispatch:
      CPU: add_tensor_cpu
      CUDA: add_tensor_cuda
    structured: True
    """)
    print()

    print("  关键字段:")
    print("    func:       函数签名 (名称 + 参数 + 返回值)")
    print("    dispatch:   各后端的内核函数名")
    print("    structured: 是否使用结构化内核模式")
    print("    variants:   生成的变体 (function, method)")
    print()


def demo_generated_code():
    print("=" * 60)
    print("3. 生成的代码示例")
    print("=" * 60)

    print("  torchgen 生成的代码位于:")
    print("    build/aten/src/ATen/          # ATen C++ 代码")
    print("    torch/_C/__init__.pyi         # Python 类型存根")
    print()

    print("  生成流程:")
    print("""
    python tools/setup_helpers/generate_code.py
        ├── 读取 native_functions.yaml
        ├── 读取 derivatives.yaml
        ├── 使用 torchgen 模板生成代码
        └── 输出到 build/ 和 torch/ 目录
    """)
    print()


if __name__ == "__main__":
    demo_torchgen_overview()
    demo_native_functions_yaml()
    demo_generated_code()
    print("e1: torchgen 学习完成!")
