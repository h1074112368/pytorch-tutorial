"""
阶段7: FX Passes 优化 - 可运行示例

演示图优化 pass: view→reshape, 常量折叠, Flash Attention, reinplace
运行方式: python demo_fx_passes.py
"""

import torch
import torch.fx


def demo_fx_passes_overview():
    """演示 FX Passes 总览"""
    print("=" * 60)
    print("1. FX Passes 总览")
    print("=" * 60)

    print("""
  FX Graph (Dynamo 捕获)
      │
      ▼
  Pre-grad Passes (前向图优化)
      │   ├── 融合模式匹配 (mm+mm, attention)
      │   ├── 量化模式匹配
      │   ├── 二元折叠
      │   └── 其他优化
      │
      ▼
  AOTAutograd (分离前向/反向)
      │
      ▼
  Post-grad Passes (反向图优化)
      │   ├── view → reshape 转换
      │   ├── 融合模式匹配 (addmm, bmm)
      │   ├── 内存绑定优化 (reinplace)
      │   └── 死代码消除
      │
      ▼
  Joint Graph Passes (联合图优化)
      │   ├── 常量折叠
      │   └── 其他优化
      │
      ▼
  Inductor Lowering → IR → Scheduler → Codegen
    """)
    print()


def demo_view_to_reshape():
    """演示 view → reshape 转换"""
    print("=" * 60)
    print("2. view → reshape 转换")
    print("=" * 60)

    x = torch.randn(6)
    y = x.view(2, 3)
    print(f"  view:   x.shape={x.shape} → y.shape={y.shape}")

    x_nc = torch.randn(2, 3).t().contiguous()
    try:
        z = x_nc.view(6)
    except RuntimeError as e:
        print(f"  view 对非 contiguous 张量报错: {e}")

    z = x_nc.reshape(6)
    print(f"  reshape: 对非 contiguous 张量正常工作, shape={z.shape}")
    print()

    print("  为什么 Inductor 需要 view → reshape?")
    print("    - view 要求输入 contiguous, 限制布局优化")
    print("    - reshape 允许非 contiguous, Inductor 可以自由优化布局")
    print("    - 例如: Inductor 可以选择 channels-last 布局")
    print()


def demo_constant_folding():
    """演示常量折叠"""
    print("=" * 60)
    print("3. 常量折叠")
    print("=" * 60)

    print("  原始代码:")
    print("""
    def forward(self, x):
        y = x + 1.0
        z = y * 2.0
        return z
    """)

    print("  常量折叠后 (1.0 * 2.0 = 2.0):")
    print("""
    def forward(self, x):
        y = x + 2.0  # 融合了加法和乘法的常量
        return y
    """)

    print("  常量折叠的好处:")
    print("    - 减少运行时计算量")
    print("    - 减少常量参数的数量")
    print("    - 可能触发更多融合机会")
    print()


def demo_reinplace():
    """演示内存绑定优化"""
    print("=" * 60)
    print("4. 内存绑定优化 (reinplace)")
    print("=" * 60)

    print("  原始代码 (out-of-place):")
    print("""
    def forward(self, x):
        y = torch.add(x, 1)  # 分配新内存
        return y
    """)

    print("  优化后 (in-place):")
    print("""
    def forward(self, x):
        y = torch.add_(x, 1)  # 原地操作, 节省内存
        return y
    """)

    print("  reinplace 的条件:")
    print("    - 输入在后续不再被使用")
    print("    - 没有其他张量依赖输入的原始值")
    print("    - 操作支持 in-place 版本")
    print()


def demo_flash_attention():
    """演示 Flash Attention 融合"""
    print("=" * 60)
    print("5. Flash Attention 融合")
    print("=" * 60)

    print("  原始代码 (多次内存访问):")
    print("""
    def forward(self, Q, K, V):
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ V
        return output
    """)

    print("  融合后 (Flash Attention):")
    print("""
    def forward(self, Q, K, V):
        output = F.scaled_dot_product_attention(Q, K, V)
        return output
    """)

    print("  Flash Attention 的好处:")
    print("    - 减少 HBM 访问次数 (O(N^2) → O(N^2d^2/M))")
    print("    - 使用 SRAM 缓存中间结果")
    print("    - 支持 causal mask 和 dropout")
    print()

    print("  SDPA 的 19 种模式:")
    print("    _sfdp_pattern_1.py ~ _sfdp_pattern_19.py")
    print("    覆盖不同的 attention 变体 (causal, dropout, etc.)")
    print()


def demo_fusion_patterns():
    """演示融合模式匹配"""
    print("=" * 60)
    print("6. 融合模式匹配")
    print("=" * 60)

    print("  mm + mm 模式 (b2b_gemm):")
    print("""
    原始: y = A @ B; z = y @ C
    融合: z = b2b_gemm(A, B, C)  # 一次 kernel 完成
    """)

    print("  addmm 模式:")
    print("""
    原始: y = A @ B; z = y + bias
    融合: z = addmm(bias, A, B)  # 一次 kernel 完成
    """)

    print("  conv + bn 模式:")
    print("""
    原始: y = conv(x); z = batch_norm(y)
    融合: z = fused_conv_bn(x)  # 融合 conv 和 bn 参数
    """)
    print()


def demo_pattern_matching_mechanism():
    """演示模式匹配机制"""
    print("=" * 60)
    print("7. 模式匹配机制")
    print("=" * 60)

    print("  Inductor 使用 torch.fx 的模式匹配:")
    print("""
    # 定义模式
    def pattern(x, y):
        return torch.add(x, y)

    # 定义替换
    def replacement(x, y):
        return custom_add_kernel(x, y)

    # 注册模式
    register_fusion_pattern(pattern, replacement)
    """)

    print("  模式匹配的流程:")
    print("    1. 遍历 FX Graph 的所有节点")
    print("    2. 对每个节点, 尝试匹配已注册的模式")
    print("    3. 匹配成功: 替换为对应的 replacement")
    print("    4. 匹配失败: 保持原样")
    print()


if __name__ == "__main__":
    demo_fx_passes_overview()
    demo_view_to_reshape()
    demo_constant_folding()
    demo_reinplace()
    demo_flash_attention()
    demo_fusion_patterns()
    demo_pattern_matching_mechanism()

    print("=" * 60)
    print("阶段7 学习完成!")
    print("下一步: 阅读 stage8_npu_inductor/guide.md 学习 NPU Inductor")
    print("=" * 60)
