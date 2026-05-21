# b1: torch.Tensor

## 核心问题

torch.Tensor 的 Python API 如何映射到 C++ 层？视图和存储的关系是什么？

## 关键源码

| 文件 | 说明 |
|------|------|
| `torch/_tensor.py` | Tensor Python 类定义 |
| `torch/_C/_tensor.py` | C++ 绑定 |
| `c10/core/TensorImpl.h` | C++ Tensor 实现 |

## Tensor 核心概念

### 1. 存储模型

```
Tensor = Storage + sizes + strides + offset
  │
  ├── Storage: 1D 连续内存 (data_ptr + nbytes)
  ├── sizes: 逻辑形状 [2, 3, 4]
  ├── strides: 步幅 [12, 4, 1] (元素数)
  └── offset: 偏移量

访问 [i, j, k] = data[(i*strides[0] + j*strides[1] + k*strides[2]) * itemsize + offset * itemsize]
```

### 2. 视图操作 (零拷贝)

```python
x = torch.randn(2, 3, 4)
y = x.view(6, 4)      # 共享存储, 只改变 sizes/strides
z = x.permute(2, 1, 0) # 共享存储, 交换 strides
w = x[:, 1, :]         # 共享存储, 调整 offset + sizes
```

### 3. 连续性

```python
x = torch.randn(2, 3, 4)
print(x.is_contiguous())  # True

y = x.permute(1, 0, 2)
print(y.is_contiguous())  # False (strides 不是递减的)

z = y.contiguous()        # 复制数据, 使其连续
```

### 4. 设备与类型

```python
x = torch.randn(2, 3)
x_cpu = x.cpu()           # 移到 CPU
x_cuda = x.cuda()         # 移到 CUDA
x_fp16 = x.half()         # 转 float16
x_bf16 = x.bfloat16()     # 转 bfloat16
```

## 学习检查点

- [ ] 理解 Tensor = Storage + sizes + strides + offset
- [ ] 知道视图操作是零拷贝的
- [ ] 理解 is_contiguous 的判断条件
- [ ] 知道 contiguous() 何时需要复制数据
