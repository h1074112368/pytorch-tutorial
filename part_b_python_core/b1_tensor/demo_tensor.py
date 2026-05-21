"""
b1: torch.Tensor - 可运行示例
运行方式: python demo_tensor.py
"""

import torch


def demo_storage_model():
    print("=" * 60)
    print("1. Tensor 存储模型")
    print("=" * 60)

    x = torch.randn(2, 3, 4)
    print(f"  shape:    {x.shape}")
    print(f"  stride:   {x.stride()}")
    print(f"  nbytes:   {x.nbytes}")
    print(f"  dtype:    {x.dtype}")
    print(f"  device:   {x.device}")
    print(f"  data_ptr: {x.data_ptr()}")
    print()

    print("  Tensor = Storage + sizes + strides + offset")
    print("  访问 [i,j,k] = data[(i*strides[0]+j*strides[1]+k*strides[2])*itemsize]")
    print()


def demo_view_operations():
    print("=" * 60)
    print("2. 视图操作 (零拷贝)")
    print("=" * 60)

    x = torch.randn(2, 3, 4)
    print(f"  原始:  shape={x.shape}, stride={x.stride()}, data_ptr={x.data_ptr()}")

    y = x.view(6, 4)
    print(f"  view:  shape={y.shape}, stride={y.stride()}, data_ptr={y.data_ptr()} (相同!)")

    z = x.permute(2, 1, 0)
    print(f"  permute: shape={z.shape}, stride={z.stride()}, data_ptr={z.data_ptr()} (相同!)")

    w = x[:, 1, :]
    print(f"  切片: shape={w.shape}, stride={w.stride()}, data_ptr={w.data_ptr()} (相同!)")
    print()


def demo_contiguity():
    print("=" * 60)
    print("3. 连续性 (contiguous)")
    print("=" * 60)

    x = torch.randn(2, 3, 4)
    print(f"  原始: is_contiguous={x.is_contiguous()}")

    y = x.permute(1, 0, 2)
    print(f"  permute后: is_contiguous={y.is_contiguous()}")

    z = y.contiguous()
    print(f"  contiguous后: is_contiguous={z.is_contiguous()}, data_ptr变化={z.data_ptr() != y.data_ptr()}")
    print()


def demo_dtype_device():
    print("=" * 60)
    print("4. 数据类型与设备")
    print("=" * 60)

    x = torch.randn(2, 3)
    print(f"  原始: dtype={x.dtype}, device={x.device}")
    print(f"  half:   dtype={x.half().dtype}")
    print(f"  double: dtype={x.double().dtype}")
    print(f"  int:    dtype={x.int().dtype}")
    if torch.cuda.is_available():
        print(f"  cuda:   device={x.cuda().device}")
    print()


if __name__ == "__main__":
    demo_storage_model()
    demo_view_operations()
    demo_contiguity()
    demo_dtype_device()
    print("b1: torch.Tensor 学习完成!")
