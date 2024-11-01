"""
在这部分中，实现单个卷积步骤，其中将滤波器应用于 input 的单个位置。这将用于构建卷积单元，该单元：
Takes an input volume  采用输入音量
Applies a filter at every position of the input
在输入的每个位置应用滤镜
Outputs another volume (usually of different size)
输出另一个卷（通常大小不同）
"""
#修正
import numpy as np
def conv_single_step(a_slice_prev, W, b):
    """
    计算单个卷积步骤的结果。

    参数:
    a_slice_prev (numpy.ndarray): 输入的切片，形状为 (f, f, n_C_prev)
    W (numpy.ndarray): 卷积核，形状为 (f, f, n_C_prev)
    b (numpy.ndarray): 偏置项，形状为 (1, 1, 1)

    返回:
    z (float): 卷积结果
    """
    # 检查输入参数的维度
    if a_slice_prev.shape != W.shape:
        raise ValueError("输入切片和卷积核的维度不匹配")

    # 假设 b 是一个二维数组，且需要第一个元素
    z = np.sum(a_slice_prev * W) + float(b[0, 0])
    return z

# 设置随机种子以复现结果
np.random.seed(1)

# 生成随机输入
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

# 计算卷积结果
try:
    z = conv_single_step(a_slice_prev, W, b)
    print("Z =", z)
except Exception as e:
    print(f"发生错误: {e}")
