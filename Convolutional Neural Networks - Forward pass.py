"""
实现以下函数，对输入激活A_prev上的滤波器 W 进行卷积。
此函数将上一层输出的激活值（对于一批 m 个输入）、F 滤波器/权重（用 W 表示）和偏置向量（用 b 表示）
作为输入A_prev，其中每个滤波器都有自己的（单个）偏置。
最后，您还可以访问超参数字典，其中包含步幅和填充。
"""
import numpy as np

def zero_pad(X, pad):
    # 确保 pad_width 的形状与 X 的维度一致
    pad_width = ((0, 0), (pad, pad), (pad, pad), (0, 0))
    return np.pad(X, pad_width, mode='constant', constant_values=0)


def conv_forward(A_prev, W, b, hparameters):
    # 假设 A_prev 的维度是 (batch_size, height, width, channels)
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int((n_H_prev + 2 * pad - f) / stride) + 1
    n_W = int((n_W_prev + 2 * pad - f) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # 循环遍历每个样本
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):  # 循环遍历输出的高度
            for w in range(n_W):  # 循环遍历输出的宽度
                for c in range(n_C):  # 循环遍历输出的通道
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = np.sum(a_slice_prev * W[:, :, :, c]) + b[0, 0, 0, c]

    cache = (A_prev, W, b, hparameters)
    return Z, cache

# 示例调用
A_prev = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {'pad': 1, 'stride': 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
