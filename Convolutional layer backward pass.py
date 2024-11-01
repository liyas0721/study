import numpy as np

# 假设输入数据
m, n_H_prev, n_W_prev, n_C_prev = 1, 4, 4, 2  # 输入数据的形状
f, n_C = 2, 3  # 卷积核的大小和输出通道数
n_H = n_H_prev - f + 1  # 输出高度
n_W = n_W_prev - f + 1  # 输出宽度

# 初始化输入数据
A_prev = np.random.randn(m, n_H_prev, n_W_prev, n_C_prev)
W = np.random.randn(f, f, n_C_prev, n_C)
b = np.random.randn(1, 1, 1, n_C)

# 初始化输出
Z = np.zeros((m, n_H, n_W, n_C))

def conv_forward(A_prev, W, b, hparameters):
    """
    实现卷积层的前向传播

    参数:
    A_prev -- 上一层的输出，形状为 (m, n_H_prev, n_W_prev, n_C_prev)
    W -- 权重矩阵，形状为 (f, f, n_C_prev, n_C)
    b -- 偏置向量，形状为 (1, 1, 1, n_C)
    hparameters -- 包含超参数的字典，如 'stride' 和 'pad'

    返回:
    Z -- 卷积层的输出，形状为 (m, n_H, n_W, n_C)
    cache -- 缓存，用于反向传播
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = np.pad(A_prev, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    for i in range(m):  # 遍历样本
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):  # 遍历高度
            for w in range(n_W):  # 遍历宽度
                for c in range(n_C):  # 遍历通道
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # 确保 a_slice 和 W[:, :, :, c] 的形状一致
                    assert a_slice.shape == W[:, :, :, c].shape, f"Shapes do not match: {a_slice.shape} and {W[:, :, :, c].shape}"

                    Z[i, h, w, c] = np.sum(a_slice * W[:, :, :, c]) + b[0, 0, 0, c]

    cache = (A_prev, W, b, hparameters)

    return Z, cache

# 超参数
hparameters = {'stride': 1, 'pad': 0}

# 调用卷积前向传播函数
Z, cache_conv = conv_forward(A_prev, W, b, hparameters)

print(Z)
