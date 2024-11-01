import numpy as np

def pool_backward(dA, cache, mode="max", hparameters=None):
    """
    实现池化层的反向传播。
    参数:
    dA -- 池化层输出的梯度，形状为 (m, n_H, n_W, n_C)
    cache -- 缓存，包含前向传播的信息 (A_prev, hparameters)
    mode -- 池化模式 ("max" 或 "average")
    hparameters -- 超参数字典，包含 "stride" 和 "f"

    返回:
    dA_prev -- 输入的梯度，形状为 (m, n_H_prev, n_W_prev, n_C_prev)
    """
    # 从缓存中提取信息
    (A_prev, hparameters) = cache

    # 获取超参数
    stride = hparameters["stride"]
    f = hparameters["f"]

    # 获取维度
    (m, n_H, n_W, n_C) = dA.shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # 初始化输出梯度
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):  # 遍历样本
        a_prev = A_prev[i]
        for h in range(n_H):  # 遍历高度
            for w in range(n_W):  # 遍历宽度
                for c in range(n_C):  # 遍历通道
                    # 计算池化窗口的起始位置
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    # 边界条件检查
                    if vert_end > n_H_prev or horiz_end > n_W_prev:
                        continue

                    # 根据模式计算梯度
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i, h, w, c]
                    elif mode == "average":
                        da = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += da / (f * f)

    return dA_prev


# 测试代码
np.random.seed(1)
dA_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride": 1, "f": 2}

# 第一次调用
dA, cache = (np.random.randn(5, 4, 2, 2), (dA_prev, hparameters))
dA_prev = pool_backward(dA, cache, mode="max")
print("mode = max")
print("mean of dA = ", np.mean(dA))
print("dA_prev[1,1] = ", dA_prev[1, 1])
print()

# 第二次调用
dA_prev = pool_backward(dA, cache, mode="average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print("dA_prev[1, 1] = ", dA_prev[1, 1])
