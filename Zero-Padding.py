import numpy as np
def x_pad(x, pad):
    # 使用 numpy 的 pad 函数进行填充
    return np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = x_pad(x, 2)

print("x.shape =", x.shape)
print("x_pad.shape =", x_pad.shape)
print("x[1,1] =", x[1,1])
print("x_pad[1,1] =", x_pad[1,1])

import matplotlib.pyplot as plt

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])
plt.show()
