import numpy as np
def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask
np.random.seed(1)
x = np.random.randn(2,3)
mask = create_mask_from_window(x)
print('x = ', x)
print("mask = ", mask)