import numpy as np
import time
import torch
from torch import Tensor

n_a_rows = 4000
n_a_cols = 3000
n_b_rows = n_a_cols
n_b_cols = 200

a = np.arange(n_a_rows * n_a_cols).reshape(n_a_rows, n_a_cols) * 1.0
b = np.arange(n_b_rows * n_b_cols).reshape(n_b_rows, n_b_cols)
a_t = Tensor(a)
b_t = Tensor(b)

start = time.time()
d = a @ b
end = time.time()

print("time taken : {}".format(end - start))

start = time.time()
d_t = torch.matmul(a_t, b_t)
end = time.time()

print("time taken : {}".format(end - start))