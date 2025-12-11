from typing import Tuple
import numpy as np
from quimb.tensor.tensor_1d import MatrixProductOperator

def fill_fun(shape: Tuple[int], fill_val: float=1.) -> np.ndarray:
    return fill_val * np.ones(shape)

L=1
chi=2
mpo1 = MatrixProductOperator.from_fill_fn(lambda shape: fill_fun(shape, 1), L, chi)
mpo2 = MatrixProductOperator.from_fill_fn(lambda shape: fill_fun(shape, -3), L, chi)
mpo_sum = mpo1 + mpo2
matrix_sum = mpo1.to_dense() + mpo2.to_dense()

print(np.allclose(mpo_sum.to_dense(), matrix_sum))