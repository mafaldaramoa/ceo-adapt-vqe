# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:47:40 2022
Implement ADAPT-VQE for the XXZ Hamiltonian using operator pool tiling.
See https://doi.org/10.48550/arXiv.2206.14215

@author: mafal
"""

from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt
from adaptvqe.hamiltonians import XXZHamiltonian

l = 3
j_xy = 1
j_z = 1
h = XXZHamiltonian(j_xy, j_z, l)
pool = FullPauliPool(n=l)

my_adapt = TensorNetAdapt(
    pool=pool,
    custom_hamiltonian=h,
    verbose=False,
    threshold=10**-5,
    max_adapt_iter=5,
    max_opt_iter=10000,
    sel_criterion="gradient",
    recycle_hessian=False,
    rand_degenerate=True,
)