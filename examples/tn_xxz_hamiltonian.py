# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:47:40 2022
Implement ADAPT-VQE for the XXZ Hamiltonian using operator pool tiling.
See https://doi.org/10.48550/arXiv.2206.14215

@author: mafal
"""

from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt, LinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian

max_mpo_bond = 100
max_mps_bond = 100

l = 3
j_xy = 1
j_z = 1
h = XXZHamiltonian(j_xy, j_z, l)
pool = FullPauliPool(n=l, max_mpo_bond=max_mpo_bond)

my_adapt = LinAlgAdapt(
    pool=pool,
    custom_hamiltonian=h,
    verbose=True,
    threshold=10**-5,
    max_adapt_iter=5,
    max_opt_iter=10000,
    sel_criterion="gradient",
    recycle_hessian=False,
    rand_degenerate=True,
    max_mpo_bond=max_mpo_bond,
    max_mps_bond=max_mps_bond
)
my_adapt.run()

print("Starting ADAPT with tensor networks.")

tn_adapt = TensorNetAdapt(
    pool=pool,
    custom_hamiltonian=h,
    verbose=True,
    threshold=10**-5,
    max_adapt_iter=5,
    max_opt_iter=10000,
    sel_criterion="gradient",
    recycle_hessian=False,
    rand_degenerate=True,
    max_mpo_bond=max_mpo_bond,
    max_mps_bond=max_mps_bond
)
tn_adapt.run()
print(tn_adapt.coefficients)