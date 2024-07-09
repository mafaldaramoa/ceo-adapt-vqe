# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:47:40 2022
Implement ADAPT-VQE for the XXZ Hamiltonian using operator pool tiling.
See https://doi.org/10.48550/arXiv.2206.14215

@author: mafal
"""
import sys
# You should replace this with your path. Alternatively, permanently modify PYTHONPATH.
sys.path.append('/mnt/c/Users/mafal/OneDrive/Computador/Doutoramento/Code/Packages/adaptvqepublic')

from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian

l = 3
j_xy = 1
j_z = 1
h = XXZHamiltonian(j_xy,j_z,l)
pool = FullPauliPool(n=l)

# Run 200 iterations of ADAPT-VQE for small problem instance, selecting randomly among degenerate gradients.
# Form a list of all unique operators ever selected for this small instance.
ixs = []
for _ in range(100):
    my_adapt = LinAlgAdapt(pool=pool,
                           custom_hamiltonian = h,
                           verbose=False,
                           threshold=10**-5,
                           max_adapt_iter=5,
                           max_opt_iter=10000,
                           sel_criterion="gradient",
                           recycle_hessian=False,
                           rand_degenerate=True,
                           )
    my_adapt.run()
    data = my_adapt.data
    for i in data.result.ansatz.indices:
        if i not in ixs:
            ixs.append(i)

print(f"Pool will be tiled from {len(ixs)} ops")

# Tile the operators to form a pool for a larger problem instance and run ADAPT-VQE with this pool
source_ops = [pool.operators[index].operator for index in ixs]
new_l = 8
tiled_pool = TiledPauliPool(n=new_l, source_ops = source_ops)

new_h = XXZHamiltonian(j_xy,j_z,new_l)

my_adapt = LinAlgAdapt(pool=tiled_pool,
                       custom_hamiltonian=new_h,
                       verbose=True,
                       threshold=10 ** -2,
                       max_adapt_iter=50,
                       max_opt_iter=10000,
                       sel_criterion="gradient",
                       recycle_hessian=False
                       )
my_adapt.run()
error = (new_h.ground_energy-my_adapt.data.result.energy)/new_h.ground_energy
print("Final relative error:",error)