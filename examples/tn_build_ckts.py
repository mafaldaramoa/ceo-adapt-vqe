import numpy as np
from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt, LinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian

max_mpo_bond = 100
dmrg_mps_bond = 5 
adapt_mps_bond = 5
l = 4

j_xy = 1
j_z = 1
h = XXZHamiltonian(j_xy, j_z, l, diag_mode="quimb", max_mpo_bond=max_mpo_bond, max_mps_bond=dmrg_mps_bond)
dmrg_energy = h.ground_energy
print(f"Got DMRG energy {dmrg_energy:4.5e}")
pool = FullPauliPool(n=l, max_mpo_bond=max_mpo_bond) 

my_adapt = LinAlgAdapt(
        pool=pool,
        custom_hamiltonian=h,
        verbose=False,
        threshold=10**-5,
        max_adapt_iter=5,
        max_opt_iter=10000,
        sel_criterion="gradient",
        recycle_hessian=False,
        rand_degenerate=True,
        max_mpo_bond=100,
        max_mps_bond = 20
    )
my_adapt.run()
data = my_adapt.data
qc = pool.get_circuit(data.result.ansatz.indices, data.result.ansatz.coefficients)
print("Depth:", qc.depth())

tn_adapt = TensorNetAdapt(
        pool=pool,
        custom_hamiltonian=h,
        verbose=False,
        threshold=10**-5,
        max_adapt_iter=5,
        max_opt_iter=10000,
        sel_criterion="gradient",
        recycle_hessian=False,
        rand_degenerate=True,
        max_mpo_bond=100,
        max_mps_bond = 20
    )
tn_adapt.run()
data = tn_adapt.data
qc = pool.get_circuit(data.result.ansatz.indices, data.result.ansatz.coefficients)
print("Depth:", qc.depth())

energy_err = abs(tn_adapt.energy - my_adapt.energy)
print(f"Energy difference: {energy_err:4.5e}")
print("LinAlg indices:\n", my_adapt.indices)
for idx in my_adapt.indices:
    print(pool.get_q_op(idx))
print("LinAlg coefficients:\n", my_adapt.coefficients)
print("TN indices:\n", tn_adapt.indices)
for idx in tn_adapt.indices:
    print(pool.get_q_op(idx))
print("TN coefficients:\n", tn_adapt.coefficients)