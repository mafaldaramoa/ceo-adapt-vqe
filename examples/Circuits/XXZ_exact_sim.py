"""Using TensorNetAdapt, optimize the Ansatz at a low bond dimension. Then
simulate the resulting circuit exactly and see what the exact energy is."""

import numpy as np
from scipy.sparse.linalg import expm, expm_multiply

from openfermion import get_sparse_operator
from qiskit.quantum_info import Operator, process_fidelity

from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.circuits import get_circuit_energy

l = 4
j_xy = 1
j_z = 1
h = XXZHamiltonian(j_xy, j_z, l)
pool = FullPauliPool(n=l)

exact_energy = h.ground_energy

max_mpo_bond = 100
max_mps_bond = 2
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
    max_mpo_bond=max_mpo_bond,
    max_mps_bond=max_mps_bond
)
my_adapt.run()
data = my_adapt.data
adapt_energy = my_adapt.energy

coefficients = data.result.ansatz.coefficients
indices = data.result.ansatz.indices

qc = data.get_circuit(pool,include_ref=True)
energy = get_circuit_energy(qc,h.operator)

print("\nEnergy from TNAdapt: ", adapt_energy)
adapt_err = abs(exact_energy - adapt_energy) / abs(exact_energy)
print(f"Error from circuit: {adapt_err:4.5e}")
print("\nEnergy from circuit: ", energy)
circuit_err = abs(exact_energy - energy) / abs(exact_energy)
print(f"Error from circuit: {circuit_err:4.5e}")