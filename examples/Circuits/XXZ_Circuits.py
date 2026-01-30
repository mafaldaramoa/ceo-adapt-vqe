# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:47:40 2022
Implement ADAPT-VQE for the XXZ Hamiltonian.

@author: mafal
"""

import numpy as np
from scipy.sparse.linalg import expm, expm_multiply

from openfermion import get_sparse_operator
from qiskit.quantum_info import Operator, process_fidelity

from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.circuits import get_circuit_energy

l = 4
j_xy = 1
j_z = 1
h = XXZHamiltonian(j_xy, j_z, l)
pool = FullPauliPool(n=l)

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
)
my_adapt.run()
data = my_adapt.data

coefficients = data.result.ansatz.coefficients
indices = data.result.ansatz.indices

# Create full ansatz unitary
goal_unitary = my_adapt.create_ansatz_unitary(coefficients,indices)

# Transform Scipy sparse matrix to Numpy array and create a Qiskit Operator
goal_unitary = Operator(goal_unitary.todense())

# Obtain unitary from circuit implementation to compare against target obtained via matrix algebra
qc = pool.get_circuit(indices, coefficients)
unitary = Operator(qc)

# Make sure the circuits match
pf = process_fidelity(unitary, goal_unitary)
assert np.abs(pf-1) < 10**-6

# Obtain energy from circuit and make sure energies match
qc = data.get_circuit(pool,include_ref=True)
energy = get_circuit_energy(qc,h.operator)
print("\nEnergy from circuit: ", energy)
assert np.abs(energy-data.result.energy) < 10**-6