# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 09:47:40 2022

@author: mafal
"""

from adaptvqe.molecules import create_h2
from adaptvqe.pools import DVE_CEO, FullPauliPool
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt

r = 1.5
molecule = create_h2(r)
pool = DVE_CEO(molecule)
# pool = FullPauliPool(molecule)

my_adapt = TensorNetAdapt(
    pool=pool,
    molecule=molecule,
    max_adapt_iter=1,
    recycle_hessian=True,
    tetris=True,
    verbose=True,
    threshold=0.1,
)

my_adapt.run()
data = my_adapt.data

# Access the final ansatz indices and coefficients
print("Evolution of ansatz indices: ", data.evolution.indices)
print(
    "Final operators in the ansatz: ",
    [pool.get_op(index) for index in data.result.ansatz.indices],
)
print("Evolution of ansatz coefficients: ", data.evolution.coefficients)

# Access the number of function evaluations, gradient evaluations, optimizer iterations for each ADAPT-VQE iteration
print("Function evaluations throughout the iterations:", data.evolution.nfevs)
print("Gradient evaluations throughout the iterations:", data.evolution.ngevs)
print("Optimizer iterations throughout the iterations:", data.evolution.nits)

# Create the circuit implementing the final ansatz
qc = pool.get_circuit(data.result.ansatz.indices, data.result.ansatz.coefficients)
print("Final ansatz circuit:\n", qc)

# Access the number of CNOTs and CNOT depth at each iteration
print("Evolution of ansatz CNOT counts: ", data.acc_cnot_counts(pool))
print("Evolution of ansatz CNOT depths: ", data.acc_cnot_depths(pool))

print("Final coefficients and indices:")
print(my_adapt.coefficients)
print(my_adapt.indices)
print("Final state/initial state fidelity:")
print(abs(my_adapt.tn_ref_state @ my_adapt.state) ** 2)
