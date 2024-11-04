from scipy.sparse.linalg import expm, expm_multiply
from openfermion import get_sparse_operator
from qiskit.quantum_info import Operator, process_fidelity
import numpy as np

from adaptvqe.pools import NoZPauliPool, DVG_CEO, QE
from adaptvqe.molecules import create_h2

# Define test case: molecule, ansatz size, coefficient list
r = 1.5
molecule = create_h2(r)
ansatz_size = 10
coefficients = np.random.random(ansatz_size)

# Decide which pools to test
pools = [NoZPauliPool(molecule), DVG_CEO(molecule), QE(molecule)]

print(f"Fidelity between unitaries generated from matrix multiplication and from circuit,"
      f" for random {ansatz_size} element ansatz with...")

for pool in pools:

    # Generate random ansatz with elements from current pool
    indices = np.random.randint(0, pool.size - 1, ansatz_size)

    # Convert list of indices to list of QubitOperators or FermionOperators
    ops = [pool.get_op(index) for index in indices]

    # Convert list of QubitOperators or FermionOperators to list of sparse matrices
    ops = [get_sparse_operator(op, molecule.n_qubits) for op in ops]

    # Obtain ansatz unitary by exponentiation and multiplication
    goal_unitary = None
    for op, c in zip(ops, coefficients):
        if goal_unitary is None:
            goal_unitary = expm(op * c)
        else:
            goal_unitary = expm_multiply(op * c, goal_unitary)

    # Transform Scipy sparse matrix to Numpy array and create a Qiskit Operator
    goal_unitary = Operator(goal_unitary.todense())

    # Obtain unitary from circuit implementation to compare against target obtained via matrix algebra
    qc = pool.get_circuit(indices, coefficients)
    unitary = Operator(qc)

    print(f"...{pool.name}: ",
          process_fidelity(unitary, goal_unitary))
