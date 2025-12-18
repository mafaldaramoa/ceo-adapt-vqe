"""Confirm that LinAlgAdapt and TensorNetAdapt produce the same states."""

import unittest
from copy import deepcopy
import numpy as np
from scipy.linalg import norm
from quimb.tensor.tensor_1d import MatrixProductState
from qiskit.quantum_info import Operator
from adaptvqe.molecules import create_h2
from adaptvqe.pools import DVE_CEO, FullPauliPool, ImplementationType
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt, TensorNetAdapt

class TestH2Molecule(unittest.TestCase):

    def test_pauli_pool(self):
        r = 1.5
        molecule = create_h2(r)
        pool = FullPauliPool(molecule)

        la_adapt = LinAlgAdapt(
            pool=deepcopy(pool),
            molecule=molecule,
            max_adapt_iter=1,
            recycle_hessian=True,
            tetris=True,
            verbose=True,
            threshold=0.1,
        )
        tn_adapt = TensorNetAdapt(
            pool=deepcopy(pool),
            molecule=molecule,
            max_adapt_iter=1,
            recycle_hessian=True,
            tetris=True,
            verbose=True,
            threshold=0.1,
        )

        coeff = 0.2
        index = len(pool.operators) - 1
        la_state = la_adapt.get_state([coeff], [index]).todense()
        la_state_mps = MatrixProductState.from_dense(la_state)
        tn_state = tn_adapt.get_state([coeff], [index])
        fidelity = abs(la_state_mps.H @ tn_state) ** 2
        self.assertTrue(fidelity >= 0.9)
    
    def test_circuit_unitary_error_pauli(self):
        r = 1.5
        molecule = create_h2(r)
        pool = FullPauliPool(molecule)
        pool.imp_type = ImplementationType.SPARSE
        index = 2
        coefficient = 0.5
        u = pool.expm(coefficient, index).todense()
        ckt = pool.get_circuit([index], [coefficient])
        ckt_op = Operator.from_circuit(ckt)
        u_ckt = ckt_op.data
        self.assertTrue(np.allclose(u_ckt, u))

    def test_circuit_unitary_error(self):
        r = 1.5
        molecule = create_h2(r)
        pool = DVE_CEO(molecule)
        pool.imp_type = ImplementationType.SPARSE
        index = 3
        coefficient = 0.5
        u = pool.expm(coefficient, index).todense()
        ckt = pool.get_circuit([index], [coefficient])
        ckt_op = Operator.from_circuit(ckt)
        u_ckt = ckt_op.data
        self.assertTrue(np.allclose(u_ckt, u))

    def test_dev_ceo_pool(self):
        r = 1.5
        molecule = create_h2(r)
        pool = DVE_CEO(molecule)

        la_adapt = LinAlgAdapt(
            pool=deepcopy(pool),
            molecule=molecule,
            max_adapt_iter=1,
            recycle_hessian=True,
            tetris=True,
            verbose=True,
            threshold=0.1,
        )
        tn_adapt = TensorNetAdapt(
            pool=deepcopy(pool),
            molecule=molecule,
            max_adapt_iter=1,
            recycle_hessian=True,
            tetris=True,
            verbose=True,
            threshold=0.1,
        )

        coeff = 0.2
        index = 0
        print(f"operator: {pool.operators[index]}")
        la_state = la_adapt.get_state([coeff], [index]).todense()
        la_state_mps = MatrixProductState.from_dense(la_state)
        tn_state = tn_adapt.get_state([coeff], [index])
        fidelity = abs(la_state_mps.H @ tn_state) ** 2
        print(f"fidelity = {fidelity:8.9e}")
        self.assertTrue(fidelity >= 0.9)

if __name__ == "__main__":
    unittest.main()