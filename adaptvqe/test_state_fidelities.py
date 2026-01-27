"""Confirm that LinAlgAdapt and TensorNetAdapt produce the same states."""

import unittest
from copy import deepcopy
import numpy as np
from scipy.linalg import norm
from quimb.tensor.tensor_1d import MatrixProductState
from qiskit.quantum_info import Operator
from cirq import equal_up_to_global_phase
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
        all_close = []
        for index in range(len(pool.operators)):
            coefficient = np.random.rand()
            u = pool.expm(coefficient, index).todense()
            ckt = pool.get_circuit([index], [coefficient])
            ckt_op = Operator.from_circuit(ckt)
            u_ckt = ckt_op.data
            all_close.append(np.allclose(u_ckt, u))
        self.assertTrue(all(all_close))

    def test_circuit_unitary_error(self):
        r = 1.5
        molecule = create_h2(r)
        pool = DVE_CEO(molecule)
        pool.imp_type = ImplementationType.SPARSE
        all_close = []
        for index in range(len(pool.operators)):
            coefficient = np.random.rand()
            u = pool.expm(coefficient, index).todense()
            ckt = pool.get_circuit([index], [coefficient])
            ckt_op = Operator.from_circuit(ckt)
            u_ckt = ckt_op.data
            # ac = np.allclose(u_ckt, u)
            ac = equal_up_to_global_phase(u_ckt, u)
            all_close.append(ac)
        self.assertTrue(all(all_close))

    def test_dev_ceo_random_state(self):
        r = 1.5
        molecule = create_h2(r)
        pool = DVE_CEO(molecule)
        pool.imp_type = ImplementationType.SPARSE
        nq = pool.n
        psi = np.random.rand(2 ** nq).astype(complex)
        psi = psi / norm(psi)
        psi_mps = MatrixProductState.from_dense(psi)
        coefficient = np.random.rand()
        all_close = []
        for index in range(len(pool.operators)):
            u_psi = pool.expm_mult(coefficient, index, psi)
            u_psi_mps = pool.tn_expm_mult_state(coefficient, index, psi_mps, 100, big_endian=False)
            u_psi_mps_vec = u_psi_mps.to_dense()
            fidelity = abs(np.vdot(u_psi_mps_vec, u_psi)) ** 2
            all_close.append(fidelity >= 0.9)
        self.assertTrue(all(all_close))

    def test_dev_ceo_random_state_circuit(self):
        r = 1.5
        molecule = create_h2(r)
        pool = DVE_CEO(molecule)
        pool.imp_type = ImplementationType.SPARSE
        nq = pool.n
        psi = np.random.rand(2 ** nq).astype(complex)
        psi = psi / norm(psi)
        all_close = []
        for index in range(len(pool.operators)):
            coefficient = np.random.rand()
            u_psi = pool.expm_mult(coefficient, index, psi)
            u_psi_ckt = pool.expm_mult_circuit(coefficient, index, psi)
            fidelity = abs(np.vdot(u_psi, u_psi_ckt)) ** 2
            all_close.append(fidelity >= 0.9)
        self.assertTrue(all(all_close))

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
            max_mps_bond=100
        )

        coeff = 0.2
        index = 2
        la_state = la_adapt.get_state([coeff], [index]).todense()
        la_state_mps = MatrixProductState.from_dense(la_state)
        tn_state = tn_adapt.get_state([coeff], [index])
        fidelity = abs(la_state_mps.H @ tn_state) ** 2
        self.assertTrue(fidelity >= 0.9)

if __name__ == "__main__":
    unittest.main()