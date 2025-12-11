import unittest
import numpy as np
from scipy.linalg import norm
import cirq
import openfermion as of
from quimb.tensor.tensor_1d import MatrixProductState, MatrixProductOperator
from adaptvqe.tensor_helpers import qubop_to_mpo, pauli_sum_to_mpo

class TestPauliSumToMPO(unittest.TestCase):

    def test_single_qubit(self):
        qs = cirq.LineQubit.range(1)
        qubit_map = {q: i for i,q in enumerate(qs)}
        qubop = cirq.PauliSum.from_pauli_strings(0.75 * cirq.X.on(qs[0]))
        nq = len(qs)
        qubop_mpo = pauli_sum_to_mpo(qubop, qs, 100)
        qubop_matrix = qubop.matrix(qubit_map)
        self.assertTrue(np.allclose(qubop_mpo.to_dense(), qubop_matrix))

    def test_single_qubit_two_terms(self):
        qs = cirq.LineQubit.range(1)
        qubit_map = {q: i for i,q in enumerate(qs)}
        qubop = 0.75 * cirq.Y.on(qs[0]) + -1.5 * cirq.Z.on(qs[0])
        nq = len(qs)
        qubop_mpo = pauli_sum_to_mpo(qubop, qs, 100)
        qubop_matrix = qubop.matrix(qubit_map)
        self.assertTrue(np.allclose(qubop_mpo.to_dense(), qubop_matrix))

    def test_two_qubits_two_terms(self):
        qs = cirq.LineQubit.range(2)
        qubit_map = {q: i for i,q in enumerate(qs)}
        qubop = 0.75 * cirq.Y.on(qs[0]) + -1.5 * cirq.X.on(qs[0]) * cirq.Z.on(qs[1])
        nq = len(qs)
        qubop_mpo = pauli_sum_to_mpo(qubop, qs, 100)
        qubop_matrix = qubop.matrix(qubit_map)
        self.assertTrue(np.allclose(qubop_mpo.to_dense(), qubop_matrix))

    def test_three_qubits_two_terms(self):
        qs = cirq.LineQubit.range(3)
        qubit_map = {q: i for i,q in enumerate(qs)}
        qubop = 0.75 * cirq.Y.on(qs[0]) + -1.5 * cirq.X.on(qs[0]) * cirq.Z.on(qs[2])
        nq = len(qs)
        qubop_mpo = pauli_sum_to_mpo(qubop, qs, 100)
        qubop_matrix = qubop.matrix(qubit_map)
        self.assertTrue(np.allclose(qubop_mpo.to_dense(), qubop_matrix))


class TestQubOpToMPO(unittest.TestCase):

    def test_single_qubit(self):
        qubop = 0.75 * of.QubitOperator("X0")
        nq = of.utils.count_qubits(qubop)
        qubop_mpo = qubop_to_mpo(qubop, 100)
        qubop_matrix = of.linalg.get_sparse_operator(qubop).todense()
        self.assertTrue(np.allclose(qubop_mpo.to_dense(), qubop_matrix))

    def test_single_qubit_two_terms(self):
        qubop = 0.75 * of.QubitOperator("Y0") + -1.5 * of.QubitOperator("Z0")
        nq = of.utils.count_qubits(qubop)
        qubop_mpo = qubop_to_mpo(qubop, 100)
        qubop_matrix = of.linalg.get_sparse_operator(qubop).todense()
        self.assertTrue(np.allclose(qubop_mpo.to_dense(), qubop_matrix))

    def test_two_qubits_two_terms(self):
        qubop = 0.75 * of.QubitOperator("Y0") - 1.5 * of.QubitOperator("X0 Z1")
        qubop_mpo = qubop_to_mpo(qubop, 100)
        qubop_matrix = of.linalg.get_sparse_operator(qubop).todense()
        self.assertTrue(np.allclose(qubop_mpo.to_dense(), qubop_matrix))

    def test_three_qubits_two_terms(self):
        # qubop = 0.75 * cirq.Y.on(qs[0]) + -1.5 * cirq.X.on(qs[0]) * cirq.Z.on(qs[2])
        qubop = 0.35 * of.QubitOperator("Y0") - 0.9 * of.QubitOperator("X0 Z2")
        qubop_mpo = qubop_to_mpo(qubop, 100)
        qubop_matrix = of.linalg.get_sparse_operator(qubop).todense()
        self.assertTrue(np.allclose(qubop_mpo.to_dense(), qubop_matrix))


if __name__ == "__main__":
    unittest.main()