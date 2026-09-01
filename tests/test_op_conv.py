import unittest
import openfermion as of
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from adaptvqe.op_conv import to_qiskit_term

class TestToTerm(unittest.TestCase):

    def test_xiii_little_endian(self):
        term = of.QubitOperator("X0")
        converted = to_qiskit_term(term, 4, True)
        target = Pauli("IIIX")
        self.assertEqual(target, converted)

    def test_xiii_big_endian(self):
        term = of.QubitOperator("X0")
        converted = to_qiskit_term(term, 4, False)
        target = Pauli("XIII")
        self.assertEqual(target, converted)

    def test_ixyz_little_endian(self):
        term = of.QubitOperator("X1 Y2 Z3")
        converted = to_qiskit_term(term, 4, True)
        target = Pauli("ZYXI")
        self.assertEqual(target, converted)

    def test_ixyz_big_endian(self):
        term = of.QubitOperator("X1 Y2 Z3")
        converted = to_qiskit_term(term, 4, False)
        target = Pauli("IXYZ")
        self.assertEqual(target, converted)

    def test_yiyiy_little_endian(self):
        term = of.QubitOperator("Y0 Y2 Y4")
        converted = to_qiskit_term(term, 5, True)
        target = Pauli("YIYIY")
        self.assertEqual(target, converted)

    def test_yiyiy_big_endian(self):
        term = of.QubitOperator("Y0 Y2 Y4")
        converted = to_qiskit_term(term, 5, False)
        target = Pauli("YIYIY")
        self.assertEqual(target, converted)

    def test_iixx_little_endian(self):
        term = of.QubitOperator("X2 X3")
        converted = to_qiskit_term(term, 4, False)
        target = Pauli("IIXX")
        self.assertEqual(target, converted)




if __name__ == "__main__":
    unittest.main()