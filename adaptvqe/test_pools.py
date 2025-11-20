import unittest
import numpy as np
import openfermion as of
from quimb.tensor.tensor_1d import MatrixProductOperator
from adaptvqe.pools import FullPauliPool, ImplementationType
from adaptvqe.tensor_helpers import computational_basis_mps
from adaptvqe.matrix_tools import ket_to_vector

class TestPauliPool(unittest.TestCase):

    def test_convert_to_mpo(self):
        nq = 2
        pool = FullPauliPool(n=nq, max_mpo_bond=10)
        pool.imp_type = ImplementationType.SPARSE
        all_close_bools = []
        for i in range(len(pool.operators)):
            mpo_op = pool.get_mpo_op(i)
            sparse_op = pool.get_imp_op(i).toarray()
            sparse_mpo = MatrixProductOperator.from_dense(sparse_op)
            all_close_bools.append((sparse_mpo - mpo_op).norm().real <= 1e-5)
        self.assertTrue(all(all_close_bools))


class TestOperatorExponential(unittest.TestCase):

    def test_one_operator_exp(self):
        cb_nums = [1, 1, 0]
        psi_mps = computational_basis_mps(cb_nums)
        psi_mat = ket_to_vector(cb_nums)
        pool = FullPauliPool(n=3, max_mpo_bond=100)
        pool.imp_type = ImplementationType.SPARSE

        test_bools = []
        for i in range(len(pool.operators)):
            psi_out_mps = pool.tn_expm_mult_state(1.0, i, psi_mps)
            psi_out_mps_vec = psi_out_mps.to_dense()
            psi_out_mat = pool.expm_mult(1.0, i, psi_mat).reshape(psi_out_mps_vec.shape)
            test_bool = np.allclose(psi_out_mps_vec, psi_out_mat)
            test_bools.append(test_bool)
        self.assertTrue(all(test_bools))

if __name__ == "__main__":
    unittest.main()