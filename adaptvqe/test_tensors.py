import unittest
import itertools as it
import numpy as np
from scipy.linalg import norm
import openfermion as of
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from adaptvqe.pools import FullPauliPool, ImplementationType
from adaptvqe.tensor_helpers import computational_basis_mps
from adaptvqe.matrix_tools import ket_to_vector
from adaptvqe.pools import FullPauliPool, ImplementationType
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt, TensorNetAdapt

class TestMPSConstruction(unittest.TestCase):

    def test_binary_cb_state(self):
        qnums = [0, 1, 1]
        generated_mps = computational_basis_mps(qnums)
        mps_dense = generated_mps.to_dense()
        # Construct the state exactly.
        psi = np.zeros(2 ** 3, dtype=complex)
        psi[3] = 1.
        psi = psi.reshape(mps_dense.shape)
        assert np.allclose(psi, mps_dense)


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

class TestGradients(unittest.TestCase):

    def test_one_operator(self):
        max_mpo_bond = 100
        max_mps_bond = 100
        l = 3
        j_xy = 1
        j_z = 1
        h = XXZHamiltonian(j_xy, j_z, l)
        pool = FullPauliPool(n=l, max_mpo_bond=max_mpo_bond)
        pool.imp_type = ImplementationType.SPARSE

        linalg_adapt = LinAlgAdapt(
            pool=deepcopy(pool),
            custom_hamiltonian=h,
            verbose=True,
            threshold=10**-5,
            max_adapt_iter=5,
            max_opt_iter=10000,
            sel_criterion="gradient",
            recycle_hessian=False,
            rand_degenerate=True,
            max_mpo_bond=max_mpo_bond,
            max_mps_bond=max_mps_bond
        )
        tn_adapt = TensorNetAdapt(
            pool=deepcopy(pool),
            custom_hamiltonian=h,
            verbose=True,
            threshold=10**-5,
            max_adapt_iter=5,
            max_opt_iter=10000,
            sel_criterion="gradient",
            recycle_hessian=False,
            rand_degenerate=True,
            max_mpo_bond=max_mpo_bond,
            max_mps_bond=max_mps_bond
        )

        test_bools = []
        for idx in range(len(pool.operators)):
            indices = [idx]
            for coeff in np.linspace(1e-4, 2.0, num=10):
                coefficients = [coeff]
                tn_grad = np.array(tn_adapt.estimate_gradients(coefficients=coefficients, indices=indices))
                linalg_grad = np.array(linalg_adapt.estimate_gradients(coefficients=coefficients, indices=indices))
                test_bools.append(norm(tn_grad - linalg_grad) <= 1e-4)
        self.assertTrue(test_bools)

    def test_two_operators(self):
        max_mpo_bond = 100
        max_mps_bond = 100
        l = 3
        j_xy = 1
        j_z = 1
        h = XXZHamiltonian(j_xy, j_z, l)
        pool = FullPauliPool(n=l, max_mpo_bond=max_mpo_bond)
        pool.imp_type = ImplementationType.SPARSE

        linalg_adapt = LinAlgAdapt(
            pool=deepcopy(pool),
            custom_hamiltonian=h,
            verbose=True,
            threshold=10**-5,
            max_adapt_iter=5,
            max_opt_iter=10000,
            sel_criterion="gradient",
            recycle_hessian=False,
            rand_degenerate=True,
            max_mpo_bond=max_mpo_bond,
            max_mps_bond=max_mps_bond
        )
        tn_adapt = TensorNetAdapt(
            pool=deepcopy(pool),
            custom_hamiltonian=h,
            verbose=True,
            threshold=10**-5,
            max_adapt_iter=5,
            max_opt_iter=10000,
            sel_criterion="gradient",
            recycle_hessian=False,
            rand_degenerate=True,
            max_mpo_bond=max_mpo_bond,
            max_mps_bond=max_mps_bond
        )

        test_bools = []
        for inds in it.combinations(range(len(pool.operators)), 2):
            indices = list(inds)
            for coeffs in it.combinations(np.linspace(1e-4, 2.0, num=10), 2):
                coefficients = list(coeffs)
                tn_grad = np.array(tn_adapt.estimate_gradients(coefficients=coefficients, indices=indices))
                linalg_grad = np.array(linalg_adapt.estimate_gradients(coefficients=coefficients, indices=indices))
                if norm(linalg_grad) <= 1e-8:
                    rel_error = norm(tn_grad)
                else:
                    rel_error = norm(tn_grad - linalg_grad) / norm(linalg_grad)
                test_bool = rel_error <= 1e-4
                if not test_bool:
                    if norm(np.abs(tn_grad) - np.abs(linalg_grad)) / norm(np.abs(linalg_grad)) <= 1e-4:
                        print(f"Off by signs.")
                    else:
                        print(f"Got relative error {rel_error:4.5e} from gradients\n{linalg_grad}, {tn_grad}\nwith indices={indices}.")
                test_bools.append(test_bool)
        self.assertTrue(test_bools)


if __name__ == "__main__":
    unittest.main()