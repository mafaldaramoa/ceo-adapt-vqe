from copy import deepcopy
import unittest
import itertools as it
import numpy as np
from scipy.linalg import norm
import openfermion as of
from quimb.tensor.tensor_1d import MatrixProductOperator
from adaptvqe.pools import FullPauliPool, ImplementationType
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt, TensorNetAdapt

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
                test_bools.append(norm(tn_grad - linalg_grad) <= 1e-4)
        self.assertTrue(test_bools)

if __name__ == "__main__":
    unittest.main()