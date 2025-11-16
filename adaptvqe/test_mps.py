import unittest
import numpy as np
from quimb.tensor.tensor_1d import MatrixProductState
from adaptvqe.tensor_helpers import computational_basis_mps

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


if __name__ == "__main__":
    unittest.main()