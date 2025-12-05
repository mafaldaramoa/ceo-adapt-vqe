from typing import List, Optional
import numpy as np
import cirq
import openfermion as of
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct

def tensor_product_mpo(matrices: List[np.ndarray]) -> MatrixProductOperator:
    """Returns an MPO of the form M1 otimes M2 otimes ... otimes Mn,
    where Mi is a matrix. This MPO has bond dimension 1."""

    if len(matrices) == 0:
        raise ValueError("No matrices were passed.")

    tensors: List[np.ndarray] = []
    for i, m in enumerate(matrices):
        if i == 0:
            if len(matrices) != 1:
                tensors.append(m.reshape((2, 2, 1)))
            else:
                tensors.append(m)
        elif i == len(matrices) - 1:
            tensors.append(m.reshape((1, 2, 2)))
        else:
            tensors.append(m.reshape((1, 2, 2, 1)))
    return MatrixProductOperator(tensors, shape="ludr")


def pauli_string_to_mpo(pstring: cirq.PauliString, qs: List[cirq.Qid]) -> MatrixProductOperator:
    """Convert a Pauli string to a matrix product operator."""

    # Make a list of matrices for each operator in the string.
    ps_dense = pstring.dense(qs)
    matrices: List[np.ndarray] = []
    for pauli_int in ps_dense.pauli_mask:
        if pauli_int == 0:
            matrices.append(np.eye(2))
        elif pauli_int == 1:
            matrices.append(cirq.unitary(cirq.X))
        elif pauli_int == 2:
            matrices.append(cirq.unitary(cirq.Y))
        else: # pauli_int == 3
            matrices.append(cirq.unitary(cirq.Z))
    # Convert the matrices into tensors. We have a bond dim chi=1 for a Pauli string MPO.
    return pstring.coefficient * tensor_product_mpo(matrices)


def pauli_sum_to_mpo(psum: cirq.PauliSum, qs: List[cirq.Qid], max_bond: int, verbose: bool = False) -> MatrixProductOperator:
    """Convert a Pauli sum to an MPO."""
    nterms = len(psum)
    for i, p in enumerate(psum):
        if verbose:
            print(f"Status: On term {i + 1} / {nterms}", end="\r")
        if i == 0:
            mpo = pauli_string_to_mpo(p, qs)
        else:
            mpo += pauli_string_to_mpo(p, qs)
            tensor_network_1d_compress_direct(mpo, max_bond=max_bond, inplace=True)
    return mpo


def mpo_mps_exepctation(mpo: MatrixProductOperator, mps: MatrixProductState) -> complex:
    """Get the expectation of an operator given the state."""

    mpo_times_mps = mpo.apply(mps)
    return mps.H @ mpo_times_mps


def computational_basis_mps(qnums: List[int], d: int=2, **kwargs) -> MatrixProductState:
    """MPS for a computational basis state.
    
    Arguments:
    qnums - Quantum numbers for the c.b. state, e.g. [0, 1, 2, 1] for d=3.
    d - The dimensionality of the sites in the MPS. (assumed uniform)
    kwargs - Arguments to pass to the MatrixProductState constructor.
    
    Returns:
    mps - The MPS representation of the state."""

    tensors: List[np.ndarray] = []
    for i, qnum in enumerate(qnums):
        if qnum >= d:
            raise ValueError(f"Quantum number {qnum} at index {i} is greater than d-1 (d={d}).")
        if i == 0:
            # This is the leftmost tensor.
            this_tensor = np.zeros((d, 1), dtype=complex)
            this_tensor[qnum, 0] = 1.
        elif i == len(qnums) - 1:
            # This is the rightmost tensor.
            this_tensor = np.zeros((1, d), dtype=complex)
            this_tensor[0, qnum] = 1.
        else:
            # This tensor is somewhere in the middle.
            this_tensor = np.zeros((1, d, 1), dtype=complex)
            this_tensor[0, qnum, 0] = 1.
        tensors.append(this_tensor.copy())
    return MatrixProductState(tensors, shape='lpr', **kwargs)


def mps_to_vector(mps: MatrixProductState) -> np.ndarray:
    """Convert an MPS into a normal vector. This assumes each index is a string
    followed by a number, e.g. three indices 'k0, k1, k2'."""

    def _idx_to_int(idx: str) -> int:
        digits = [c for c in idx if c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
        if len(digits) == 0:
            raise ValueError(f"Index {str} has no digits in it.")
        return int(''.join(digits))
    
    # Contract the MPS into a tensor, then sort the indices. Convert that to a vector.
    contracted_tensor = mps.contract()
    sorted_inds = sorted(contracted_tensor.inds, key=_idx_to_int)
    contracted_tensor.transpose(*sorted_inds, inplace=True)
    tensor_data = contracted_tensor.data
    return tensor_data.reshape((tensor_data.size,))


def qubop_to_mpo(qubop: of.QubitOperator, max_bond: int, nq: Optional[int]=None) -> MatrixProductOperator:
    """Convert an openfermion QubitOperator to an MPO."""

    if nq is None:
        nq = of.utils.count_qubits(qubop)

    for i, (paulis, coeff) in enumerate(qubop.terms.items()):
        matrices = [np.eye(2).astype(complex) for _ in range(nq)]
        for idx, pauli in paulis:
            if pauli == 'X':
                matrices[idx] = np.array([[0., 1.], [1., 0.]]).astype(complex)
            elif pauli == 'Y':
                matrices[idx] = np.array([[0., -1j], [1j, 0.]])
            else: # By default this is a Z.
                matrices[idx] = np.array([[1., 0.], [0, -1.]]).astype(complex)
        term_mpo = tensor_product_mpo(matrices)
        if i == 0:
            total_mpo = coeff * term_mpo
        else:
            total_mpo += coeff * term_mpo
    return total_mpo