from typing import List
import numpy as np
import cirq
import openfermion as of
import quimb.tensor as qtn
from quimb.tensor.tensor_1d import MatrixProductOperator, MatrixProductState
from quimb.tensor.tensor_1d_compress import tensor_network_1d_compress_direct

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
    tensors: List[np.ndarray] = []
    for i, m in enumerate(matrices):
        if i == 0:
            tensors.append(m.reshape((2, 2, 1)))
        elif i == len(matrices) - 1:
            tensors.append(m.reshape((1, 2, 2)))
        else:
            tensors.append(m.reshape((1, 2, 2, 1)))
    return pstring.coefficient * MatrixProductOperator(tensors, shape="ludr")


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