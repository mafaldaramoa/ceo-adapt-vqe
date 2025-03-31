# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:49:30 2022

@author: mafal
"""

import numpy as np

from openfermion import FermionOperator

# Chemical accuracy threshold (au)
chemical_accuracy = 1.5936 * 10**-3


def get_hf_det(electron_number, qubit_number):
    """
    Get the Hartree Fock ket |1>|1>...|0>|0>.

    Arguments:
    electron_number (int): the number of electrons of the molecule.
    qubit_number (int): the number of qubits necessary to represent the molecule
      (equal to the number of spin orbitals we're considering active).

    Returns:
    reference_ket (list): a list of lenght qubit_number, representing the
      ket of the adequate computational basis state in big-endian ordering.
    """

    # Consider occupied the lower energy orbitals, until enough one particle
    # states are filled
    reference_ket = [1 for _ in range(electron_number)]

    # Consider the remaining orbitals empty
    reference_ket += [0 for _ in range(qubit_number - electron_number)]

    return reference_ket


def normalize_op(operator):
    """
    Normalize Qubit or Fermion Operator by forcing the absolute values of the coefficients to sum to zero.
    This function modifies the operator.

    Arguments:
        operator (Union[FermionOperator,QubitOperator]): the operator to normalize

    Returns:
        operator (Union[FermionOperator,QubitOperator]): the same operator, now normalized0
    """

    if operator:
        coeff = 0
        for t in operator.terms:
            coeff_t = operator.terms[t]
            # coeff += np.abs(coeff_t * coeff_t)
            coeff += np.abs(coeff_t)

        # operator = operator/np.sqrt(coeff)
        operator = operator / coeff

    return operator


def create_spin_adapted_one_body_op(p, q):
    """
    Returns a spin-adapted excitation from spatial orbital q to spatial orbital p.
    We assume the OF orbital ordering (alternating up and down spin, spin-orbitals
    corresponding to the same spatial orbital are adjacent numbers).
    """
    e_pq = FermionOperator(f"{2 * p}^ {2 * q}", 1) + FermionOperator(
        f"{2 * p + 1}^ {2 * q + 1}"
    )
    e_qp = FermionOperator(f"{2 * q}^ {2 * p}", 1) + FermionOperator(
        f"{2 * q + 1}^ {2 * p + 1}"
    )

    op = e_pq - e_qp
    op = normalize_op(op)

    return op

def convert_orbital_index(i,n):
    """
    Converts the index of a spin-orbital
    From: all-alpha then all-beta (e.g. Qiskit)
    To: alternating alpha/beta (e.g. Openfermion)

    Arguments:
        i (int): original index in all-alpha then all-beta ordering
        n (int): total number of qubits
    """
    if i >= n/2:
        return int(1 + 2*(i - n/2))
    else:
        return int(2*i)