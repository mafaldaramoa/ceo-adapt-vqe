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
    

def my_jordan_wigner(fermion_operator):

    # Initialize full qubit operator
    transformed_operator = QubitOperator()
    n = count_qubits(fermion_operator)

    # Initialize dictionary mapping fermionic ladder operators to linear combinations of Pauli strings
    lookup_ladder_terms = dict()

    for term in fermion_operator.terms:

        # Initialize qubit operator corresponding to this product of fermionic ladder operators
        transformed_term = QubitOperator((), fermion_operator.terms[term])

        # Loop through individual operators, transform and multiply.
        for ladder_operator in term:

            if ladder_operator not in lookup_ladder_terms:
                # We haven't transformed this ladder term yet

                # Get the index of the qubit the fermionic ladder operator acts on
                i = ladder_operator[0] 

                # Create anticommutation string
                z_string_ixs = [k for k in range(n) if k<i] # Regular JW transform
                # Use all-alpha then all-beta ordering for the Z string:
                #z_string_ixs = [k for k in range(n) if convert_orbital_from_of(k,n)<convert_orbital_from_of(i,n)]
                z_factors = tuple((index, 'Z') for index in z_string_ixs)

                # Get the Pauli X + Z string
                pauli_x_component = QubitOperator(z_factors + ((i, 'X'),), 0.5)

                # Get the Pauli Y + Z string with appropriated coefficient (- for creation, + for annihilation)
                if ladder_operator[1]:
                    pauli_y_component = QubitOperator(
                        z_factors + ((i, 'Y'),), -0.5j
                    )
                else:
                    pauli_y_component = QubitOperator(
                        z_factors + ((i, 'Y'),), 0.5j
                    )
                
                # Construct the full qubit operator associated with this fermionic ladder operator
                lookup_ladder_terms[ladder_operator] = pauli_x_component + pauli_y_component

            transformed_term *= lookup_ladder_terms[ladder_operator]

        transformed_operator += transformed_term

    return transformed_operator