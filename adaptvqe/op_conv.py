# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 09:20:36 2022

@author: mafal
"""

import numpy as np

from openfermion import (
    count_qubits,
    FermionOperator,
    QubitOperator,
    get_fermion_operator,
    InteractionOperator,
    jordan_wigner,
)
import qiskit
from qiskit.qasm3 import dumps

# todo: use stable version of qiskit only
if int(qiskit.__version__[0]) >= 1:
    # opflow has been deprecated
    from qiskit.quantum_info import Pauli

    I = Pauli("I")
    X = Pauli("X")
    Y = Pauli("Y")
    Z = Pauli("Z")
else:
    from qiskit.opflow import I, Z, X, Y

from .matrix_tools import string_to_matrix


def get_qasm(qc):
    """
    Converts a Qiskit QuantumCircuit to qasm.
    Args:
        qc (QuantumCircuit): a Qiskit QuantumCircuit

    Returns:
        qasm (str): the QASM string for this circuit
    """

    if int(qiskit.__version__[0]) >= 1:
        qasm = dumps(qc)
    else:
        qasm = qc.qasm()

    return qasm


def endian_conv(index, n):
    """
    Converts an index labeling a qubit in an n qubit register from big endian
    to little endian, or vice-versa.

    Arguments:
        index (int): the index of the qubit
        n (int): the total number of qubits of the register

    Returns:
        new_index (int): the index of the qubit, in the opposite endianness
    """
    new_index = n - index - 1

    return new_index


def to_qiskit_pauli(letter):
    """
    Transforms a letter representing a Pauli operator to the corresponding
    Qiskit observable.

    Arguments:
        letter (str): the letter representing the Pauli operator
    Returns:
        qiskit_Pauli (PauliOp): the corresponding operator in Qiskit
    """
    if letter == "X":
        qiskit_pauli = X
    elif letter == "Y":
        qiskit_pauli = Y
    elif letter == "Z":
        qiskit_pauli = Z
    else:
        raise ValueError(
            "Letter isn't recognized as a Pauli operator" " (must be X, Y or Z)."
        )

    return qiskit_pauli


def to_qiskit_term(of_term, n, switch_endianness):
    """
    Transforms an Openfermion term into a Qiskit Operator.
    Only works for individual Pauli strings. For generic operators, see to_qiskit_operator.

    Arguments:
        of_term (QubitOperator): a Pauli string multiplied by a coefficient, given as an Openfermion operator
        n (int): the size of the qubit register
        switch_endianness (bool): whether to revert the endianness
    Returns:
        qiskit_op (PauliSumOp): the original operator, represented in Qiskit
    """

    pauli_strings = list(of_term.terms.keys())

    if len(pauli_strings) > 1:
        raise ValueError(
            "Input must consist of a single Pauli string."
            " Use to_qiskit_operator for other operators."
        )
    pauli_string = pauli_strings[0]

    coefficient = of_term.terms[pauli_string]

    qiskit_op = None

    previous_index = -1

    for qubit_index, pauli in pauli_string:

        id_count = qubit_index - previous_index - 1

        if switch_endianness:
            new_ops = to_qiskit_pauli(pauli)
            for _ in range(id_count):
                new_ops = new_ops ^ I
            if qiskit_op is None:
                qiskit_op = new_ops
            else:
                qiskit_op = new_ops ^ qiskit_op
        else:
            new_ops = (I ^ id_count) ^ to_qiskit_pauli(pauli)
            qiskit_op = qiskit_op ^ new_ops

        previous_index = qubit_index

    id_count = (n - previous_index - 1)
    if switch_endianness:
        for _ in range(id_count):
            qiskit_op = I ^ qiskit_op
    else:
        for _ in range(id_count):
            qiskit_op = qiskit_op ^ I

    qiskit_op = coefficient * qiskit_op

    return qiskit_op


def to_qiskit_operator(of_operator, n=None, little_endian=True):
    """
    Transforms an Openfermion operator into a Qiskit Operator.

    Arguments:
        of_operator (QubitOperator): a linear combination of Pauli strings as an Openfermion operator
        n (int): the size of the qubit register
        little_endian (bool): whether to revert use little endian ordering
    Returns:
        qiskit_operator (PauliSumOp): the original operator, represented in Qiskit
    """

    # If of_operator is an InteractionOperator, shape it into a FermionOperator
    if isinstance(of_operator, InteractionOperator):
        of_operator = get_fermion_operator(of_operator)

    if not n:
        n = count_qubits(of_operator)

    # Now use the Jordan Wigner transformation to map the FermionOperator into
    # a QubitOperator
    if isinstance(of_operator, FermionOperator):
        of_operator = jordan_wigner(of_operator)

    qiskit_operator = None

    # Iterate through the terms in the operator. Each is a Pauli string
    # multiplied by a coefficient
    for term in of_operator.get_operators():
        qiskit_term = to_qiskit_term(term, n, little_endian)
        if qiskit_operator is None:
            qiskit_operator = qiskit_term
        else:
            qiskit_operator += qiskit_term

    return qiskit_operator


def find_substrings(main_string, hamiltonian, checked=[]):
    """
    Finds and groups all the strings in a Hamiltonian that only differ from
    main_string by identity operators.

    Arguments:
      main_string (str): a Pauli string (e.g. "XZ)
      hamiltonian (dict): a Hamiltonian (with Pauli strings as keys and their
        coefficients as values)
      checked (list): a list of the strings in the Hamiltonian that have already
        been inserted in another group

    Returns:
      grouped_operators (dict): a dictionary whose keys are boolean strings
        representing substrings of the main_string (e.g. if main_string = "XZ",
        "IZ" would be represented as "01"). It includes all the strings in the
        hamiltonian that can be written in this form (because they only differ
        from main_string by identities), except for those that were in checked
        (because they are already part of another group of strings).
      checked (list):  the same list passed as an argument, with extra values
        (the strings that were grouped in this function call).
    """

    grouped_operators = {}

    # Go through the keys in the dictionary representing the Hamiltonian that
    # haven't been grouped yet, and find those that only differ from main_string
    # by identities
    for pauli_string in hamiltonian:

        if pauli_string not in checked:
            # The string hasn't been grouped yet

            if all(
                (op1 == op2 or op2 == "I")
                for op1, op2 in zip(main_string, pauli_string)
            ):
                # The string only differs from main_string by identities

                # Represent the string as a substring of the main one
                boolean_string = "".join(
                    [
                        str(int(op1 == op2))
                        for op1, op2 in zip(main_string, pauli_string)
                    ]
                )

                # Add the boolean string representing this string as a key to
                # the dictionary of grouped operators, and associate its
                # coefficient as its value
                grouped_operators[boolean_string] = hamiltonian[pauli_string]

                # Mark the string as grouped, so that it's not added to any
                # other group
                checked.append(pauli_string)

    return grouped_operators, checked


def group_hamiltonian(hamiltonian):
    """
    Organizes a Hamiltonian into groups where strings only differ from
    identities, so that the expectation values of all the strings in each
    group can be calculated from the same measurement array.

    Arguments:
      hamiltonian (dict): a dictionary representing a Hamiltonian, with Pauli
        strings as keys and their coefficients as values.

    Returns:
      grouped_hamiltonian (dict): a dictionary of subhamiltonians, each of
        which includes Pauli strings that only differ from each other by
        identities.
        The keys of grouped_hamiltonian are the main strings of each group: the
        ones with least identity terms. The value associated to a main string is
        a dictionary, whose keys are boolean strings representing substrings of
        the respective main string (with 1 where the Pauli is the same, and 0
        where it's identity instead). The values are their coefficients.
    """
    grouped_hamiltonian = {}
    checked = []

    # Go through the hamiltonian, starting by the terms that have less
    # identity operators
    for main_string in sorted(
        hamiltonian, key=lambda pauli_string: pauli_string.count("I")
    ):

        # Call find_substrings to find all the strings in the dictionary that
        # only differ from main_string by identities, and organize them as a
        # dictionary (grouped_operators)
        grouped_operators, checked = find_substrings(main_string, hamiltonian, checked)

        # Use the dictionary as a value for the main_string key in the
        # grouped_hamiltonian dictionary
        grouped_hamiltonian[main_string] = grouped_operators

        # If all the strings have been grouped, exit the for cycle
        if len(checked) == len(hamiltonian.keys()):
            break

    return grouped_hamiltonian


def convert_hamiltonian(openfermion_hamiltonian):
    """
    Formats a qubit Hamiltonian obtained from openfermion, so that it's a suitable
    argument for functions such as measure_expectation_estimation.

    Arguments:
      openfermion_hamiltonian (openfermion.qubitOperator): the Hamiltonian.

    Returns:
      formatted_hamiltonian (dict): the Hamiltonian as a dictionary with Pauli
        strings (eg 'YXZI') as keys and their coefficients as values.
    """

    formatted_hamiltonian = {}
    qubit_number = count_qubits(openfermion_hamiltonian)

    # Iterate through the terms in the Hamiltonian
    for term in openfermion_hamiltonian.get_operators():

        operators = []
        coefficient = list(term.terms.values())[0]
        pauli_string = list(term.terms.keys())[0]
        previous_qubit = -1

        for qubit, operator in pauli_string:

            # If there are qubits in which no operations are performed, add identities
            # as necessary, to make sure that the length of the string will match the
            # number of qubits
            identities = qubit - previous_qubit - 1
            if identities > 0:
                operators.append("I" * identities)

            operators.append(operator)
            previous_qubit = qubit

        # Add final identity operators if the string still doesn't have the
        # correct length (because no operations are performed in the last qubits)
        operators.append("I" * (qubit_number - previous_qubit - 1))

        formatted_hamiltonian["".join(operators)] = coefficient

    return formatted_hamiltonian


def hamiltonian_to_matrix(hamiltonian):
    """
    Convert a Hamiltonian (from OpenFermion) to matrix form.

    Arguments:
      hamiltonian (openfermion.InteractionOperator): the Hamiltonian to be
        transformed.

    Returns:
      matrix (np.ndarray): the Hamiltonian, as a matrix in the computational
        basis

    """

    qubit_number = hamiltonian.n_qubits

    hamiltonian = jordan_wigner(hamiltonian)

    formatted_hamiltonian = convert_hamiltonian(hamiltonian)
    grouped_hamiltonian = group_hamiltonian(formatted_hamiltonian)

    matrix = np.zeros((2**qubit_number, 2**qubit_number), dtype=complex)

    # Iterate through the strings in the Hamiltonian, adding the respective
    # contribution to the matrix
    for string in grouped_hamiltonian:

        for substring in grouped_hamiltonian[string]:
            pauli = "".join(
                "I" * (not int(b)) + a * int(b) for (a, b) in zip(string, substring)
            )

            matrix += string_to_matrix(pauli) * grouped_hamiltonian[string][substring]

    return matrix


def read_of_qubit_operator(operator):
    """
    Given a QubitOperator from openfermion, return lists of the coefficients,
    strings and qubit indices representing the qubit operator.
    It is assumed that the coefficients are immaginary.
    E.g. "1j(X0 Y1) - 3j(Z3)" -> ([1.0, -3.0], ['XY', 'Z'], [[0, 1], [3]])
    """

    qubit_lists = []
    strings = []
    coefficients = []

    for term in operator.get_operators():

        qubits = []
        string = ""

        op = list(term.terms.keys())[0]

        for qubit, pauli in op:
            coefficient = term.terms[op]
            qubits.append(qubit)
            string += pauli

        strings.append(string)
        qubit_lists.append(qubits)
        assert np.abs(coefficient.real) < 10**-8
        coefficient = coefficient.imag
        coefficients.append(coefficient)

    return coefficients, strings, qubit_lists


def string_to_qop(string):
    """
    Transforms a string into an Openfermion QubitOperator.

    Arguments:
        string (str): a Pauli string, e.g. "XYIIZ"
    Returns:
        The same string as an Openfermion QubitOperator
    """

    op = ""
    for i, pauli in enumerate(string):
        if pauli != "I":
            op += f"{pauli}{i} "

    return QubitOperator(op)
