#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:31:26 2022

@author: mafal
"""
import numpy as np
import pickle

from openfermion import (jordan_wigner,
                         QubitOperator,
                         count_qubits)
from .op_conv import convert_hamiltonian, string_to_qop


def get_operator_qubits(operator):
    """
    Obtains the support of an operator.

    Args:
        operator (Union[FermionOperator, QubitOperator]): the operator in question

    Returns:
        qubits (Set): List containing the indices of the qubits in which operator acts on non-trivially
    """
    qubits = set()

    for string in list(operator.terms.keys()):
        for qubit, pauli in string:
            if qubit not in qubits:
                qubits.add(qubit)

    return qubits


def remove_z_string(operator):
    """
    Removes the anticommutation string from Jordan-Wigner transformed excitations. This is equivalent to removing
    all Z operators.
    This function does not change the original operator.

    Args:
        operator (Union[FermionOperator, QubitOperator]): the operator in question

    Returns:
        new_operator (Union[FermionOperator, QubitOperator]): the same operator, with Pauli-Zs removed
    """

    if isinstance(operator, QubitOperator):
        qubit_operator = operator
    else:
        qubit_operator = jordan_wigner(operator)

    new_operator = QubitOperator()

    for term in qubit_operator.get_operators():

        coefficient = list(term.terms.values())[0]
        pauli_string = list(term.terms.keys())[0]

        new_pauli = QubitOperator((), coefficient)

        for qubit, operator in pauli_string:
            if operator != 'Z':
                new_pauli *= QubitOperator((qubit, operator))

        new_operator += new_pauli

    return new_operator


def bfgs_update(hk, gfkp1, gfk, xkp1, xk):
    """
    Performs a BFGS update.

    Arguments:
        hk (np.ndarray): the previous inverse Hessian (iteration k)
        gfkp1 (np.ndarray): the new gradient vector (iteration k + 1)
        gfk (np.ndarray): the old gradient vector (iteration k)
        xkp1 (np.ndarray): the new parameter vector (iteration k + 1)
        xk (np.ndarray):  the old parameter vector (iteration k)

    Returns:
        hkp1 (np.darray): the new inverse Hessian (iteration k + 1)
    """

    gfkp1 = np.array(gfkp1)
    gfk = np.array(gfk)
    xkp1 = np.array(xkp1)
    xk = np.array(xk)

    n = len(xk)
    id_mat = np.eye(n, dtype=int)

    sk = xkp1 - xk
    yk = gfkp1 - gfk

    rhok_inv = np.dot(yk, sk)
    if rhok_inv == 0.:
        rhok = 1000.0
        print("Divide-by-zero encountered: rhok assumed large")
    else:
        rhok = 1. / rhok_inv

    a1 = id_mat - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
    a2 = id_mat - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
    hkp1 = np.dot(a1, np.dot(hk, a2)) + (rhok * sk[:, np.newaxis] *
                                         sk[np.newaxis, :])

    return hkp1


def create_deg2_taylor_fun(energy_0, coefficients, gradients, hessian):
    """
    Creates a second degree approximation to a function.

    Args:
        energy_0 (float): the energy of the function at coefficients
        coefficients (list): coefficients at which the Taylor series is being created
        gradients (list): the gradients of the function at coefficients
        hessian (np.ndarray): the hessian of the function at coefficients

    Returns:
        fun (function): a second degree function approximating the original function, as described by the
        input arguments.
            Receives a coefficient list and returns the value of the approximate quadratic function at that point
    """

    def fun(x):
        # Returns: degree 2 Taylor approximation to the energy

        vec = np.subtract(x, coefficients)
        new_energy = energy_0 + np.dot(vec, gradients) + 1 / 2 * np.matmul(
            np.matmul(vec, hessian), vec.transpose())

        return new_energy

    return fun


def save_to_file(data, file_name):
    """
    Saves data to a file named "file_name" (can be a directory)
    """

    my_file = open(file_name, "wb")
    pickle.dump(data, my_file)

    my_file.close()


def load_from_file(file_name):
    """
    Loads data from a file named "file_name" (can be a directory)
    """

    my_file = open(file_name, "rb")
    data = pickle.load(my_file)

    my_file.close()

    return data


def tile_1s(q_op, n, new_n):
    """
    q_op must be a single string
    """

    n_ids = new_n - n
    op = convert_hamiltonian(q_op)
    string = list(op.keys())[0]

    if len(string) < n:
        string = string + "I" * (n - len(string))

    tiled_ops = []
    for i in range(n_ids + 1):
        tiled_string = "I" * i + string + "I" * (n_ids - i)
        q_op = 1j * string_to_qop(tiled_string)
        tiled_ops.append(q_op)

        assert len(tiled_string) == new_n

    return tiled_ops


def tile(q_op, n, new_n):
    """
    q_op can be linear combination of strings
    """

    assert n >= count_qubits(q_op)

    n_ids = new_n - n
    op = convert_hamiltonian(q_op)
    tiled_ops = []

    for i in range(n_ids + 1):

        q_op = QubitOperator()

        for string in op.keys():

            coeff = op[string]

            if len(string) < n:
                # We choose to consider identities when tiling. This could be
                # done differently. E.g. X0 acting on two qubits, when tiled for
                # four qubits, will yield X0, X1, X2, but not X3 (because we
                # view the tiling unit as XI, not just X).
                string = string + "I" * (n - len(string))

            tiled_string = "I" * i + string + "I" * (n_ids - i)
            assert len(tiled_string) == new_n

            q_op += coeff * string_to_qop(tiled_string)

        tiled_ops.append(q_op)

    return tiled_ops


def tile2(q_op, n, new_n):
    """
    q_op can be linear combination of strings
    This one tiles without considering identities. E.g. X0 I1 is tiled into X0, X1, X2, X3 if n=2, new_n=4.
    """

    assert n >= count_qubits(q_op)

    op = convert_hamiltonian(q_op)
    tiled_ops = []

    string = list(op.keys())[0]
    k = 0
    while string[k] == "I":
        k += 1
    string = string[k:]
    n_ids = new_n - len(string)

    for i in range(n_ids + 1):

        q_op = QubitOperator()

        for string in op.keys():

            coeff = op[string]

            k = 0
            while string[k] == "I":
                k += 1
            string = string[k:]

            tiled_string = "I" * i + string + "I" * (n_ids - i)
            assert len(tiled_string) == new_n

            q_op += coeff * string_to_qop(tiled_string)

        tiled_ops.append(q_op)

    return tiled_ops
