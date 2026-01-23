#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:31:26 2022

@author: mafal
"""
import numpy as np
import pickle
from pyscf import ao2mo
from pyscf.tools.fcidump import read

from openfermion import (jordan_wigner,
                         FermionOperator,
                         QubitOperator,
                         InteractionOperator,
                         hermitian_conjugated,
                         normal_ordered,
                         count_qubits)
from openfermion.chem.molecular_data import spinorb_from_spatial
<<<<<<< HEAD

import qiskit
=======
>>>>>>> mafalda-main

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


def create_excitations(p, q, r, s, fermionic = False, jw_transform = False):
    """
    Creates all unique qubit excitations acting on the set of spin-orbitals p,q,r,s.

    If aaaa or bbbb, all possible source/orbital pair combinations are valid.
    In this case, all the ifs apply and we get 6 distinct operators.

    In the other cases, only two source/orbital pair combinations are valid.
    In this case, only one of the ifs applies and we get 2 distinct operators.

    Arguments:
        p, q, r, s (int): the spin-orbital indices
        fermionic (bool): whether to keep the Jordan-Wigner anticommutation Z string
        jw_transform (bool): whether to apply JW transformation in case fermionic = True

    Returns:
        operators (list): list of lists containing pairs of qubit excitations. If p,q,r,s are aaaa or bbbb, the list
            contains three pairs of qubit excitations. Otherwise it contains one.
        orbs (list): list of lists containing pairs of source/target orbitals. Length: same as described above.
            The source (target) orbitals for q_operators[0] are returned in orbs[0][0] (orbs[1][0]).
            The source (target) orbitals for q_operators[1] are returned in orbs[0][1] (orbs[1][1]).
    """


    operators = []
    orbs = []

    if (p + r) % 2 == 0:
        # pqrs is abab or baba, or aaaa or bbbb

        operator_1 = FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)))
        operator_2 = FermionOperator(((q, 1), (r, 1), (p, 0), (s, 0)))

        operator_1 -= hermitian_conjugated(operator_1)
        operator_2 -= hermitian_conjugated(operator_2)

        operator_1 = normal_ordered(operator_1)
        operator_2 = normal_ordered(operator_2)

        if not fermionic:
            operator_1 = remove_z_string(operator_1)
            operator_2 = remove_z_string(operator_2)
        elif jw_transform:
            operator_1 = jordan_wigner(operator_1)
            operator_2 = jordan_wigner(operator_2)

        source_orbs = [[r, s], [p, s]]
        target_orbs = [[p, q], [q, r]]

        operators.append([operator_1, operator_2])
        orbs.append([source_orbs, target_orbs])

    if (p + q) % 2 == 0:
        # aabb or bbaa, or aaaa or bbbb

        operator_1 = FermionOperator(((p, 1), (r, 1), (q, 0), (s, 0)))
        operator_2 = FermionOperator(((q, 1), (r, 1), (p, 0), (s, 0)))

        operator_1 -= hermitian_conjugated(operator_1)
        operator_2 -= hermitian_conjugated(operator_2)

        operator_1 = normal_ordered(operator_1)
        operator_2 = normal_ordered(operator_2)

        if not fermionic:
            operator_1 = remove_z_string(operator_1)
            operator_2 = remove_z_string(operator_2)
        elif jw_transform:
            operator_1 = jordan_wigner(operator_1)
            operator_2 = jordan_wigner(operator_2)

        source_orbs = [[q, s], [p, s]]
        target_orbs = [[p, r], [q, r]]

        operators.append([operator_1, operator_2])
        orbs.append([source_orbs, target_orbs])

    if (p + s) % 2 == 0:
        # abba or baab, or aaaa or bbbb

        operator_1 = FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)))
        operator_2 = FermionOperator(((p, 1), (r, 1), (q, 0), (s, 0)))

        operator_1 -= hermitian_conjugated(operator_1)
        operator_2 -= hermitian_conjugated(operator_2)

        operator_1 = normal_ordered(operator_1)
        operator_2 = normal_ordered(operator_2)

        if not fermionic:
            operator_1 = remove_z_string(operator_1)
            operator_2 = remove_z_string(operator_2)
        elif jw_transform:
            operator_1 = jordan_wigner(operator_1)
            operator_2 = jordan_wigner(operator_2)

        source_orbs = [[r, s], [q, s]]
        target_orbs = [[p, q], [p, r]]

        operators.append([operator_1, operator_2])
        orbs.append([source_orbs, target_orbs])

    return operators, orbs


def swap(l,i,j):
    """
    Swaps two elements in a list

    Arguments:
        l (list): the list:
        i (int): index of one element to be swapped
        j (int): index of the other element to be swapped

    Returns:
        l (list): the modified list. This is not a copy.
    """
    temp = l[i]
    if j >= len(l) or i >= len(l):
        print("H")
    l[i] = l[j]
    l[j] = temp
    return l

def find_lnn_singles(spin_orb_order):
    """
    Given a spin-orbital ordering, outputs the indices of (spin preserving)
    single excitations acting on 2 adjacent qubits.

    Arguments:
        spin_orb_order (list): the ordered list of spin-orbital indices. Even/odd
            indices are assumed to correspond to alpha/beta orbitals

    Returns:
        single_indices (list): a list of ordered lists of 2 indices that are
            adjacent in the input list and correspond to the same type of
            spin-orbital (i.e. same parity)
    """
    single_indices = []
    for i in range(len(spin_orb_order)-1):
        a,b = sorted(spin_orb_order[i:i+2])
        if a%2 == b%2:
            single_indices.append([a,b])

    return single_indices

def find_lnn_paired_doubles(spin_orb_order):
    """
    Given a spin-orbital ordering, outputs the indices of total spin preserving
    double excitations acting on 4 adjacent qubits

    Arguments:
        spin_orb_order (list): the ordered list of spin-orbital indices. Even/odd
            indices are assumed to correspond to alpha/beta orbitals

    Returns:
        double_indices (list): a list of ordered lists of 4 indices that are
            adjacent in the input list and correspond to two pairs of orbitals
            with the same spatial part
    """
    double_indices = []
    for i in range(len(spin_orb_order)-3):
        a,b,c,d = sorted(spin_orb_order[i:i+4])
        if a%2 == 0 and c%2 == 0 and a+1==b and c+1==d:
            # a,b and c,d are each a pair of alpha, beta spin-orbitals with the
            #same spatial part
            double_indices.append([a,b,c,d])

    return double_indices

def find_spin_preserving_exc_indices(order_list):
    """
    Given a list of spin-orbital orderings, outputs a list of the corresponding
    indices of single and paired double excitations that can be executed
    assuming we have a LNN architecture. The order of the output list is the
    order in which the excitations would appear in a circuit described by the
    successive orderings. The list of orderings may represent e.g. a swap
    network.

    Arguments:
        order_list (list): list of lists, each of which represents an ordering
            of spin-orbital indices. Even/odd indices are assumed to correspond
            to alpha/beta orbitals

    Returns:
        double_indices (list): a list of ordered lists of 4 indices that are
            adjacent in the input list and correspond to two pairs of orbitals
            with the same spatial part
    """

    excitation_indices = []

    for order in order_list:

        for double_exc in find_lnn_paired_doubles(order):
            if double_exc not in excitation_indices:
                excitation_indices.append(double_exc)

        for single_exc in  find_lnn_singles(order):
            if single_exc not in excitation_indices:
                excitation_indices.append(single_exc)

    return excitation_indices

def appears_in(list1,list2):
    """
    Checks if list1 appears in list2 (all elements together)
    """

    cutoff = len(list2) - len(list1) + 1
    for i in range(cutoff):
        if list2[i:i+len(list1)] == list1:
            return True

    return False


def invert_circuit_qubits(ckt: qiskit.QuantumCircuit) -> qiskit.QuantumCircuit:
    """Swap all qubits in a circuit following the permutation [n-1, n-2, ..., 1, 0].
    This is implemented with a qiskit Permute instruction at the beginning."""

    nq = ckt.num_qubits
    swap_ckt = qiskit.QuantumCircuit(nq)
    perm = list(range(nq))
    perm.reverse()
    swap_ckt.append(qiskit.circuit.library.PermutationGate(perm), range(nq))
    return swap_ckt.compose(ckt)


def hamiltonian_from_fcidump(path):
    """
    Gets the Openfermion Hamiltonian from an FCIDUMP file
    """

    data = read(path,verbose=False)

    ecore = data['ECORE']
    nelec = data['NELEC']
    norb = data['NORB']
    h1 = data['H1']

    # Rebuild 4-index tensor from flattened H2
    h2_packed = data["H2"]
    h2 = ao2mo.restore(1, h2_packed, norb)  # (norb,norb,norb,norb)

    """
    # openfermion seems to expect the integrals reordered as p^ r^ s q
    # whereas pyscf stores them as (p,q,r,s)
    h2_temp = np.zeros((4, 4, 4, 4))
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    h2_temp[p, r, s, q] = h2[p, q, r, s]

    h1_new, h2_new = get_tensors_from_integrals(h1, h2_temp)
    """

    # Shorter code:
    h2 = 0.5 * np.asarray(h2.transpose(0, 2, 3, 1), order="C")
    h1_new, h2_new = spinorb_from_spatial(h1, h2)

    h = InteractionOperator(ecore, h1_new, h2_new)

    return h, norb*2, nelec



def hamiltonian_from_npz(path):
    """
    Gets the Openfermion Hamiltonian from an NPZ file
    """
    data = np.load(path)

    # Get core energy and one-/two-body tensors
    ecore = data['ECORE']
    h1 = np.array(data['H1'])
    h2 = np.array(data['H2'])
    norb = data["NORB"]
    nelec = data["NELEC"]

    # Get number of alpha and beta electrons
    n_a = int(nelec/2)
    n_b = int(nelec/2)

    h2 = 0.5 * np.asarray(h2.transpose(0, 2, 3, 1), order="C")
    h1_new, h2_new = spinorb_from_spatial(h1, h2)

    # Get the Hamiltonian and transform it to FermionOperator
    h = InteractionOperator(ecore.item(), h1_new, h2_new)

    return h, norb*2, nelec