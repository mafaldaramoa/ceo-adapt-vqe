from warnings import warn

import re

import numpy as np
from copy import deepcopy
from math import floor

from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from openfermion import get_sparse_operator

from .op_conv import convert_hamiltonian, read_of_qubit_operator
from .utils import swap, appears_in


def get_gate_depth(qasm, n, gate_name):
    """
    Counts the depth of a circuit on n qubits represented by a QASM string, considering only a particular type of
    two-qubit gates. All other gates are ignored.

    Aguments:
        qasm (str): the QASM representation of the circuit
        n (int): the number of qubits
    Returns:
        The depth of the circuit, in terms of this type of gates
    """
    # n = int(re.search(r"(?<=q\[)[0-9]+(?=\])", qasm.splitlines()[2]).group())
    depths = [0 for _ in range(n)]

    for line in qasm.splitlines()[3:]:
        # Remove ;
        line = line[:-1]

        # Split line by spaces
        line_elems = line.split(" ")

        # First element is operation type
        op = line_elems[0]
        if op[:2] != gate_name:
            continue

        # Next element is qubits
        # Depending on the QASM version, we may have qubits in the same entry or separate ones
        if len(line_elems) == 2:
            qubits = [int(qubit) for qubit in re.findall(r"[0-9]+",line_elems[1])]
        elif len(line_elems) == 3:
            qubits = [
                int(re.search(r"[0-9]+", qubit_string).group())
                for qubit_string in line_elems[1:]
            ]
        else:
            raise ValueError

        assert len(qubits) == 2

        max_depth = max([depths[qubit] for qubit in qubits])
        new_depth = max_depth + 1

        for qubit in qubits:
            depths[qubit] = new_depth

    return max(depths)

def cnot_depth(qasm,n):
    return get_gate_depth(qasm,n,"cx")

def ecr_depth(qasm,n):
    return get_gate_depth(qasm,n,"ecr")

def cz_depth(qasm,n):
    return get_gate_depth(qasm,n,"cz")

def count_gates(qasm,gate_name):
    """
    Counts the number of gates of a particular type in a circuit represented by a QASM string.
    """
    count = 0

    for line in qasm.splitlines()[3:]:
        # Remove ;
        line = line[:-1]
        line_elems = line.split(" ")
        op = line_elems[0]

        if op[:2] == gate_name:
            count += 1

    return count

def cnot_count(qasm):
    """
    Counts the CNOTs in a circuit represented by a QASM string.
    """

    return count_gates(qasm,"cx")

def ecr_count(qasm):
    """
    Counts the ECR gates in a circuit represented by a QASM string.
    """

    return count_gates(qasm,"ecr")

def cz_count(qasm):
    """
    Counts the ECR gates in a circuit represented by a QASM string.
    """

    return count_gates(qasm,"cz")


def qe_circuit(source_orbs, target_orbs, theta, n, big_endian=False):
    """
    Creates a qubit excitation circuit. See https://doi.org/10.1103/PhysRevA.102.062612

    Arguments:
        source_orbs (list): the spin-orbitals from which the excitation removes electrons
        target_orbs (list): the spin-orbitals to which the excitation adds electrons
        theta (float): the coefficient of the excitation
        n (int): the number of qubits
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    if len(source_orbs) == 2:
        return double_qe_circuit(source_orbs, target_orbs, theta, n, big_endian)
    else:
        return single_qe_circuit(source_orbs, target_orbs, theta, n, big_endian)


def double_qe_circuit(source_orbs, target_orbs, theta, n, big_endian=False):
    """
    Creates a qubit excitation circuit. See https://doi.org/10.1103/PhysRevA.102.062612

    Arguments:
        source_orbs (list): the spin-orbitals from which the excitation removes electrons
        target_orbs (list): the spin-orbitals to which the excitation adds electrons
        theta (float): the coefficient of the excitation
        n (int): the number of qubits
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    a, b = source_orbs
    c, d = target_orbs

    # Circuit requires sorted orbitals
    if a>b:
        a,b = b,a
        theta = -theta
    if c>d:
        c,d = d,c
        theta = -theta

    if big_endian:
        # Qiskit's default is little endian - switch
        a = n - a - 1
        b = n - b - 1
        c = n - c - 1
        d = n - d - 1

    qc = QuantumCircuit(n)

    qc.cx(a, b)
    qc.cx(c, d)
    qc.x(b)
    qc.x(d)
    qc.cx(a, c)
    qc.ry(2 * theta / 8, a)

    qc.h(b)
    qc.cx(a, b)
    qc.h(d)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, d)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, b)
    qc.h(c)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, c)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, b)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, d)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, b)
    qc.ry(-2 * theta / 8, a)

    qc.h(d)
    qc.h(b)
    qc.rz(+np.pi / 2, c)
    qc.cx(a, c)

    qc.rz(-np.pi / 2, a)
    qc.rz(+np.pi / 2, c)
    qc.ry(+np.pi / 2, c)

    qc.x(b)
    qc.x(d)
    qc.cx(a, b)
    qc.cx(c, d)

    return qc


def single_qe_circuit(source_orb, target_orb, theta, n, big_endian=False):
    """
    Creates a qubit excitation circuit. See https://doi.org/10.1103/PhysRevA.102.062612
    Example: if source_orb = [0] and target_orb = [1], this implements theta * 1/2 (X1 Y0 - Y1 X0)

    Arguments:
        source_orb (list): the spin-orbital from which the excitation removes electrons
        target_orb (list): the spin-orbital to which the excitation adds electrons
        theta (float): the coefficient of the excitation
        n (int): the number of qubits
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    a = source_orb[0]
    b = target_orb[0]

    if big_endian:
        a = n - a - 1
        b = n - b - 1

    qc = QuantumCircuit(n)

    qc.rz(np.pi / 2, a)
    qc.rx(np.pi / 2, a)
    qc.rx(np.pi / 2, b)
    qc.cx(a, b)

    qc.rx(theta, a)
    qc.rz(theta, b)
    qc.cx(a, b)

    qc.rx(-np.pi / 2, b)
    qc.rx(-np.pi / 2, a)
    qc.rz(-np.pi / 2, a)

    return qc



def fe_circuit(source_orbs, target_orbs, theta, n, big_endian=False):
    """
    Creates a fermionic excitation circuit. See https://doi.org/10.1103/PhysRevA.102.062612

    Arguments:
        source_orbs (list): the spin-orbitals from which the excitation removes electrons
        target_orbs (list): the spin-orbitals to which the excitation adds electrons
        theta (float): the coefficient of the excitation
        n (int): the number of qubits
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    if len(source_orbs) == 2:
        return double_fe_circuit(source_orbs, target_orbs, theta, n, big_endian)
    else:
        return single_fe_circuit(source_orbs, target_orbs, theta, n, big_endian)

def double_fe_circuit(source_orbs, target_orbs, theta, n, big_endian=False):
    """
    Creates a double fermionic excitation circuit. See https://doi.org/10.1103/PhysRevA.102.062612

    Arguments:
        source_orbs (list): the spin-orbitals from which the excitation removes electrons
        target_orbs (list): the spin-orbitals to which the excitation adds electrons
        theta (float): the coefficient of the excitation
        n (int): the number of qubits
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    a, b = source_orbs
    c, d = target_orbs

    if big_endian:
        # Qiskit's default is little endian - switch
        a = n - a - 1
        b = n - b - 1
        c = n - c - 1
        d = n - d - 1

    qc = QuantumCircuit(n)

    qc.cx(a, b)
    qc.cx(c, d)

    # Find the qubits that have a Jordan-Wigner Z string
    x1, x2, x3, x4 = sorted([a, b, c, d])
    z_qubits = list(range(x1+1, x2)) + list(range(x3+1, x4))

    # Calculate parity into last qubit using CNOTs and apply phase
    for i, j in zip(z_qubits[:-1],z_qubits[1:]):
        qc.cx(i,j)
    if z_qubits:
        qc.cz(z_qubits[-1],a)

    qc.x(b)
    qc.x(d)
    qc.cx(a, c)
    qc.ry(2 * theta / 8, a)

    qc.h(b)
    qc.cx(a, b)
    qc.h(d)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, d)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, b)
    qc.h(c)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, c)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, b)
    qc.ry(-2 * theta / 8, a)

    qc.cx(a, d)
    qc.ry(2 * theta / 8, a)

    qc.cx(a, b)
    qc.ry(-2 * theta / 8, a)

    qc.h(d)
    qc.h(b)
    qc.rz(+np.pi / 2, c)
    qc.cx(a, c)

    qc.rz(-np.pi / 2, a)
    qc.rz(+np.pi / 2, c)
    qc.ry(+np.pi / 2, c)

    qc.x(b)
    qc.x(d)

    qc.cx(a, b)
    qc.cx(c, d)

    # Revert parity calculation
    if z_qubits:
        qc.cz(z_qubits[-1], a)
    for i, j in zip(reversed(z_qubits[:-1]),reversed(z_qubits[1:])):
        qc.cx(i,j)

    return qc

def single_fe_circuit(source_orb, target_orb, theta, n, big_endian=False):
    """
    Creates a fermionic excitation circuit. See https://doi.org/10.1103/PhysRevA.102.062612
    Example: if source_orb = [0] and target_orb = [1], this implements theta * 1/2 (X1 Y0 - Y1 X0)

    Arguments:
        source_orb (list): the spin-orbital from which the excitation removes electrons
        target_orb (list): the spin-orbital to which the excitation adds electrons
        theta (float): the coefficient of the excitation
        n (int): the number of qubits
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    a = source_orb[0]
    b = target_orb[0]

    if big_endian:
        a = n - a - 1
        b = n - b - 1

    qc = QuantumCircuit(n)

    x1, x2 = sorted([a,b])
    z_qubits = range(x1+1,x2)

    # Calculate parity into last qubit using CNOTs and apply phase
    for i, j in zip(z_qubits[:-1], z_qubits[1:]):
        qc.cx(i,j)
    if z_qubits:
        qc.cz(z_qubits[-1],a)

    qc.rz(np.pi / 2, a)
    qc.rx(np.pi / 2, a)
    qc.rx(np.pi / 2, b)
    qc.cx(a, b)

    qc.rx(theta, a)
    qc.rz(theta, b)
    qc.cx(a, b)

    qc.rx(-np.pi / 2, b)
    qc.rx(-np.pi / 2, a)
    qc.rz(-np.pi / 2, a)

    # Revert parity calculation
    if z_qubits:
        qc.cz(z_qubits[-1], a)
    for i, j in zip(reversed(z_qubits[:-1]), reversed(z_qubits[1:])):
        qc.cx(i,j)

    return qc

def pauli_exp_circuit(qubit_operator, n, revert_endianness=False):
    """
    Implements the exponential of an operator, which must be anti-Hermitian.
    If the Paulis commute, the circuit is exact. If they don't, it's an approximation with 1 Trotter step.

    Arguments:
        qubit_operator (QubitOperator): the generator of the unitary (exponent)
        n (int): the number of qubits of the output circuit (must be >= number of qubits qubit_operator acts on)
        revert_endianness (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        Circuit implementing e^qubit_operator (approximately if strings in generator do not commute)
    """

    pauli_dict = convert_hamiltonian(qubit_operator)
    qc = QuantumCircuit(n)

    for pauli_string in pauli_dict.keys():

        coeff = pauli_dict[pauli_string]

        if revert_endianness:
            pauli_string = pauli_string + "I" * (n - len(pauli_string))
            pauli_string = pauli_string[::-1]

        assert coeff.real < 10**-8
        coeff = coeff.imag

        active_qubits = []
        for i, pauli in enumerate(pauli_string):

            if pauli != "I":
                active_qubits.append(i)

            if pauli == "X":
                qc.h(i)

            if pauli == "Y":
                qc.rx(np.pi / 2, i)

        for i in range(len(active_qubits) - 1):
            qc.cx(active_qubits[i], active_qubits[i + 1])

        qc.rz(-2 * coeff, active_qubits[-1])

        for i in reversed(range(len(active_qubits) - 1)):
            qc.cx(active_qubits[i], active_qubits[i + 1])

        for i, pauli in enumerate(pauli_string):

            if pauli == "X":
                qc.h(i)

            if pauli == "Y":
                qc.rx(-np.pi / 2, i)

    return qc


def mvp_ceo_circuit(qubit_operator, n, big_endian=False):
    """
    Implements an operator of the form:
    (c0 * XXXY +
     c1 * XXYX +
     c2 * YXYY +
     c3 * YXXX +
     c4 * YYXY +
     c5 * YYYX +
     c6 * XYYY +
     c7 * XYXX)/8

    Arguments:
        qubit_operator (QubitOperator): operator of the form above
        n (int): the number of qubits of the output circuit (must be >= number of qubits qubit_operator acts on)
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        Circuit implementing e^qubit_operator
    """

    coefficients, strings, qubit_lists = read_of_qubit_operator(qubit_operator)
    if len(qubit_lists) == 0:
        return QuantumCircuit(n)

    qubits = qubit_lists[0]
    assert all([qubit_list == qubits for qubit_list in qubit_lists])
    a, b, c, d = qubits

    if big_endian:
        # Qiskit's default is little endian - switch
        a = n - a - 1
        b = n - b - 1
        c = n - c - 1
        d = n - d - 1

    thetas = []
    for string in ["YXXX", "XYXX", "YYXY", "XXXY", "YXYY", "XYYY", "YYYX", "XXYX"]:
        index = strings.index(string)
        thetas.append(8 * coefficients[index])

    qc = QuantumCircuit(n)

    qc.rz(-np.pi / 2, a)
    qc.cx(a, d)
    qc.cx(a, c)
    qc.cx(a, b)
    qc.h(a)

    qc.rz(-2 * thetas[0] / 8, a)
    qc.cx(b, a)

    qc.rz(-2 * thetas[1] / 8, a)
    qc.cx(d, a)

    qc.rz(2 * thetas[2] / 8, a)
    qc.cx(b, a)

    qc.rz(-2 * thetas[3] / 8, a)
    qc.rz(-np.pi / 2, c)
    qc.cx(c, a)

    qc.rz(+2 * thetas[4] / 8, a)
    qc.cx(b, a)

    qc.rz(+2 * thetas[5] / 8, a)
    qc.cx(d, a)

    qc.rz(2 * thetas[6] / 8, a)
    qc.cx(b, a)

    qc.rz(-2 * thetas[7] / 8, a)

    qc.h(a)
    qc.cx(a, b)
    qc.cx(a, c)
    qc.cx(a, d)
    qc.rz(np.pi / 2, c)

    return qc

def sort_orbitals_ovp_ceo(source_orbs,target_orbs,theta,ceo_type):
    """
    Rearranges the source and target orbitals of a CEO operator such that they are sorted.
    Example: if source_orbs = [[r,s],[p,s]] and target_orbs = [[p,q],[q,r]],
    it will make sure that r<s, p<s, p<q, q<r. If this is not the case, the CEO type (sum/diff)
    and the sign of the coefficient will me modified so that this it is.

        Arguments:
            source_orbs (list): the spin-orbitals from which the excitation removes electrons
            target_orbs (list): the spin-orbitals to which the excitation adds electrons
            theta (float): the coefficient of the excitation
            ceo_type (str): "sum"/"diff" depending on the type of OVP-CEO

        Returns:
            source_orbs (list): the sorted source spin-orbitals 
            target_orbs (list): the sorted spin-orbitals 
            ceo_type (bool): the final CEO type
            theta (float): the final CEO coefficient
    """

    source_orbs_qe1 = source_orbs[0]
    source_orbs_qe2 = source_orbs[1]
    target_orbs_qe1 = target_orbs[0]
    target_orbs_qe2 = target_orbs[1]

    flip_sign_qe1 = False
    flip_sign_qe2 = False

    # The circuit assumes orbitals are sorted. Sort them and correct for the sign flip that might appear
    if not sorted(source_orbs_qe1) == source_orbs_qe1:
        source_orbs_qe1 = sorted(source_orbs_qe1)
        flip_sign_qe1 = not flip_sign_qe1

    if not sorted(target_orbs_qe1) == target_orbs_qe1:
        target_orbs_qe1 = sorted(target_orbs_qe1)
        flip_sign_qe1 = not flip_sign_qe1

    if not sorted(source_orbs_qe2) == source_orbs_qe2:
        source_orbs_qe2 = sorted(source_orbs_qe2)
        flip_sign_qe2 = not flip_sign_qe2

    if not sorted(target_orbs_qe2) == target_orbs_qe2:
        target_orbs_qe2 = sorted(target_orbs_qe2)
        flip_sign_qe2 = not flip_sign_qe2
        
    if flip_sign_qe1 and not flip_sign_qe2:
        # We had e.g. CEO_P(qe1,qe2) = qe1 + qe2, now have -qe1 + qe2 = -CEO_M(qe1,qe2)
        theta = -theta
        ceo_type = 'sum' if ceo_type == 'diff' else 'diff'
    elif not flip_sign_qe1 and flip_sign_qe2:
        # We had e.g. CEO_P(qe1,qe2) = qe1 + qe2, now have qe1 - qe2 = CEO_M(qe1,qe2)
        ceo_type = 'sum' if ceo_type == 'diff' else 'diff'
    elif flip_sign_qe1 and flip_sign_qe2:
        # We had e.g. CEO_P(qe1,qe2) = qe1 + qe2, now have -qe1 - qe2 = -CEO_P(qe1,qe2)
        theta = - theta

    source_orbs = [source_orbs_qe1, source_orbs_qe2]
    target_orbs = [target_orbs_qe1,target_orbs_qe2]

    return source_orbs, target_orbs, theta, ceo_type


def ovp_ceo_circuit(source_orbs, target_orbs, n, theta, ceo_type, big_endian=False):
    """
    Implements the OVP-CEO defined by the input arguments using CNOT and single
    qubit gates.
    Example: if source_orbs = [[r,s],[p,s]] and target_orbs = [[p,q],[q,r]],
    it implements the unitary generated by QE(r,s->p,q) +/- QE(p,s->q,r).
    It is assumed that source/target orbitals are ordered (in this example,
    r<s, p<s, p<q, q<r -> p<q<r<s).

        Arguments:
            source_orbs (list): the spin-orbitals from which the excitation removes electrons
            target_orbs (list): the spin-orbitals to which the excitation adds electrons
            n (int): the number of qubits
            theta (float): the coefficient of the excitation
            ceo_type (str): "sum"/"diff" depending on the type of OVP-CEO
            big_endian (bool): if True/False, big/little endian ordering will be assumed

        Returns:
            QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    if len(source_orbs) == 1:
        # Same as QEs
        return single_qe_circuit(source_orbs, target_orbs, theta, n, big_endian)

    # Sort orbitals, since it is assumed by the circuit that they are in ascending order
    source_orbs, target_orbs, theta, ceo_type = sort_orbitals_ovp_ceo(source_orbs,target_orbs,theta,ceo_type)

    # We need a unique attribution of qubits to spin-orbitals. We define c
    # as the common source orbital and d as the common target orbital.
    c = np.intersect1d(source_orbs[0], source_orbs[1])[0]
    d = np.intersect1d(target_orbs[0], target_orbs[1])[0]

    a = source_orbs[0].copy()
    a.remove(c)
    a = a[0]
    b = target_orbs[0].copy()
    b.remove(d)
    b = b[0]

    # The rotation sign depends on the CEO type
    theta = theta * (-1) ** (ceo_type == "sum")

    if ceo_type == "sum":
        # Changing the CEO type is equivalent to changing endianness on the
        # involved qubits
        a, b, c, d = d, c, b, a

    if big_endian:
        # Qiskit's default is little endian - switch
        a = n - a - 1
        b = n - b - 1
        c = n - c - 1
        d = n - d - 1

    qc = QuantumCircuit(n)

    qc.cx(a, b)
    qc.cx(c, d)
    qc.cx(a, c)

    qc.h(b)
    qc.h(d)

    qc.ry(+theta / 2, a)
    qc.cx(a, d)

    qc.ry(-theta / 2, a)
    qc.cx(a, b)

    qc.ry(+theta / 2, a)
    qc.cx(a, d)

    qc.ry(-theta / 2, a)
    qc.cx(a, c)

    qc.h(d)

    qc.rz(np.pi / 2, a)
    qc.ry(-np.pi / 2, b)
    qc.rz(-np.pi / 2, b)

    qc.cx(a, b)
    qc.cx(c, d)

    qc.rz(-np.pi / 2, b)

    return qc


def ovp_ceo_cr_circuit(source_orbs, target_orbs, n, theta, ceo_type, big_endian=False):
    """
    Implements the OVP-CEO defined by the input arguments using CNOT and single
    qubit gates using a multi-control rotation and CNOTs.
    Example: if source_orbs = [[r,s],[p,s]] and target_orbs = [[p,q],[q,r]],
    it implements the unitary generated by QE(r,s->p,q) +/- QE(p,s->q,r).

        Arguments:
            source_orbs (list): the spin-orbitals from which the excitation removes electrons
            target_orbs (list): the spin-orbitals to which the excitation adds electrons
            n (int): the number of qubits
            theta (float): the coefficient of the excitation
            ceo_type (str): "sum"/"diff" depending on the type of OVP-CEO
            big_endian (bool): if True/False, big/little endian ordering will be assumed

        Returns:
            QuantumCircuit (the circuit implementing the operator in Qiskit)
    """

    if len(source_orbs) == 1:
        # Same as QEs
        return single_qe_circuit(source_orbs, target_orbs, theta, n, big_endian)

    # We need a unique attribution of qubits to spin-orbitals. We define c
    # as the common source orbital and d as the common target orbital.
    c = np.intersect1d(source_orbs[0], source_orbs[1])[0]
    d = np.intersect1d(target_orbs[0], target_orbs[1])[0]

    a = source_orbs[0].copy()
    a.remove(c)
    a = a[0]
    b = target_orbs[0].copy()
    b.remove(d)
    b = b[0]

    theta = theta * (-1) ** (ceo_type == "sum")

    if big_endian:
        # Qiskit's default is little endian - switch
        a = n - a - 1
        b = n - b - 1
        c = n - c - 1
        d = n - d - 1

    if ceo_type == "sum":
        # Changing the CEO type is equivalent to changing endianness on the
        # involved qubits
        a, b, c, d = d, c, b, a

    qc = QuantumCircuit(n)

    qc.cx(a, b)
    qc.cx(c, d)
    qc.cx(a, c)

    c2ry = RYGate(-2 * theta).control(2)
    qc.append(c2ry, [d, b, a])

    qc.cx(a, c)
    qc.cx(a, b)
    qc.cx(c, d)

    return qc

def paired_f_swap_network_orderings(n):
    """
    Finds the list of spin-orbital orderings at each step of a fermionic swap
    network for a  UpCCGSD circuit. Note that since fermionic swaps are used,
    this effectively swaps spin-orbitals, not just qubits - anticommutation
    effects are taken into account.
    It is assumed that orbitals 1a, 1b, 2a, 2b,... are labeled as 0, 1, 2, 3,...
    See Fig. 7 of https://arxiv.org/pdf/1905.05118

    Arguments:
        n (int): the number of qubits/spin-orbitals

    Returns:
        orders: a list of lists of length n representing spin-orbital
            orderings. The first/last lists are the initial/final orderings.
    """
    print("HELLO")

    if (n/2)%2 != 0:
        # Not sure how the authors define the initial ordering for an odd number of spatial orbitals?
        raise NotImplementedError

    orders = []

    # Create initial ordering
    order = [None for _ in range(n)]
    for spatial_orb in range(0,int(n/2),2):
        # We place two alpha orbitals from subsequent spatial orbitals, then
        #the two beta counterparts
        order[spatial_orb*2] = spatial_orb*2
        order[spatial_orb*2+1] = (spatial_orb + 1)*2
        order[spatial_orb*2+2] = spatial_orb*2 + 1
        order[spatial_orb*2+3] = (spatial_orb + 1)*2 + 1

    orders.append(order.copy())

    # The overall effect of the network is to reverse the ordering
    final_order = order[::-1]

    while order != final_order:

        # Apply layer of even-odd fermionic swaps
        for i in range(0,n-1,2):
            swap(order,i,i+1)
        orders.append(order.copy())

        # Apply layer of odd-even fermionic swaps
        for i in range(1,n-1,2):
            swap(order,i,i+1)
        orders.append(order.copy())

    return orders


def get_swaps(qubits,qubit_order,fermionic):
    """
    Gets the swap circuit that is necessary to bring the qubits in the input list together.

    Arguments:
        qubits (list): the list of qubits we want to bring together. These are not necessary represented by
        physical qubits with the same index. qubit_order is assumed for the original mapping of physical to
            logical qubits.
        qubit_order (list): current physical qubit mapping.
        fermionic (bool): whether we're exchanging fermionic modes or qubits
    """
    n = len(qubit_order)

    # Get the list of physical qubits associated with these qubits
    physical_qubits = [qubit_order.index(q) for q in qubits]

    if max(physical_qubits) - min(physical_qubits) > len(physical_qubits) - 1:

        # We need to apply swap gates to bring the qubits together
        # Find the optimal target physical qubits to implement the operation
        target_physical_qubits = prepare_lnn_op(physical_qubits)

        # Find the swaps that must be applied to bring these qubits together and the qubit ordering after those swaps
        n_swaps = count_lnn_swaps(physical_qubits, target_physical_qubits)
        qubit_order, swap_net_circuit = swap_lnn(qubit_order, physical_qubits, target_physical_qubits,
                                                        fermionic = fermionic)

    else:
        n_swaps = 0
        swap_net_circuit = QuantumCircuit(n)

    return n_swaps, swap_net_circuit, qubit_order

    
def get_swap_circuits(n, pool, indices, fermionic):
    """
    Given a list of K operators, returns a list of K-1 circuits performing the necessary swaps to bring qubits
    together for each operator to act on qubits that are adjacent in the connectivity map.
    Also returns the associated qubit orders.

    Arguments:
        pool (OperatorPool): the pool the operators come from
        indices (list): list of lists of indices labeling the operators
        qubit_order (list): current physical qubit mapping
        fermionic (bool): whether we're exchanging fermionic modes or qubits
    """

    qubit_order = list(range(n))
    qubit_orders = []
    swap_net_circuits = []

    for index in indices:
        iteration_qubits = pool.get_orb_qubits(index)
        n_swaps, swap_net_circuit, qubit_order = get_swaps(iteration_qubits, qubit_order, fermionic)
        qubit_orders.append(deepcopy(qubit_order))
        swap_net_circuits.append(swap_net_circuit)

    return swap_net_circuits, qubit_orders


def prepare_lnn_op(og_phys_qubits):
    """
    Finds a new set of adjacent physical qubits to represent the qubits previously represented by the input physical
    qubits. This is necessary to prepare the application of an operation in a LNN connectivity.

    Arguments:
        og_phys_qubits (list): the original physical qubit mapping
    """

    if len(og_phys_qubits) == 2:
        a,b = sorted(og_phys_qubits)
        return [b-1,b]
    elif len(og_phys_qubits) != 4:
        raise NotImplementedError

    if max(og_phys_qubits) - min(og_phys_qubits) == len(og_phys_qubits) - 1:
        # Already adjacent
        return og_phys_qubits

    a,b,c,d = sorted(og_phys_qubits)

    float_k = sum(og_phys_qubits)/4
    k = round(float_k)
    if k > float_k and k+2 < max(og_phys_qubits):
        target_phys_qubits = [k-1, k, k+1, k+2]
    else:
        target_phys_qubits = [k-2, k-1, k, k+1]

    assert target_phys_qubits[0] >= 0 and target_phys_qubits[-1] <= max(og_phys_qubits)

    if appears_in([b,c], target_phys_qubits):
        target_phys_qubits = [b-1, b, c, c+1]

    # Order the target qubits according to the original order, s.t. og_phys_qubits[i] will move to target_phys_qubits[i]
    target_phys_qubits = [x for _, x in sorted(zip(og_phys_qubits, target_phys_qubits))]

    return target_phys_qubits

def count_lnn_swaps(og_qubits,target_qubits):
    """
    Counts the number of swaps necessary to implement swap_lnn on hardware.

    Arguments:
        og_qubits (list): phsyical qubits representing the qubits before the swaps
        target_qubits (list): phsyical qubits we want to represent the qubits after the swaps
    """

    if len(og_qubits) == 2:
        a,b = sorted(og_qubits)
        return np.abs(b-a-1)
    elif len(og_qubits) == 4:
        return sum([np.abs(x1-x2) for x1,x2 in zip(target_qubits,og_qubits)])
    else:
        raise NotImplementedError
    

def apply_swap(circuit,i,j,fermionic=False):
    """
    Appends a (possible fermionic) swap between i and j to a circuit.

    Arguments:
        circuit (QuantumCircuit): the circuit to apply the swap to
        i, j (ints): the qubits the swap acts on
        fermionic (bool): whether this is a fermionic swap or a regular swap
    """
    if fermionic:
        circuit.compose(f_swap(i,j,circuit.num_qubits, big_endian=False),inplace=True)
    else:
        circuit.compose(swap_gate(i,j,circuit.num_qubits, big_endian=False),inplace=True)
        #circuit = circuit.swap(i,j)

    return circuit

def move_lnn(qubit_order,i,j,swap_net_circuit,fermionic=False):
    """
    Given a qubit order/layout, swaps qubits until qubit currently represented by physical qubit i is
    represented by physical qubit j. This is done using a cascade of swaps, assuming that the connectiviy is linear.
    This function modifies the input list and circuit.

    Arguments:
        qubit_order (list): the original mapping of physical qubits. qubit_order[i]=j means that physical qubit i
            represents qubit j.
        i (int): the current physical qubit representing the qubit we want to represent by a different physical one
        j (int): the physical qubit that should represent i in the end
        swap_net_circuit (QuantumCircuit): the quantum circuit applying the swap network
    """

    if i<j:
        for index in range(i,j):
            swap(qubit_order,index,index+1)
            apply_swap(swap_net_circuit,index,index+1,fermionic)
    else:
        for index in range(i,j,-1):
            swap(qubit_order,index,index-1)
            apply_swap(swap_net_circuit,index,index-1,fermionic)

def swap_lnn_2(qubit_order,og_phys_qubits,target_phys_qubits,swap_net_circuit,fermionic):
    """
    Performs cascades of swap gates such that qubits previously represented by the two physical qubits
    in input list og_phys_qubits will be represented by the four physical qubits in input list target_phys_qubits.
    This function modifies the input list. The lists of physical qubits must be ordered.

    Arguments:
        qubit_order (list): the initial mapping of physical qubits. Indices correspond to physical qubits, values to the
            qubits they represent.
        og_phys_qubits (list): the original physical qubit mapping
        target_phys_qubits (list): the target physical qubit mapping
        fermionic (bool): whether this is a fermionic swap or a regular swap

    Returns:
        qubit_order (list): the final mapping of physical qubits. Indices correspond to physical qubits, values to the
            qubits they represent.
        swap_net_circuit (QuantumCircuit): the QuantumCircuit applying the necessary swaps
    """

    target_phys_qubits = sorted(target_phys_qubits)
    og_phys_qubits = sorted(og_phys_qubits)

    a,b = og_phys_qubits
    og_qubit_order = deepcopy(qubit_order)

    if og_phys_qubits == target_phys_qubits:
        return qubit_order, swap_net_circuit

    move_lnn(qubit_order, a, b-1, swap_net_circuit,fermionic)

    assert [og_qubit_order[i] for i in og_phys_qubits] == [qubit_order[i] for i in target_phys_qubits]

    return qubit_order, swap_net_circuit

def update_indices(indices,original_qubit_order,new_qubit_order):
    """
    Given an original and final qubit ordering, transforms the indices in the input list such that they index the
    same qubit as before (i.e., transforms i->i' s.t. original_qubit_order[i] == new_qubit_order[i']).

    Arguments:
        indices (list): the indices to be transformed
        qubit_order (list): the original ordering of qubits
        new_qubit_order (list): the new ordering of qubits

    Returns:
        list: list with the transformed input indices
    """
    return [new_qubit_order.index(original_qubit_order[i]) for i in indices]


def move_and_update(i, i2, qubits, qubit_order, swap_net_circuit,fermionic):
    """
    Swaps qubits until the one currently represented by physical qubit i is represented by physical qubit j,
    and updates the indices of the qubits to represent the new ordering.
    The input circuit and qubit_order are modified to reflect the changes.

    Arguments:
        i (int): original physical qubit
        i2 (int): target physical qubit
        qubits (list): indices of physical qubits we want to keep track of
        qubit_order (list): the original ordering of qubits (will be modified to reflect the changes)
        swap_net_circuit (QuantumCircuit): the circuit to add the swaps to (will be modified to include them)
        fermionic (bool): whether this is a fermionic swap or a regular swap
    """

    # Make a copy of qubit_order before it's modified by move_lnn
    old_qubit_order = deepcopy(qubit_order)

    # Move i to i2, modifying qubit_order and swap_net_circuit
    move_lnn(qubit_order, i, i2, swap_net_circuit, fermionic)

    # Update the indices of the qubits to reflect the new ordering
    new_qubits = update_indices(qubits, old_qubit_order, qubit_order)

    return new_qubits

def swap_lnn_4(qubit_order,og_phys_qubits,target_phys_qubits,swap_net_circuit,fermionic):
    """
    Performs cascades of swap gates such that qubits previously represented by the four physical qubits
    in input list og_phys_qubits will be represented by the four physical qubits in input list target_phys_qubits.
    This function modifies the input list. The lists of physical qubits must be ordered.

    Arguments:
        qubit_order (list): the initial mapping of physical qubits. Indices correspond to physical qubits, values to the
            qubits they represent.
        og_phys_qubits (list): the original physical qubit mapping
        target_phys_qubits (list): the target physical qubit mapping
        fermionic (bool): whether this is a fermionic swap or a regular swap

    Returns:
        qubit_order (list): the final mapping of physical qubits. Indices correspond to physical qubits, values to the
            qubits they represent.
        swap_net_circuit (QuantumCircuit): the QuantumCircuit applying the necessary swaps
    """

    # We define the indices s.t. a<b<c<d. Order will be important
    og_phys_qubits = sorted(og_phys_qubits)
    target_phys_qubits = sorted(target_phys_qubits)
    a,b,c,d = og_phys_qubits
    a2,b2,c2,d2 = target_phys_qubits
    initial_qubit_order = deepcopy(qubit_order)

    if og_phys_qubits == target_phys_qubits:
        return qubit_order, swap_net_circuit

    # Swaps can interfere. Imagine b, c are both before b2, c2. If you change b2<->b then c2<->c, the ladder of swaps
    # for c2 will affect the previous mode attribution. Hence, we leave a and d for last. They can't interfere because
    # they're either before/after the target qubits, or the same as before.
    if b < b2 and c < c2:
        # We move c to c2 first because the path from b to b2 includes c
        if c != c2:
            [a,b,c,d] = move_and_update(c, c2, [a,b,c,d], qubit_order, swap_net_circuit,fermionic)

        if b != b2:
            [a,b,c,d] = move_and_update(b, b2, [a,b,c,d], qubit_order, swap_net_circuit,fermionic)
    else:
        # We move b to b2 first because the path from c to c2 includes b if b > b2 and c > c2
        # For other cases (e.g., b < b2 and c > c2), the order doesn't matter
        if b != b2:
            [a,b,c,d] = move_and_update(b, b2, [a,b,c,d], qubit_order, swap_net_circuit,fermionic)
        if c != c2:
            [a,b,c,d] = move_and_update(c, c2, [a,b,c,d], qubit_order, swap_net_circuit,fermionic)

    # Finally, swap a and d. These don't impact the previous b and c assignments because a<b and c<d.
    if a != a2:
            [a,b,c,d] = move_and_update(a, a2, [a,b,c,d], qubit_order, swap_net_circuit,fermionic)
    if d != d2:
            [a,b,c,d] = move_and_update(d, d2, [a,b,c,d], qubit_order, swap_net_circuit,fermionic)

    assert [initial_qubit_order[i] for i in og_phys_qubits] == [qubit_order[i] for i in target_phys_qubits]

    return qubit_order, swap_net_circuit

def swap_lnn(qubit_order,og_phys_qubits,target_phys_qubits,fermionic):
    """
    Performs cascades of swap gates such that qubits previously represented by physical qubits
    in input list og_phys_qubits will be represented by physical qubits in input list target_phys_qubits.
    This function modifies the input list. The lists of physical qubits must be ordered.

    Arguments:
        qubit_order (list): the initial mapping of physical qubits. Indices correspond to physical qubits, values to the
            qubits they represent.
        og_phys_qubits (list): the original physical qubit mapping
        target_phys_qubits (list): the target physical qubit mapping
        fermionic (bool): whether this is a fermionic swap or a regular swap

    Returns:
        qubit_order (list): the final mapping of physical qubits. Indices correspond to physical qubits, values to the
            qubits they represent.
        swap_net_circuit (QuantumCircuit): the QuantumCircuit applying the necessary swaps
    """
    #if not sorted(target_phys_qubits) == target_phys_qubits or not sorted(og_phys_qubits) == og_phys_qubits:
        #raise ValueError("Input physical qubit lists must be ordered.")

    swap_net_circuit = QuantumCircuit(len(qubit_order))

    if len(og_phys_qubits) == 2:
        return swap_lnn_2(qubit_order,og_phys_qubits,target_phys_qubits,swap_net_circuit,fermionic)

    if len(og_phys_qubits) == 4:
        return swap_lnn_4(qubit_order,og_phys_qubits,target_phys_qubits,swap_net_circuit,fermionic)

    else:
        raise NotImplementedError

def transpile_lnn(circuit,opt_level=3,basis_gates=["rz","cx","x","sx","h","s"],apply_border_swaps=False,initial_layout=None):
    """
    Transpile a quantum circuit into a linear nearest neighbor architecture.

    Arguments:
        circuit (QuantumCircuit): the quantum circuit we want to transpile
        opt_level (int): the optimization level (up to 3, which is the one that attempts most gate cancellations etc)
        basis_gates (list): the target basis gates
        apply_border_swaps (bool): whether to apply initial and final swaps such that the circuit exactly matches
            the unitary we want. Note that initial and final swaps may instead be applied classically by changing
            which index represents each qubit. Therefore, they shouldn't count towards circuit costs.

    Returns:
        QuantumCircuit: the transpiled circuit
        layout (Layout): final circuit's layout
    """

    n = circuit.num_qubits

    # Create linear nearest neighbour coupling map
    lnn_cm = []
    for i in range(n - 1):
        lnn_cm.append([i, i + 1])
        lnn_cm.append([i + 1, i])

    # Create the pass manager with the linear coupling map and the desired optimization level and basis gates
    pass_manager = generate_preset_pass_manager(opt_level, coupling_map=lnn_cm, basis_gates=basis_gates,
                                                approximation_degree=1,initial_layout=initial_layout)

    # Transpile it by calling the run method of the pass manager
    transpiled_circuit = pass_manager.run(circuit)
    layout = transpiled_circuit.layout

    if apply_border_swaps:
        transpiled_circuit = restore_qubit_ordering(transpiled_circuit,transpiled_circuit.layout)

    return transpiled_circuit, layout

def find_initial_swaps(circuit, layout):
    """
    Given a circuit transpiled using Qiskit, finds the initial swap circuit that restores the qubit ordering
    to [0,1,2,...]. Qiskit's initial layout is in general different from this, because it is optimized to decrease
    circuit costs.

    Arguments:
        circuit (QuantumCircuit): the transpiled circuit we want to restore the initial ordering of
        layout (Layout): the layout we want to restore to trivial

    Returns:
        QuantumCircuit: the swap circuit that restores the ordering if prepended to the original circuit
    """

    n = circuit.num_qubits
    swap_circuit = QuantumCircuit(n)

    # For our functions (e.g. move_lnn), qubit_order[i]=j means that qubit i is represented by physical qubit j
    # In qiskit.compose, the qubits argument is the reverse: physical qubit i represents qubit j
    # Hence, we need to redefine the layout using the index() method
    initial_layout = [layout.initial_layout[i]._index for i in range(n)]
    initial_layout = [initial_layout.index(i) for i in range(n)]

    for q in range(n):
        # Move qubit q back to the qth physical qubit
        i = initial_layout.index(q)
        move_lnn(initial_layout, i, q, swap_circuit)

    return swap_circuit

def find_final_swaps(circuit, layout):
    """
    Given a circuit transpiled using Qiskit, finds the final swap circuit that restores the qubit ordering
    to [0,1,2,...]. Qiskit's final layout is in general different from this, because it is optimized to decrease
    circuit costs.

    Arguments:
        circuit (QuantumCircuit): the transpiled circuit we want to restore the final ordering of
        layout (Layout): the layout we want to restore to trivial

    Returns:
        QuantumCircuit: the swap circuit that restores the ordering if appended to the original circuit
    """

    n = circuit.num_qubits
    swap_circuit = QuantumCircuit(n)

    if layout is None or layout.final_layout is None:
        final_layout = range(n)
    else:
        final_layout = [layout.final_layout[i]._index for i in range(n)]

    for q in range(n):
        # Move qubit q back to the qth physical qubit
        i = final_layout.index(q)
        move_lnn(final_layout, i, q, swap_circuit)

    return swap_circuit

def restore_qubit_ordering(circuit, layout, include_initial_swaps=True):
    """
    Restores the qubit ordering of a circuit transpiled using Qiskit by applying swaps at the beginning and at the end.
    After applying the swaps, qubits are ordered as [0,1,2,3,...] both as input and as output.

    Arguments:
        circuit (QuantumCircuit): the transpiled circuit we want to restore the ordering of
        layout (Layout): the layout we want to restore to trivial

    Returns:
        QuantumCircuit: the same circuit, with additional
    """

    initial_swap_circuit = find_initial_swaps(circuit, layout)
    final_swap_circuit = find_final_swaps(circuit, layout)

    circuit = circuit.compose(final_swap_circuit).compose(initial_swap_circuit.reverse_ops())
    if include_initial_swaps:
        circuit = initial_swap_circuit.compose(circuit)

    return circuit

def transform_to_qiskit_order(qubit_order, n):
    """
    Transform our qubit ordering convention to Qiskit's. The output can be directly used in Qiskit's
    QuantumCircuit.compose
    For us, qubit_order[i]=j means that qubit i is represented by physical qubit j.
    In qiskit.compose, the qubits argument is the reverse: physical qubit i represents qubit j.
    """

    # Since our pool circuits are converted such that qubit k is represented by qubit N-k-1 in Qiskit, we need to revert
    # the list and the endianness of the argument of the index method
    qiskit_ordering = [qubit_order[::-1].index(n - i - 1) for i in range(n)]

    return qiskit_ordering

def swap_gate(a,b,n,big_endian=True):
    """
    Returns circuit applying swap gate:
    [[1,0,0,0],
     [0,0,1,0],
     [0,1,0,0],
     [0,0,0,1]]

     Arguments:
        a, b (int): qubits to swap
        n (int): number of qubits in the circuit
        big_endian (bool): if to revert endianness wrt Qiskit
    """

    if big_endian:
        a = n-1-a
        b = n-1-b

    circuit = QuantumCircuit(n)

    #circuit.cx(a,b)
    #circuit.cx(b,a)
    #circuit.cx(a,b)
    circuit.swap(a,b)

    return circuit


def f_swap(a,b,n,big_endian=True):
    """
    Returns circuit applying fermionic swap gate:
    [[1,0,0,0],
     [0,0,1,0],
     [0,1,0,0],
     [0,0,0,-1]]

     Arguments:
        a, b (int): qubits to swap
        n (int): number of qubits in the circuit
        big_endian (bool): if to revert endianness wrt Qiskit
    """

    if big_endian:
        a = n-1-a
        b = n-1-b

    circuit = QuantumCircuit(n)

    circuit.s(a)
    circuit.s(b)
    circuit.h(a)
    circuit.cx(a,b)
    circuit.cx(b,a)
    circuit.h(b)
    circuit.rz(-np.pi/2,a)
    circuit.rz(-np.pi/2,b)

    return circuit


def get_order_restoring_circuit(current_qubit_order, fermionic=False):
    """
    Get the swap network circuit that restores the qubit order to [0,1,2,...].

    Arguments:
        current_qubit_order (list): the mapping of physical qubits that we want to convert back to [0,1,2,...]
        fermionic (bool): whether the swaps are fermionic or regular

    Returns:
        QuantumCircuit: the swap circuit that restores the order
    """

    n = len(current_qubit_order)
    qubit_order = current_qubit_order.copy()
    swap_circuit = QuantumCircuit(n)

    for q in range(n):
        i = qubit_order.index(q)
        move_lnn(qubit_order, i, q, swap_circuit, fermionic=fermionic)

    return swap_circuit


def correct_signs(operators, coefficients, qubit_order):
    """
    Corrects the signs of the coefficients associated with an operator given by a list of indices, given a
    qubit order representing an ordering of fermionic modes.
    When swaps are fermionic, there might be a phase with respect to the desired circuit if the qubit_order
     - which in that case corresponds to the mode order - places source and target orbitals in distinct
    directions from the original ones in the standard mode order (e.g., the first source orbital comes 
    before the second, but the first target orbital comes after the second; but when the mode order was
    just [0,1,2,...], the first of each came before the second). 
    We need to flip the coefficient in this case for the circuits to match the unitary.
    In the case of OVP-CEOs, if the sign for one of the constituent QEs is flipped but not the other,
    we additionally have to convert the operator type (sum <-> difference).

    Arguments:
        operators (list): list of PoolOperators
        coefficients (list): ansatz coefficients
        qubit_order (list): the fermionic mode mapping in which the operator will be implemented

    Returns:
        coefficients (list): the corrected coefficients
        switch (bool): whether to switch operator type in the case of OVP-CEOs (sum <-> difference)
    """

    original_qubit_order = list(range(len(qubit_order)))
    for k, op in enumerate(operators):
        source_orbs, target_orbs = op.source_orbs, op.target_orbs
        switch = False
        if len(source_orbs) == 2:

            if isinstance(source_orbs[0], list):
                # OVP-CEO

                s1, s2 = qubit_order.index(source_orbs[0][0]), qubit_order.index(source_orbs[0][1])
                t1, t2 = qubit_order.index(target_orbs[0][0]), qubit_order.index(target_orbs[0][1])
                os1, os2 = original_qubit_order.index(source_orbs[1][0]), original_qubit_order.index(source_orbs[1][1])
                ot1, ot2 = original_qubit_order.index(target_orbs[1][0]), original_qubit_order.index(target_orbs[1][1])
                flip1 = (s1 < s2) != (os1 < os2)
                flip2 = (t1 < t2) != (ot1 < ot2)
                # Whether to flip sign of first QE 
                flip_qe1 = (flip1 + flip2 == 1)

                s1, s2 = qubit_order.index(source_orbs[1][0]), qubit_order.index(source_orbs[1][1])
                t1, t2 = qubit_order.index(target_orbs[1][0]), qubit_order.index(target_orbs[1][1])
                os1, os2 = original_qubit_order.index(source_orbs[1][0]), original_qubit_order.index(source_orbs[1][1])
                ot1, ot2 = original_qubit_order.index(target_orbs[1][0]), original_qubit_order.index(target_orbs[1][1])
                flip1 = (s1 < s2) != (os1 < os2)
                flip2 = (t1 < t2) != (ot1 < ot2)
                # Whether to flip sign of second QE 
                flip_qe2 = (flip1 + flip2 == 1)

                if flip_qe1 and not flip_qe2:
                    switch = True
                    coefficients[k] = -coefficients[k]
                if not flip_qe1 and flip_qe2:
                    switch = True
                if flip_qe1 and flip_qe2:
                    coefficients[k] = -coefficients[k]

            else:
                # Double excitation
                s1, s2 = qubit_order.index(source_orbs[0]), qubit_order.index(source_orbs[1])
                t1, t2 = qubit_order.index(target_orbs[0]), qubit_order.index(target_orbs[1])
                os1, os2 = original_qubit_order.index(source_orbs[0]), original_qubit_order.index(source_orbs[1])
                ot1, ot2 = original_qubit_order.index(target_orbs[0]), original_qubit_order.index(target_orbs[1])
                flip1 = (s1 < s2) != (os1 < os2)
                flip2 = (t1 < t2) != (ot1 < ot2)
                if (flip1 + flip2 == 1):
                    coefficients[k] = - coefficients[k]

    return coefficients, switch


def get_circuit_energy(circuit,hamiltonian):
    """
    Gets the expectation value of a Hamiltonian in a circuit.

    Arguments:
        circuit (QuantumCircuit): the circuit we want to measure the energy in
        hamiltonian (Union[InteractionOperator,QubitOperator,FermionOperator]): the Hamiltonian we want the expectation
            value of

    Returns:
        energy (float): the energy in this circuit
    """

    sv = Statevector(circuit).data
    h = get_sparse_operator(hamiltonian).todense()
    energy = sv.dot(h).dot(sv.transpose().conjugate())[0,0]
    assert np.abs(energy.imag) < 10**-10
    energy = energy.real

    return energy