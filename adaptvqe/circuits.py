import re

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate

from adaptvqe.op_conv import convert_hamiltonian, read_of_qubit_operator


def cnot_depth(qasm):
    """
    Counts the depth of a circuit represented by a QASM string, considering only cx gates.
    Circuit must be decomposed into a cx + single qubit rotations gate set.
    """

    n = int(re.search(r"(?<=q\[)[0-9]+(?=\])", qasm.splitlines()[2]).group())
    depths = [0 for _ in range(n)]

    for line in qasm.splitlines()[3:]:
        # Remove ;
        line = line[:-1]

        op, qubits = line.split(" ")

        if op[:2] != "cx":
            continue

        qubits = qubits.split(",")
        for i, qubit in enumerate(qubits):
            # Get qubit indices as list
            qubits[i] = int(re.search(r"(?<=q\[)[0-9]+(?=\])", qubit).group())

        max_depth = max([depths[qubit] for qubit in qubits])
        new_depth = max_depth + 1

        for qubit in qubits:
            depths[qubit] = new_depth

    return max(depths)


def cnot_count(qasm):
    """
    Counts the CNOTs in a circuit represented by a QASM string.
    """
    count = 0

    for line in qasm.splitlines()[3:]:
        # Remove ;
        line = line[:-1]

        op, qubits = line.split(" ")

        if op[:2] == "cx":
            count += 1

    return count


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


def pauli_exp_circuit(qubit_operator, n, big_endian=False):
    """
    Implements the exponential of an operator, which must be anti-Hermitian.
    If the Paulis commute, the circuit is exact. If they don't, it's an approximation with 1 Trotter step.

    Arguments:
        qubit_operator (QubitOperator): the generator of the unitary (exponent)
        n (int): the number of qubits of the output circuit (must be >= number of qubits qubit_operator acts on)
        big_endian (bool): if True/False, big/little endian ordering will be assumed

    Returns:
        Circuit implementing e^qubit_operator (approximately if strings in generator do not commute)
    """

    pauli_dict = convert_hamiltonian(qubit_operator)
    qc = QuantumCircuit(n)

    for pauli_string in pauli_dict.keys():

        coeff = pauli_dict[pauli_string]

        if big_endian:
            pauli_string = pauli_string + "I" * (n - len(pauli_string))
            pauli_string = pauli_string[::-1]

        assert coeff.real < 10 ** -8
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


def ovp_ceo_circuit(source_orbs, target_orbs, n, theta, ceo_type, big_endian=False):
    """
    Implements the OVP-CEO defined by the input arguments using CNOT and single
    qubit gates.
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
