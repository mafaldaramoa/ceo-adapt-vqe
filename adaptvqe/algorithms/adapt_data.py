# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:19:38 2022

@author: mafal
"""

from copy import deepcopy

import numpy as np
from qiskit import QuantumCircuit

from ..circuits import (cnot_depth, cnot_count, transpile_lnn, transform_to_qiskit_order, get_order_restoring_circuit,
                        correct_signs, find_final_swaps, get_swap_circuits)
from ..op_conv import get_qasm


class AnsatzData:

    def __init__(self, coefficients=[], indices=[], sel_gradients=[]):
        self.coefficients = coefficients
        self.indices = indices
        self.sel_gradients = sel_gradients

    def grow(self, indices, new_coefficients, sel_gradients):
        self.indices = indices
        self.coefficients = new_coefficients
        self.sel_gradients = np.append(self.sel_gradients, sel_gradients)

    def remove(self, index, new_coefficients):
        self.indices.pop(index)
        self.coefficients = new_coefficients
        rem_grad = self.sel_gradients.pop(index)

        return rem_grad

    @property
    def size(self):
        return len(self.indices)


class IterationData:

    def __init__(
        self,
        ansatz=None,
        energy=None,
        error=None,
        energy_change=None,
        gradient_norm=None,
        prep_gradient_norm=None,
        sel_gradients=None,
        inv_hessian=None,
        gradients=None,
        nfevs=None,
        ngevs=None,
        nits=None,
        qubit_order=None,
        n_swaps=None,
        swap_net_circuit=None
    ):
        """
        Ansatz and energy at the end of the iteration
        Gradient of selected operator (before optimization)
        Gradient norm (before optimization)
        Number of function evaluations
        Number o gradient vector evaluations
        """
        if ansatz:
            self.ansatz = deepcopy(ansatz)
        else:
            self.ansatz = AnsatzData()

        self.energy = energy
        self.energy_change = energy_change
        self.error = error
        self.gradient_norm = gradient_norm
        self.prep_gradient_norm = prep_gradient_norm
        self.sel_gradients = sel_gradients
        self.inv_hessian = inv_hessian
        self.gradients = gradients
        self.nfevs = nfevs
        self.ngevs = ngevs
        self.nits = nits
        self.qubit_order = qubit_order
        self.n_swaps = n_swaps
        self.swap_net_circuit = swap_net_circuit


class EvolutionData:

    def __init__(self, initial_energy, prev_ev_data=None):

        self.initial_energy = initial_energy

        if prev_ev_data:
            self.its_data = prev_ev_data.its_data
        else:
            # List of IterationData objects
            self.its_data = []

    def reg_it(
        self,
        coefficients,
        indices,
        energy,
        error,
        gradient_norm,
        prep_gradient_norm,
        sel_gradients,
        inv_hessian,
        gradients,
        nfevs,
        ngevs,
        nits,
        qubit_order,
        n_swaps,
        swap_net_circuit
    ):

        if self.its_data:
            previous_energy = self.last_it.energy
        else:
            previous_energy = self.initial_energy

        energy_change = energy - previous_energy

        ansatz = deepcopy(self.last_it.ansatz)
        ansatz.grow(indices, coefficients, sel_gradients)

        it_data = IterationData(
            ansatz,
            energy,
            error,
            energy_change,
            gradient_norm,
            prep_gradient_norm,
            sel_gradients,
            inv_hessian,
            gradients,
            nfevs,
            ngevs,
            nits,
            qubit_order,
            n_swaps,
            swap_net_circuit
        )

        self.its_data.append(it_data)

        return

    @property
    def coefficients(self):
        return [it_data.ansatz.coefficients for it_data in self.its_data]

    @property
    def energies(self):
        return [it_data.energy for it_data in self.its_data]

    @property
    def inv_hessians(self):
        return [it_data.inv_hessian for it_data in self.its_data]

    @property
    def gradients(self):
        return [it_data.gradients for it_data in self.its_data]

    @property
    def errors(self):
        return [it_data.error for it_data in self.its_data]

    @property
    def energy_changes(self):
        return [it_data.energy_change for it_data in self.its_data]

    @property
    def gradient_norms(self):
        return [it_data.gradient_norm for it_data in self.its_data]

    @property
    def prep_gradient_norms(self):
        return [it_data.prep_gradient_norm for it_data in self.its_data]

    @property
    def indices(self):
        return [it_data.ansatz.indices for it_data in self.its_data]

    @property
    def nfevs(self):
        return [it_data.nfevs for it_data in self.its_data]

    @property
    def ngevs(self):
        return [it_data.ngevs for it_data in self.its_data]

    @property
    def nits(self):
        return [it_data.nits for it_data in self.its_data]

    @property
    def sel_gradients(self):
        return [it_data.sel_gradients for it_data in self.its_data]

    @property
    def qubit_orders(self):
        return [it_data.qubit_order for it_data in self.its_data]

    @property
    def sizes(self):
        return [len(it_data.ansatz.indices) for it_data in self.its_data]

    @property
    def n_swaps(self):
        return [it_data.n_swaps for it_data in self.its_data]

    @property
    def swap_net_circuits(self):
        return [it_data.swap_net_circuit for it_data in self.its_data]

    @property
    def last_it(self):

        if self.its_data:
            return self.its_data[-1]
        else:
            # No data yet. Return empty IterationData object
            return IterationData()


class AdaptData:
    """
    Class meant to store data from an Adapt VQE run.

    Methods:
      process_iteration: to be called by the AdaptVQE class at the end of each
        iteration
      close: to be called by the AdaptVQE class at the end of the run
      plot: to be called to plot data after the run
    """

    def __init__(
        self, initial_energy, pool, ref_det, sparse_ref_state, file_name, fci_energy, n, hamiltonian, lnn, lnn_qiskit
    ):
        """
        Initialize class instance

        Arguments:
          initial_energy (float): energy of the reference state
          pool (OperatorPool): operator pool
          ref_det (list): the length n Slater determinant that will be used as the reference state
          sparse_ref_state (csc_matrix): the state to be used as the reference state (e.g. Hartree-Fock)
          file_name (str): a string describing the ADAPT implementation type and molecule
          fci_energy (float): the exact ground energy
          n (int): the size of the system (number of qubits)
          hamiltonian (csc_matrix): the Hamiltonian of the system
          lnn (bool): if to compile to LNN using swaps
          lnn_qiskit (bool): if to compile to LNN using Qiskit's transpiler
        """

        self.pool_name = pool.name

        # The initial energy is stored apart from the remaining ones, so that
        # the gradient norms in the beginning of an iteration are stored associated
        # with the ansatz and energy in that iteration
        self.initial_energy = initial_energy
        self.initial_error = initial_energy - fci_energy
        self.ref_det = ref_det
        self.sparse_ref_state = sparse_ref_state

        self.evolution = EvolutionData(initial_energy)
        self.file_name = file_name
        self.iteration_counter = 0
        self.fci_energy = fci_energy
        self.n = n
        self.hamiltonian = hamiltonian
        self.lnn = lnn
        self.lnn_qiskit = lnn_qiskit

        self.closed = False
        self.success = False

    def process_iteration(
        self,
        indices,
        energy,
        gradient_norm,
        prep_gradient_norm,
        selected_gradients,
        coefficients,
        inv_hessian,
        gradients,
        nfevs,
        ngevs,
        nits,
        qubit_order,
        n_swaps=None,
        swap_net_circuit=None
    ):
        """
        Receives and processes the values fed to it by an instance of the AdaptVQE
        class at the end of each run.

        Arguments:
          indices: indices of the ansatz elements at the end of the iteration
          energy (float): energy at the end of the iteration
          gradient_norm (int): the norm of the total gradient norm at the beggining
            of this iteration
          prep_gradient_norm (int): same as above, but prepending instead of appending
          selected_gradients (float): the absolute values of the gradient of the
            operators that were added in this iteration
          coefficients (list): a list of the coefficients selected by the optimizer
            in this iteration
          inv_hessian (np.ndarray): the approximate inverse Hessian at the end of the
            iteration
          gradients (list): the gradients of the ansatz elements at the end of the
            iteration
          nfevs (list): the number of function evaluations during the
            optimization. List length should match the number of optimizations
          ngevs (list): the number of evaluations of operator gradients during the
            optimization. List length should match the number of optimizations
          qubit_order (list): order of the modes. qubit_order[i] is the mode represented
            by qubit i.
          n_swaps (int): number of swaps necessary to implement the last operation, if
            the connectivity isn't all-to-all
          swap_net_circuit:
        """

        if not isinstance(energy, float):
            raise TypeError("Expected float, not {}.".format(type(energy).__name__))

        if not isinstance(gradient_norm, (float, np.floating)):
            raise TypeError(
                "Expected float, not {}.".format(type(gradient_norm).__name__)
            )

        if prep_gradient_norm is not None and not isinstance(prep_gradient_norm, (float, np.float64)):
            raise TypeError(
                "Expected float, not {}.".format(type(prep_gradient_norm).__name__)
            )

        if not (
            isinstance(selected_gradients, list)
            or isinstance(selected_gradients, np.ndarray)
        ):
            raise TypeError(
                "Expected list, not {}.".format(type(selected_gradients).__name__)
            )

        if not isinstance(coefficients, list):
            raise TypeError(
                "Expected list, not {}.".format(type(coefficients).__name__)
            )

        if not isinstance(nfevs, list):
            raise TypeError("Expected list, not {}.".format(type(nfevs).__name__))

        if not isinstance(ngevs, list):
            raise TypeError("Expected list, not {}.".format(type(ngevs).__name__))

        if not isinstance(nits, list):
            raise TypeError("Expected list, not {}.".format(type(ngevs).__name__))

        if not isinstance(qubit_order, list):
            raise TypeError("Expected list, not {}.".format(type(qubit_order).__name__))

        if not isinstance(n_swaps, (int, np.integer, None)):
            raise TypeError("Expected int, not {}.".format(type(n_swaps).__name__))

        if len(coefficients) != len(indices):
            raise ValueError(
                "The length of the coefficient list should match the"
                " ansatz size ({} != {}).".format(len(coefficients), len(indices))
            )
        """
        if gradients is not None:
            if len(gradients) != len(indices):
                raise ValueError("The length of the gradient vector match the"
                                 " ansatz size ({} != {})."
                                 .format(len(gradients),
                                         len(indices)))"""

        if gradient_norm < 0:
            raise ValueError(
                "Total gradient norm should be positive; its {}".format(gradient_norm)
            )
        if prep_gradient_norm is not None and prep_gradient_norm < 0:
            raise ValueError(
                "Total gradient norm should be positive; its {}".format(gradient_norm)
            )

        error = energy - self.fci_energy
        self.evolution.reg_it(
            coefficients,
            indices,
            energy,
            error,
            gradient_norm,
            prep_gradient_norm,
            selected_gradients,
            inv_hessian,
            gradients,
            nfevs,
            ngevs,
            nits,
            qubit_order,
            n_swaps,
            swap_net_circuit
        )

        self.iteration_counter += 1

        return energy

    def acc_depths(self, pool):
        """
        Outputs the list of accumulated depth through the iterations.
        Depth for iteration 0 (reference state), then for 1, then for 2, etc.
        Depth is the total number of gate layers - entangling or not, all gates are
        considered equal
        """
        assert pool.name == self.pool_name

        acc_depths = [0]
        ansatz_size = 0
        circuit = QuantumCircuit(pool.n)

        for iteration in self.evolution.its_data:
            indices = iteration.ansatz.indices
            coefficients = iteration.ansatz.coefficients
            new_indices = indices[ansatz_size:]
            new_coefficients = coefficients[ansatz_size:]
            ansatz_size += len(new_indices)

            new_circuit = pool.get_circuit(new_indices, new_coefficients)
            circuit = circuit.compose(new_circuit)
            depth = circuit.depth()
            acc_depths.append(depth)

        return acc_depths

    def acc_cnot_depths(self, pool, fake_params=False):
        """
        Outputs the list of accumulated CNOT depth through the iterations.
        Depth for iteration 0 (reference state), then for 1, then for 2, etc.
        All single qubit gates are ignored.
        """

        acc_depths = [0]
        ansatz_size = 0
        circuit = QuantumCircuit(pool.n)

        for iteration in self.evolution.its_data:
            indices = iteration.ansatz.indices
            coefficients = iteration.ansatz.coefficients
            new_indices = indices[ansatz_size:]
            new_coefficients = coefficients[ansatz_size:]
            ansatz_size += len(new_indices)

            if fake_params:
                # Sometimes if the coefficient is too small Openfermion will read the operator as zero, so this is
                # necessary for the circuit functions not to raise an error
                new_coefficients = [np.random.rand() for _ in coefficients]

            new_circuit = pool.get_circuit(new_indices, new_coefficients)
            circuit = circuit.compose(new_circuit)
            qasm_circuit = get_qasm(circuit)
            depth = cnot_depth(qasm_circuit, self.n)
            acc_depths.append(depth)

        return acc_depths

    def acc_cnot_counts(self, pool, fake_params=False):
        """
        Get accumulated CNOT counts across iterations.

        Arguments:
            pool (OperatorPool): pool used to define the ansatz
            fake_params (bool): whether to replace parameters with random ones (if parameters are too small,
                operators may be viewed as zero by OpenFermion

        Returns:
            list: a list of the evolution of the CNOT count along ADAPT iterations
        """

        acc_counts = [0]
        ansatz_size = 0
        count = 0

        for iteration in self.evolution.its_data:
            indices = iteration.ansatz.indices
            coefficients = iteration.ansatz.coefficients
            new_indices = indices[ansatz_size:]
            new_coefficients = coefficients[ansatz_size:]
            ansatz_size += len(new_indices)

            if fake_params:
                # Sometimes if the coefficient is too small Openfermion will read the operator as zero, so this is
                # necessary for the circuit functions not to raise an error
                new_coefficients = [np.random.rand() for _ in coefficients]

            new_circuit = pool.get_circuit(new_indices, new_coefficients)
            qasm_circuit = get_qasm(new_circuit)
            count += cnot_count(qasm_circuit)
            acc_counts.append(count)

        return acc_counts

    def acc_lnn_cnot_counts(self, pool, fake_params=False):
        """
        Get accumulated CNOT counts across iterations.

        Arguments:
            pool (OperatorPool): pool used to define the ansatz
            fake_params (bool): whether to replace parameters with random ones (if parameters are too small,
                operators may be viewed as zero by OpenFermion

        Returns:
            list: a list of the evolution of the CNOT count along ADAPT iterations
        """

        _, acc_counts, _ = self.get_lnn_circuit(pool, fake_params=True)

        return acc_counts


    def get_iteration_range(self, iteration):
        """
        Returns the range of positions in the indices and coefficients list associated with a given iteration.
        For most pools, iteration i corresponds to position i only. For pools where each element may have more than
        1 variational parameter, e.g. MVP-CEO, one iteration may correspond to more than one position.
        """

        if iteration:
            n_old_indices = len(self.evolution.indices[iteration - 1])
        else:
            n_old_indices = 0

        n_new_indices = len(self.evolution.indices[iteration])

        return range(n_old_indices,n_new_indices)


    def get_iteration_ixs(self, iteration):
        """
        Returns the list of indices added in a given iteration. May be up to three in the case of MVP-CEOs.
        """

        return [self.result.ansatz.indices[i] for i in self.get_iteration_range(iteration)]

    def get_circuit(self,pool,indices=None,coefficients=None,include_ref=False):
        """
        Returns the QuantumCircuit implementing the ansatz defined by indices and coefficients. If they are None, final
        values are used.
        """

        if pool.fermionic_swaps:
            raise ValueError("This method does not support fermionic swap implementations. Use get_lnn_circuit instead.")

        if indices is None:
            indices = self.result.ansatz.indices
        if coefficients is None:
            coefficients = self.result.ansatz.coefficients

        qc = pool.get_circuit(indices, coefficients)

        if include_ref:
            qc = self.get_ref_circuit().compose(qc)

        return qc

    def get_ref_circuit(self):

        ref_circuit = QuantumCircuit(self.n)
        for q, s in enumerate(self.ref_det):
            if s:
                ref_circuit.x(self.n - 1 - q)

        return ref_circuit

    def get_lnn_circuit(self,pool,iteration=None,apply_border_swaps=False,fake_params=False,include_ref=False,basis_gates=["rz","cx","x","sx","h","s"],):
        """
        Get the circuit representing the final ansatz, transpiled to LNN connectivity.
        The circuit will be composed of CNOTs + single qubit gates, and possibly swaps if apply_border_swaps flag is set.

        Arguments:
            pool (OperatorPool): the pool to use to construct the ansatz. Must match the pool that was used to run
                the instance of AdaptVQE corresponding to this AdaptData instance.
            iteration (int): the iteration at which to compute the circuit. Iteration 0 is the first one
            apply_border_swaps (bool): whether to apply initial and final swaps such that the circuit exactly matches
                the unitary we want. Note that initial and final swaps may instead be applied classically by changing
                which index represents each qubit. Therefore, they shouldn't count towards circuit costs.
            fake_params (bool): whether to replace parameters with random ones (if parameters are too small,
                operators may be viewed as zero by OpenFermion

        Returns:
            transpiled_circuit (QuantumCircuit): the final circuit
            acc_cnot_counts (list): list of accumulated CNOT counts
            layout (Layout): final circuit's layout

        """
        assert self.pool_name == pool.name

        if iteration is None:
            # Set iteration to last one (final ansatz)
            iteration = self.iteration_counter - 1

        if not self.lnn:
            return self.qiskit_lnn_circuit(pool, iteration, apply_border_swaps,fake_params,include_ref=include_ref,basis_gates=basis_gates)

        return self.swap_based_lnn_circuit(pool, iteration, apply_border_swaps,fake_params,include_ref=include_ref,basis_gates=basis_gates)

    def qiskit_lnn_circuit(self,pool,iteration,apply_border_swaps,fake_params,include_ref,basis_gates=["rz","cx","x","sx","h","s"],):
        """
        Get the circuit representing the final ansatz, transpiled to LNN connectivity using Qiskit's transpiler.
        The circuit will be composed of CNOTs + single qubit gates, and possibly swaps if apply_border_swaps flag is set.

        Arguments:
            pool (OperatorPool): the pool to use to construct the ansatz. Must match the pool that was used to run
                the instance of AdaptVQE corresponding to this AdaptData instance.
            iteration (int): the iteration at which to compute the circuit. Iteration 0 is the first one
            apply_border_swaps (bool): whether to apply initial and final swaps such that the circuit exactly matches
                the unitary we want. Note that initial and final swaps may instead be applied classically by changing
                which index represents each qubit. Therefore, they shouldn't count towards circuit costs.
            fake_params (bool): whether to replace parameters with random ones (if parameters are too small,
                operators may be viewed as zero by OpenFermion

        Returns:
            transpiled_circuit (QuantumCircuit): the final circuit
            acc_cnot_counts (list): list of accumulated CNOT counts
            layout (Layout): final circuit's layout

        """
        if fake_params:
            ansatz_coefficients = [np.random.rand() for _ in range(len(self.evolution.coefficients[iteration]))]
        else:
            ansatz_coefficients = self.evolution.coefficients[iteration]

        if include_ref:
            circuit = self.get_ref_circuit()
        else:
            circuit = QuantumCircuit(self.n)

        acc_cnot_counts = [0]
        for i in range(iteration + 1):
            indices = self.get_iteration_ixs(i)
            coefficients = [ansatz_coefficients[k] for k in self.get_iteration_range(i)]

            op_circuit = self.get_circuit(pool, indices, coefficients)
            circuit.compose(op_circuit,inplace=True)

            # Transpile and optimize to calculate CNOT count for this iteration
            lnn_circuit, layout = transpile_lnn(circuit, basis_gates=basis_gates, apply_border_swaps=apply_border_swaps)

            qasm_circuit = get_qasm(lnn_circuit)
            count = cnot_count(qasm_circuit)
            acc_cnot_counts.append(count)

        return lnn_circuit, acc_cnot_counts, layout
    
    
    def swap_based_lnn_circuit(self,pool,iteration,apply_border_swaps,fake_params,include_ref,basis_gates=["rz","cx","x","sx","h","s"]):
        """
        Get the circuit representing the final ansatz in a LNN architecture, implemented using (possibly fermionic)
        swap gates.
        The circuit will be composed of CNOTs + single qubit gates, and possibly swaps if apply_border_swaps flag is set.

        Arguments:
            pool (OperatorPool): the pool to use to construct the ansatz. Must match the pool that was used to run
                the instance of AdaptVQE corresponding to this AdaptData instance.
            iteration (int): the iteration at which to compute the circuit. Iteration 0 is the first one
            apply_border_swaps (bool): whether to apply initial and final swaps such that the circuit exactly matches
                the unitary we want. Note that initial and final swaps may instead be applied classically by changing
                which index represents each qubit. Therefore, they shouldn't count towards circuit costs.
            fake_params (bool): whether to replace parameters with random ones (if parameters are too small,
                operators may be viewed as zero by OpenFermion

        Returns:
            transpiled_circuit (QuantumCircuit): the final circuit
            acc_cnot_counts (list): list of accumulated CNOT counts
            layout (Layout): final circuit's layout

        """

        assert self.lnn

        if fake_params:
            ansatz_coefficients = [np.random.rand() for _ in range(len(self.evolution.coefficients[iteration]))]
        else:
            ansatz_coefficients = self.evolution.coefficients[iteration]

        if include_ref:
            circuit = self.get_ref_circuit()
        else:
            circuit = QuantumCircuit(self.n)

        # If we have MVP-CEOs, we have multiple indices, but they all act on the same qubits so we don't need to
        #know all of them to decide which qubits to swap. Take the first one
        iteration_ixs = [self.get_iteration_ixs(i)[0] for i in range(self.iteration_counter)]
        swap_net_circuits, qubit_orders = get_swap_circuits(self.n, pool, iteration_ixs, pool.fermionic_swaps)

        acc_cnot_counts = [0]
        for i in range(iteration + 1):
            # Add the circuit for this iteration's operator

            # Get the indices and coefficients added in this iteration
            indices = self.get_iteration_ixs(i)
            coefficients = [ansatz_coefficients[k] for k in self.get_iteration_range(i)]

            # Get the circuit that is necessary to bring the qubits involved in this operator together
            swap_net_circuit = swap_net_circuits[i]

            # Append swap circuit to ansatz to bring relevant qubits together
            # Our qubit ordering uses big endian, we revert it using the qubits kwarg to match with Qiskit
            circuit.compose(swap_net_circuit, inplace=True, qubits=range(self.n)[::-1])

            qubit_order = qubit_orders[i]
            qiskit_ordering = transform_to_qiskit_order(qubit_order, self.n)

            if pool.fermionic_swaps:
            # Mode ordering might introduce a phase with respect to the typical circuit
                coefficients, switch = correct_signs([pool.operators[i] for i in indices], coefficients, qubit_order)
                if switch:
                    # Switch CEO type to account for this phase
                    assert len(indices) == 1
                    indices = [pool.find_dual_op(indices[0])]

            op_circuit = QuantumCircuit(self.n).compose(pool.get_circuit(indices, coefficients), qubits=qiskit_ordering)
            op_circuit, _ = transpile_lnn(op_circuit, opt_level=0, initial_layout=range(self.n), apply_border_swaps=False, basis_gates=basis_gates)
            swaps = find_final_swaps(op_circuit,op_circuit.layout)
            op_circuit = op_circuit.compose(swaps)
            circuit.compose(op_circuit, inplace=True)

            # Optimize for this iteration
            opt_circuit, _ = transpile_lnn(circuit,apply_border_swaps=apply_border_swaps, basis_gates=basis_gates)
            acc_cnot_counts.append(cnot_count(get_qasm(opt_circuit)))

            # This would not keep initial or final ordering:
            # circuit.compose(pool.get_circuit(indices,coefficients),qubits=qiskit_ordering,inplace=True)

        # Optimize final circuit
        lnn_circuit, layout = transpile_lnn(circuit, apply_border_swaps=apply_border_swaps,basis_gates=basis_gates)

        if apply_border_swaps:
            # Apply swaps to restore the qubit ordering to [0,1,2,...]
            swap_circuit = get_order_restoring_circuit(qubit_order, fermionic=pool.fermionic_swaps)
            # We again revert the ordering to match the endianness
            lnn_circuit.compose(swap_circuit, inplace=True, qubits=range(self.n)[::-1])

        return lnn_circuit, acc_cnot_counts, layout


    def swap_based_lnn_circuit_old(self,pool,iteration,apply_border_swaps,fake_params,include_ref,basis_gates=["rz","cx","x","sx","h","s"]):
        """
        Get the circuit representing the final ansatz in a LNN architecture, implemented using (possibly fermionic)
        swap gates.
        The circuit will be composed of CNOTs + single qubit gates, and possibly swaps if apply_border_swaps flag is set.

        Arguments:
            pool (OperatorPool): the pool to use to construct the ansatz. Must match the pool that was used to run
                the instance of AdaptVQE corresponding to this AdaptData instance.
            iteration (int): the iteration at which to compute the circuit. Iteration 0 is the first one
            apply_border_swaps (bool): whether to apply initial and final swaps such that the circuit exactly matches
                the unitary we want. Note that initial and final swaps may instead be applied classically by changing
                which index represents each qubit. Therefore, they shouldn't count towards circuit costs.
            fake_params (bool): whether to replace parameters with random ones (if parameters are too small,
                operators may be viewed as zero by OpenFermion

        Returns:
            transpiled_circuit (QuantumCircuit): the final circuit
            acc_cnot_counts (list): list of accumulated CNOT counts
            layout (Layout): final circuit's layout

        """

        assert self.lnn

        if fake_params:
            ansatz_coefficients = [np.random.rand() for _ in range(len(self.evolution.coefficients[iteration]))]
        else:
            ansatz_coefficients = self.evolution.coefficients[iteration]

        if include_ref:
            circuit = self.get_ref_circuit()
        else:
            circuit = QuantumCircuit(self.n)

        acc_cnot_counts = [0]
        for i in range(iteration + 1):
            # Add the circuit for this iteration's operator

            # Get the indices and coefficients added in this iteration
            indices = self.get_iteration_ixs(i)
            coefficients = [ansatz_coefficients[k] for k in self.get_iteration_range(i)]

            # Get the circuit that is necessary to bring the qubits involved in this operator together
            swap_net_circuit = self.evolution.swap_net_circuits[i]

            # Append swap circuit to ansatz to bring relevant qubits together
            # Our qubit ordering uses big endian, we revert it using the qubits kwarg to match with Qiskit
            circuit.compose(swap_net_circuit, inplace=True, qubits=range(self.n)[::-1])

            qubit_order = self.evolution.qubit_orders[i]
            qiskit_ordering = transform_to_qiskit_order(qubit_order, self.n)

            if pool.fermionic_swaps:
            # Mode ordering might introduce a phase with respect to the typical circuit
                coefficients, switch = correct_signs([pool.operators[i] for i in indices], coefficients, qubit_order)
                if switch:
                    # Switch CEO type to account for this phase
                    assert len(indices) == 1
                    indices = [pool.find_dual_op(indices[0])]

            op_circuit = QuantumCircuit(self.n).compose(pool.get_circuit(indices, coefficients), qubits=qiskit_ordering)
            op_circuit, _ = transpile_lnn(op_circuit, opt_level=0, initial_layout=range(self.n), apply_border_swaps=False, basis_gates=basis_gates)
            swaps = find_final_swaps(op_circuit,op_circuit.layout)
            op_circuit = op_circuit.compose(swaps)
            circuit.compose(op_circuit, inplace=True)

            # Optimize for this iteration
            opt_circuit, _ = transpile_lnn(circuit,apply_border_swaps=apply_border_swaps, basis_gates=basis_gates)
            acc_cnot_counts.append(cnot_count(get_qasm(opt_circuit)))

            # This would not keep initial or final ordering:
            # circuit.compose(pool.get_circuit(indices,coefficients),qubits=qiskit_ordering,inplace=True)

        # Optimize final circuit
        lnn_circuit, layout = transpile_lnn(circuit, apply_border_swaps=apply_border_swaps,basis_gates=basis_gates)

        if apply_border_swaps:
            # Apply swaps to restore the qubit ordering to [0,1,2,...]
            swap_circuit = get_order_restoring_circuit(qubit_order, fermionic=pool.fermionic_swaps)
            # We again revert the ordering to match the endianness
            lnn_circuit.compose(swap_circuit, inplace=True, qubits=range(self.n)[::-1])

        return lnn_circuit, acc_cnot_counts, layout


    def close(self, success, file_name=None):
        """
        To be called at the end of the run, to close the data structures

        Arguments:
          success (bool): True if the convergence condition was met, False if not
            (the maximum number of iterations was met before that)
          file_name (str): the final file name to assume
        """

        self.result = self.evolution.last_it
        self.closed = True
        self.success = success
        if file_name is not None:
            self.file_name = file_name

    def delete_hessians(self, preserve_last=True):
        """
        Delete the inverse Hessians stored from a run of ADAPT-VQE. By default, it keeps the last one, in case a new
        instance of ADAPT-VQE will need the data.

        Arguments:
            preserve_last (bool): if to preserve the last inverse Hessian
        """

        for it_data in self.evolution.its_data[:-preserve_last]:
            it_data.inv_hessian = None

    @property
    def current(self):
        """
        Current iteration
        """
        if self.evolution.its_data:
            return self.evolution.last_it
        else:
            return IterationData(energy=self.initial_energy)
