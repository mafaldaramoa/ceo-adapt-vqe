# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:19:38 2022

@author: mafal
"""

from copy import deepcopy

import numpy as np
from qiskit import QuantumCircuit

from ..circuits import cnot_depth, cnot_count
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
    def sizes(self):
        return [len(it_data.ansatz.indices) for it_data in self.its_data]

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
        self, initial_energy, pool, ref_det, sparse_ref_state, file_name, fci_energy, n, hamiltonian
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
