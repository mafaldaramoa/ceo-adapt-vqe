# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:00:03 2022

@author: mafal
"""

from copy import copy, deepcopy
import abc
import numpy as np
import scipy

from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import expm, expm_multiply

from openfermion import get_sparse_operator, count_qubits
from openfermion.transforms import get_fermion_operator, freeze_orbitals

from .adapt_data import AdaptData
from ..chemistry import chemical_accuracy, get_hf_det, create_spin_adapted_one_body_op
from ..matrix_tools import ket_to_vector
from ..minimize import minimize_bfgs
from ..pools import ImplementationType
from ..utils import bfgs_update


class AdaptVQE(metaclass=abc.ABCMeta):
    """
    Class for running the ADAPT-VQE algorithm.
    """

    def __init__(
        self,
        pool,
        molecule=None,
        custom_hamiltonian=None,
        verbose=False,
        max_adapt_iter=50,
        max_opt_iter=10000,
        full_opt=True,
        threshold=0.1,
        convergence_criterion="total_g_norm",
        tetris=False,
        progressive_opt=False,
        candidates=1,
        orb_opt=False,
        sel_criterion="gradient",
        recycle_hessian=False,
        penalize_cnots=False,
        rand_degenerate=False,
        frozen_orbitals=[],
        shots=None,
        track_prep_g=False,
        previous_data=None
    ):
        """
        Arguments:
            pool (OperatorPool): The operator pool to use in the algorithm
            molecule (MolecularData): The molecule for which to run calculations
            custom_hamiltonian (Union[XXZHamiltonian,HubbardHamiltonian]): Alternative to molecule, for condensed matter
            verbose (bool): Whether to display detailed info (e.g. gradients)
            max_adapt_iter (int): Maximum number of iterations of the ADAPT-VQE algorithm
            max_opt_iter (int): Maximum number of iterations of the optimization algorithm in each ADAPT-VQE iteration
            full_opt (bool): Whether to perform the full optimization or to only optimize the last parameter
            threshold (float): The convergence threshold
            convergence_criterion (str): The convergence criterion. Can be "total_g_norm" (total gradient norm) or
                "max_g" (maximum absolute gradient). When this falls below the threshold, the algorithm terminates
            tetris (bool): Whether to do TETRIS-ADAPT (see https://doi.org/10.1103/PhysRevResearch.6.013254)
            progressive_opt (bool): If doing TETRIS, whether to optimize all new parameters in a given iteration at once
                or in a phased manner
            candidates (int): The number of candidates to consider per iteration. From these, one will be selected based
                on the selection criterion
            orb_opt (bool): Whether to perform orbital optimization (see https://doi.org/10.48550/arXiv.2212.11405)
            sel_criterion (str): The selection criterion. If it is gradient-based, selection will be done among all
                operators. In this case, candidates must be set to 1. If it is energy-based, selection will be done only
                among a number of largest gradient operators, where the number is defined by candidates.
                Valid choices:
                    "gradient": largest absolute value of the gradient
                    "summed_gradient": Largest sum of absolute values of the gradients (MVP-CEO pool only)
                    "1d_energy": Largest energy decrease from optimizing just the last parameter
                    "energy": Largest energy decrease from full optimization
                    "1d_quad_fit": Largest energy decrease predicted from quadratic fit
                    "line_search": Largest energy decrease from a single line search
                    "sample": Sample from a probability distribution defined by the absolute values of the gradients
                        (higher gradient -> more likely)
                    "random": Choose randomly among the candidates
            recycle_hessian (bool): Whether to recycle the Hessian (see https://doi.org/10.48550/arXiv.2401.05172)
            penalize_cnots (bool): Whether to penalize the CNOTs when selecting operator. If True, the selection
                criterion will be divided by the number of CNOTs in the circuit implementation of the operator
            rand_degenerate (bool): Whether to select randomly among operators with identical gradients (difference
                below 10**-8). If False, the largest gradient is decided by the ">" operator.
            frozen_orbitals (list): Indices of orbitals that are considered to be permanently occupied. Note that
                virtual orbitals are not yet implemented.
            previous_data (AdaptData): data from a previous run of ADAPT we wish to continue
        """

        self.pool = pool
        self.molecule = copy(molecule)
        self.custom_hamiltonian = custom_hamiltonian
        self.verbose = verbose
        self.max_adapt_iter = max_adapt_iter
        self.max_opt_iter = max_opt_iter
        self.full_opt = full_opt
        self.threshold = threshold
        self.convergence_criterion = convergence_criterion
        self.tetris = tetris
        self.progressive_opt = progressive_opt
        self.candidates = candidates
        self.orb_opt = orb_opt
        self.sel_criterion = sel_criterion
        self.recycle_hessian = recycle_hessian
        self.penalize_cnots = penalize_cnots
        self.rand_degenerate = rand_degenerate
        self.frozen_orbitals = frozen_orbitals
        self.track_prep_g = track_prep_g

        # Attributes describing type of CEO pool, when applicable. The algorithm runs differently for each of them
        self.dvg = "DVG" in self.pool.name
        self.dve = "DVE" in self.pool.name
        self.mvp = "MVP" in self.pool.name

        self.data = previous_data  # AdaptData object
        self.initialize_hamiltonian()  # Initialize and store Hamiltonian and initial energy
        self.create_orb_rotation_ops()  # Create list of orbital rotation operators
        self.gradients = np.array(())
        self.orb_opt_dim = len(self.orb_ops)
        if previous_data is None:
            self.detail_file_name()  # Create detailed file name including all options
        self.energy_meas = self.observable_to_measurement(
            self.hamiltonian
        )  # Transform Hamiltonian into measurement
        self.set_window()  # Set the number of top gradient operators we will need to consider in each iteration
        self.shots = shots

        if self.recycle_hessian:
            self.inv_hessian = np.eye(
                self.orb_opt_dim
            )  # Initialize inverse Hessian at identity
        else:
            self.inv_hessian = None

        self.verify_inputs()

        if self.verbose:
            print(f"\n{self.name} prepared with the following settings:")
            self.print_settings()

    def verify_inputs(self):

        if self.convergence_criterion not in ["total_g_norm", "max_g"]:
            raise ValueError(
                "Convergence criterion {self.convergence_criterion} not supported."
                'Supported criteria: "total_g_norm", "max_g"'
            )

        if self.candidates < 1:
            raise ValueError(
                f"{self.candidates} is not an accepted number of candidates."
            )

        if "gradient" in self.sel_criterion:
            if self.candidates != 1:
                raise ValueError("Gradient selection method only accepts 1 candidate.")
        else:
            if self.candidates == 1:
                raise ValueError(
                    f"Selection criterion {self.sel_criterion} must have more than one candidate per"
                    "iteration."
                )

        if (self.dvg or self.dve) and self.pool.parent_pool is None:
            raise ValueError(
                f"{self.pool.name} pool must keep track of QE constituents."
            )

        if self.dvg and self.dve:
            raise ValueError(
                "DVG and DVE-CEO pools cannot be implemented simultaneously."
            )

        if self.mvp and not self.pool.couple_exchanges:
            raise ValueError(
                "MVP-CEO pool must keep track of QEs with the same support."
            )

        if self.progressive_opt and not self.tetris:
            raise ValueError("Progressive optimization is only defined by TETRIS mode.")

        if not self.sel_criterion in (
            [
                "gradient",
                "summed_gradient",
                "1d_energy",
                "energy",
                "1d_quad_fit",
                "line_search",
                "sample",
                "random",
            ]
        ):
            raise ValueError(
                f"Selection criterion {self.sel_criterion} not supported.\nSupported criteria: "
                '"gradient", "summed_gradient", "1d_energy", "energy", "1d_quad_fit"',
                '"line_search", "sample", "random"',
            )

        if self.sel_criterion == "summed_gradient" and self.pool.name != "mvp_ceo":
            raise ValueError(
                "summed_gradient selection criterion only defined for MVP-CEO pool."
            )

        if self.frozen_orbitals != self.pool.frozen_orbitals:
            raise ValueError("Frozen orbitals must match the pool's.")

        if (self.molecule is not None) + (self.custom_hamiltonian is not None) + (self.data is not None) != 1:
            raise ValueError(
                "Exactly one out of molecule / custom Hamiltonian / previous AdaptData must be provided."
            )

        if not self.recycle_hessian and self.sel_criterion == "line_search":
            raise ValueError(
                "line_search selection criterion is only defined with Hessian recycling."
            )

        if not self.full_opt and (self.mvp or self.dvg or self.dve):
            raise NotImplementedError(
                "1D optimization is not implemented for CEO pools other than OVP-CEO."
            )

    def initialize_hamiltonian(self):
        """
        Initialize attributes associated with the Hamiltonian.
        """
        if self.data is not None:
            hamiltonian = self.initialize_with_previous_data()
        elif self.molecule is not None:
            hamiltonian = self.initialize_with_molecule()
        else:
            hamiltonian = self.initialize_with_hamiltonian()

        self.save_hamiltonian(hamiltonian)

    def initialize_with_molecule(self):
        """
        Initialize attributes associated with a molecular Hamiltonian.
        """

        self.n = self.molecule.n_qubits - len(self.frozen_orbitals)
        self.molecule.n_electrons -= len(self.frozen_orbitals)

        # Set the Hartree Fock state as reference
        self.ref_det = get_hf_det(self.molecule.n_electrons, self.n)

        self.sparse_ref_state = csc_matrix(
            ket_to_vector(self.ref_det), dtype=complex
        ).transpose()

        hamiltonian = self.molecule.get_molecular_hamiltonian()

        if self.frozen_orbitals:
            hamiltonian = get_fermion_operator(hamiltonian)
            hamiltonian = freeze_orbitals(hamiltonian, self.frozen_orbitals)

        self.file_name = (
            f"{self.molecule.description}_r={self.molecule.geometry[1][1][2]}"
        )
        self.exact_energy = self.molecule.fci_energy

        return hamiltonian

    def initialize_with_previous_data(self):
        """
        Initialize attributes using previous AdaptData object.
        """

        self.n = self.data.n

        self.ref_det = self.data.ref_det
        self.sparse_ref_state = self.data.sparse_ref_state

        hamiltonian = self.data.hamiltonian


        pre_it, post_it = self.data.file_name.split(str(self.data.iteration_counter) + "i")
        self.file_name = "".join(
            [pre_it, str(self.max_adapt_iter) + "i", post_it]
        )

        self.exact_energy = self.data.fci_energy
        self.load(previous_data=self.data)

        return hamiltonian

    def initialize_with_hamiltonian(self):
        """
        Initialize attributes associated with a custom Hamiltonian.
        """

        self.n = count_qubits(self.custom_hamiltonian.operator)
        self.file_name = f"{self.custom_hamiltonian.description}_{self.n}"
        self.sparse_ref_state = self.custom_hamiltonian.ref_state
        self.ref_det = self.custom_hamiltonian.ref_state
        hamiltonian = self.custom_hamiltonian.operator
        self.exact_energy = self.custom_hamiltonian.ground_energy

        return hamiltonian

    def detail_file_name(self):
        """
        Add details concerning the ADAPT-VQE implementation to the file name.
        """

        mode = "_tetris" if self.tetris else ""
        mode += "(prog)" if self.progressive_opt else ""

        self.file_name += (
            f"{mode}_{self.pool.name}_{self.max_adapt_iter}i_{self.sel_criterion}"
        )

        if self.orb_opt:
            self.file_name += "_oopt"
        if not self.full_opt:
            self.file_name += "_1D"
        if self.candidates > 1:
            self.file_name += str(self.candidates)
        if self.recycle_hessian:
            self.file_name += "_rec_hess"
        if self.penalize_cnots:
            self.file_name += "_pen_cnots"
        if self.convergence_criterion != "total_g_norm":
            self.file_name += f"_{self.convergence_criterion}_cc"

    def load(self, previous_data=None, eig_decomp=None):
        """
        Load data from previous results or an eigendecomposition of the pool.

        Arguments:
          previous_data (AdaptData): data from a previous run, that will be continued
          eig_decomp (list): list of eigendecompositions of pool operators
        """

        if previous_data is not None:

            self.data = deepcopy(previous_data)
            self.data.file_name = self.file_name

            # Make sure we're continuing the run of ADAPT with the same settings
            assert self.pool.name == previous_data.pool_name
            assert bool("rec_hess" in previous_data.file_name) == self.recycle_hessian
            assert bool("tetris" in previous_data.file_name) == self.tetris
            assert bool("prog" in previous_data.file_name) == self.progressive_opt
            assert bool("oopt" in previous_data.file_name) == self.orb_opt
            assert bool("1D" in previous_data.file_name) == (not self.full_opt)
            assert bool("pen_cnots" in  previous_data.file_name) == self.penalize_cnots
            if self.candidates > 1:
                assert str(self.candidates) in previous_data.file_name
            if self.convergence_criterion != "total_g_norm":
                assert str(self.convergence_criterion) in previous_data.file_name

            # Set current state to the last iteration of the loaded data
            self.indices = self.data.evolution.indices[-1]
            self.coefficients = self.data.evolution.coefficients[-1]
            self.gradients = self.data.evolution.gradients[-1]
            self.inv_hessian = self.data.evolution.inv_hessians[-1]

            # Calculate current state
            self.state = self.compute_state()

            print("Loaded indices: ", self.indices)
            print("Loaded coefficients: ", self.coefficients)

        if eig_decomp is not None:

            if len(eig_decomp) != self.pool.size:
                raise ValueError("List of eigendecompositions does not match pool size")
            self.pool.eig_decomp = eig_decomp

            if self.verbose:
                print(
                    f"\nLoaded diagonalized pool containing {len(eig_decomp)} operators."
                )

    def print_settings(self):
        """
        Prints the options that were chosen for the Adapt VQE run.
        """
        print(f"> Pool: {self.pool.name}")
        if self.custom_hamiltonian is not None:
            print(f"> Custom Hamiltonian: {self.custom_hamiltonian.description}")
        else:
            molecule, r = self.file_name.split("_")[:2]
            print(
                f"> Molecule: {molecule}"
                f" (interatomic distance {r}Å)"
            )
        print(f"> Orbital Optimization: {self.orb_opt}")
        print(f"> Selection method: {self.sel_criterion}")
        print(f"> Convergence criterion: {self.convergence_criterion}")
        print(f"> Recycling Hessian: {self.recycle_hessian}")
        print(
            f"> Tetris: {self.tetris}"
            f" (progressive optimization: {self.progressive_opt})"
        )
        print("> Convergence threshold (gradient norm): ", self.threshold)
        print("> Maximum number of iterations: ", self.max_adapt_iter)
        print("> candidates per iteration: ", self.candidates)

    def get_state(self, coefficients=None, indices=None, ref_state=None):
        """
        Return the state obtained by acting on ref_state with the ansatz defined by the provided coefficients and
        indices. If ref_state is nor provided, self.ref_state will be used.
        If coefficients and indices are None, it will be assumed that the state is the current one, self.state. Storing
        this as an attribute allows for the calculation of the state to be reused, which happens for the gradient
        measurements. For those measurements, the state is fixed; we're just measuring different observables in it.
        In contrast, the energy is measured in different states along the optimization, so we don't want the state to be
        the current one - self.current_state is the state from the previous iteration, as we haven't chosen the
        parameters of this one yet.

        Arguments:
            coefficients (list): ansatz coefficients
            indices (list): ansatz indices
            ref_state (csc_matrix): reference state for the ansatz
        """

        if coefficients is None or indices is None:
            # No ansatz provided, return current state
            if ref_state is not None:
                raise ValueError("Resulting state is just input reference state.")
            if coefficients is not None or indices is not None:
                raise ValueError("Cannot provide only coefficients or only indices.")
            state = self.state
        else:
            # Calculate state from scratch
            state = self.compute_state(coefficients, indices, ref_state)

        return state

    def rank_gradients(self, coefficients=None, indices=None, silent=False):
        """
        Selects the operators that currently have the largest gradients. The number of selected operators depends on
        mode - TETRIS or not - and instance variable self.candidates.
        If coefficients and indices are None, the gradients will be calculated in the current state. This is what
        happens in the gradient screening of ADAPT-VQE.

        Arguments:
            coefficients (list): ansatz coefficients
            indices (list): ansatz indices
            silent (bool): whether to print information along the execution

        Returns:
          selected_index (int): the index that labels the selected operator
          selected_gradient (float): the norm of the gradient of that operator
          total_norm (float): the total gradient norm
          max_norm (float): the maximum gradient magnitude
        """

        sel_gradients = []
        sel_indices = []
        total_norm = 0

        if not silent:
            print(
                f"Creating list of up to {self.window} operators ordered by gradient magnitude..."
            )

        if self.verbose and not silent:
            print("\nNon-Zero Gradients (tolerance E-8):")

        for index in range(self.pool.size):

            gradient = self.eval_candidate_gradient(index, coefficients, indices)
            gradient = self.penalize_gradient(gradient, index, silent)

            if np.abs(gradient) < 10**-8:
                continue

            # Find the index of the operator in the ordered gradient list
            sel_gradients, sel_indices = self.place_gradient(
                gradient, index, sel_gradients, sel_indices
            )

            if index not in self.pool.parent_range:
                total_norm += gradient**2

        total_norm = np.sqrt(total_norm)

        if sel_gradients:
            max_norm = sel_gradients[0]
        else:
            # All operators have negligeable gradients
            max_norm = 0

        print("Total gradient norm: {}".format(total_norm))

        return sel_indices, sel_gradients, total_norm, max_norm

    def place_gradient(self, gradient, index, sel_gradients, sel_indices):
        """
        Find the position the operator associated with gradient belongs to, when being inserted in an ordered list
        containing the highest gradient operators selected so far.

        Arguments:
            gradient (float): the gradient to be placed in the list
            index (int): the index of the operator associated with this gradient
            sel_gradients (list): the previous list of ordered gradients
            sel_indices (list): the indices of the operators associated with the gradients in sel_gradients
        """

        i = 0

        for sel_gradient in sel_gradients:

            if np.abs(np.abs(gradient) - np.abs(sel_gradient)) < 10**-8:
                # Very similar gradients
                condition = self.break_gradient_tie(gradient, sel_gradient)

                if condition:
                    break

            elif np.abs(gradient) - np.abs(sel_gradient) >= 10**-8:
                # Next op has significantly lower gradient - this is the final position
                break

            # Next op has higher gradient, continue to a later place in list
            i = i + 1

        # i now contains the position of the operator in the list
        if i < self.window:
            # The new operator has a place in the list; its gradient is higher than the gradient of at least the lowest
            # gradient operator in the list. Insert it in the proper position

            sel_indices = sel_indices[:i] + [index] + sel_indices[i : self.window - 1]

            sel_gradients = (
                sel_gradients[:i] + [gradient] + sel_gradients[i : self.window - 1]
            )

        return sel_gradients, sel_indices

    def set_window(self):
        """
        Saves the number of highest gradient operators that should be screened in each iteration to an attribute. This
        will be the number of operators in the list output by rank_gradients.
        On TETRIS mode, we might use all operators with nonzero gradient, hence the list should contain the whole pool.
        The same is true for the MVP-, DVG- and DVE-CEO pools, because we might add to the ansatz QEs acting on the same
        spin-orbitals as the highest gradient operator, as long as they have nonzero gradient.
        Otherwise, this is just the number of candidates as provided by the user.
        """

        if self.tetris or self.mvp or self.dvg or self.dve:
            window = self.pool.size
        else:
            # We will use at most self.candidates operators
            window = self.candidates

        self.window = window

    def break_gradient_tie(self, gradient, sel_gradient):
        """
        Decide whether to select the operator associated with gradient or sel_gradient when the
        values are identical (difference under 10**-8).
        If attribute self.rand_degenerate is True, the choice is random. Otherwise, we always keep
        the largest one as decided by the ">" operator.

        Arguments:
            gradient (float): gradient associated with one of the operators
            sel_gradient (float): gradient associated with the other operator

        Returns:
            condition (bool): a condition which, if True, indicates that the operator associated with gradient will be
                favored over the one associated with sel_gradient
        """

        assert np.abs(np.abs(gradient) - np.abs(sel_gradient)) < 10**-8

        if self.rand_degenerate:
            # Position before/after with 50% probability
            condition = np.random.rand() < 0.5
        else:
            # Just place the highest first even if the difference is small
            condition = np.abs(gradient) > np.abs(sel_gradient)

        return condition

    def penalize_gradient(self, gradient, index, silent=False):

        if self.penalize_cnots:
            penalty = self.pool.get_cnots(index)
        else:
            penalty = 1

        gradient = gradient / penalty

        if np.abs(gradient) > 10**-8:

            if self.verbose and not silent:
                print(
                    "".join(
                        [
                            f"Operator {index}: {gradient}",
                            f" (penalty: {penalty})" if penalty != 1 else "",
                        ]
                    )
                )

        return gradient

    def eval_candidate_gradient(self, index, coefficients=None, indices=None):
        """
        Calculates the norm of the gradient of a candidate operator if appended to the state defined by the provided
        coefficients and indices. If those are not provided, the current state is used; this is all that is necessary in
        a normal run of ADAPT-VQE.
        This method uses the formula dexp(c*A)/dc = <psi|[H,A]|psi> = 2 * real(<psi|HA|psi>).
        This is the gradient calculated at c = 0, which will be the initial value of the coefficient in the
        optimization. Only the absolute value is returned.

        Arguments:
          index (int): the index that labels this operator
          coefficients (list): coefficients of the ansatz
          indices (list): indices of the ansatz

        Returns:
          gradient (float): the norm of the gradient of the operator labeled by index, in the specified state.
        """

        measurement = self.pool.get_grad_meas(index)

        if measurement is None:
            # Gradient observable for this operator has not been created yet

            operator = self.pool.get_imp_op(index)
            observable = self.hamiltonian @ operator - operator @ self.hamiltonian

            # Process so that it's ready to evaluate
            measurement = self.observable_to_measurement(observable)

            # Store to avoid recalculating next time
            self.pool.store_grad_meas(index, measurement)

        gradient = self.evaluate_observable(measurement, coefficients, indices)

        return gradient

    def eval_candidate_gradient_prepending(
        self,
        index,
        method="an",
        dx=10**-8,
        orb_params=None):
        """
        Estimates the gradient of unitary generated by pool operator if they are prepended to the ansatz (added
        right after the reference state, beginning of the circuit) at point zero.

        Args:
            index (int): index of pool operator
            method (str): the method for estimating the gradient
            dx (float): the step size used for the finite difference approximation
            orb_params (list): the parameters for the orbital optimization, if applicable

        Returns:
            gradient (float): the gradient
        """

        if self.data is not None:
            coefficients = self.coefficients.copy()
            indices = self.indices
        else:
            coefficients = []
            indices = []

        return self.estimate_gradient(
            0,
            coefficients = [0] + coefficients,
            indices = np.concatenate(([index],indices)).astype('int'),
            method=method,
            dx=dx,
            orb_params=orb_params,
        )

    def eval_candidate_gradients_prepending(
        self,
        method="an",
        dx=10 ** -8,
        orb_params=None,
        ):
        """
        Estimates the gradient of unitaries generated by each pool operator if they are prepended to the ansatz (added
        right after the reference state, beginning of the circuit) at point zero.

        Args:
            index (int): index of pool operator
            method (str): the method for estimating the gradient
            dx (float): the step size used for the finite difference approximation
            orb_params (list): the parameters for the orbital optimization, if applicable

        Returns:
            gradients (list): the list of gradients, in the same order as the pool operator list
            norm (float): the norm of the gradient
        """

        gradients = []
        norm = 0

        for index in range(self.pool.size):
            gradient = self.eval_candidate_gradient_prepending(index, method, dx,orb_params)
            gradients.append(gradient)
            norm += gradient**2

        return gradients, np.sqrt(norm)

    def estimate_gradient(
        self,
        operator_pos=None,
        coefficients=None,
        indices=None,
        method="fd",
        dx=10**-8,
        orb_params=None,
        ref_state=None,
    ):
        """
        Estimates the gradient of the operator in position operator_pos of the ansatz defined by coefficients and
        indices. Default is finite differences; child classes may define other methods.

        Args:
            operator_pos (int): the position of the operator whose gradient we want to estimate. It may be higher than
                the length of indices, in which case it returns the gradient of the orbital optimization parameter
                indexed by operator_pos - len(indices).
            coefficients (list): the coefficients of the ansatz. If not, current coefficients will be used.
            indices (list): the indices of the ansatz. If not, current indices will be used.
            method (str): the method for estimating the gradient
            dx (float): the step size used for the finite difference approximation
            orb_params (list): the parameters for the orbital optimization, if applicable
            ref_state (csc_matrix): the reference state the ansatz acts on

        Returns:
            gradient (float): the approximation to the gradient
        """

        if method != "fd":
            raise ValueError(f"Method {method} is not supported.")

        if indices is None:
            # Assume current indices and coefficients
            assert coefficients is None
            indices = self.indices.copy()
            coefficients = self.coefficients.copy()

        assert len(coefficients) == len(indices)

        coefficients_plus = copy(coefficients)
        orb_params_plus = copy(orb_params)

        if operator_pos < self.orb_opt_dim:
            # We want the gradient of an orbital optimization parameter. Coefficients stay the same
            orb_params_plus[operator_pos] += dx
        else:
            # We want the gradient of an ansatz parameter. Orbital parameters stay the same
            coefficients_plus[operator_pos - self.orb_opt_dim] += dx

        energy = self.evaluate_energy(
            coefficients, indices, orb_params=orb_params, ref_state=ref_state
        )
        energy_plus = self.evaluate_energy(
            coefficients_plus, indices, orb_params=orb_params_plus
        )
        gradient = (energy_plus - energy) / dx

        return gradient

    def estimate_snd_derivative_1var(
        self,
        operator_pos,
        coefficients=None,
        indices=None,
        method="fd",
        formula="central",
        dx=10**-4,
    ):
        """
        Estimates the second derivative of the energy with respect to the coefficient of the operator in position
        operator_pos of the ansatz defined by coefficients and indices. Default is finite differences; child classes may
        define other methods.
        If coefficients and indices are None, the current ones will be used.

        Args:
            operator_pos (int): the position of the operator whose gradient we want to estimate
            coefficients (list): the coefficients of the ansatz
            indices (list): the indices of the ansatz. If not, current indices will be used
            method (str): the method for estimating the gradient
            formula (str): the formula for the FD calculation. May be "central", "forward" or "backward"
            dx (float): the step size used for the finite difference approximation

        Returns:
            gradient (float): the approximation to the gradient
        """

        if self.orb_opt:
            raise NotImplementedError

        if method != "fd":
            raise ValueError(f"Method {method} is not supported.")

        if indices is None:
            assert coefficients is None
            indices = self.indices.copy()
            coefficients = self.coefficients.copy()

        assert len(coefficients) == len(indices)

        energy = self.evaluate_energy(coefficients, indices)

        if formula in ["central", "forward"]:
            # We will need the energy evaluated at a positive shift
            coefficients_plus = copy(coefficients)
            coefficients_plus[operator_pos] += dx
            energy_plus = self.evaluate_energy(coefficients_plus, indices)

        if formula in ["central", "backward"]:
            # We will need the energy evaluated at a negative shift
            coefficients_minus = copy(coefficients)
            coefficients_minus[operator_pos] -= dx
            energy_minus = self.evaluate_energy(coefficients_minus, indices)

        if formula == "central":
            snd_derivative = (energy_plus - 2 * energy + energy_minus) / dx**2

        elif formula == "forward":
            coefficients_plus2 = copy(coefficients)
            coefficients_plus2[operator_pos] += 2 * dx
            energy_plus2 = self.evaluate_energy(coefficients_plus2, indices)
            snd_derivative = (energy_plus2 - 2 * energy_plus + energy) / dx**2

        elif formula == "backward":
            coefficients_minus2 = copy(coefficients)
            coefficients_minus2[operator_pos] -= 2 * dx
            energy_minus2 = self.evaluate_energy(coefficients_minus2, indices)
            snd_derivative = (energy + energy_minus2 - 2 * energy_minus) / dx**2

        return snd_derivative

    def estimate_snd_derivative(
        self,
        op1_pos,
        op2_pos=None,
        coefficients=None,
        indices=None,
        method="fd",
        formula="central",
        dx=10**-4,
    ):
        """
        Estimates the second derivative of the energy with respect to the coefficient of the operator in position
        operator_pos of the ansatz defined by coefficients and indices. Default is finite differences; child classes may
        define other methods.
        If coefficients and indices are None, the current ones will be used.

        Args:
            op1_pos (int): the position of one of the operators with respect to which we wish to take the 2nd derivative
            op2_pos (int): the position of the other operator with respect to which we wish to take the 2nd derivative
            coefficients (list): the coefficients of the ansatz. If not, current coefficients will be used.
            indices (list): the indices of the ansatz. If not, current indices will be used.
            method (str): the method for estimating the gradient
            formula (str): the formula for the FD calculation. Currently must be "central"
            dx (float): the step size used for the finite difference approximation

        Returns:
            gradient (float): the approximation to the gradient
        """

        if formula != "central" or self.orb_opt or method != "fd":
            raise NotImplementedError

        if indices is None:
            assert coefficients is None
            indices = self.indices.copy()
            coefficients = self.coefficients.copy()

        assert len(coefficients) == len(indices)

        if op1_pos == op2_pos or op2_pos is None:
            # We want a second order derivative concerning only one operator - use specific function
            return self.estimate_snd_derivative_1var(
                op1_pos,
                coefficients=coefficients,
                indices=indices,
                method=method,
                formula=formula,
                dx=dx,
            )

        coefficients_pp = copy(coefficients)
        coefficients_pm = copy(coefficients)
        coefficients_mp = copy(coefficients)
        coefficients_mm = copy(coefficients)

        coefficients_pp[op1_pos] += dx
        coefficients_pp[op2_pos] += dx

        coefficients_pm[op1_pos] += dx
        coefficients_pm[op2_pos] -= dx

        coefficients_mp[op1_pos] -= dx
        coefficients_mp[op2_pos] += dx

        coefficients_mm[op1_pos] -= dx
        coefficients_mm[op2_pos] -= dx

        e_pp = self.evaluate_energy(coefficients_pp, indices)
        e_pm = self.evaluate_energy(coefficients_pm, indices)
        e_mp = self.evaluate_energy(coefficients_mp, indices)
        e_mm = self.evaluate_energy(coefficients_mm, indices)

        snd_derivative = (e_pp - e_pm - e_mp + e_mm) / (4 * dx**2)

        return snd_derivative

    def estimate_hessian(
        self, coefficients=None, indices=None, method="fd", formula="central", dx=10**-4
    ):
        """
        Estimates the Hessian of the energy with respect to operator parameters. Default is finite differences; child
        classes may define other methods.
        If coefficients and indices are None, the current ones will be used.

        Args:
            coefficients (list): the coefficients of the ansatz. If not, current coefficients will be used.
            indices (list): the indices of the ansatz. If not, current indices will be used.
            method (str): the method for estimating the gradient
            formula (str): the formula for the FD calculation. Currently must be "central"
            dx (float): the step size used for the finite difference approximation

        Returns:
            hessian (np.ndarray): the approximation to the Hessian
        """

        if formula != "central" or self.orb_opt or method != "fd":
            raise NotImplementedError

        if indices is None:
            assert coefficients is None
            indices = self.indices.copy()
            coefficients = self.coefficients.copy()

        assert len(coefficients) == len(indices)

        size = len(indices)
        hessian = np.zeros((size, size))

        for i in range(size):
            for j in range(i, size):
                snd_derivative = self.estimate_snd_derivative(
                    i, j, coefficients, indices, method, formula, dx
                )
                hessian[i, j] = snd_derivative
                hessian[j, i] = snd_derivative

        return hessian

    def estimate_partial_hessian(
        self,
        coefficients=None,
        indices=None,
        diagonal=True,
        lines=None,
        entries=None,
        method=None,
        formula="central",
        dx=10**-4,
    ):
        """
        Estimates some components of the Hessian of the energy with respect to operator parameters. Default is finite
        differences; child classes may define other methods.
        If coefficients and indices are None, the current ones will be used.

        Args:
            coefficients (list): the coefficients of the ansatz. If not, current coefficients will be used.
            indices (list): the indices of the ansatz. If not, current indices will be used.
            diagonal (bool): whether to calculate the full diagonal. If diagonal=False, the only calculated diagonal
                elements will be those in the specified lines.
            lines (list): the lines (and columns, since H[i,j] = H[j,i]) that should be calculated. Other lines will be
                left as zero, except for diagonal terms if diagonal=True. If None, all lines will be calculated
            entries (list): entries within these lines/columns and the diagonal that should be calculate. If None,
                all entries will be calculated
            method (str): the method for estimating the gradient
            formula (str): the formula for the FD calculation. Currently must be "central"
            dx (float): the step size used for the finite difference approximation

        Returns:
            hessian (np.ndarray): the approximation to the Hessian
        """

        if formula != "central" or self.orb_opt or method != "fd":
            raise NotImplementedError

        if indices is None:
            assert coefficients is None
            indices = self.indices.copy()
            coefficients = self.coefficients.copy()

        assert len(coefficients) == len(indices)

        size = len(indices)
        hessian = np.zeros((size, size))

        if lines is None:
            lines = range(size)
        if entries is None:
            entries = range(size)

        for i in lines:
            for j in entries:
                if j >= i or j not in lines:
                    snd_derivative = self.estimate_snd_derivative(
                        i, j, coefficients, indices, method, dx=dx
                    )
                    # Make sure we are not overwriting
                    assert (hessian[i, j]) == 0
                    assert (hessian[j, i]) == 0
                    hessian[i, j] = snd_derivative
                    hessian[j, i] = snd_derivative

        if diagonal:
            for i in entries:
                if i not in lines:
                    # Make sure we are not overwriting
                    assert hessian[i, i] == 0
                    hessian[i, i] = self.estimate_snd_derivative(
                        i, i, coefficients, indices, method=method
                    )

        return hessian

    def update_hessian_line(
        self, hessian, line, coefficients, indices, method=None, formula=None, dx=10**-4
    ):
        """
        Updates Hessian by recalculating a line/column. Default is finite differences; child classes may define other
        methods. If coefficients and indices are None, the current ones will be used.

        Args:
            hessian (np.ndarray): the previous Hessian
            coefficients (list): the coefficients of the ansatz. If None, current coefficients will be used.
            indices (list): the indices of the ansatz. If None, current indices will be used.
            line (int): the line to update
            method (str): the method for estimating the gradient
            formula (str): the formula for the FD calculation. Currently must be "central"
            dx (float): the step size used for the finite difference approximation

        Returns:
            hessian (np.ndarray): the updated Hessian
        """

        if formula != "central" or self.orb_opt or method != "fd":
            raise NotImplementedError

        hessian = hessian.copy()

        # Calculate the new line/column
        line_only_hess = self.estimate_partial_hessian(
            coefficients,
            indices,
            lines=[line],
            diagonal=False,
            method=method,
            formula=formula,
            dx=dx,
        )

        # Update matrix
        hessian[line, :] = line_only_hess[line, :]
        hessian[:, line] = line_only_hess[:, line]

        return hessian

    def estimate_gradients(
        self, coefficients=None, indices=None, method="fd", dx=10**-8, orb_params=None
    ):
        """
        Estimates the gradients of all operators in the ansatz defined by coefficients and indices. If they are None,
        the current state is assumed.
        If self.orb_opt is set, the gradients of the orbital optimization parameters are also included (at the start of
        the list). The output is a single vector because it is more convenient for the optimization.
        Default method is finite differences; child classes may define other methods.

        Args:
            coefficients (list): the coefficients of the ansatz. If None, current coefficients will be used
            indices (list): the indices of the ansatz. If None, current indices will be used
            method (str): the method for estimating the gradient
            dx (float): the step size used for the finite difference approximation
            orb_params (list): the parameters for the orbital optimization, if applicable

        Returns:
            gradient (float): the approximation to the gradient
        """

        if method != "fd":
            raise NotImplementedError

        if indices is None:
            assert coefficients is None
            indices = self.indices.copy()
            coefficients = self.coefficients.copy()

        if self.orb_opt:
            assert len(coefficients) == len(indices) + self.orb_opt_dim
            # Orbital parameters are at the start of the list by convention
            orb_params = coefficients[: self.orb_opt_dim]
            coefficients = coefficients[self.orb_opt_dim :]
        else:
            assert len(coefficients) == len(indices)
            orb_params = None

        gradients = []
        for operator_pos in range(self.orb_opt_dim + len(coefficients)):
            gradient = self.estimate_gradient(
                operator_pos=operator_pos,
                coefficients=coefficients,
                indices=indices,
                method="fd",
                dx=dx,
                orb_params=orb_params,
            )
            gradients.append(gradient)

        return gradients

    def evaluate_energy(
        self, coefficients=None, indices=None, ref_state=None, orb_params=None
    ):
        """
        Calculates the energy in a specified state using matrix algebra.
        If coefficients and indices are not specified, the current ones are used.

        Arguments:
          coefficients (list): coefficients of the ansatz
          indices (list): indices of the ansatz
          ref_state (csc_matrix): the reference state to which to append the ansatz
          orb_params (list): if self.orb_params, the parameters of the orbital rotation operators

        Returns:
          energy (float): the energy in this state.
        """

        energy = self.evaluate_observable(
            self.energy_meas, coefficients, indices, ref_state, orb_params
        )

        return energy

    def run(self):
        """
        Run the full ADAPT-VQE algorithm, until either the convergence condition is met or the maximum number of
        iterations is reached.
        """

        # Initialize data structures
        self.initialize()

        finished = False
        while not finished and self.data.iteration_counter < self.max_adapt_iter:
            # Run one iteration and check if we have converged. This might add one operator or more depending on
            # the algorithm configuration  (e.g. if self.tetris=True, we may add multiple operators)
            finished = self.run_iteration()

        if not finished:
            # The last time we measured the gradients was in the beginning of the last iteration. Remeasure in the end
            print("Performing final convergence check...")
            viable_candidates, viable_gradients, total_norm, max_norm = (
                self.rank_gradients(silent=True)
            )
            finished = self.probe_termination(total_norm, max_norm)

        if finished:
            print("\nConvergence condition achieved!\n")
            error = self.energy - self.exact_energy
            print(
                f"Final Energy: {self.energy}\nError: {error}\n"
                f"(in % of chemical accuracy: {(error / chemical_accuracy * 100):.3f}%)\n"
                f"Iterations completed: {self.data.iteration_counter}\n"
                f"Ansatz indices: {self.indices}\nCoefficiens: {self.coefficients}"
            )
        else:
            print(
                f"\nThe maximum number of iterations ({self.max_adapt_iter}) was hit before the convergence criterion "
                f"was satisfied.\n(current gradient norm is {self.data.current.gradient_norm} > {self.threshold})"
            )
            self.data.close(False)

        return

    def run_iteration(self):
        """
        Run one iteration of the ADAPT-VQE algorithm.
        """

        # Screen viable candidates and calculate current total gradient norm
        finished, viable_candidates, viable_gradients, total_norm, prep_norm = self.start_iteration()

        if finished:
            # We have converged; do not complete the iteration
            return finished

        while viable_candidates:
            # Select operators, track costs, and screen other viable candidates to add in this iteration
            # (e.g. if self.tetris=True, viable candidates are those whose supports are disjoint from those of any other
            # operators added in this iteration)
            energy, g, viable_candidates, viable_gradients = self.grow_and_update(
                viable_candidates, viable_gradients
            )

        if energy is None:
            # Energy is not yet optimized, because the selection criterion is not energy-based. Optimize it.
            energy = self.optimize(g)

        # Save and print iteration data and update state
        self.complete_iteration(energy, total_norm, prep_norm, self.iteration_sel_gradients)

        return finished

    def update_viable_candidates(self, viable_candidates, viable_gradients):
        """
        Update list of candidates that may still be added in this iteration.

        Arguments:
            viable_candidates (list): indices of the candidates that were viable prior to the last addition
            viable_gradients (list): corresponding gradients

        Returns:
            viable_candidates (list): indices of still viable candidates
            viable_gradients (list): corresponding gradients
            ngevs (list): total number of gradient evaluations used by this method
        """

        if self.tetris:
            print("\nScreening operators with disjoint supports...")
            new_qubits = self.pool.get_qubits(
                self.indices[-1]
            )  # Qubits acted on non-trivially by last operator added
            viable_candidates, viable_gradients = self.tetris_screening(
                new_qubits, viable_candidates, viable_gradients
            )
        else:
            # Finish iteration; we don't want to add more than 1 operator
            viable_candidates = []

        ngevs = 0
        if viable_candidates and self.sel_criterion in ["line_search", "energy"]:
            # Update gradient of these operators inside the current iteration
            # Our selection method is energy-based, so we've added a new non-zero coefficient, which makes the
            # action of the operator on the ansatz non-trivial at the starting point
            viable_gradients = [
                self.estimate_gradient(-1, self.coefficients + [0], self.indices + [c])
                for c in viable_candidates
            ]
            ngevs = len(viable_gradients)

        return viable_candidates, viable_gradients, ngevs

    def update_iteration_costs(self, new_nfevs=None, new_ngevs=None, new_nits=None):
        """
        Update the total number of function/gradient evaluations and optimizer iterations in a given ADAPT-VQE
        iteration.

        Arguments:
            new_nfevs (int): the number of function evaluations to add
            new_ngevs (int): the number of gradient evaluations to add
            new_nits (int): the number of optimizer iterations to add
        """

        if new_nfevs:
            self.iteration_nfevs = self.iteration_nfevs + new_nfevs
        if new_ngevs:
            self.iteration_ngevs = self.iteration_ngevs + new_ngevs
        if new_nits:
            self.iteration_nits = self.iteration_nits + new_nits

    def initialize(self):
        """
        Initialize data structures.
        """

        if not self.data:

            initial_energy = self.evaluate_energy()
            self.energy = initial_energy

            # We're starting a fresh ADAPT-VQE; create new AdaptData instance and initialize indices, coefficients
            self.data = AdaptData(
                initial_energy,
                self.pool,
                self.ref_det,
                self.sparse_ref_state,
                self.file_name,
                self.exact_energy,
                self.n,
                self.hamiltonian
            )

            self.indices = []
            self.coefficients = []
            self.old_coefficients = []
            self.old_gradients = []

        else:

            self.state = self.compute_state()
            initial_energy = self.evaluate_energy()
            self.energy = initial_energy

            # Make sure the current energy is consistent with the energy of the ADAPT-VQE run we are continuing.
            # Sometimes there's a discrepancy because the PySCF Hamiltonian is not deterministic for some molecules
            # See https://github.com/pyscf/pyscf/issues/1935
            assert self.energy - self.data.evolution.energies[-1] < 10**-12

        print(f"\nInitial energy: {initial_energy}")

        return

    def start_iteration(self):
        """
        Start a new ADAPT-VQE iteration by screening viable candidates, checking if we've converged, and initializing
        data structures.
        """

        print(f"\n*** ADAPT-VQE Iteration {self.data.iteration_counter + 1} ***\n")

        viable_candidates, viable_gradients, total_norm, max_norm = (
            self.rank_gradients()
        )

        if self.track_prep_g:
            prep_gradients, prep_norm = self.eval_candidate_gradients_prepending()
        else:
            prep_norm = None

        # Check if we've reached a termination condition
        finished = self.probe_termination(total_norm, max_norm)

        if finished:
            return finished, viable_candidates, viable_gradients, total_norm, prep_norm

        print(
            f"Operators under consideration ({len(viable_gradients)}):\n{viable_candidates}"
            f"\nCorresponding gradients (ordered by magnitude):\n{viable_gradients}"
        )

        self.iteration_nfevs = []  # Function evaluations by optimizer
        self.iteration_ngevs = []  # Gradient evaluations by optimizer
        self.iteration_nits = []  # Iterations by optimizer
        self.iteration_sel_gradients = []  # Operators selected so far in this iteration
        self.iteration_qubits = (
            set()
        )  # Qubits acted upon by the operators selected in the current iteration

        return finished, viable_candidates, viable_gradients, total_norm, prep_norm

    def grow_and_update(self, viable_candidates, viable_gradients):
        """
        Grow the current ansatz by adding an operator and update the list of viable candidates and corresponding
        gradients.

        Arguments:
            viable_candidates (list): list of indices of pool operators that are still viable for the current iteration,
                ordered by gradient magnitude
            viable_gradients (list): corresponding gradients
        """

        # Decide which operator to add to the ansatz, keeping track of costs. The optimization is NOT performed
        # here unless the selection criterion is energy-based
        energy, gradient = self.grow_ansatz(viable_candidates, viable_gradients)

        # Update the ordered list of ansatz elements that we may still add in this iteration
        viable_candidates, viable_gradients, extra_ngevs = (
            self.update_viable_candidates(viable_candidates, viable_gradients)
        )
        if extra_ngevs:
            # Condition is necessary because if not extra_ngevs, ngevs[-1] may not be defined. This can happen
            # in the first iteration, if the selection criterion is not energy-based
            self.iteration_ngevs[-1] += extra_ngevs

        self.iteration_sel_gradients = np.append(self.iteration_sel_gradients, gradient)

        return energy, gradient, viable_candidates, viable_gradients

    def complete_iteration(self, energy, total_norm, prep_norm, sel_gradients):
        """
        Complete iteration by storing the relevant data and updating the state.

        Arguments:
            energy (float): final energy for this iteration
            total_norm (float): final gradient norm for this iteration
            prep_norm (float): same but prepending instead of appending
            sel_gradients(list): gradients selected in this iteration
        """

        energy_change = energy - self.energy
        self.energy = energy

        # Save iteration data
        self.data.process_iteration(
            self.indices,
            self.energy,
            total_norm,
            prep_norm,
            sel_gradients,
            self.coefficients,
            self.inv_hessian,
            self.gradients,
            self.iteration_nfevs,
            self.iteration_ngevs,
            self.iteration_nits,
        )

        # Update current state
        self.state = self.compute_state()

        print("\nCurrent energy:", self.energy)
        print(f"(change of {energy_change})")
        print(f"Current ansatz: {list(self.indices)}")

        return

    def probe_termination(self, total_norm, max_norm):
        """
        Check if the termination condition has been satisfied.

        Arguments:
            total_norm (float): current gradient norm
            max_norm (float): current maximum gradient magnitude

        Returns:
            finished (bool): True if we have reached convergence, False otherwise
        """

        finished = False

        if total_norm < self.threshold and self.convergence_criterion == "total_g_norm":
            self.converged()
            finished = True

        if max_norm < self.threshold and self.convergence_criterion == "max_g":
            self.converged()
            finished = True

        return finished

    def converged(self):
        """
        To call when convergence is reached. Updates file name to include the total number of iterations executed.
        """

        # Update iteration number on file_name - we didn't reach the maximum
        pre_it, post_it = self.file_name.split(str(self.max_adapt_iter) + "i")
        self.file_name = "".join(
            [pre_it, str(self.data.iteration_counter) + "i", post_it]
        )
        self.data.close(True, self.file_name)

        return

    def inter_it_bfgs_update(self):
        """
        Perform a BFGS-like update on the Hessian matrix between the last and new iteration's points
        """

        # Update with $x_{k-1}=x_{it-1}^{(2)}$
        # (coefficients of previous iteration, before previous iteration's Hessian update;
        # new operator gets zero)
        gkp1 = self.estimate_gradients()
        xkp1 = self.coefficients
        xk = self.old_coefficients + [0]
        gk = self.old_gradients + self.estimate_gradient(
            operator_pos=-1, coefficients=xk, indices=self.indices
        )

        # Update with $x_{k-1}=x_{it}^{(1)}$ (coefficients of current iteration, after previous
        # iteration's Hessian update but before appending 1D minimizer)
        # xk = self.coefficients.copy()
        # xk[-1] = 0
        # gk = self.estimate_gradients(xk,self.indices)
        self.bfgs_update(gkp1, gk, xkp1, xk)

        # Extra costs: len(gkp1)-1 for gkp1 (last one is from 1D opt); 1 for gk
        extra_ngevs = [len(gkp1) + 1]
        self.old_coefficients = xkp1
        self.old_gradients = gkp1
        self.gradients = list(gkp1)

        return extra_ngevs

    def perform_quad_step(self):
        """
        Perform one Newton step. The change in coordinates corresponds to the vector pointing at the minimum of the
        current quadratic model of the cost function, as defined by the Hessian matrix, the gradient vector and the
        parameters.

        Returns:
            energy (float): the energy after the step
            extra_nfevs (int): the cost in terms of the number of function evaluations
        """

        # Obtain change in coordinates as the gradient, pre-conditioned by the inverse Hessian
        dx = -self.inv_hessian.dot(self.gradients)

        # Update coefficients
        self.coefficients = list(np.add(self.coefficients, dx))

        # Update energy
        energy = self.evaluate_energy(self.coefficients, self.indices)

        # Cost: a single energy evaluation
        extra_nfevs = 1

        return energy, extra_nfevs

    def perform_line_search(self):
        """
        Perform a line search along the search direction dictated by current Hessian and gradients.

        Returns:
            energy (float): the energy after the line search
            extra_nfevs (int): the cost in terms of the number of function evaluations
        """

        # Obtain search direction as the gradient, pre-conditioned by the inverse Hessian
        pk = -self.inv_hessian.dot(self.gradients)
        # print(f"Norm of pk: {np.linalg.norm(pk)}")
        alpha = 1  # Start with unit step
        energy = self.energy
        extra_nfevs = 0

        while energy - self.energy > -(10**-12):
            # Calculate possible new coefficient vector and measure corresponding energy
            test_x = np.add(self.coefficients, alpha * pk)
            energy = self.evaluate_energy(test_x, self.indices)
            extra_nfevs += 1
            alpha = alpha / 2  # Decrease step
            if alpha < 1 / 2**10:
                # Give up
                self.data.close(False)
                return

        self.coefficients = list(test_x)

        return energy, extra_nfevs

    def partial_optim(self, gradient):
        """
        Perform a partial optimization. Only the last parameter is optimized. If self.recycle_hessian is True, an
        additional BFGS step is performed.

        Arguments:
            gradient (float): the gradient of the last added operator
        Returns:
            energy (float):
            nfev (int): total number of performed function evaluations
            g1ev (int): total number of performed gradient evaluations
            nit (int): total number of performed optimizer iterations
        """

        # Peform optimization on the last parameter
        energy, coef, new_gradient, nfev, g1ev, nit = self.minimize_1d(
            e0=self.energy, g0=[gradient]
        )
        # Replace coefficient and gradient of last parameter with new ones
        self.coefficients[-1] = coef
        self.gradients[-1] = new_gradient
        print(f"New coefficient list: {self.coefficients}")

        if self.recycle_hessian:
            # Update coefficients, gradients and Hessian via BFGS update
            extra_ngevs = self.inter_it_bfgs_update()

            # Opt 1: step is dictated by H*g, no line search
            # energy, extra_nfevs = self.perform_quad_step()

            # Opt 2: step is dictated by backtracking line search along H*g; stopping condition is that
            # the energy is decreased
            energy, extra_nfevs = self.perform_line_search()
            nfev = nfev + extra_nfevs
            g1ev = g1ev + extra_ngevs

        return energy, nfev, g1ev, nit

    def get_quad_fit_minimizers(self, indices, gradients):
        """
        Estimate the minimizers and minima of the cost function for the input pool operators, if added to the ansatz,
        assuming a one-dimensional quadratic model.

        Arguments:
            indices (list): the indices of the pool operators to consider
            gradients (list): corresponding gradients

        Returns:
            minimizers_1d (list): the minimizers according to the quadratic model, in the same order as provided
            minima_1d (list):  the corresponding minima
        """

        minimizers_1d = []
        energies_1d = []

        for index, gradient in zip(indices, gradients):
            print(f"\n*Operator {index}*")
            position = len(
                self.indices
            )  # Assume operators would go to the end of the ansatz
            snd_der = self.estimate_snd_derivative(
                position,
                coefficients=self.coefficients + [0],
                indices=self.indices + [index],
                method="an",
            )

            # Calculate minimizer and minimum of the energy as a function of the candidate parameter
            minimizer = -gradient / snd_der
            energy_1d = self.energy - gradient**2 / (2 * snd_der)

            minimizers_1d.append(minimizer)
            energies_1d.append(energy_1d)

            print(
                "\nPrediction"
                f"\nMinimum: {energy_1d}"
                f"\nMinimizer: {minimizer}"
                f"\nActual energy at minimizer:"
                f"{self.evaluate_energy(self.coefficients + [minimizer], self.indices + [index])}"
            )

        return minimizers_1d, energies_1d

    def expand_inv_hessian(self, added_dim=None):
        """
        Expand the current approximation to the inverse Hessian by adding ones in the diagonal, zero elsewhere.

        Arguments:
            added_dim (int): the number of added dimensions (equal to the number of added lines/columns). If None,
                it is assumed that the Hessian is expanded so that its dimension is consistent with the current ansatz.

        Returns:
            inv_hessian (np.ndarray): the expanded inverse Hessian
        """

        if not self.recycle_hessian:
            return None

        size, size = self.inv_hessian.shape

        if added_dim is None:
            # Expand Hessian to have as many columns as the number of ansatz + orbital optimization parameters
            added_dim = len(self.indices) + self.orb_opt_dim - size

        size += added_dim

        # Expand inverse Hessian with zeros
        inv_hessian = np.zeros((size, size))
        inv_hessian[:-added_dim, :-added_dim] += self.inv_hessian

        # Add ones in the diagonal
        while added_dim > 0:
            inv_hessian[-added_dim, -added_dim] = 1
            added_dim -= 1

        return inv_hessian

    def grow_ansatz(self, viable_candidates, viable_gradients, max_additions=1):
        """
        Grow the ansatz according to the selection criterion and track costs.

        Arguments:
            viable_candidates (list): indices of the candidates that were viable prior to the last addition
            viable_gradients (list): corresponding gradients
            max_additions (int): maximum number of operators to add in a given iteration

        Returns:
            energy (float): energy after the new additions
            gradient (float): gradient of the newly selected operator
        """

        if not self.sel_criterion == "summed_gradient" and not (
            self.dvg or self.dve or self.mvp
        ):
            # We only need to consider self.candidate operators. This is not true for CEOs with multiple variational
            # parameters, because we might add a CEO including qubit excitations with lower gradients
            viable_candidates = viable_candidates[: self.candidates]
            viable_gradients = viable_gradients[: self.candidates]
        if self.candidates > 1:
            print(f"\nCandidates for next ansatz addition: {viable_candidates}")

        total_new_nfevs = []
        total_new_ngevs = []
        total_new_nits = []
        gradients = []
        while max_additions > 0:
            # Select operators according to the selection criterion, and track costs
            energy, gradient, new_nfevs, new_ngevs, new_nits = self.select_operators(
                viable_candidates, viable_gradients
            )

            # Obtain complete list of indices of pool operators added to the ansatz
            if self.data.evolution.indices:
                old_size = len(self.data.evolution.indices[-1])
            else:
                old_size = 0
            new_indices = self.indices[old_size:]

            # Update costs
            if new_nfevs:
                total_new_nfevs.append(new_nfevs)
            if new_ngevs:
                total_new_ngevs.append(new_ngevs)
            if new_nits:
                total_new_nits.append(new_nits)

            gradients.append(gradient)
            max_additions -= 1

        print(f"Operator(s) added to ansatz: {new_indices}")
        self.update_iteration_costs(total_new_nfevs, total_new_ngevs, total_new_nits)

        return energy, gradient

    def select_operators(self, max_indices, max_gradients):
        """
        Select operators to add to the ansatz, based on the selection criterion.

        Arguments:
            max_indices (list): list of operators ordered by gradient magnitudes, possibly cropped to self.candidates
                elements if no more are needed considering the particular ADAPT implementation (TETRIS and CEO need
                more)
            max_gradients (list): the corresponding gradients
        """

        new_nfevs = []
        new_ngevs = []
        new_nits = []
        energy = None

        if self.sel_criterion == "random":
            gradient = self.select_via_sampling(max_indices, max_gradients)

        if self.sel_criterion == "sample":
            abs_gradients = [np.abs(gradient) for gradient in max_gradients]
            norm = sum(abs_gradients)
            probabilities = [g / norm for g in abs_gradients]
            gradient = self.select_via_sampling(
                max_indices, max_gradients, probabilities=probabilities
            )

        if self.sel_criterion == "1d_quad_fit":
            gradient = self.select_via_1d_quad_fit(max_indices, max_gradients)
            new_nfevs.append(
                4
            )  # Not worth creating a whole new g'' counter? Each eval requires four fevs

        if self.sel_criterion == "line_search":
            gradient, new_nfevs, new_ngevs, new_nits = self.select_via_line_search(
                max_indices, max_gradients, maxiters=1
            )

        elif self.sel_criterion == "1d_energy":
            gradient, new_nfevs, new_ngevs = self.select_via_1d_energy(
                max_indices, max_gradients
            )

        elif self.sel_criterion == "energy":
            energy, gradient, new_nfevs, new_ngevs, new_nits = self.select_via_energy(
                max_indices, max_gradients
            )

        elif self.sel_criterion == "summed_gradient":
            gradient = self.select_via_summed_gradient(max_indices, max_gradients)

        elif self.sel_criterion == "gradient":
            if self.dvg or self.dve or self.mvp:
                energy, gradient, new_nfevs, new_ngevs, new_nits = (
                    self.select_via_gradient_ceo(max_indices, max_gradients)
                )
            else:
                gradient = self.select_via_gradient(max_indices, max_gradients)

        return energy, gradient, new_nfevs, new_ngevs, new_nits

    def tetris_screening(self, new_qubits, viable_candidates, viable_gradients):
        """
        Filter viable candidates to include only operators which have disjoint supports from operators previously added
        in this iteration.
        When we run this method, viable_candidates includes all pool operators with nonzero gradients, because with
        TETRIS any of those might be added.

        Arguments:
            new_qubits (list): indices of the qubits that the last added operator acts on
            viable_candidates (list): indices of the candidates that were viable prior to the last addition
            viable_gradients (list): corresponding gradients

        Returns:
            still_viable_candidates (list): list of viable candidates, filtered to include only operators also in
                twin_ops
            still_viable_gradients (list): corresponding gradients
        """

        # Update list of qubits acted on non-trivially by operators in the current iteration
        self.iteration_qubits = self.iteration_qubits.union(new_qubits)

        still_viable_candidates = []
        still_viable_gradients = []
        for i, op in enumerate(viable_candidates):
            op_qubits = self.pool.get_qubits(op)

            if op_qubits.isdisjoint(self.iteration_qubits):
                still_viable_candidates.append(viable_candidates[i])
                still_viable_gradients.append(viable_gradients[i])

        if still_viable_candidates:
            print(
                f"{len(still_viable_candidates)} viable candidates: {still_viable_candidates}"
            )
        else:
            print("None found.\n")

        return still_viable_candidates, still_viable_gradients

    def select_via_1d_quad_fit(self, indices, gradients):
        """
        Select among operators based on a one-dimensional quadratic fit.

        Arguments:
            indices (list): the indices  of pool operators to select from
            gradients (list): the corresponding gradients

        Returns:
            gradient (float): the gradient of the selected operator
        """

        # Get 1D minimizers and minima
        minimizers_1d, e_changes_1d = self.get_quad_fit_minimizers(indices, gradients)

        # Find the best energy change
        e_change_1d = min(e_changes_1d)

        # Find the corresponding minimizer, operator index and gradient
        grad_rank = e_changes_1d.index(e_change_1d)
        coefficient = minimizers_1d[grad_rank]
        index = indices[grad_rank]
        gradient = gradients[grad_rank]

        # Augment ansatz with selected operator and the corresponding 1d minimizer as the new coefficient
        self.indices.append(index)
        self.coefficients.append(coefficient)

        return gradient

    def select_via_sampling(self, indices, gradients, probabilities=None):
        """
        Select among operators by sampling according to a probability distribution.

        Arguments:
            indices (list): the indices  of pool operators to select from
            gradients (list): the corresponding gradients
            probabilities (list): probabilities associated with each of the indices. If None, a uniform distribution
                is assumed

        Returns:
            gradient (float): the gradient of the selected operator
        """

        # Choose one of the operators by sampling according to the provided probability distribution
        gradient = np.random.choice(gradients, p=probabilities)
        grad_rank = gradients.index(gradient)
        index = indices[grad_rank]

        # Grow the ansatz and the parameter and gradient vectors
        self.indices.append(index)
        self.coefficients.append(0)
        self.gradients = np.append(self.gradients, gradient)

        return gradient

    def select_via_line_search(self, indices, gradients, maxiters=1):
        """
        Select among operators by performing one (or more) line searches and selecting the highest energy change.

        Arguments:
            indices (list): the indices  of pool operators to select from
            gradients (list): the corresponding gradients
            maxiters (int): the maximum number of line searches to be performed

        Returns:
            gradient (float): the gradient of the selected operator
        """

        # Equivalent to energy selection, but restricted to the desired number of line searches
        energy, gradient, nfevs, ngevs, nits = self.select_via_energy(
            indices, gradients, maxiters=maxiters
        )

        return gradient, nfevs, ngevs, nits

    def select_via_gradient(self, indices, gradients):
        """
        Select among operators based on the gradient (highest magnitude gradient wins).

        Arguments:
            indices (list): the indices  of pool operators to select from
            gradients (list): the corresponding gradients

        Returns:
            gradient (float): the gradient of the selected operator
        """

        index, gradient = self.find_highest_gradient(indices, gradients)

        # Grow the ansatz and the parameter and gradient vectors
        self.indices.append(index)
        self.coefficients.append(0)
        self.gradients = np.append(self.gradients, gradient)

        return gradient

    def find_highest_gradient(self, indices, gradients, excluded_range=[]):
        """
        Find the operator with the highest gradient magnitude.

        Arguments:
            indices (list): the indices  of pool operators to select from
            gradients (list): the corresponding gradients
            excluded_range (range): range of pool indices to be ignored

        Returns:
            index (float): the index of the operator with the highest gradient magnitude
            gradient (float): the corresponding gradient
        """

        # Filter out operators in the excluded range
        viable_indices = []
        viable_gradients = []
        for index, gradient in zip(indices, gradients):
            if index not in excluded_range:
                viable_indices.append(index)
                viable_gradients.append(gradient)

        # Find maximum absolute gradient
        abs_gradients = [np.abs(gradient) for gradient in viable_gradients]
        max_abs_gradient = max(abs_gradients)

        # Find corresponding gradient and index
        grad_rank = abs_gradients.index(max_abs_gradient)
        index = viable_indices[grad_rank]
        gradient = viable_gradients[grad_rank]

        return index, gradient

    def select_via_gradient_ceo(self, indices, gradients):
        """
        Select among operators based on the gradient (highest magnitude gradient wins).

        Arguments:
            indices (list): the indices  of pool operators to select from
            gradients (list): the corresponding gradients

        Returns:
            gradient (float): the gradient of the selected operator
        """

        # We want to select among OVP-CEOs, not parent pool (individual QEs) - hence the excluded range
        index, gradient = self.find_highest_gradient(
            indices, gradients, excluded_range=self.pool.parent_range
        )

        # Find "parents" - QE operators which act on the same qubits, and from which CEOs are derived
        if self.dve or self.dvg:
            parents = self.pool.get_parents(index)
        elif self.mvp:
            parents = self.pool.get_twin_ops(index) + [index]

        # Filter parents that also belong to the indices we want to select from
        parent_gradients, viable_parents = self.screen_parents(
            parents, indices, gradients
        )

        # Find list of candidate operators, which are CEOs of different types, and possibly formed from different QEs
        # acting on the same qubits. This method chooses the final operators (len(candidates)==1) unless we are running
        # DVE-CEO, in which case we still have to run an optimization to choose the winner
        candidates, candidate_gradients = self.filter_ceo_components(
            index, gradient, viable_parents, parent_gradients
        )

        if self.progressive_opt or self.dve:
            # We perform the optimization here for all candidates
            energy, gradient, nfevs, ngevs, nits = self.select_via_energy(
                candidates, candidate_gradients
            )
        else:
            # We are running DVG- or MVP-CEO, so the candidate was already selected by filter_ceo_components
            # Optimization will be performed later
            gradient = self.add_ceo_candidate(candidates, candidate_gradients)
            energy = None
            nfevs = None
            ngevs = None
            nits = None

        return energy, gradient, nfevs, ngevs, nits

    def add_ceo_candidate(self, candidate, gradient):
        """
        Add a CEO to the ansatz.

        Arguments:
            candidate (list): the CEO to add. It might be a single index (if it's an OVP-CEO) or multiple
            gradient (float): the corresponding gradient
        """

        if isinstance(candidate[0], list):
            # We have a list inside a list, but this function is only meant to add one CEO. Make sure we have only one
            assert len(candidate) == 1
            candidate = candidate[0]

        indices = np.array(self.indices.copy(), dtype=int)

        # Grow ansatz and current gradient/coefficient vectors
        self.gradients = np.append(self.gradients, gradient)
        self.indices = np.append(indices, candidate)
        self.coefficients = list(np.append(self.coefficients, [0 for _ in candidate]))

        return gradient

    def filter_ceo_components(self, index, gradient, viable_parents, parent_gradients):
        """
        Find all possible QEs to add to an MVP-CEO. The MVP-CEO is the one acting on the same qubits as the operator
        labeled by "index".

        Arguments:
            index (int): index of the OVP-CEO or QE whose qubits will define the qubits the final MVP-CEO acts on.
                This will typically be the operator with the highest gradient magnitude in the pool
            gradient (float): corresponding gradient
            viable_parents (list): operators with nonzero gradients acting on the same indices as this operator
                (including index, if it corresponds to a QE)
            parent_gradients (list): corresponding gradients

        Returns:
            candidates (list): list of candidates to add to the ansatz. Elements may be int (index of one candidate
            operator, OVP-CEO or QE) or list of int (index of QEs composing one MVP-CEO). Length is one for DVG- or MVP-
                CEO, where decision is made in this method (based on gradients), or two for DVE-CEO, where decision
                will be made later based on an optimization.
            candidate_gradients (list): corresponding gradients
        """

        if self.dve:
            # We either add the original OVP-CEO (index) or the QEs that it is formed from (viable_parents)
            candidates = [index, viable_parents]
            candidate_gradients = [gradient, parent_gradients]
            if len(viable_parents) > 1:
                print(
                    f"\nTesting OVP-CEO {index} and separately MVP-CEO formed from {viable_parents}"
                )
            else:
                print(f"\nOnly one QE with nonzero gradients. Will add QE {index}.")
                candidates = [index]
                candidate_gradients = [gradient]

        if self.dvg or self.mvp:
            # We add an MVP-CEO formed from all QEs with nonzero gradients acting on the concerned qubits
            if len(viable_parents) <= 1:
                # Either zero parents have nonzero gradients or only one, better to just use the more
                # circuit-efficient CEO
                # (if there are no parents, it's a single excitation operator)
                print(f"\nAdding OVP-CEO.")
                candidates = [index]
                candidate_gradients = [gradient]

            elif len(viable_parents) > 1:
                if self.dvg or self.mvp or len(viable_parents) == 3:
                    print(f"\nAdding MVP-CEO.")
                    candidates = [viable_parents]
                    candidate_gradients = [parent_gradients]

            # no else - at least one parent has non zero gradient

        return candidates, candidate_gradients

    def screen_parents(self, parents, indices, gradients):
        """
        Filter the intersection between the parents of a given CEO and the indices we are considering in the
        current iteration.

        Arguments:
            parents (list): the indices identifying the parents of the CEO we are concerned with
            indices (list): the indices  of pool operators under consideration
            gradients (list): the corresponding gradients
        """

        parent_gradients = []
        viable_parents = []

        if parents is not None:
            for parent in parents:
                if parent in indices:
                    viable_parents.append(parent)
                    i = indices.index(parent)
                    parent_gradients.append(gradients[i])

        return parent_gradients, viable_parents

    def select_via_summed_gradient(self, indices, gradients):
        """
        Select among operators based on a summed gradient criterion (highest sum of magnitudes gradient wins).

        Arguments:
            indices (list): the indices  of pool operators to select from
            gradients (list): the corresponding gradients

        Returns:
            gradient (float): the gradient of the selected operator
        """

        # For each group of operators acting on the same qubits, find the sum of the gradient magnitudes
        checked = []
        summed_gradients = []
        for index, gradient in zip(indices, gradients):

            if index in checked:
                # Already considered this in another group
                continue
            summed_gradient = np.abs(gradient)

            # Find operators acting on the same qubits and add their gradient magnitudes
            for i in self.pool.operators[index].twin_string_ops:
                if i in indices:
                    ix = indices.index(i)
                    summed_gradient += np.abs(gradients[ix])
                # else: gradient is zero
                checked.append(i)

            summed_gradients.append(summed_gradient)
            checked.append(index)

        max_summed_gradient = max(summed_gradients)

        # Find operator with the highest gradient magnitude within the group with the highest sum of gradient magnitudes
        grad_rank = summed_gradients.index(max_summed_gradient)
        index = indices[grad_rank]

        # Grow the ansatz with this operator
        self.indices.append(index)
        gradient = gradients[grad_rank]
        self.coefficients.append(0)
        self.gradients = np.append(self.gradients, gradient)

        return gradient

    def select_via_1d_energy(self, indices, gradients):
        """
        Select among operators based on the energy change they produce through a 1D optimization (only their parameter
        is optimized, the rest of the ansatz is frozen).

        Arguments:
            indices (list): the indices  of pool operators to select from
            gradients (list): the corresponding gradients

        Returns:
            gradient (float): the gradient of the selected operator
        """

        energies = []
        nfevs = []
        ngevs = []
        state = self.state

        # Optimize the coefficient of each possible operator, if appended to the current ansatz with current parameters
        for pos, gradient in enumerate(gradients):
            index = indices[pos]
            energy, coef, new_gradient, nfev, ngev, nit = self.minimize_1d(
                index, state, e0=self.energy, g0=[gradient]
            )

            energies.append(energy)
            nfevs.append(nfev)
            ngevs.append(ngev)

        echanges = [e - self.energy for e in energies]
        print("1D Optimization Energy Changes: ", echanges)

        # Identify operator corresponding to the best energy reduction
        energy = min(energies)
        grad_rank = energies.index(energy)
        gradient = gradients[grad_rank]
        index = indices[grad_rank]

        # Grow ansatz
        self.indices.append(index)
        self.coefficients.append(0)  # Could also append the new optimized coefficient
        self.gradients = np.append(self.gradients, gradient)

        return gradient, nfevs, ngevs

    def select_via_energy(self, indices, gradients, maxiters=None):
        """
        Select among operators based on the energy change they produce through a full optimization (all parameters are
        optimized - ansatz + new candidate)

        Arguments:
            indices (list): the indices  of pool operators to select from. Elements can be indices or lists of
                indices. E.g., if indices = [9,[22,23]], this method selects whichever produces the highest energy
                impact: adding 9 or adding both 22 and 23
            gradients (list): the corresponding gradients
            maxiters (int): maximum number of optimizer iterations.

        Returns:
            gradient (float): the gradient of the selected operator
        """

        inv_hessians = []
        energies = []
        energy_changes = []
        echanges_per_cnot = []
        coefficient_lists = []
        gs = []
        nfevs = []
        ngevs = []
        nits = []

        # Perform full optimization for each candidate operator
        for pos, gradient in enumerate(gradients):
            index = indices[pos]
            initial_inv_hessian, initial_coeffs, test_indices, e0, g0, maxiters = (
                self.prepare_parallel_opt(index, gradient, maxiters)
            )

            coefficients, energy, inv_hessian, gradients, nfev, ngev, nit = (
                self.full_optim(
                    test_indices,
                    initial_coeffs,
                    e0=e0,
                    g0=g0,
                    initial_inv_hessian=initial_inv_hessian,
                    maxiters=maxiters,
                )
            )

            energy_change = energy - self.energy
            print("Energy change: ", energy_change)
            gs.append(gradients)
            coefficient_lists.append(coefficients)
            energies.append(energy)
            energy_changes.append(energy_change)
            nfevs.append(nfev)
            ngevs.append(ngev)
            nits.append(nit)
            inv_hessians.append(inv_hessian)

            echange_per_cnot = self.divide_by_cnots(energy_change, index)
            echanges_per_cnot.append(echange_per_cnot)

        if self.penalize_cnots or (self.dve and self.candidates == 1):
            # Choose operator that produces the best energy change per CNOT
            echange_per_cnot = min(echanges_per_cnot)
            grad_rank = echanges_per_cnot.index(echange_per_cnot)
            energy = energies[grad_rank]
        else:
            # Choose operator that produces the best energy change
            energy = min(energies)
            grad_rank = energies.index(energy)

        # Grow ansatz
        gradient = gradients[grad_rank]
        index = indices[grad_rank]
        indices = np.array(self.indices.copy(), dtype=int)
        self.indices = np.append(indices, index)
        self.coefficients = coefficient_lists[grad_rank]
        self.gradients = gs[grad_rank]
        if self.recycle_hessian:
            self.inv_hessian = inv_hessians[grad_rank]

        return energy, gradient, nfevs, ngevs, nits

    def prepare_parallel_opt(self, index, gradient, maxiters=None):
        """
        Prepare input arguments for optimization function when testing multiple candidates in parallel without actually
        adding them to the ansatz.

        Arguments:
            index (int): the index of the pool operator that will be tested
            gradient (float): the corresponding gradient

        Returns:
            initial_inv_hessian (np.ndarray): initial inverse Hessian for the optimization
            initial_coefficients (list): initial coefficients for the optimization
            test_indices (list): list of pool operator indices that defines the ansatz that will be optimized
            e0 (float): initial energy
            g0 (float): initial gradient
            maxiters (int): maximum number of optimizer iterations

        """

        if isinstance(gradient, float):
            gradient = [gradient]
        if self.recycle_hessian:
            initial_inv_hessian = self.expand_inv_hessian(
                len(gradient) + self.orb_opt_dim
            )
        else:
            initial_inv_hessian = self.inv_hessian

        initial_coefficients = self.coefficients.copy() + [
            0 for _ in range(len(gradient))
        ]
        print(f"\nInitial coefficients: {initial_coefficients}")

        test_indices = np.array(self.indices.copy(), dtype=int)
        test_indices = np.append(test_indices, index)

        if self.recycle_hessian:
            g0 = np.append(self.gradients, gradient)
            e0 = self.energy
        else:
            g0 = None
            e0 = None

        if maxiters is None:
            maxiters = self.max_opt_iter

        return initial_inv_hessian, initial_coefficients, test_indices, e0, g0, maxiters

    def divide_by_cnots(self, value, index):
        """
        Divides a value by the number of CNOTs in the circuit implementationf of a given operator

        Arguments:
            value (float): the value to be divided
            index (int): the index of the pool operator to consider the CNOTs of

        Returns:
            value_per_cnot (float): value, divided by the number of CNOTs of the operator
        """

        cnots = self.pool.get_cnots(index)
        value_per_cnot = value / cnots

        return value_per_cnot

    def optimize(self, gradient):
        """
        Optimize current ansatz. May optimize full parameter vector or part, depending on attribute self.full_opt.

        Arguments:
            gradient (float): the gradient of the last-added operator

        Returns:
            energy (float): the optimized energy
        """

        if not self.full_opt:
            energy, nfev, g1ev, nit = self.partial_optim(gradient)

        else:
            self.inv_hessian = self.expand_inv_hessian()
            (
                self.coefficients,
                energy,
                self.inv_hessian,
                self.gradients,
                nfev,
                g1ev,
                nit,
            ) = self.full_optim()

        self.iteration_nfevs.append(nfev)
        self.iteration_ngevs.append(g1ev)
        self.iteration_nits.append(nit)

        return energy

    def full_optim(
        self,
        indices=None,
        initial_coefficients=None,
        initial_inv_hessian=None,
        e0=None,
        g0=None,
        maxiters=None,
    ):
        """
        Minimizes the energy of a new ansatz, created from the previous one by appending a new element labeled by
        new_index. The minimization is done using BFGS.

        Args:
            indices (list): the indices defining the ansatz before the new addition. If None, current ansatz is assumed
            initial_coefficients (list): the initial point for the optimization. If None, the initial point will be the
                previous coefficients with zeros appended.
            initial_inv_hessian (np.ndarray): an approximation for the initial inverse Hessian for the optimization
            e0 (float): initial energy
            g0 (list): initial gradient vector
            maxiters (int): maximum number of optimizer iterations. If None, self.max_opt_iters is assumed.

        Returns:
            ansatz_coefficients (list): the newly optimized coefficients
            opt_energy (float): the optimized energy
            inv_hessian (np.ndarray): the final approximation to the inverse Hessian
            nfev (int): total number of function evaluations
            ngev (int): total number of gradient evaluations
            nit (int): total number of optimizer iterations
        """
        initial_coefficients, indices, initial_inv_hessian, g0, e0, maxiters = (
            self.prepare_opt(
                initial_coefficients, indices, initial_inv_hessian, g0, e0, maxiters
            )
        )

        print(
            f"Initial energy: {self.energy}"
            f"\nOptimizing energy with indices {list(indices)}..."
            f"\nStarting point: {list(initial_coefficients)}"
        )

        # Define callback to collect data during optimization
        def callback(args):
            evolution["parameters"].append(args.x)
            evolution["energy"].append(args.fun)
            evolution["inv_hessian"].append(args.inv_hessian)
            evolution["gradient"].append(args.gradient)

        evolution = {"parameters": [], "energy": [], "inv_hessian": [], "gradient": []}

        # Define cost function
        e_fun = lambda x, ixs: self.evaluate_energy(
            coefficients=x[self.orb_opt_dim :],
            indices=ixs,
            orb_params=x[: self.orb_opt_dim],
        )

        extra_njev = 0
        if self.recycle_hessian and (not self.data.iteration_counter and self.orb_opt):
            # First case: we know the gradient of the ansatz parameter, but not orbital optimization coefficients
            # Second case: we're not recycling orbital parameters, so we must recalculate gradients
            g0 = self.estimate_gradients(initial_coefficients, indices)
            extra_njev = 1

        # Perform optimization
        opt_result = minimize_bfgs(
            e_fun,
            initial_coefficients,
            [indices],
            jac=self.estimate_gradients,
            initial_inv_hessian=initial_inv_hessian,
            disp=self.verbose,
            gtol=10**-8,
            maxiter=self.max_opt_iter,
            callback=callback,
            f0=e0,
            g0=g0,
        )

        # Retrieve solution and perform similarity transform accordingly
        opt_coefficients = list(opt_result.x)
        orb_params = opt_coefficients[: self.orb_opt_dim]
        ansatz_coefficients = opt_coefficients[self.orb_opt_dim :]
        opt_energy = self.evaluate_energy(
            ansatz_coefficients, indices, orb_params=orb_params
        )
        self.perform_sim_transform(orb_params)

        # Add costs
        nfev = opt_result.nfev
        njev = opt_result.njev + extra_njev
        ngev = njev * len(indices)
        nit = opt_result.nit

        if self.recycle_hessian:
            inv_hessian = opt_result.hess_inv
        else:
            inv_hessian = None

        if opt_result.nit:
            gradients = evolution["gradient"][-1]
        else:
            gradients = g0

        return ansatz_coefficients, opt_energy, inv_hessian, gradients, nfev, ngev, nit

    def prepare_opt(
        self, initial_coefficients, indices, initial_inv_hessian, g0, e0, maxiters
    ):
        """
        Prepares the arguments for the optimization by replacing None arguments with defaults.

        Args:
            initial_coefficients (list): the initial point for the optimization. If None, the initial point will be the
                previous coefficients with zeros appended.
            indices (list): the indices defining the ansatz before the new addition. If None, current ansatz is assumed
            initial_inv_hessian (np.ndarray): an approximation for the initial inverse Hessian for the optimization
            e0 (float): initial energy
            g0 (list): initial gradient vector
            maxiters (int): maximum number of optimizer iterations. If None, self.max_opt_iters is assumed.
        """

        if initial_coefficients is None:
            initial_coefficients = deepcopy(self.coefficients)
        if indices is None:
            indices = self.indices.copy()
        if initial_coefficients is None and indices is None:
            # Use current Hessian, gradient and energy for the starting point
            if initial_inv_hessian is None:
                initial_inv_hessian = self.inv_hessian
            if g0 is None and self.recycle_hessian:
                g0 = self.gradients
            if e0 is None and self.recycle_hessian:
                e0 = self.energy
        if maxiters is None:
            maxiters = self.max_opt_iter

        initial_coefficients = np.append(
            [0 for _ in range(self.orb_opt_dim)], initial_coefficients
        )

        return initial_coefficients, indices, initial_inv_hessian, g0, e0, maxiters

    def perform_sim_transform(self, orb_params):
        """
        Perform a similarity transform on the Hamiltonian.

        Arguments:
            orb_params (list): the parameters for the orbital tansformation
        """

        generator = self.create_orb_rotation_generator(orb_params)
        orb_rotation = expm(generator)
        self.hamiltonian = (
            orb_rotation.transpose().conj().dot(self.hamiltonian).dot(orb_rotation)
        )
        self.energy_meas = self.observable_to_measurement(self.hamiltonian)

        return

    def minimize_1d(self, index=None, state=None, max_iter=100, e0=None, g0=None):
        """
        Minimizes the energy of a new ansatz, created from the previous one by appending a new element labeled by
        new_inde, by optimizing only the last parameter

        Args:
            index (list): the index of the operator to be appended to the ansatz and optimized. If None, last ansatz
                element is assumed
            state (csc_matrix): the state in which the optimization is to be performed. Since we are freezing the ansatz
                there is no need to recalculate the state each time. If None, the current state will be assumed
            max_iter (int): maximum number of optimizer iterations
            e0 (float): initial energy
            g0 (list): initial gradient

        Returns:
            coef (int): the optimizd coefficient
            gradient (float): the final gradient of the operator
            nfev (int): total number of function evaluations
            ngev (int): total number of gradient evaluations
            nit (int): total number of optimizer iterations
        """

        if state is None:
            state = self.state
        if index is None:
            index = self.indices[-1]

        def callback(args):
            evolution["parameters"].append(args.x)
            evolution["energy"].append(args.fun)
            evolution["inv_hessian"].append(args.inv_hessian)
            evolution["gradient"].append(args.gradient)

        evolution = {"parameters": [], "energy": [], "inv_hessian": [], "gradient": []}
        evaluate_energy = lambda x: self.evaluate_energy(
            x, indices=[index], ref_state=state
        )
        estimate_gradient = lambda x: self.estimate_gradient(
            x, -1, indices=[index], coefficients=[0], ref_state=state
        )

        print(f"\nPerforming 1D optimization with operator {index}...")
        opt_result = minimize_bfgs(
            evaluate_energy,
            [0],
            jac=estimate_gradient,
            disp=self.verbose,
            gtol=10**-6,
            maxiter=max_iter,
            callback=callback,
            f0=e0,
            g0=g0,
        )
        energy = opt_result.fun
        coef = list(opt_result.x)[0]
        nfev = opt_result.nfev
        ngev = opt_result.njev
        nit = opt_result.nit

        print(f"Energy: {energy}\nMinimizer: {coef}")

        if opt_result.nit:
            gradient = evolution["gradient"][-1][0]
        else:
            gradient = g0

        return energy, coef, gradient, nfev, ngev, nit

    def bfgs_update(self, gfkp1, gfk, xkp1, xk):
        """
        Perform a BFGS update on the inverse Hessian matrix.

        Arguments:
            gfkp1 (Union[list,np.ndarray]): the gradient at the new iterate
            gfk (Union[list,np.ndarray]): the gradient at the old iterate
            xkp1 (nion[list,np.ndarray]): the coefficients at the new iterate
            xk (Union[list,np.ndarray]): the coefficients at the old iterate

        Returns:
            inv_hessian (np.ndarray): the updated hessian. Also updates self.inv_hessian
        """

        gfkp1 = np.array(gfkp1)
        gfk = np.array(gfk)
        xkp1 = np.array(xkp1)
        xk = np.array(xk)

        self.inv_hessian = bfgs_update(self.inv_hessian, gfkp1, gfk, xkp1, xk)

        return self.inv_hessian

    @abc.abstractmethod
    def compute_state(self, coefficients=None, indices=None, ref_state=None):
        """
        Calculates the state specified by the list of coefficients and indices

        Arguments:
            coefficients (list): the coefficients of the ansatz
            indices (list): the indices of the pool operators defining the ansatz
            ref_state (csc_matrix): the reference state in which the ansatz acts
        """
        pass

    @abc.abstractmethod
    def save_hamiltonian(self, hamiltonian):
        """
        Transform Hamiltonian as an InteractionOperator into the Hamiltonian used in a specific simulation
         (e.g. sparse matrix) and save it as an attribute

         Arguments:
             hamiltonian (InteractionOperator): the original Hamiltonian
        """
        pass

    @abc.abstractmethod
    def evaluate_observable(
        self,
        measurement,
        coefficients=None,
        indices=None,
        ref_state=None,
        orb_params=None,
    ):
        """
        Evaluates the observable in the state specified by the list of coefficients and indices

        Arguments:
            measurement: the measurement to be done. Type depends on specific subclass
            coefficients (list): the coefficients of the ansatz
            indices (list): the indices of the pool operators defining the ansatz
            ref_state (csc_matrix): the state in which the ansatz acts
            orb_params (list): the parameters for the orbital optimization
        """
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def observable_to_measurement(self, observable):
        """
        Transform an observable into its processed form, ready for evaluation.
        For sparse matrix simulations it will be the same matrix. Circuit implementations are more complicated.
        """
        pass

    @abc.abstractmethod
    def create_orb_rotation_ops(self):
        """
        Create list of orbital rotation operators.
        See https://doi.org/10.48550/arXiv.2212.11405
        """
        pass

    @abc.abstractmethod
    def create_orb_rotation_generator(self, orb_params):
        """
        Create orbital rotation generator from self.orb_ops and the provided parameters.

        Arguments:
            orb_params (list): the parameters to multiply each operator in self.orb_ops

        Returns:
            generator (csc_matrix): the generator of the orbital rotation
        """
        pass


class LinAlgAdapt(AdaptVQE):
    """
    Class for ADAPT-VQEs in which the energy and gradients are calculated using linear algebra.
    Expectation values are calculated exactly, with no sampling noise
    """

    hamiltonian = None

    def __init__(self, *args, **kvargs):

        kvargs["pool"].imp_type = ImplementationType.SPARSE

        super().__init__(*args, **kvargs)

        self.state = self.sparse_ref_state
        self.ref_state = self.sparse_ref_state

    def evaluate_observable(
        self,
        observable,
        coefficients=None,
        indices=None,
        ref_state=None,
        orb_params=None,
    ):
        """
        Evaluates the observable in the state specified by the list of coefficients and indices

        Arguments:
            observable (csc_matrix): the hermitian operator to be measured
            coefficients (list): the coefficients of the ansatz
            indices (list): the indices of the pool operators defining the ansatz
            ref_state (csc_matrix): the reference state to consider
            orb_params (list): the parameters for the orbital optimization

        Returns:
            exp_value (float): the exact expectation value of the observable
        """

        ket = self.get_state(coefficients, indices, ref_state)

        if orb_params is not None:
            orb_rotation_generator = self.create_orb_rotation_generator(orb_params)
            ket = expm_multiply(orb_rotation_generator, ket)

        # Get the corresponding bra and calculate the energy: |<bra| H |ket>|
        bra = ket.transpose().conj()
        # exp_value = (bra * observable * ket)[0, 0].real # slower
        exp_value = (bra.dot(observable.dot(ket)))[0, 0].real

        return exp_value

    def compute_state(self, coefficients=None, indices=None, ref_state=None, bra=False):
        """
        Calculates the state specified by the list of coefficients and indices. If coefficients or indices are None, the
        current ones are assumed.

        Arguments:
            coefficients (list): the coefficients of the ansatz
            indices (list): the indices of the pool operators defining the ansatz
            ref_state (np.ndarray): the reference state in which the ansatz acts
            bra (bool): whether to return the adjoint of the state

        Returns:
            state (csc_matrix): the desired state
        """

        if indices is None:
            indices = self.indices
        if coefficients is None:
            coefficients = self.coefficients

        if ref_state is None:
            ref_state = self.sparse_ref_state
        state = ref_state.copy()

        if bra:
            coefficients = [-c for c in reversed(coefficients)]
            indices = reversed(indices)

        # Apply e ** (coefficient * operator) to the state (ket) for each
        # operator in the ansatz, following the order of the list
        for coefficient, index in zip(coefficients, indices):
            # Exponentiate the operator and update ket to represent the state after
            # this operator has been applied
            state = self.pool.expm_mult(coefficient, index, state)
        if bra:
            state = state.transpose().conj()

        return state

    def create_ansatz_unitary(self, coefficients, indices):
        """
        Create a unitary e^(C_N*Op_N)...e^(C_1*Op_1).

        Arguments:
          coefficients (list): the coefficients of the exponentials
          indices(list): the indices representing the operators

        Returns:
            ansatz_unitary (csc_matrix): the desired unitary
        """

        n, _ = self.pool.get_imp_op(
            0
        ).shape  # Could be any - we just want the dimension
        ansatz_unitary = scipy.sparse.identity(n)

        # Apply e ** (coefficient * operator) for each operator in
        # operators, following the order of the list
        for coefficient, index in zip(coefficients, indices):
            ansatz_unitary = self.pool.expm_mult(coefficient, index, ansatz_unitary)

        return ansatz_unitary

    def create_orb_rotation_ops(self):
        """
        Create list of orbital rotation operators.
        See https://doi.org/10.48550/arXiv.2212.11405
        """

        n_spatial = int(self.n / 2)

        k = 0
        self.orb_ops = []
        self.sparse_orb_ops = []

        if not self.orb_opt:
            return

        for p in range(n_spatial):
            for q in range(p + 1, n_spatial):
                new_op = create_spin_adapted_one_body_op(p, q)
                # new_op = get_sparse_operator(new_op, n_spatial * 2)
                self.orb_ops.append(new_op)
                sparse_op = get_sparse_operator(new_op, self.n)
                self.sparse_orb_ops.append(sparse_op)
                k += 1

        assert len(self.orb_ops) == int((n_spatial * (n_spatial - 1)) / 2)

        return

    def create_orb_rotation_generator(self, orb_params):
        """
        Create orbital rotation generator from self.orb_ops and the provided parameters.

        Arguments:
            orb_params (list): the parameters to multiply each operator in self.orb_ops

        Returns:
            generator (csc_matrix): the generator of the orbital rotation
        """

        assert len(orb_params) == len(self.orb_ops)

        generator = None
        for param, op in zip(orb_params, self.sparse_orb_ops):

            if not np.abs(param):
                continue

            if generator is None:
                generator = param * op
            else:
                generator = generator + param * op

        if generator is None:
            generator = np.zeros((2**self.n, 2**self.n), dtype=complex)
            generator = csc_matrix(generator)

        return generator

    def eval_candidate_gradient_prepending(
        self,
        index,
        method="an",
        dx=10**-8,
        orb_params=None,
    ):
        """
        Estimates the gradient of unitary generated by pool operator if they are prepended to the ansatz (added
        right after the reference state, beginning of the circuit) at point zero.

        Args:
            index (int): index of pool operator
            method (str): the method for estimating the gradient
            dx (float): the step size used for the finite difference approximation
            orb_params (list): the parameters for the orbital optimization, if applicable

        Returns:
            gradient (float): the gradient
        """

        if method == "fd":
            # Finite differences are implemented in parent class
            return super().eval_candidate_gradient_prepending(index, method, dx, orb_params)

        if method != "an":
            raise ValueError(f"Method {method} is not supported.")

        if self.orb_opt:
            raise NotImplementedError

        if self.data is not None:
            coefficients = self.coefficients.copy()
            indices = self.indices
        else:
            coefficients = []
            indices = []

        operator = self.pool.get_imp_op(index)

        left_matrix = (
            self.compute_state(coefficients, indices)
            .transpose()
            .conj()
        )

        if self.pool.eig_decomp[index] is None:
            hamiltonian = self.hamiltonian

        else:
            hamiltonian = self.hamiltonian.todense()

        left_matrix = left_matrix.dot(hamiltonian)
        right_matrix = self.ref_state
        right_matrix = self.compute_state(
            coefficients,
            indices,
            ref_state=operator.dot(right_matrix)
        )
        gradient = 2 * (left_matrix.dot(right_matrix))[0, 0].real

        return gradient

    def eval_candidate_gradients_prepending(
        self,
        method="an",
        dx=10**-8,
        orb_params=None,
    ):
        """
        Estimates the gradient of unitaries generated by each pool operator if they are prepended to the ansatz (added
        right after the reference state, beginning of the circuit) at point zero.

        Args:
            index (int): index of pool operator
            method (str): the method for estimating the gradient
            dx (float): the step size used for the finite difference approximation
            orb_params (list): the parameters for the orbital optimization, if applicable

        Returns:
            gradients (list): the list of gradients, in the same order as the pool operator list
            norm (float): the norm of the gradient
        """

        if method == "fd":
            # Finite differences are implemented in parent class
            return super().eval_candidate_gradients_prepending(method,dx,orb_params)

        if method != "an":
            raise ValueError(f"Method {method} is not supported.")

        if self.orb_opt:
            raise NotImplementedError

        if self.data is not None:
            coefficients = self.coefficients.copy()
            indices = self.indices
        else:
            coefficients = []
            indices = []

        gradients = []
        norm = 0

        left_matrix = csc_matrix(
            self.compute_state(coefficients, indices)
            .transpose()
            .conj()
        )

        if self.pool.eig_decomp[0] is None:
            hamiltonian = self.hamiltonian

        else:
            hamiltonian = csc_matrix(self.hamiltonian)#.todense()

        left_matrix = left_matrix.dot(hamiltonian)
        right_matrix = self.ref_state

        for index in range(self.pool.size):
            operator = self.pool.get_imp_op(index)
            new_right_matrix = self.compute_state(
                coefficients,
                indices,
                ref_state=operator.dot(right_matrix)
            )
            new_right_matrix = csc_matrix(new_right_matrix)
            gradient = 2 * (left_matrix.dot(new_right_matrix))[0, 0].real
            gradients.append(gradient)
            norm += gradient**2

        return gradients, np.sqrt(norm)

    def estimate_gradients(
        self, coefficients=None, indices=None, method="an", dx=10**-8, orb_params=None
    ):
        """
        Estimates the gradients of all operators in the ansatz defined by coefficients and indices. If they are None,
        the current state is assumed. Default method is analytical (with unitary recycling for faster execution).

        Args:
            coefficients (list): the coefficients of the ansatz. If None, current coefficients will be used.
            indices (list): the indices of the ansatz. If None, current indices will be used.
            method (str): the method for estimating the gradient
            dx (float): the step size used for the finite difference approximation
            orb_params (list): the parameters for the orbital optimization, if applicable

        Returns:
            gradients (list): the gradient vector
        """

        if method == "fd":
            # Finite differences are implemented in parent class
            return super().estimate_gradients(
                coefficients=coefficients, indices=indices, method=method, dx=dx
            )

        if method != "an":
            raise ValueError(f"Method {method} is not supported.")

        if indices is None:
            assert coefficients is None
            indices = self.indices
            coefficients = self.coefficients

        if self.orb_opt:
            orb_params = coefficients[: self.orb_opt_dim]
            coefficients = coefficients[self.orb_opt_dim :]
        else:
            orb_params = None

        if not len(indices):
            return []

        # Define orbital rotation
        hamiltonian = self.hamiltonian
        if orb_params is not None:
            generator = self.create_orb_rotation_generator(orb_params)
            orb_rotation = expm(generator)
            hamiltonian = (
                orb_rotation.transpose().conj().dot(hamiltonian).dot(orb_rotation)
            )
        else:
            orb_rotation = np.eye(2**self.n)
            orb_rotation = csc_matrix(orb_rotation)

        gradients = []
        state = self.compute_state(coefficients, indices)
        right_matrix = self.sparse_ref_state
        left_matrix = self.compute_state(
            coefficients, indices, hamiltonian.dot(state), bra=True
        )

        # Ansatz gradients
        for operator_pos in range(len(indices)):
            operator = self.pool.get_imp_op(indices[operator_pos])
            coefficient = coefficients[operator_pos]
            index = indices[operator_pos]

            left_matrix = (
                self.pool.expm_mult(coefficient, index, left_matrix.transpose().conj())
                .transpose()
                .conj()
            )
            right_matrix = self.pool.expm_mult(coefficient, index, right_matrix)

            gradient = 2 * (left_matrix.dot(operator.dot(right_matrix)))[0, 0].real
            gradients.append(gradient)

        right_matrix = csc_matrix(orb_rotation.dot(right_matrix))
        left_matrix = csc_matrix(right_matrix.transpose().conj())

        # Orbital gradients
        orb_gradients = []
        for operator in self.sparse_orb_ops:
            gradient = (
                2
                * left_matrix.dot(self.hamiltonian)
                .dot(operator)
                .dot(right_matrix)[0, 0]
                .real
            )
            orb_gradients.append(gradient)

        # Remember that orbital optimization coefficients come first
        gradients = orb_gradients + gradients

        return gradients

    def estimate_snd_derivative_1var(
        self,
        operator_pos,
        coefficients=None,
        indices=None,
        method="an",
        formula=None,
        dx=10**-4,
    ):
        """
        Estimates the second derivative dE^2/dx^2 of the operator in position operator_pos of the ansatz defined by
        coefficients and indices. Default is finite differences; child classes may define other methods.

        Args:
            operator_pos (int): the position of the operator whose derivative we want to estimate. It may be higher than
                the length of indices, in which case it returns the gradient of the orbital optimization parameter
                indexed by operator_pos - len(indices).
            coefficients (list): the coefficients of the ansatz. If not, current coefficients will be used.
            indices (list): the indices of the ansatz. If not, current indices will be used.
            method (str): the method for estimating the gradient
            formula (str): the finite difference formula to use
            dx (float): the step size used for the finite difference approximation

        Returns:
            snd_derivative (float): the desired second derivative
        """

        if self.orb_opt:
            raise NotImplementedError

        if method == "fd":
            # Finite differences are implemented in parent class
            return super().estimate_snd_derivative_1var(
                operator_pos,
                coefficients=coefficients,
                indices=indices,
                method=method,
                formula=formula,
                dx=dx,
            )

        if method != "an":
            raise ValueError(f"Method {method} is not supported.")

        if formula is not None:
            raise ValueError(
                f"Analytical differentiation using formula {formula} is not supported."
            )

        if indices is None:
            assert coefficients is None
            indices = self.indices
            coefficients = self.coefficients

        assert len(coefficients) == len(indices)

        operator = self.pool.get_imp_op(indices[operator_pos])

        ant_unitary = self.create_ansatz_unitary(
            coefficients[: operator_pos + 1], indices[: operator_pos + 1]
        )

        post_unitary = self.create_ansatz_unitary(
            coefficients[operator_pos + 1 :], indices[operator_pos + 1 :]
        )

        right_matrix = ant_unitary.dot(self.ref_state)
        left_matrix = (
            (post_unitary.dot(right_matrix))
            .transpose()
            .conj()
            .dot(self.hamiltonian.dot(post_unitary))
        )

        term_1 = (
            2 * (left_matrix.dot(operator).dot(operator).dot(right_matrix))[0, 0].real
        )

        right_matrix = post_unitary.dot(operator).dot(ant_unitary).dot(self.ref_state)
        left_matrix = -right_matrix.transpose().conj()

        term_2 = -2 * (left_matrix.dot(self.hamiltonian).dot(right_matrix))[0, 0].real

        snd_derivative = term_1 + term_2

        return snd_derivative

    def estimate_snd_derivative(
        self,
        op1_pos,
        op2_pos=None,
        coefficients=None,
        indices=None,
        method="an",
        formula=None,
        dx=10**-4,
    ):
        """
        Estimates the second derivative dE^2/dxdy of the operators in position op1_pos, op2_pos of the ansatz defined by
        coefficients and indices. Default is finite differences; child classes may define other methods.

        Args:
            op1_pos (int): the position of one of the operator in the derivative.
            op2_pos (int): the position of the second operator in the derivative.
            coefficients (list): the coefficients of the ansatz. If not, current coefficients will be used.
            indices (list): the indices of the ansatz. If not, current indices will be used.
            method (str): the method for estimating the gradient
            formula (str): the finite difference formula to use
            dx (float): the step size used for the finite difference approximation

        Returns:
            snd_derivative (float): the desired second derivative
        """

        if self.orb_opt:
            raise NotImplementedError

        if method == "fd":
            # Finite differences are implemented in parent class
            return super().estimate_snd_derivative(
                op1_pos,
                op2_pos,
                coefficients=coefficients,
                indices=indices,
                method=method,
                formula=formula,
                dx=dx,
            )

        if method != "an":
            raise ValueError(f"Method {method} is not supported.")

        if formula is not None:
            raise ValueError(
                f"Analytical differentiation using formula {formula} is not supported."
            )

        if indices is None:
            assert coefficients is None
            indices = self.indices
            coefficients = self.coefficients

        assert len(coefficients) == len(indices)

        if self.pool.eig_decomp[0] is None:
            snd_derivative = self.estimate_snd_der_no_eig_decomp(
                op1_pos, op2_pos, coefficients, indices
            )
        else:
            snd_derivative = self.estimate_snd_der_eig_decomp(
                op1_pos, op2_pos, coefficients, indices
            )

        return snd_derivative

    def estimate_snd_der_no_eig_decomp(
        self, op1_pos, op2_pos=None, coefficients=None, indices=None
    ):
        """
        Estimates the second derivative dE^2/dxdy of the operators in position op1_pos, op2_pos of the ansatz defined by
        coefficients and indices when there is no available eigendecomposition of the pool.

        Args:
            op1_pos (int): the position of one of the operator in the derivative.
            op2_pos (int): the position of the second operator in the derivative.
            coefficients (list): the coefficients of the ansatz. If not, current coefficients will be used.
            indices (list): the indices of the ansatz. If not, current indices will be used.

        Returns:
            snd_derivative (float): the desired second derivative
        """

        if self.orb_opt:
            raise NotImplementedError

        if indices is None:
            assert coefficients is None
            indices = self.indices.copy()
            coefficients = self.coefficients.copy()

        assert len(coefficients) == len(indices)

        if op2_pos is None:
            op2_pos = op1_pos

        if op1_pos > op2_pos:
            temp = op1_pos
            op1_pos = op2_pos
            op2_pos = temp

        op1 = self.pool.get_imp_op(indices[op1_pos])
        op2 = self.pool.get_imp_op(indices[op2_pos])

        ant_unitary = self.create_ansatz_unitary(
            coefficients[: op1_pos + 1], indices[: op1_pos + 1]
        )

        int_unitary = self.create_ansatz_unitary(
            coefficients[op1_pos + 1 : op2_pos + 1], indices[op1_pos + 1 : op2_pos + 1]
        )

        post_unitary = self.create_ansatz_unitary(
            coefficients[op2_pos + 1 :], indices[op2_pos + 1 :]
        )

        right_matrix = ant_unitary.dot(self.ref_state)
        left_matrix = (
            (post_unitary.dot(int_unitary).dot(right_matrix))
            .transpose()
            .conj()
            .dot(self.hamiltonian)
            .dot(post_unitary)
        )

        term_1 = (
            2
            * (left_matrix.dot(op2).dot(int_unitary).dot(op1).dot(right_matrix))[
                0, 0
            ].real
        )

        left_matrix = right_matrix.transpose().conj()
        int_matrix = (
            (post_unitary.dot(int_unitary))
            .transpose()
            .conj()
            .dot(self.hamiltonian)
            .dot(post_unitary)
        )

        term_2 = (
            -2
            * left_matrix.dot(op1)
            .dot(int_matrix)
            .dot(op2)
            .dot(int_unitary)
            .dot(right_matrix)[0, 0]
            .real
        )

        snd_derivative = term_1 + term_2

        return snd_derivative

    def estimate_snd_der_eig_decomp(
        self, op1_pos, op2_pos=None, coefficients=None, indices=None
    ):
        """
        Estimates the second derivative dE^2/dxdy of the operators in position op1_pos, op2_pos of the ansatz defined by
        coefficients and indices when there is an available eigendecomposition of the pool.
        This is faster than estimate_snd_der_eig_decomp.
        Avoid matrix-matrix operations, prever matrix-vector for increased efficiency

        Args:
            op1_pos (int): the position of one of the operator in the derivative.
            op2_pos (int): the position of the second operator in the derivative.
            coefficients (list): the coefficients of the ansatz. If not, current coefficients will be used.
            indices (list): the indices of the ansatz. If not, current indices will be used.

        Returns:
            snd_derivative (float): the desired second derivative
        """

        if self.orb_opt:
            raise NotImplementedError

        if indices is None:
            assert coefficients is None
            indices = self.indices.copy()
            coefficients = self.coefficients.copy()

        assert len(coefficients) == len(indices)

        if op2_pos is None:
            op2_pos = op1_pos

        if op1_pos > op2_pos:
            temp = op1_pos
            op1_pos = op2_pos
            op2_pos = temp

        op1 = self.pool.get_imp_op(indices[op1_pos])
        op2 = self.pool.get_imp_op(indices[op2_pos])

        m = self.compute_state(
            coefficients[: op1_pos + 1], indices[: op1_pos + 1], self.ref_state
        )  # right_matrix
        m = op1.dot(m)
        m = self.compute_state(
            coefficients[op1_pos + 1 : op2_pos + 1],
            indices[op1_pos + 1 : op2_pos + 1],
            m,
        )  # int_unitary
        m = op2 * m
        m = self.compute_state(
            coefficients[op2_pos + 1 :], indices[op2_pos + 1 :], m
        )  # post_unitary
        m = self.hamiltonian.dot(m)
        m = (
            self.compute_state(coefficients, indices, m, bra=True).transpose().conj()
        )  # post_unitary * int * right .t.c
        term_1 = 2 * (self.ref_state.transpose().conj().dot(m))[0, 0].real

        m = self.compute_state(
            coefficients[: op2_pos + 1], indices[: op2_pos + 1], self.ref_state
        )  # int_unitary * right_matrix
        m = op2.dot(m)
        m = self.compute_state(
            coefficients[op2_pos + 1 :], indices[op2_pos + 1 :], m
        )  # post_unitary
        m = self.hamiltonian.dot(m)
        m = (
            self.compute_state(
                coefficients[op1_pos + 1 :], indices[op1_pos + 1 :], m, bra=True
            )
            .transpose()
            .conj()
        )  # post_unitary * int_unitary .t.c
        m = op1.dot(m)
        m = (
            self.compute_state(
                coefficients[: op1_pos + 1], indices[: op1_pos + 1], m, bra=True
            )
            .transpose()
            .conj()
        )  # right_matrix.t.c.
        term_2 = -2 * (self.ref_state.transpose().conj().dot(m))[0, 0].real

        snd_derivative = term_1 + term_2

        return snd_derivative

    def estimate_hessian(
        self, coefficients=None, indices=None, method="an", formula="central", dx=10**-4
    ):
        """
        Estimates the Hessian of the energy.

        Args:
            coefficients (list): the coefficients of the ansatz. If not, current coefficients will be used.
            indices (list): the indices of the ansatz. If not, current indices will be used.
            method (str): the method for estimating the gradient
            formula (str): the finite difference formula to use
            dx (float): the step size used for the finite difference approximation

        Returns:
            hessian (np.ndarray): the Hessian
        """

        if indices is None:
            assert coefficients is None
            indices = self.indices.copy()
            coefficients = self.coefficients.copy()

        assert len(coefficients) == len(indices)

        if method == "fd":
            return super().estimate_hessian(
                coefficients=coefficients, indices=indices, method=method, dx=dx
            )

        size = len(indices)
        hessian = np.zeros((size, size))

        ket = self.ref_state
        bra_tc = self.ref_state

        for i in range(size):
            for j in range(i, size):

                op_i = self.pool.get_imp_op(indices[i])
                op_j = self.pool.get_imp_op(indices[j])

                if j == i:
                    # Update the vectors we'll recycle from previous iteration
                    ket = self.pool.expm_mult(coefficients[i], indices[i], ket)
                    left_matrix_tc = self.compute_state(coefficients, indices)
                    left_matrix_tc = self.hamiltonian.dot(left_matrix_tc)
                    left_matrix_tc = (
                        self.compute_state(
                            coefficients[i:], indices[i:], left_matrix_tc, bra=True
                        )
                        .transpose()
                        .conj()
                    )

                    ket2 = self.compute_state(coefficients[:j], indices[:j])
                    bra_tc = self.pool.expm_mult(coefficients[i], indices[i], bra_tc)

                right_matrix = op_i.dot(ket)
                right_matrix = self.compute_state(
                    coefficients[i + 1 : j + 1], indices[i + 1 : j + 1], right_matrix
                )
                right_matrix = op_j * right_matrix

                left_matrix_tc = self.pool.expm_mult(
                    coefficients[j], indices[j], left_matrix_tc
                )
                left_matrix = left_matrix_tc.transpose().conj()

                term_1 = 2 * (left_matrix.dot(right_matrix))[0, 0].real

                ket2 = self.pool.expm_mult(coefficients[j], indices[j], ket2)

                right_matrix = op_j.dot(ket2)
                right_matrix = self.compute_state(
                    coefficients[j + 1 :], indices[j + 1 :], right_matrix
                )
                right_matrix = self.hamiltonian.dot(right_matrix)
                right_matrix = (
                    self.compute_state(
                        coefficients[i + 1 :], indices[i + 1 :], right_matrix, bra=True
                    )
                    .transpose()
                    .conj()
                )
                right_matrix = op_i.dot(right_matrix)
                bra = bra_tc.transpose().conj()

                term_2 = -2 * (bra.dot(right_matrix))[0, 0].real

                snd_derivative = term_1 + term_2

                hessian[i, j] = snd_derivative
                hessian[j, i] = snd_derivative

        return hessian

    def save_hamiltonian(self, hamiltonian):
        """
        Store the provided hamiltonian as self.hamiltonian attribute after converting it to a sparse matrix.

        Arguments:
            hamiltonian (Union[InteractionOperator,csc_matrix])
        """

        if not issparse(hamiltonian):
            hamiltonian = get_sparse_operator(hamiltonian, self.n)

        self.hamiltonian = hamiltonian


    @property
    def name(self):
        return "linalg_adapt"

    def observable_to_measurement(self, observable):
        return observable

    def eval_candidate_gradient(self, index, coefficients=None, indices=None):
        """
        Calculates the norm of the gradient of a candidate operator if appended to the current state (or a different
        one, if coefficients and indices are provided; however this is never necessary in a normal run).
        Uses dexp(c*A)/dc = <psi|[H,A]|psi> = 2 * real(<psi|HA|psi>).
        This is the gradient calculated at c = 0, which will be the initial value of the coefficient in the
        optimization.
        Only the absolute value is returned.

        Arguments:
            index (int): the index that labels this operator
            coefficients (list): the coefficients of the ansatz. If not, current coefficients will be used.
            indices (list): the indices of the ansatz. If not, current indices will be used.

        Returns:
          gradient (float): the norm of the gradient of this operator in
          the current state
        """

        measurement = self.pool.get_grad_meas(index)

        if measurement is None:
            # Gradient observable for this operator has not been created yet

            operator = self.pool.get_imp_op(index)
            observable = 2 * self.hamiltonian @ operator

        gradient = self.evaluate_observable(observable, coefficients, indices)

        return gradient


from qiskit import QuantumCircuit
from adaptvqe.op_conv import to_qiskit_operator


class SampledLinAlgAdapt(LinAlgAdapt):
    """
    Do everything without noise, but then sample from the statevector instead of calculating the exact expectation value
    Only works for the qubit pool as of now because the finite differences are implemented in the simplest way possible
    If shots is None implements sampling noise free algorithm
    """

    def __init__(self, *args, **kvargs):

        super().__init__(*args, **kvargs)
        assert self.pool.name == "no_z_pauli_pool"
        assert not self.orb_opt

    def save_hamiltonian(self, hamiltonian):
        self.hamiltonian = to_qiskit_operator(hamiltonian, little_endian=False)

    def evaluate_observable(
        self,
        observable,
        coefficients=None,
        indices=None,
        ref_state=None,
        orb_params=None,
    ):
        from scipy.sparse import issparse
        from qiskit.primitives import Estimator

        ket = self.get_state(coefficients, indices, ref_state)

        if issparse(ket):
            ket = ket.toarray()
        else:
            ket = np.array(ket)

        ket = ket[:, 0]
        qc = QuantumCircuit(self.molecule.n_qubits)
        qc.initialize(ket)
        estimator = Estimator()
        job = estimator.run(qc, observable, shots=self.shots)
        result = job.result()
        exp_value = result.values[0]

        return exp_value

    def eval_candidate_gradient(self, op_index, coefficients=None, indices=None):

        observable = self.pool.get_grad_meas(op_index)

        if observable is None:
            # Gradient observable for this operator has not been created yet

            operator = self.pool.get_q_op(op_index)
            operator = to_qiskit_operator(operator, little_endian=False)
            observable = 2 * self.hamiltonian @ operator
            # observable = self.hamiltonian @ operator - operator @ self.hamiltonian
            self.pool.store_grad_meas(op_index, observable)

        gradient = self.evaluate_observable(observable, coefficients, indices)

        return gradient

    def estimate_gradients(
        self, coefficients=None, indices=None, method="psr", dx=10**-8
    ):
        """
        Estimates the gradients of all operators in the ansatz defined by coefficients and
        indices. If they are None, the current state is assumed. Default method is analytical (with unitary recycling
        for faster execution).
        Formula (14) in https://arxiv.org/pdf/1811.11184.pdf

        Args:
            coefficients (list): the coefficients of the ansatz. If not, current coefficients will be used.
            indices (list): the indices of the ansatz. If not, current indices will be used.
            method (str): the method for estimating the gradient
            dx (float): the step size used for the finite difference approximation

        Returns:
            gradient (float): the approximation to the gradient
        """

        if method == "fd":
            # Finite differences are implemented in parent class
            return super().estimate_gradients(
                coefficients=coefficients, indices=indices, method=method, dx=dx
            )

        if method != "psr":
            raise ValueError(f"Method {method} is not supported.")

        if indices is None:
            assert coefficients is None
            indices = self.indices
            coefficients = self.coefficients

        assert len(coefficients) == len(indices)

        if not indices:
            return []

        gradients = []
        for i in range(len(indices)):
            coefs_plus = coefficients.copy()
            coefs_plus[i] += np.pi / 4
            coefs_minus = coefficients.copy()
            coefs_minus[i] -= np.pi / 4

            energy_plus = self.evaluate_energy(coefs_plus, indices)
            energy_minus = self.evaluate_energy(coefs_minus, indices)

            g = energy_plus - energy_minus

            gradients.append(g)

        return gradients

    @property
    def name(self):
        return "sampled_linal_adapt"
