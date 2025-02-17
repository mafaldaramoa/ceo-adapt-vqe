import abc
import numpy as np

from openfermion import get_sparse_operator
from scipy.optimize import minimize
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply

from adaptvqe.chemistry import get_hf_det
from adaptvqe.matrix_tools import ket_to_vector
from adaptvqe.pools import SD, PairedGSD, PairedGSD_SwapNet, ImplementationType

class UCC():
    """
    A class for representing ansatze from the Unitary Coupled Cluster family.
    """

    def __init__(self,molecule,trotter=False, n_layers=1, eig_decomp=None):

        self.molecule = molecule
        self.trotter = trotter
        self.n_layers = n_layers
        self.n = self.molecule.n_qubits
        hamiltonian = self.molecule.get_molecular_hamiltonian()
        self.sparse_hamiltonian = get_sparse_operator(hamiltonian, self.n)
        self.ref_det = get_hf_det(self.molecule.n_electrons,self.n)
        self.ref_state = csc_matrix(
            ket_to_vector(self.ref_det), dtype=complex
        ).transpose()
        self.energy = molecule.hf_energy

        self.build_pool()
        self.pool.imp_type = ImplementationType.SPARSE

        self.eig_decomp = [None for _ in range(self.pool.size)]
        if eig_decomp is not None:
            if len(eig_decomp) != self.pool.size:
                raise ValueError("List of eigendecomposition list does not match pool size")
            self.eig_decomp = eig_decomp

        self.sparse_operators = [get_sparse_operator(op.operator,self.n) for op in self.pool.operators]
        self.num_params = len(self.sparse_operators) * self.n_layers

    @abc.abstractmethod
    def build_pool(self):
        """
        Choose self.pool
        """
        pass

    def calculate_state(self, parameters):
        """
        Calculates the state associated with the given parameters. If self.trotter, implements one Trotter step.
        Otherwise, exponentiates the whole generator for each layer.
        """

        if len(parameters) != self.num_params:
            raise ValueError("Length of supplied parameter list does not match ansatz parameter count.")

        if self.trotter:

            state = self.ref_state

            for index, parameter in zip(list(range(self.pool.size))*self.n_layers, parameters):
                state = self.pool.expm_mult(parameter, index, state)
        else:
            # For each layer, build the full generator, then take the exponential. This might actually be slower if
            #there's an eigendecomposition of the pool
            state = self.ref_state

            layer_param_lists = np.array_split(parameters, self.n_layers)
            for layer_params in layer_param_lists:

                generator = csc_matrix((2 ** self.n, 2 ** self.n), dtype=complex)

                for operator, parameter in zip(self.sparse_operators, layer_params):
                    generator += parameter * operator

                state = expm_multiply(generator, state)

        return state

    def calculate_energy(self, parameters):

        if len(parameters) != self.num_params:
            raise ValueError("Length of supplied parameter list does not match ansatz parameter count.")

        state = self.calculate_state(parameters)
        energy = state.transpose().conj().dot(self.sparse_hamiltonian.dot(state))[0, 0].real

        return energy

    def minimize(self, initial_parameters, maxiter=1000):

        if len(initial_parameters) != self.num_params:
            raise ValueError("Length of supplied parameter list does not match ansatz parameter count.")

        energy_fun = lambda parameters: self.calculate_energy(parameters)

        opt_result = minimize(energy_fun,
                            initial_parameters,
                            jac=self.calculate_gradients,
                            method="BFGS",
                            # callback=callback
                            options={"disp": True,
                                     "gtol": 10 ** -8,
                                     "maxiter": maxiter})

        self.energy = opt_result.fun

        return opt_result

    def create_eig_decomp(self):

        self.pool.create_eig_decomps()

    def estimate_fd_gradient(self, position, parameters, dx=10 ** -8):

        if len(parameters) != self.num_params:
            raise ValueError("Length of supplied parameter list does not match ansatz parameter count.")

        params_plus = parameters.copy()
        params_plus[position] += dx

        energy = self.calculate_energy(parameters)
        energy_plus = self.calculate_energy(params_plus)
        gradient = (energy_plus - energy) / dx

        return gradient

    def estimate_fd_gradients(self, parameters, dx=10 ** -8):

        if len(parameters) != self.num_params:
            raise ValueError("Length of supplied parameter list does not match ansatz parameter count.")

        gradients = []

        for i in range(self.num_params):
            gradient = self.estimate_fd_gradient(i, parameters, dx)
            gradients.append(gradient)

        return gradients

    def calculate_gradients(self, parameters):

        if self.trotter:
            return self.calculate_gradients_trotterized(parameters)
        else:
            return self.estimate_fd_gradients(parameters)

    def calculate_gradients_trotterized(self, parameters):

        assert self.trotter

        if len(parameters) != self.num_params:
            raise ValueError("Length of supplied parameter list does not match ansatz parameter count.")

        gradients = []
        right_matrix = self.ref_state

        # Calculate left_matrix
        # <HF|e^(-A_nc_n) ... e^(-A_1c_1) H e^(A_nc_n) ... e^(A_1c_1)
        left_matrix = self.calculate_state(parameters)
        left_matrix = self.sparse_hamiltonian.dot(left_matrix)
        for index, parameter in zip(reversed(list(range(self.pool.size))*self.n_layers),
                                    reversed(parameters)):
            left_matrix = self.pool.expm_mult(-parameter, index, left_matrix)
        left_matrix = left_matrix.transpose().conj()

        for index, parameter in zip(list(range(self.pool.size))*self.n_layers, parameters):
            sparse_op = self.pool.get_imp_op(index)
            left_matrix = self.pool.adj_expm_mult(parameter, index, left_matrix)
            right_matrix = self.pool.expm_mult(parameter, index, right_matrix)
            gradient = 2 * (left_matrix.dot(sparse_op.dot(right_matrix)))[0, 0].real
            gradients.append(gradient)

        return gradients

class UCCSD(UCC):
    """
    Unitary coupled cluster singles and doubles. Excitations are restricted to occupied-to-virtual (in the reference
    state).
    """

    def build_pool(self):

        self.pool = SD(self.molecule)

    def minimize(self,initial_parameters=None, maxiter=1000):
        """

        """

        opt_result = super().minimize(initial_parameters=[0 for _ in range(self.num_params)], maxiter=1000)

        return opt_result

class k_UpCCGSD(UCC):
    """
    Unitary paired coupled cluster generalized singles and doubles. This is a spin-preserving ansatz.
    See https://pubs.acs.org/doi/10.1021/acs.jctc.8b01004
    """

    def build_pool(self):
        self.pool = PairedGSD(self.molecule)

    def minimize(self,initial_parameters=None, maxiter=1000):

        # Choose parameters randomly between [-pi,pi]
        initial_parameters = [-np.pi + 2 * np.pi * np.random.rand() for _ in range(self.num_params)]
        opt_result = super().minimize(initial_parameters=initial_parameters, maxiter=1000)

        return opt_result

class k_UpCCGSD_SwapNet(k_UpCCGSD):
    """
    The unitary paired coupled cluster generalized singles and doubles, Trotterized and implemented using a fermionic
    swap network.
    See https://arxiv.org/abs/1905.05118
    """

    def build_pool(self):
        """
        Choose pool
        """
        self.pool = PairedGSD_SwapNet(self.molecule)