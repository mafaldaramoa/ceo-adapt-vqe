from openfermion import get_sparse_operator
from scipy.optimize import minimize
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply

from adaptvqe.chemistry import get_hf_det
from adaptvqe.matrix_tools import ket_to_vector
from adaptvqe.pools import SD, ImplementationType

class UCCSD():

  def __init__(self,molecule):

    self.molecule = molecule
    self.n = self.molecule.n_qubits
    hamiltonian = self.molecule.get_molecular_hamiltonian()
    self.hamiltonian = get_sparse_operator(hamiltonian, self.n)
    self.ref_det = get_hf_det(self.molecule.n_electrons,self.n)
    self.ref_state = csc_matrix(
        ket_to_vector(self.ref_det), dtype=complex
    ).transpose()
    self.energy = molecule.hf_energy

    pool = SD(molecule)
    self.operators = [get_sparse_operator(op.operator,self.n) for op in pool.operators]
    self.num_params = len(self.operators)

    self.pool = SD(molecule)
    self.pool.imp_type = ImplementationType.SPARSE
    self.num_params = self.pool.size

  def calculate_energy(self, parameters):

    generator = csc_matrix((2 ** self.n, 2 ** self.n), dtype=complex)

    '''for index, parameter in zip(range(self.pool.size), parameters):
      operator = self.pool.get_imp_op(index)
      generator += parameter * operator'''

    for operator, parameter in zip(self.operators, parameters):
      generator += parameter * operator

    state = expm_multiply(generator, self.ref_state)
    energy = state.transpose().conj().dot(self.hamiltonian.dot(state))[0, 0].real

    return energy

  def calculate_energy_trotterized(self, parameters):

    state = self.ref_state

    for index, parameter in zip(range(self.pool.size), parameters):
      state = self.pool.expm_mult(parameter, index, state)

    energy = state.transpose().conj().dot(self.hamiltonian.dot(state))[0, 0].real

    return energy

  def minimize(self, initial_parameters=None, maxiter=1000, trotter=False):

    if initial_parameters is None:
      initial_parameters = [0 for _ in range(self.num_params)]

    if trotter:
      energy_fun = self.calculate_energy_trotterized
    else:
      energy_fun = self.calculate_energy

    opt_result = minimize(energy_fun,
                          initial_parameters,
                          # jac='3-point',
                          method="BFGS",
                          # callback=callback
                          options={"disp": True,
                                   "gtol": 10 ** -4,
                                   "maxiter": maxiter})

    self.energy = opt_result.fun

    return opt_result