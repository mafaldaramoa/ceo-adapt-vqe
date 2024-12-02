import numpy as np
from scipy.sparse import csc_matrix

from openfermion import QubitOperator

try:
    from quspin.operators import hamiltonian
    from quspin.basis import spin_basis_1d
except:
    pass
    # Quspin not installed. Will try to use precalculated Hamiltonian ground energies if necessary.
    # Todo: fix, it's ugly

from openfermion import (
    fermi_hubbard,
    get_ground_state,
    get_sparse_operator,
    get_quadratic_hamiltonian,
)

from .matrix_tools import ket_to_vector


class HubbardHamiltonian:
    """
    Class for Hubbard Hamiltonians.
    See https://quantumai.google/openfermion/fqe/tutorials/fermi_hubbard
    """

    def __init__(self, x_dim, y_dim, t, u, p_b_conds, ph_sym):
        """
        Initialize class instance.

        Arguments:
            x_dim (int): x dimension of the lattice
            y_dim (int): y dimension of the lattice
            t (float): tunneling amplitude
            u (float): Coulomb potential
            p_b_conds (bool): if to consider periodic boundary conditions
            ph_sym (bool): if to consider particle-hole symmetry
        """

        self.description = f"HH_{x_dim}_{y_dim}_{t}_{u}"

        h = fermi_hubbard(
            x_dim, y_dim, t, u, periodic=p_b_conds, particle_hole_symmetry=ph_sym
        )
        self.operator = h

        self._ground_energy = None
        self._ground_state = None
        neel_state_cb = [0, 1, 1, 0] * (y_dim // 2) + [0, 1] * (y_dim % 2)
        neel_state = ket_to_vector(neel_state_cb)
        neel_state = csc_matrix(neel_state).transpose()
        self.ref_state = neel_state
        self.ref_det = neel_state_cb

        """
        # Use non interacting ground state instead of Néel as the reference state:
        h = fermi_hubbard(x_dim, l, t, 0, periodic=p_b_conds, particle_hole_symmetry=ph_sym)
        quad_ham = get_quadratic_hamiltonian(h)
        sparse_h = get_sparse_operator(quad_ham)
        non_int_ground_energy, non_int_ground_state = get_ground_state(sparse_h)
        non_int_ground_state = csc_matrix(non_int_ground_state).transpose()
        self.ref_state = non_int_ground_state
        """

    @property
    def ground_state(self):
        """
        Returns the exact ground state of the Hamiltonian.
        """

        if self._ground_state is None:
            ground_energy, ground_state = get_ground_state(
                get_sparse_operator(self.operator)
            )
            self._ground_state = ground_state
            self._ground_energy = ground_energy

        return self._ground_state

    @property
    def ground_energy(self):
        """
        Returns the exact ground energy of the Hamiltonian.
        """

        if self._ground_energy is None:
            ground_energy, ground_state = get_ground_state(
                get_sparse_operator(self.operator)
            )
            self._ground_state = ground_state
            self._ground_energy = ground_energy

        return self._ground_energy


class XXZHamiltonian:
    """
    Class for XXZ Hamiltonians.
    See https://doi.org/10.48550/arXiv.2206.14215.
    """

    def __init__(self, j_xy, j_z, l):

        self.description = f"XXZ_{j_xy}_{j_z}"

        # Define Hamiltonian in Openfermion
        h = QubitOperator()
        for i in range(l - 1):
            h += j_xy * (
                QubitOperator(f"X{i} X{i + 1}") + QubitOperator(f"Y{i} Y{i + 1}")
            )
            h += j_z * QubitOperator(f"Z{i} Z{i + 1}")
        self.operator = h

        # Try to load precomputed ground energy
        self.ground_energy = self.load_ground_energy(l, j_z, j_xy)

        if self.ground_energy is None:
            # Define Hamiltonian in Quspin
            hz = 0  # z external field
            basis = spin_basis_1d(l, pauli=True)
            j_zz = [[j_z, i, i + 1] for i in range(l - 1)]  # OBC
            j_xy = [[j_xy / 2.0, i, i + 1] for i in range(l - 1)]  # OBC
            static = [["+-", j_xy], ["-+", j_xy], ["zz", j_zz]]
            dynamic = []
            h_xxz = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
            emin, emax = h_xxz.eigsh(
                k=2, which="BE", maxiter=1e4, return_eigenvectors=False
            )
            self.ground_energy = emin

        neel_state_cb = [i % 2 for i in range(l)]
        neel_state = ket_to_vector(neel_state_cb)
        neel_state = csc_matrix(neel_state).transpose()
        self.ref_state = neel_state
        self.ref_det = neel_state_cb

    def diagonalize_np(self):
        """
        Find ground state using Numpy.
        Much slower than Quspin, but useful to double-check results for small instances.
        """

        eigvals, eigvecs = np.linalg.eigh(get_sparse_operator(self.operator).todense())
        ground_energy = min(eigvals)
        ground_ix = np.where(eigvals == ground_energy)[0][0]
        ground_state = eigvecs[:, ground_ix]

        return ground_state, ground_energy

    @classmethod
    def load_ground_energy(cls, l, j_z, j_xy):
        """
        Just a patch up in case we want the ground state in a machine where QuSpin is not installed.
        Precalculated values.
        """

        ground_energy = None

        if l == 3 and j_z == 0.5 and j_xy == 1:
            ground_energy = -3.372281323269014

        if l == 3 and j_z == 1.0 and j_xy == 1:
            ground_energy = -4.0

        if l == 3 and j_z == 1.5 and j_xy == 1:
            ground_energy = -4.701562118716425

        if l == 4 and j_z == 0.5 and j_xy == 1:
            ground_energy = -5.424343992020248

        if l == 4 and j_z == 1.0 and j_xy == 1:
            ground_energy = -6.464101615137756

        if l == 4 and j_z == 1.5 and j_xy == 1:
            ground_energy = -7.5746735825151195

        if l == 6 and j_z == 0.5 and j_xy == 1:
            ground_energy = -8.391550553031378

        if l == 6 and j_z == 1.0 and j_xy == 1:
            ground_energy = -9.974308535551693

        if l == 6 and j_z == 1.5 and j_xy == 1:
            ground_energy = -11.709343563572977

        if l == 8 and j_z == 0.5 and j_xy == 1:
            ground_energy = -11.372996104030012

        if l == 8 and j_z == 1.0 and j_xy == 1:
            ground_energy = -13.499730394751541

        if l == 8 and j_z == 1.5 and j_xy == 1:
            ground_energy = -15.864651738684806

        if l == 10 and j_z == 0.5 and j_xy == 1:
            ground_energy = -14.361002811946285

        if l == 10 and j_z == 1.0 and j_xy == 1:
            ground_energy = -17.032140829131514

        if l == 10 and j_z == 1.5 and j_xy == 1:
            ground_energy = -20.029815648025142

        if l == 12 and j_z == 0.5 and j_xy == 1:
            ground_energy = -17.35259332370827

        if l == 12 and j_z == 1.0 and j_xy == 1:
            ground_energy = -20.568362531362073

        if l == 12 and j_z == 1.5 and j_xy == 1:
            ground_energy = -24.200560590841633

        if l == 14 and j_z == 0.5 and j_xy == 1:
            ground_energy = -20.34636164583642

        if l == 14 and j_z == 1.0 and j_xy == 1:
            ground_energy = -24.10689864744868

        if l == 14 and j_z == 1.5 and j_xy == 1:
            ground_energy = -28.374822014932327

        if l == 16 and j_z == 0.5 and j_xy == 1:
            ground_energy = -23.34155470512117

        if l == 16 and j_z == 1.0 and j_xy == 1:
            ground_energy = -27.646948582300418

        if l == 16 and j_z == 1.5 and j_xy == 1:
            ground_energy = -32.55146881041078

        return ground_energy

    @classmethod
    def print_ground_energy_range(cls, l_range, j_z_range, j_xy_range):
        """
        Print code lines for load_ground_energy. This is useful because installing quspin on Google Colab is not trivial
        The calculations are pretty light, so you can run them locally. E.g. for fig 1 of the paper:

        l_range = [3] + list(range(4, 17, 2))
        j_z_range = np.linspace(0.5, 1.5, 3)
        j_xy_range = [1]
        XXZHamiltonian.print_ground_energy_range(l_range,j_z_range,j_xy_range)
        """

        ls = []
        j_zs = []
        j_xys = []
        es = []

        for l in l_range:
            for j_z in j_z_range:
                for j_xy in j_xy_range:
                    h = XXZHamiltonian(j_xy, j_z, l)
                    e = h.ground_energy
                    ls.append(l)
                    j_zs.append(j_z)
                    j_xys.append(j_xy)
                    es.append(e)

        for i in range(len(ls)):
            l = ls[i]
            j_z = j_zs[i]
            e = es[i]
            print(f"if l=={l} and j_z=={j_z} and j_xy=={j_xy}:\n\tground_energy={e}\n")
