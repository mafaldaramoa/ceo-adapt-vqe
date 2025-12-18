from qiskit.quantum_info import Operator, process_fidelity
import numpy as np

from adaptvqe.molecules import create_h4
from adaptvqe.pools import DVG_CEO
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.circuits import get_circuit_energy
from adaptvqe.op_conv import get_remapped_f_hamiltonian, remap_hamiltonian_by_layout

r = 3
molecule = create_h4(r)
pool = DVG_CEO(molecule,fermionic_swaps=True)

# Run ADAPT-VQE with a CNOT penalty applied to operator gradients during the selection stage.
# This favors the selection of operators that require lower gate counts to be implemented.
# The connectivity is assumed to be LNN. The mapping of fermionic modes to qubits is dynamically updated throughout the
# algorithm (using swap gates) to bring the modes involved in each excitation together.
max_iter = 20
my_adapt = LinAlgAdapt(pool=pool,
                       molecule=molecule,
                       threshold=10**-6,
                       max_adapt_iter=max_iter,
                       recycle_hessian=True,
                       verbose=True,
                       penalize_cnots=True,
                       lnn=True
                       )

my_adapt.pool.create_eig_decomps()
my_adapt.run()
data = my_adapt.data

# Get the final ansatz indices and coefficients
indices = data.result.ansatz.indices
coefficients = data.result.ansatz.coefficients

# Calculate the ansatz unitary
goal_unitary = pool.get_unitary(coefficients,indices).todense()

# Obtain the ansatz circuit in linear-nearest-neighbor connectivity
# Reference state is not included so we can compare with unitary formed from ansatz operators only
lnn_qc_no_ref, _, layout = data.get_lnn_circuit(pool,apply_border_swaps=True)

# Calculate process fidelity of LNN ansatz circuit with goal unitary
pf_lnn = process_fidelity(Operator(goal_unitary),Operator(lnn_qc_no_ref))
print("Process fidelity of LNN circuit: ",pf_lnn)

# Make sure fidelity is close to 1
assert np.abs(1 - pf_lnn) < 10**-5

# Get the CNOT count of the circuit in LNN connectivity
# Border swaps are irrelevant - they can be applied classically (see below)
lnn_qc_no_swaps_no_ref, acc_cnot_counts, layout = data.get_lnn_circuit(pool,apply_border_swaps=False)
print("Final LNN ansatz circuit (without reference state):\n", lnn_qc_no_swaps_no_ref)
print("LNN Final CNOT counts:", acc_cnot_counts[-1])

# Get the energy from the circuit including reference state preparation and swap gates
hamiltonian = molecule.get_molecular_hamiltonian()
lnn_qc, _, layout = data.get_lnn_circuit(pool,apply_border_swaps=True,include_hf=True)
energy_swaps = get_circuit_energy(lnn_qc,hamiltonian)
error_swaps = np.abs(energy_swaps-data.evolution.energies[-1])
print("|Energy from circuit - ADAPT-VQE energy| for...\n"
      "* Circuit with swaps, original Hamiltonian: ", error_swaps)

# Make sure error is close to 1
assert error_swaps < 10**-8

# Get the energy from the circuit including reference state preparation but no swap gates
lnn_qc_no_swaps, acc_cnot_counts, layout = data.get_lnn_circuit(pool,apply_border_swaps=False,include_hf=True)
energy_no_swaps = get_circuit_energy(lnn_qc_no_swaps,hamiltonian)
error_no_swaps = np.abs(energy_no_swaps-data.evolution.energies[-1])
print("* Circuit without swaps, original Hamiltonian: ", error_no_swaps)

# Get the energy from the circuit including reference state preparation but no swap gates, but remapping the Hamiltonian
#to account for the nontrivial layout and mode order

# First: remap fermionic modes to account for fermionic swap gates
qubit_order = data.evolution.qubit_orders[-1]
remapped_hamiltonian = get_remapped_f_hamiltonian(hamiltonian, qubit_order)

# Second: remap qubits to account for nontrivial layout
remapped_hamiltonian = remap_hamiltonian_by_layout(remapped_hamiltonian,layout)
energy_no_swaps_remapped = get_circuit_energy(lnn_qc_no_swaps,remapped_hamiltonian)
error_no_swaps_remapped = np.abs(energy_no_swaps_remapped-data.evolution.energies[-1])
print("* Circuit without swaps, remapped Hamiltonian:", error_no_swaps_remapped)

# Make sure error is close to 1
assert error_no_swaps_remapped < 10**-8