from qiskit.quantum_info import Operator, process_fidelity
import numpy as np

from adaptvqe.molecules import create_h4
from adaptvqe.pools import QE
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.circuits import cnot_count, get_circuit_energy
from adaptvqe.op_conv import get_qasm, remap_hamiltonian_by_layout
from openfermion import jordan_wigner

r = 3
molecule = create_h4(r)
pool = QE(molecule)

# Run ADAPT-VQE with a CNOT penalty applied to operator gradients during the selection stage.
# This favors the selection of operators that require lower gate counts to be implemented.
# The connectivity is assumed to be LNN. Operators are transpiled using Qiskit.
max_iter = 20
my_adapt = LinAlgAdapt(pool=pool,
                       molecule=molecule,
                       threshold=10**-6,
                       max_adapt_iter=max_iter,
                       recycle_hessian=True,
                       verbose=True,
                       penalize_cnots=True,
                       lnn_qiskit=True
                       )

my_adapt.pool.create_eig_decomps()
my_adapt.run()
data = my_adapt.data

# Get the final ansatz indices and coefficients
indices = data.result.ansatz.indices
coefficients = data.result.ansatz.coefficients

# Calculate the ansatz unitary
goal_unitary = pool.get_unitary(coefficients,indices).todense()

# Obtain the ansatz circuit in all-to-all connectivity
qc = pool.get_circuit(indices,coefficients)
print("Final ansatz circuit:\n", qc)

# Get the CNOT count of the circuit in all-to-all connectivity
ata_cnot_count = cnot_count(get_qasm(qc))
print("ATA final CNOT count: ", ata_cnot_count)

# Calculate process fidelity of ansatz circuit with goal unitary
pf = process_fidelity(Operator(goal_unitary),Operator(qc))
print("Process fidelity of non-transpiled circuit: ",pf)

# Obtain the ansatz circuit in linear-nearest-neighbor connectivity
# Reference state is not included so we can compare with unitary formed from ansatz operators only
lnn_qc_no_ref, _, layout = data.get_lnn_circuit(pool,apply_border_swaps=True)

# Calculate process fidelity of LNN ansatz circuit with goal unitary
pf_lnn = process_fidelity(Operator(goal_unitary),Operator(lnn_qc_no_ref))
print("Process fidelity of LNN circuit: ",pf_lnn)

# Get the CNOT count of the circuit in LNN connectivity
# Border swaps are irrelevant - they can be applied classically (see below)
lnn_qc_no_swaps_no_ref, acc_cnot_counts, layout = data.get_lnn_circuit(pool,apply_border_swaps=False)
print("Final LNN ansatz circuit (without reference state):\n", lnn_qc_no_swaps_no_ref)
print("LNN Final CNOT counts:", acc_cnot_counts[-1])

# Make sure fidelities are close to 1
assert np.abs(1 - pf) < 10**-5
assert np.abs(1 - pf_lnn) < 10**-5

# Get the energy from the circuit including reference state preparation and swap gates
hamiltonian = jordan_wigner(molecule.get_molecular_hamiltonian())
lnn_qc, _, _ = data.get_lnn_circuit(pool,apply_border_swaps=True,include_ref=True)
energy_swaps = get_circuit_energy(lnn_qc,hamiltonian)
error_swaps = np.abs(energy_swaps-data.evolution.energies[-1])
print("|Energy from circuit - ADAPT-VQE energy| for...\n"
      "* Circuit with swaps, original Hamiltonian: ", error_swaps)

# Make sure error is close to 1
assert error_swaps < 10**-8

# Get the energy from the circuit including reference state preparation but no swap gates
lnn_qc_no_swaps, acc_cnot_counts, layout = data.get_lnn_circuit(pool,apply_border_swaps=False,include_ref=True)
energy_no_swaps = get_circuit_energy(lnn_qc_no_swaps,hamiltonian)
error_no_swaps = np.abs(energy_no_swaps-data.evolution.energies[-1])
print("* Circuit without swaps, original Hamiltonian: ", error_no_swaps)

# Get the energy from the circuit including reference state preparation but no swap gates, but remapping the Hamiltonian
#to account for the nontrivial layout and qubit order

# Remap qubits in the Hamiltonian to account for nontrivial layout
remapped_hamiltonian = remap_hamiltonian_by_layout(hamiltonian,layout)
energy_no_swaps_remapped = get_circuit_energy(lnn_qc_no_swaps,remapped_hamiltonian)
error_no_swaps_remapped = np.abs(energy_no_swaps_remapped-data.evolution.energies[-1])
print("* Circuit without swaps, remapped Hamiltonian:", error_no_swaps_remapped)

# Make sure error is close to 1
assert error_no_swaps_remapped < 10**-8