
from openfermion import hermitian_conjugated, jordan_wigner, FermionOperator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from adaptvqe.circuits import pauli_exp_circuit, fe_circuit, qe_circuit, cnot_count, transpile_lnn
from adaptvqe.utils import remove_z_string
from adaptvqe.op_conv import get_qasm

n = 100
source_orbs = [0]
target_orbs = [n-1]
coefficient = 1
f_exc = FermionOperator(f"{source_orbs[0]} {target_orbs[0]}")
f_exc = (f_exc - hermitian_conjugated(f_exc))/2
f_exc = jordan_wigner(f_exc)

qc_fp = pauli_exp_circuit(f_exc, n, revert_endianness=True)
qc_fcr = fe_circuit(source_orbs, target_orbs, coefficient, n, True)
qc_qp = pauli_exp_circuit(remove_z_string(f_exc), n, revert_endianness=True)
qc_qcr = qe_circuit(source_orbs, target_orbs, coefficient, n, True)

# Create the pass manager with the linear coupling map and the desired optimization level and basis gates
pass_manager = generate_preset_pass_manager(3, basis_gates=["rz","cx","x","sx","h","s"], approximation_degree=1)

for label,circuit in zip(["FE, Pauli","Fe, CR","QE, Pauli","QE, CR"],[qc_fp, qc_fcr, qc_qp, qc_qcr]):
    print(label)
    print("All to all, no optimization:", cnot_count(get_qasm(circuit)))
    opt_circuit = pass_manager.run(circuit)
    print("All to all, max optimization:", cnot_count(get_qasm(opt_circuit)))
    t_qc, _ = transpile_lnn(circuit, opt_level=0, initial_layout=range(n),apply_border_swaps=True)
    print("Transpiled, fixed final layout, no optimization: ",cnot_count(get_qasm(t_qc)))
    t_qc, _ = transpile_lnn(circuit, opt_level=3, initial_layout=range(n),apply_border_swaps=True)
    print("Transpiled, fixed final layout, max optimization: ",cnot_count(get_qasm(t_qc)))
    t_qc, _ = transpile_lnn(circuit, opt_level=0, initial_layout=range(n))
    print("Transpiled, arbitrary final layout, no optimization: ",cnot_count(get_qasm(t_qc)))
    t_qc, _ = transpile_lnn(circuit, opt_level=3, initial_layout=range(n))
    print("Transpiled, arbitrary final layout, max optimization: ",cnot_count(get_qasm(t_qc)))
    print("\n")