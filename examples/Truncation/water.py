from openfermion.chem import MolecularData, geometry_from_pubchem
from openfermionpyscf import run_pyscf

from time import perf_counter_ns
from adaptvqe.pools import CEO, PairedDoubleCEO
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt

geometry = geometry_from_pubchem("water")
basis = 'sto-3g'
multiplicity = 1
charge = 0
mol = MolecularData(geometry, basis, multiplicity, charge, description='BeH2')
mol = run_pyscf(mol, run_fci=True, run_ccsd=True)

start_time = perf_counter_ns()
ceo_pool = CEO(mol)
end_time = perf_counter_ns()
elapsed_time_ceo_pool = abs(end_time - start_time)

start_time = perf_counter_ns()
paired_pool = PairedDoubleCEO(mol)
end_time = perf_counter_ns()
elapsed_time_paired_pool = abs(end_time - start_time)

print("CEO pool:")
for op in ceo_pool.operators:
    print(op.operator, "\n")
print("paired CEO pool:")
for op in paired_pool.operators:
    print(op.operator, "\n")

start_time = perf_counter_ns()
ceo_adapt = LinAlgAdapt(
    pool=ceo_pool,
    molecule=mol,
    max_adapt_iter=10,
    recycle_hessian=True,
    tetris=True,
    verbose=True,
    threshold=0.1,
)

ceo_adapt.run()
ceo_energy = ceo_adapt.energy
end_time = perf_counter_ns()
elapsed_time_ceo_adapt = abs(end_time - start_time)

start_time = perf_counter_ns()
paired_adapt = LinAlgAdapt(
    pool=paired_pool,
    molecule=mol,
    max_adapt_iter=10,
    recycle_hessian=True,
    tetris=True,
    verbose=True,
    threshold=0.1,
)

paired_adapt.run()
paired_energy = paired_adapt.energy
end_time = perf_counter_ns()
elapsed_time_paired_adapt = abs(end_time - start_time)

print("Sizes of pools:")
print(len(ceo_pool.operators), len(paired_pool.operators))

print("Times for regular CEO pool:")
print(f"Building pool: {elapsed_time_ceo_pool:5.4e}\nRun: {elapsed_time_paired_adapt:5.4e}")
print("Times for paired CEO pool:")
print(f"Building pool: {elapsed_time_paired_pool:5.4e}\nRun: {elapsed_time_paired_adapt:5.4e}")

err = abs(paired_energy - ceo_energy)
print(f"CEO vs. paired CEO energy error: {err:5.4e}")

exact_err = abs(mol.fci_energy - ceo_energy)
print(f"Exact vs. CEO energy error: {exact_err:5.4e}")
