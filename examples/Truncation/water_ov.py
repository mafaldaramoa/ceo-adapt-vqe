from time import perf_counter_ns

from openfermion.chem import MolecularData, geometry_from_pubchem
from openfermionpyscf import run_pyscf

from adaptvqe.pools import CEO, PairedDoubleCEO, OccupiedVirtualCEO
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt

geometry = geometry_from_pubchem("water")
basis = 'sto-3g'
multiplicity = 1
charge = 0
mol = MolecularData(geometry, basis, multiplicity, charge, description='BeH2')
mol = run_pyscf(mol, run_fci=True, run_ccsd=True)
nelec = mol.n_electrons

start_time = perf_counter_ns()
ceo_pool = CEO(mol)
end_time = perf_counter_ns()
elapsed_time_ceo_pool = abs(end_time - start_time)

start_time = perf_counter_ns()
ov_pool = OccupiedVirtualCEO(mol, n_occ=nelec)
end_time = perf_counter_ns()
elapsed_time_ov_pool = abs(end_time - start_time)

print("CEO pool:")
for op in ceo_pool.operators:
    print(op.operator, "\n")
print("OV CEO pool:")
for op in ov_pool.operators:
    print(op.operator, "\n")

start_time = perf_counter_ns()
ceo_adapt = LinAlgAdapt(
    pool=ceo_pool,
    molecule=mol,
    max_adapt_iter=1,
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
ov_adapt = LinAlgAdapt(
    pool=ov_pool,
    molecule=mol,
    max_adapt_iter=1,
    recycle_hessian=True,
    tetris=True,
    verbose=True,
    threshold=0.1,
)

ov_adapt.run()
ov_energy = ov_adapt.energy
end_time = perf_counter_ns()
elapsed_time_ov_adapt = abs(end_time - start_time)

print("Sizes of pools:")
print(len(ceo_pool.operators), len(ov_pool.operators))

print("Times for regular CEO pool:")
print(f"Building pool: {elapsed_time_ceo_pool:5.4e}\nRun: {elapsed_time_ceo_adapt:5.4e}")
print("Times for paired CEO pool:")
print(f"Building pool: {elapsed_time_ov_pool:5.4e}\nRun: {elapsed_time_ov_adapt:5.4e}")

err = abs(ov_energy - ceo_energy)
print(f"Energy error: {err:5.4e}")
fci_err = abs(mol.fci_energy - ov_energy)
print(f"Error wrt FCI: {fci_err:5.4e}")
