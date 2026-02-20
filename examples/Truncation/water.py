from openfermion.chem import MolecularData, geometry_from_pubchem
from openfermionpyscf import run_pyscf

from adaptvqe.pools import CEO, PairedDoubleCEO
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt

geometry = geometry_from_pubchem("water")
basis = 'sto-3g'
multiplicity = 1
charge = 0
mol = MolecularData(geometry, basis, multiplicity, charge, description='BeH2')
mol = run_pyscf(mol, run_fci=True, run_ccsd=True)

ceo_pool = CEO(mol)
paired_pool = PairedDoubleCEO(mol)

print("CEO pool:")
for op in ceo_pool.operators:
    print(op.operator, "\n")
print("paired CEO pool:")
for op in paired_pool.operators:
    print(op.operator, "\n")

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

print("Sizes of pools:")
print(len(ceo_pool.operators), len(paired_pool.operators))

err = abs(paired_energy - ceo_energy)
print(f"CEO vs. paired CEO energy error: {err:5.4e}")

exact_err = abs(mol.fci_energy - ceo_energy)
print(f"Exact vs. CEO energy error: {exact_err:5.4e}")
