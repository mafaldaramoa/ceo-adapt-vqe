from adaptvqe.molecules import create_h2
from adaptvqe.pools import CEO, PairedDoubleCEO, OccupiedVirtualCEO
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt

r = 0.1
mol = create_h2(r)
nelec = mol.n_electrons

ceo_pool = CEO(mol)
ov_pool = OccupiedVirtualCEO(mol, n_occ=nelec)

print("CEO pool:")
for op in ceo_pool.operators:
    print(op.operator, "\n")
print("OV CEO pool:")
for op in ov_pool.operators:
    print(op.operator, "\n")

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

err = abs(ov_energy - ceo_energy)
print(f"Energy error: {err:5.4e}")
fci_err = abs(mol.fci_energy - ov_energy)
print(f"Error wrt FCI: {fci_err:5.4e}")
