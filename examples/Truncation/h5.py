from adaptvqe.molecules import create_h5
from adaptvqe.pools import CEO, PairedDoubleCEO
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt

r = 0.1
mol = create_h5(r)

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
print(f"Energy error: {err:5.4e}")
