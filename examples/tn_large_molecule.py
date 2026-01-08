import h5py
import numpy as np
from quimb.tensor.tensor_dmrg import DMRG
from adaptvqe.molecules import create_h7
from adaptvqe.pools import DVE_CEO
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt

r = 1.5
molecule = create_h7(r)
pool = DVE_CEO(molecule)

my_adapt = TensorNetAdapt(
    pool=pool,
    molecule=molecule,
    max_adapt_iter=3,
    recycle_hessian=True,
    tetris=True,
    verbose=True,
    threshold=0.1,
    max_mpo_bond=100,
    max_mps_bond = 20
)
my_adapt.initialize()

# Get the ground state energy with DMRG
ham_mpo = my_adapt.hamiltonian_mpo
dmrg_max_bond = 50
dmrg = DMRG(ham_mpo, dmrg_max_bond)
converged = dmrg.solve(max_sweeps=100)
if not converged:
    print("DMRG did not converge.")
dmrg_energy = dmrg.energy
print(f"Got energy {dmrg_energy}")

energies = []
num_iter = 15
energies.append(my_adapt.energy)
for _ in range(num_iter):
    my_adapt.run_iteration()
    energies.append(my_adapt.energy)

f = h5py.File("h7_out.hdf5", "w")
f.create_dataset("dmrg_energy", data=dmrg_energy)
f.create_dataset("energies", data=np.array(energies))
f.close()