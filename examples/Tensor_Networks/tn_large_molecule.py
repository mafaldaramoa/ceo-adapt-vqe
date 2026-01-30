import argparse
from time import perf_counter_ns
import h5py
import numpy as np
from quimb.tensor.tensor_dmrg import DMRG
from adaptvqe.molecules import create_h7
from adaptvqe.pools import DVE_CEO
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt

parser = argparse.ArgumentParser()
parser.add_argument("max_bond", type=int, help="Maximum bond dimension of the MPS.")
args = parser.parse_args()
mps_max_bond = args.max_bond

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
    max_mps_bond=mps_max_bond
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
times = []
num_iter = 15
energies.append(my_adapt.energy)
for _ in range(num_iter):
    start_time = perf_counter_ns()
    my_adapt.run_iteration()
    end_time = perf_counter_ns()
    elapsed_time = float(abs(end_time - start_time))
    energies.append(my_adapt.energy)
    times.append(elapsed_time)

f = h5py.File("h7_out.hdf5", "w")
f.create_dataset("dmrg_energy", data=dmrg_energy)
f.create_dataset("energies", data=np.array(energies))
f.create_dataset("times", data=np.array(times))
f.close()