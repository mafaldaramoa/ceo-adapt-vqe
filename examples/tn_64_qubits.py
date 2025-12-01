from time import perf_counter_ns
import argparse
import h5py
import numpy as np
from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt, LinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian

parser = argparse.ArgumentParser()
parser.add_argument("nqubits", type=int, help="Number of qubits/spins.")
parser.add_argument("output_file", type=str, help="HDF5 output file.")
args = parser.parse_args()

n_iter = 4
max_mpo_bond = 100
dmrg_mps_bond = 5 
adapt_mps_bond = 5
l = 4

j_xy = 1
j_z = 1
h = XXZHamiltonian(j_xy, j_z, l, diag_mode="quimb", max_mpo_bond=max_mpo_bond, max_mps_bond=dmrg_mps_bond)
dmrg_energy = h.ground_energy
print(f"Got DMRG energy {dmrg_energy:4.5e}")
pool = FullPauliPool(n=l, max_mpo_bond=max_mpo_bond)

# Run 200 iterations of ADAPT-VQE for small problem instance, selecting randomly among degenerate gradients.
# Form a list of all unique operators ever selected for this small instance.
ixs = []
for _ in range(100):
    my_adapt = TensorNetAdapt(
        pool=pool,
        custom_hamiltonian=h,
        verbose=False,
        threshold=10**-5,
        max_adapt_iter=5,
        max_opt_iter=10000,
        sel_criterion="gradient",
        recycle_hessian=False,
        rand_degenerate=True,
        max_mpo_bond=100,
        max_mps_bond = 20
    )
    my_adapt.run()
    data = my_adapt.data
    for i in data.result.ansatz.indices:
        if i not in ixs:
            ixs.append(i)

print(f"Pool will be tiled from {len(ixs)} ops")
source_ops = [pool.operators[index].operator for index in ixs]

# Now go to the larger size.
new_l = args["nqubits"]
j_xy = 1
j_z = 1
h = XXZHamiltonian(
    j_xy, j_z, new_l,
    store_ref_vector=False,
    diag_mode="quimb", max_mpo_bond=max_mpo_bond, max_mps_bond=dmrg_mps_bond
)
dmrg_energy = h.ground_energy
print(f"Got DMRG energy {dmrg_energy:4.5e}")
tiled_pool = TiledPauliPool(n=new_l, source_ops=source_ops)

tn_adapt = TensorNetAdapt(
    pool=tiled_pool,
    custom_hamiltonian=h,
    verbose=True,
    threshold=10**-5,
    max_adapt_iter=30,
    max_opt_iter=10000,
    sel_criterion="gradient",
    recycle_hessian=False,
    rand_degenerate=True,
    max_mpo_bond=max_mpo_bond,
    max_mps_bond=adapt_mps_bond
)
tn_adapt.initialize()

adapt_energies = []
adapt_times = []
for _ in range(n_iter):
    start_time = perf_counter_ns()
    tn_adapt.run_iteration()
    end_time = perf_counter_ns()
    elapsed_time = end_time - start_time
    adapt_energies.append(tn_adapt.energy)
    adapt_times.append(elapsed_time)

adapt_energies = np.array(adapt_energies)
adapt_times = np.array(adapt_times)

f = h5py.File(args["output_file"], "w")
f.create_dataset("dmrg_energy", data=dmrg_energy)
f.create_dataset("adapt_energies", data=adapt_energies)
f.create_dataset("adapt_times", data=adapt_times)
f.close()