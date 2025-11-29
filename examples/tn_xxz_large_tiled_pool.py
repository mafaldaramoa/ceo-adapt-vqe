# -*- coding: utf-8 -*-

from time import perf_counter_ns
import pandas as pd
from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt, TensorNetAdapt
from adaptvqe.hamiltonians import XXZHamiltonian

dmrg_mpo_bond = 100
dmrg_mps_bond = 20

l = 5
j_xy = 1
j_z = 1
h = XXZHamiltonian(j_xy, j_z, l, diag_mode="quimb", max_mpo_bond=dmrg_mpo_bond, max_mps_bond=dmrg_mps_bond)
pool = FullPauliPool(n=l)

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

tiled_energies = {}
tiled_runtimes = {}
tiled_dmrg_energies = {}
system_sizes = [5 * i for i in range(1, 6)]
for new_l in system_sizes:
    # Tile the operators to form a pool for a larger problem instance and run ADAPT-VQE with this pool
    source_ops = [pool.operators[index].operator for index in ixs]
    tiled_pool = TiledPauliPool(n=new_l, source_ops=source_ops)

    new_h = XXZHamiltonian(j_xy, j_z, new_l, diag_mode="quimb", max_mpo_bond=dmrg_mpo_bond, max_mps_bond=dmrg_mps_bond)
    tiled_dmrg_energies[new_l] = new_h.ground_energy

    start_time = perf_counter_ns()
    my_adapt = TensorNetAdapt(
        pool=tiled_pool,
        custom_hamiltonian=new_h,
        verbose=True,
        threshold=10**-2,
        max_adapt_iter=50,
        max_opt_iter=10000,
        sel_criterion="gradient",
        recycle_hessian=False,
        max_mpo_bond = 100,
        max_mps_bond = 20
    )
    my_adapt.run()
    end_time = perf_counter_ns()
    elapsed_time = end_time - start_time
    tiled_runtimes[new_l] = elapsed_time
    tiled_energies[new_l] = my_adapt.data.result.energy
    error = (new_h.ground_energy - my_adapt.data.result.energy) / new_h.ground_energy
    print("Final relative error:", error)

# Output to csv.
records = []
for l in system_sizes:
    records.append((l, tiled_dmrg_energies[l], tiled_energies[l], tiled_runtimes[l]))
df = pd.DataFrame.from_records(records, columns=["l", "dmrg_energy", "adapt_energy", "runtime"])
df.to_csv("../data/tn_stress_results.csv")
