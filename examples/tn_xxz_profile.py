import cProfile
from typing import Dict
from time import perf_counter_ns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cirq
import openfermion as of
import quimb.tensor as qtn
from adaptvqe.pools import FullPauliPool, TiledPauliPool
from adaptvqe.algorithms.adapt_vqe import TensorNetAdapt, LinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.tensor_helpers import pauli_sum_to_mpo

def main():
    max_mpo_bond = 100
    max_mps_bond = 30
    l = 6
    dmrg_energies: Dict[int, float] = {}

    j_xy = 1
    j_z = 1
    h = XXZHamiltonian(j_xy, j_z, l)
    qs = cirq.LineQubit.range(l)
    h_cirq = of.transforms.qubit_operator_to_pauli_sum(h.operator)
    h_mpo = pauli_sum_to_mpo(h_cirq, qs, max_mpo_bond)

    # DMRG
    dmrg = qtn.DMRG(h_mpo, max_mps_bond)
    converged = dmrg.solve()
    if not converged:
        print("DRMG did not converge.")
    ground_energy = dmrg.energy
    print(f"At size {l} got energy {ground_energy}")
    dmrg_energies[l] = ground_energy

    # ADAPT
    start_time = perf_counter_ns()
    pool = FullPauliPool(n=l, max_mpo_bond=max_mpo_bond)
    tn_adapt = TensorNetAdapt(
        pool=pool,
        custom_hamiltonian=h,
        verbose=True,
        threshold=10**-5,
        max_adapt_iter=30,
        max_opt_iter=10000,
        sel_criterion="gradient",
        recycle_hessian=False,
        rand_degenerate=True,
        max_mpo_bond=max_mpo_bond,
        max_mps_bond=max_mps_bond
    )
    tn_adapt.run()
    end_time = perf_counter_ns()
    elapsed_time = end_time - start_time
    print(f"For l={l} got energy {tn_adapt.energy} in {elapsed_time:4.5e} ns.")

if __name__ == "__main__":
    cProfile.run('main()')