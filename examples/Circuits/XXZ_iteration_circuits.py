from adaptvqe.pools import FullPauliPool
from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.hamiltonians import XXZHamiltonian
from adaptvqe.circuits import get_circuit_energy

l = 4
j_xy = 1
j_z = 1
h = XXZHamiltonian(j_xy, j_z, l)
pool = FullPauliPool(n=l)

my_adapt = LinAlgAdapt(
    pool=pool,
    custom_hamiltonian=h,
    verbose=False,
    threshold=10**-5,
    max_adapt_iter=3,
    max_opt_iter=10000,
    sel_criterion="gradient",
    recycle_hessian=False,
    rand_degenerate=True,
)
my_adapt.run()
data = my_adapt.data

circuits = []
for i,(indices, coefficients) in enumerate(zip(data.evolution.indices,
                                               data.evolution.coefficients)):
    circuit = data.get_circuit(pool,
                               indices,
                               coefficients,
                               include_ref=True)
    circuits.append(circuit)
    assert (data.evolution.energies[i] - 
            get_circuit_energy(circuit,h.operator)
            < 10**-8)
