from openfermion import MolecularData, jordan_wigner
from openfermionpyscf import run_pyscf

from adaptvqe.algorithms.adapt_vqe import LinAlgAdapt
from adaptvqe.pools import NoZPauliPool

# Create molecule (linear chain of hydrogens)
r = 3 # Interatomic distance
n_atoms = 3 # Number of atoms
geometry = [['H', [0, 0, r*i]] for i in range(n_atoms)]
basis = 'sto-3g'
multiplicity = 1 + (n_atoms % 2) # 1 (2) if even (odd) number of electrons
charge = 0 # Neutral 
molecule = MolecularData(geometry, basis, multiplicity, charge, description='H2')
molecule = run_pyscf(molecule, run_fci=True)

# Create Pauli pool (individual Pauli strings, weight 2 or 4)
pool = NoZPauliPool(molecule)

# Run ADAPT-VQE
my_adapt = LinAlgAdapt(pool=pool, molecule=molecule, threshold=10**-1)
my_adapt.run()

# Retrieve run data
data = my_adapt.data

# Obtain list of ansatz operators
operators = [pool.get_op(i) for i in data.result.ansatz.indices]
print(f"\nOperator list:\n{operators}")

# Obtain 
qubit_hamiltonian = jordan_wigner(molecule.get_molecular_hamiltonian())
print(f"\n Qubit Hamiltonian:\n",qubit_hamiltonian)