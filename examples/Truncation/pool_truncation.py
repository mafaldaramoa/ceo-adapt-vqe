from adaptvqe.utils import create_excitations

p = 5
q = 0
r = 0
s = 5
operators, orbs = create_excitations(p, q, r, s, fermionic=True)
print(operators)