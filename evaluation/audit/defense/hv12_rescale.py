import sympy
from sympy import Symbol, Rational, pi, simplify, nsimplify

lam = Symbol("lambda", real=True)
x = 1/(1 - 2*lam)

# EH builder
N_eh = Rational(4,3)*x + 8 - 8*x**2
D_eh = Rational(1,3)*x + 2 - 2*x**2

# anomalous_dim path (Litim closed forms substituted)
A_m = Rational(1,3)/pi * (5*x - 4 + 6*x**2)
B_m = -Rational(1,6)/pi * (Rational(5,2)*x - 2 + 3*x**2)

# literature (Reuter d=4 Litim): A=(1/3pi)(5x-9x^2-7), B=-(1/12pi)(5x-6x^2)
A_lit = Rational(1,3)/pi*(5*x - 9*x**2 - 7)
B_lit = -Rational(1,12)/pi*(5*x - 6*x**2)

# Is A_m = c * N_eh for constant c?  (then lambda*, theta preserved under g->g/c)
r1 = simplify(A_m / N_eh)
r2 = simplify(B_m / D_eh)
print("A_m/N_eh constant?", r1.free_symbols == set(), "->", r1)
print("B_m/D_eh constant?", r2.free_symbols == set(), "->", r2)
r3 = simplify(A_m / A_lit); r4 = simplify(B_m / B_lit)
print("A_m/A_lit constant?", r3.free_symbols == set())
print("sign of x^2 coefficient: A_m: +6/(3pi), A_lit: -9/(3pi), N_eh: -8  (TT-sector term, gauge-independent)")
