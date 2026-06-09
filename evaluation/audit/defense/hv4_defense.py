"""HV-4 defense: exhaustively test alternative conventions for the S^4 transverse
vector degeneracy formula (2l+3)(l+2)(l+1)/2."""
import itertools

code = lambda l: (2*l+3)*(l+2)*(l+1)//2 if l >= 1 else 0

# Candidate legitimate towers on spheres (representation theory; integer facts):
def rubin_ordonez_transverse(N, l):  # transverse vector on S^N
    from math import factorial
    if l < 1: return 0
    return l*(l+N-1)*(2*l+N-1)*factorial(l+N-3)//(factorial(N-2)*factorial(l+1))

def scalar_SN(N, l):
    from math import factorial
    return (2*l+N-1)*factorial(l+N-2)//(factorial(N-1)*factorial(l))

def longitudinal_S4(l):  # gradients of scalar harmonics, l>=1
    return scalar_SN(4, l) if l >= 1 else 0

def total_vector_S4(l):  # transverse + longitudinal at level l (NOT same eigenvalue)
    return rubin_ordonez_transverse(4, l) + longitudinal_S4(l)

towers = {
    "transverse S^4 (Rubin-Ordonez)": lambda l: rubin_ordonez_transverse(4, l),
    "transverse S^5": lambda l: rubin_ordonez_transverse(5, l),
    "transverse S^3": lambda l: rubin_ordonez_transverse(3, l),
    "longitudinal S^4": longitudinal_S4,
    "total vector S^4": total_vector_S4,
    "scalar S^4": lambda l: scalar_SN(4, l),
    "scalar S^5": lambda l: scalar_SN(5, l),
    "3 x scalar S^4": lambda l: 3*scalar_SN(4, l),
    "4 x scalar S^4": lambda l: 4*scalar_SN(4, l),
}

print("code tower l=1..6:", [code(l) for l in range(1, 7)])
print("\nMatching attempts (including label offsets l -> l+s, s in -2..2):")
for name, f in towers.items():
    for s in range(-2, 3):
        vals = [f(l+s) for l in range(1, 7)]
        if vals == [code(l) for l in range(1, 7)]:
            print(f"  MATCH: {name} with offset l->l+{s}")
            break
    else:
        print(f"  no match: {name}  (offset-0 values: {[f(l) for l in range(1,7)]})")

# Line-100 formula check: is (N-1)(2l+N-1)(l+N-2)!/((N-1)! l!) ever Rubin-Ordonez?
from math import factorial
line100 = lambda N, l: (N-1)*(2*l+N-1)*factorial(l+N-2)//(factorial(N-1)*factorial(l))
print("\nLine-100 'cited' formula with N=4 vs (N-1) x scalar_S4:")
print("  line100(4,l):", [line100(4, l) for l in range(1, 7)])
print("  3*scalar_S4 :", [3*scalar_SN(4, l) for l in range(1, 7)])
print("  -> line-100 formula IS algebraically (N-1) x scalar deg, not any transverse formula")

# Killing vector anchor (zero convention freedom):
print("\nKilling vectors of S^4 = dim SO(5) =", 5*4//2, "; code l=1 =", code(1))

# Hodge vs Bochner vs Lichnerowicz only shifts eigenvalues, never degeneracies.
# Internal consistency: sum over symmetric-tensor decomposition at level l must
# reproduce dim of (l,2)+(l,1)+(l,0)... check completeness with TRUE formulas:
TT = lambda l: 5*(l-1)*(l+4)*(2*l+3)//6 if l >= 2 else 0
print("\nCompleteness check, symmetric 2-tensor modes on S^4 (true towers):")
print("  10 x scalar(l) (naive component count) vs TT + vec-derived + 2 scalars per level")
for l in range(2, 6):
    lhs = 10*scalar_SN(4, l)
    rhs = TT(l) + rubin_ordonez_transverse(4, l) + 2*scalar_SN(4, l)
    print(f"  l={l}: TT={TT(l)} (code TT identical), code vec={code(l)} vs true {rubin_ordonez_transverse(4,l)}")
