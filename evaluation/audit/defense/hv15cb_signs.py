import sys
import numpy as np
import sympy
sys.path.insert(0, "/root/cdev/exasymptoticsafety/src")
from asymsafety.frg.threshold import ThresholdFunctions
from asymsafety.frg.regulator import LitimRegulator, ExponentialRegulator

lam = sympy.Symbol("lambda", real=True)
w = -2*lam

for name, tf in [("Litim", ThresholdFunctions(LitimRegulator())),
                 ("Exponential", ThresholdFunctions(ExponentialRegulator()))]:
    try:
        A = sympy.Rational(1,3)/sympy.pi*(5*tf.Phi(1,1,w) - 4*tf.Phi(1,1,0) + 6*tf.Phi(2,1,w))
        B = -sympy.Rational(1,6)/sympy.pi*(5*tf.Phi_tilde(1,1,w) - 4*tf.Phi_tilde(1,1,0) + 6*tf.Phi_tilde(2,1,w))
        fA = sympy.lambdify(lam, A, "numpy"); fB = sympy.lambdify(lam, B, "numpy")
        ls = np.linspace(-0.49, 0.49, 25)
        Av = np.array([float(fA(x)) for x in ls]); Bv = np.array([float(fB(x)) for x in ls])
        print(f"{name}: min A_fol on (-0.49,0.49) = {Av.min():.4f}  (A>0 everywhere: {bool((Av>0).all())})")
        print(f"{name}: max B_fol on (-0.49,0.49) = {Bv.max():.4f}  (B<0 everywhere: {bool((Bv<0).all())})")
        print(f"{name}: A(0.2)={float(fA(0.2)):.4f}, B(0.2)={float(fB(0.2)):.4f}")
        # eta = gA/(1-gB) with A>0, B<0 -> eta>0 for all g>0 -> beta_g >= 2g > 0
    except Exception as e:
        print(f"{name}: symbolic path failed ({type(e).__name__}: {e}); trying numerical")
        for func, lbl in [("Phi","A")]:
            pass
        # numerical fallback
        def phi(p,n,x):
            return tf.evaluate_numerical("Phi", p, n, x)
        def phit(p,n,x):
            return tf.evaluate_numerical("Phi_tilde", p, n, x)
        ls = np.linspace(-0.49, 0.49, 13)
        Av=[]; Bv=[]
        for x in ls:
            wv = -2*x
            Av.append((1/3/np.pi)*(5*phi(1,1,wv)-4*phi(1,1,0.0)+6*phi(2,1,wv)))
            Bv.append(-(1/6/np.pi)*(5*phit(1,1,wv)-4*phit(1,1,0.0)+6*phit(2,1,wv)))
        Av=np.array(Av); Bv=np.array(Bv)
        print(f"{name} (numerical): min A={Av.min():.4f} (A>0: {bool((Av>0).all())}), max B={Bv.max():.4f} (B<0: {bool((Bv<0).all())})")
        i02 = np.argmin(abs(ls-0.2))
        print(f"{name} (numerical): A(~0.2)={Av[i02]:.4f}, B(~0.2)={Bv[i02]:.4f}")

# Also: what if eta entered beta_g with the OPPOSITE sign (eta-convention flip)?
# beta_g = (2 - eta)g = 0 -> eta = +2. Solve g*A/(1-gB)=2 with Litim values at lambda=0.2:
tf = ThresholdFunctions(LitimRegulator())
A = sympy.Rational(1,3)/sympy.pi*(5*tf.Phi(1,1,w) - 4*tf.Phi(1,1,0) + 6*tf.Phi(2,1,w))
B = -sympy.Rational(1,6)/sympy.pi*(5*tf.Phi_tilde(1,1,w) - 4*tf.Phi_tilde(1,1,0) + 6*tf.Phi_tilde(2,1,w))
g = sympy.Symbol("g", positive=True)
sol = sympy.solve(sympy.Eq(g*A/(1-g*B), 2), g)
f = sympy.lambdify(lam, sol[0], "numpy")
print("\nHypothetical flipped convention beta_g=(2-eta)g: g*(lambda) where eta=+2:")
for lv in (0.0, 0.1, 0.2, 0.3):
    print(f"  lambda={lv}: g*={float(f(lv)):.4f}")

# Confirm the sole non-Gaussian root and its exponents
import numpy as np
from asymsafety.beta.foliated import build_foliated_eh_beta_system
system = build_foliated_eh_beta_system(d=4)
pt = {"g": 16.277563, "lambda": -0.671226, "lambda_ADM": 1.0}
print("\nresidual at (16.278, -0.671, 1.0):", system.evaluate(pt))
M = system.jacobian_numerical(pt)
print("theta_i =", np.round(-np.linalg.eigvals(M), 3))
