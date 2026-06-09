"""HV-2b: integrate the CODE's exact Phi integrand (threshold.py:131-141) with mpmath
(removing the float-overflow failure) to quantify the wrong-numerator error."""
import mpmath as mp
mp.mp.dps = 30

R0 = lambda y: y / mp.expm1(y)
NUM = lambda y: y**2 * mp.exp(y) / mp.expm1(y)**2   # R0 - y R0'  (correct Reuter numerator)
P = lambda y: y + R0(y)

def Phi(p, n, w):
    return mp.quad(lambda y: y**(n-1)*NUM(y)/(P(y)+w)**p, [mp.mpf("1e-12"),1,10,60])/mp.gamma(n)
def Phit(p, n, w):
    return mp.quad(lambda y: y**(n-1)*R0(y)/(P(y)+w)**p, [mp.mpf("1e-12"),1,10,60])/mp.gamma(n)

# CODE integrand, threshold.py lines 131-141 with k=1 (y=z):
#   r = 1/(e^y - 1); dr = -e^y/(e^y-1)^2; R = z r; dR_dt = 2 z (r - y dr)
def code_phi(p, n, w):
    def f(z):
        y = z
        r = 1/mp.expm1(y)
        dr = -mp.exp(y)/mp.expm1(y)**2
        Rv = z*r
        dRdt = 2*z*(r - y*dr)
        return z**(n-1)*dRdt/(z + Rv + w)**p
    return mp.quad(f, [mp.mpf("1e-12"),1,10,60])/mp.gamma(n)

print(f"{'p,n,w':14s} {'code-integrand':>16s} {'truth Phi':>14s} {'ratio':>8s} {'2Phi+2Phit':>14s}")
for (p,n,w) in [(1,1,0.0),(1,1,0.5),(1,1,-0.3),(2,1,0.0),(2,1,0.5),(1,2,0.0),(2,2,0.5),(1,3,0.0),(2,3,0.0)]:
    c = code_phi(p,n,mp.mpf(w)); t = Phi(p,n,mp.mpf(w)); pred = 2*t + 2*Phit(p,n,mp.mpf(w))
    print(f"({p},{n},{w:+.1f})    {mp.nstr(c,12):>16s} {mp.nstr(t,10):>14s} {mp.nstr(c/t,6):>8s} "
          f"{mp.nstr(pred,10):>14s}  code==2Phi+2Phit: {abs(c-pred) < 1e-12}")
