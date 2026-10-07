import numpy as np, mpmath as mp
from scipy.special import eval_genlaguerre, gammaln

print("=== Theorem: beta-sensitivity norm ===")
for (th, c1, c2) in [(2.7, 1.8, 1.9), (3.9, 2.2, 2.2001), (1.0, 1.3, 4.0)]:
    n = np.arange(0, 2_000_000)
    d = np.sqrt(n + c1) - np.sqrt(n + c2)
    sup = np.max(np.abs(np.exp(-1j*th*d) - 1))
    pred = 2*abs(np.sin(th*(np.sqrt(c1)-np.sqrt(c2))/2))
    print(f"theta={th}, c1={c1}, c2={c2}: sup={sup:.15f}  formula={pred:.15f}")

print("=== Theorem: theta-norm = 2 ===")
for dth in [1e-1, 1e-2, 1e-3]:
    n = np.arange(0, 50_000_000, dtype=np.float64)
    v = np.abs(np.exp(-1j*dth*np.sqrt(n+1.8)) - 1)
    print(f"dtheta={dth}: sup over n<5e7 = {v.max():.12f}")

print("=== Theorem: non-closure asymptotics ===")
mp.mp.dps = 40
th, ph, c1, c2 = mp.mpf('2.7'), mp.mpf('1.3'), mp.mpf('1.8'), mp.mpf('2.6')
psi = th+ph; c = (th*c1+ph*c2)/psi
b = -th*ph*(c1-c2)**2/(8*psi)
for N in [10**2, 10**4, 10**6, 10**8]:
    g = th*mp.sqrt(N+c1)+ph*mp.sqrt(N+c2)-psi*mp.sqrt(N+c)
    print(f"n={N}: n^1.5 g(n) = {mp.nstr(N**mp.mpf(1.5)*g, 12)}   predicted b = {mp.nstr(b, 12)}")

print("=== Gauss-Laguerre (Golub-Welsch) discretisation ===")
from wlcore import golub_welsch   # robust sign convention (works for N >= 256)
for N in [64, 256, 512]:
    x, V = golub_welsch(N, 1.6)
    print(f"N={N}: ||V V^T - I||_max = {np.abs(V@V.T-np.eye(N)).max():.2e}, ||V^T V - I||_max = {np.abs(V.T@V-np.eye(N)).max():.2e}")
# check V entries vs orthonormal Laguerre values
N, gam = 32, 1.6
x, V = golub_welsch(N, gam)
w = V[0, :]**2 * np.exp(gammaln(gam+1))   # Gauss weights
n = 5
pn = eval_genlaguerre(n, gam, x)*np.exp(-0.5*(gammaln(n+gam+1)-gammaln(n+1)))
sgn = np.sign(pn[0]*V[n, 0])
print("V[n,j] vs sqrt(w_j) p_n(x_j) (n=5) max diff:", np.abs(V[n, :]*sgn - np.sqrt(w)*pn).max(), " sign", sgn)
print("sum of weights vs Gamma(gam+1):", w.sum(), np.exp(gammaln(gam+1)))

print("=== Discrete vs continuous half-wave on f = sqrt(x) e^{-s x} ===")
al, be, th, s = 0.3, 0.25, 2.7, 1/3
gam = 2*al+1; lam0 = al+1+2*be
def cont_coeffs(M):    # orthonormal coefficients of f_Lag(x)=e^{-s x} in L^2(x^gam e^{-x})
    m = np.arange(M)
    logc = gammaln(m+gam+1)-gammaln(m+1) + m*np.log(s) - (m+gam+1)*np.log(1+s)
    return np.exp(logc - 0.5*(gammaln(m+gam+1)-gammaln(m+1)))
for N in [16, 32, 64, 128]:
    x, V = golub_welsch(N, gam)
    w = V[0, :]**2*np.exp(gammaln(gam+1))
    g = np.sqrt(w)*np.exp(-s*x)
    u = np.sqrt(np.arange(N)+lam0)
    Dg = V.T @ (np.exp(-1j*th*u)*(V @ g))
    # continuous: W_theta f evaluated at nodes, in node-vector convention
    M = 400
    cc = cont_coeffs(M); uu = np.sqrt(np.arange(M)+lam0)
    P = np.array([eval_genlaguerre(m, gam, x)*np.exp(-0.5*(gammaln(m+gam+1)-gammaln(m+1))) for m in range(M)])
    cont = np.sqrt(w)*((np.exp(-1j*th*uu)*cc) @ P)
    print(f"N={N}: max |discrete - continuous| at nodes = {np.abs(Dg-cont).max():.2e}")
