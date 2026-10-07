import numpy as np
from scipy.special import eval_genlaguerre, gammaln
from wlcore import *
for N in [64, 256, 512]:
    x, V = golub_welsch(N, 1.6)
    print(f"N={N}: ||VV^T-I||={np.abs(V@V.T-np.eye(N)).max():.2e} ||V^TV-I||={np.abs(V.T@V-np.eye(N)).max():.2e}, x_max={x.max():.1f}")
# consistency of sign with small-N reference (V[0,j]>0)
x, V = golub_welsch(40, 1.6); print("min V[0,:] (N=40) =", V[0].min())
al, be, th, s = 0.3, 0.25, 2.7, 1/3
gam = 2*al+1; lam0 = al+1+2*be
print("Discrete vs continuous at nodes, f=sqrt(x)e^{-x/3}:")
for N in [8, 12, 16, 24, 32, 48]:
    x, V = golub_welsch(N, gam)
    w = V[0]**2*np.exp(gammaln(gam+1))
    g = np.sqrt(w)*np.exp(-s*x)
    u = phases(N, al, be)
    Dg = V.T @ (np.exp(-1j*th*u)*(V@g))
    M = 300; m = np.arange(M)
    cc = np.exp(0.5*(gammaln(m+gam+1)-gammaln(m+1)) + m*np.log(s) - (m+gam+1)*np.log(1+s))
    # orthonormal p_n with positive leading coefficient = (-1)^n L_n / norm ; coefficient sign follows
    cc = cc*(-1.0)**m
    P = np.array([(-1)**k*eval_genlaguerre(k, gam, x)*np.exp(-0.5*(gammaln(k+gam+1)-gammaln(k+1))) for k in m])
    cont = np.sqrt(w)*((np.exp(-1j*th*np.sqrt(m+lam0))*cc) @ P)
    print(f"  N={N}: max err = {np.abs(Dg-cont).max():.2e}")
# exactness on polynomials of degree < N
N = 24; x, V = golub_welsch(N, gam); w = V[0]**2*np.exp(gammaln(gam+1))
coef = np.random.default_rng(1).normal(size=N)
P = np.array([(-1)**k*eval_genlaguerre(k, gam, x)*np.exp(-0.5*(gammaln(k+gam+1)-gammaln(k+1))) for k in range(N)])
g = np.sqrt(w)*(coef*np.array([(-1)**k for k in range(N)]))@P
print("Exactness on degree<N polynomial: coefficient recovery err =", np.abs(V@g - coef*np.array([(-1)**k for k in range(N)])).max())
R = rp_matrix(256, al, be, th)
print("RP: ||R R^T - I|| =", np.abs(R@R.T-np.eye(256)).max(), " real:", np.isrealobj(R))
R2 = rp_matrix(256, al, be, 1.1); R3 = rp_matrix(256, al, be, th+1.1)
print("RP group law ||R(th)R(1.1)-R(th+1.1)|| =", np.abs(R@R2-R3).max())
