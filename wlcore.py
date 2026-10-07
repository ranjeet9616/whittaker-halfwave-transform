import numpy as np
from scipy.special import gammaln

def golub_welsch(N, gam):
    """Nodes x_j and orthogonal V with V[n,j] = sqrt(w_j) p_n(x_j), p_n orthonormal
    w.r.t. x^gam e^{-x} with positive leading coefficient; sign fixed robustly."""
    k = np.arange(N)
    a = 2*k + gam + 1.0
    b = np.sqrt(np.arange(1, N)*(np.arange(1, N) + gam))
    J = np.diag(a) + np.diag(b, 1) + np.diag(b, -1)
    x, V = np.linalg.eigh(J)
    # sign of p_n(x_j) for the index of largest |V[n,j]|, via rescaled recurrence
    kmax = np.argmax(np.abs(V), axis=0)
    sgn = np.empty(N)
    for j in range(N):
        prev, cur = 0.0, 1.0           # p_{-1}, p_0 (scaled)
        target = kmax[j]
        for n in range(target):
            nxt = ((x[j]-a[n])*cur - (b[n-1] if n > 0 else 0.0)*prev)/b[n]
            prev, cur = cur, nxt
            if abs(cur) > 1e100:
                prev, cur = prev*1e-100, cur*1e-100
        sgn[j] = np.sign(cur)*np.sign(V[target, j])
    return x, V*sgn

def phases(N, alpha, beta):
    return np.sqrt(np.arange(N) + alpha + 1 + 2*beta)

def halfwave_matrix(N, alpha, beta, theta):
    x, V = golub_welsch(N, 2*alpha+1)
    return V.T @ (np.exp(-1j*theta*phases(N, alpha, beta))[:, None]*V)

def rp_matrix(N, alpha, beta, theta):
    """Reality-preserving half-wave matrix: V^T B V, B = blockdiag rotations by theta*u_{2k}."""
    assert N % 2 == 0
    x, V = golub_welsch(N, 2*alpha+1)
    u = phases(N, alpha, beta)
    B = np.zeros((N, N))
    for k in range(N//2):
        f = theta*u[2*k]
        c, s = np.cos(f), np.sin(f)
        B[2*k:2*k+2, 2*k:2*k+2] = [[c, s], [-s, c]]
    return V.T @ B @ V
