import numpy as np, hashlib
from wlcore import rp_matrix, golub_welsch, phases
from skimage import data
def load(name, N):
    im = getattr(data, name)(); f = im.shape[0]//N
    return np.round(im[:f*N,:f*N].reshape(N,f,N,f).mean(axis=(1,3))).astype(np.uint8)
KEY = dict(a1=0.3, b1=0.25, t1=2.7, seed="NMIMS-2026")
def perm(N, seed):
    s = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], 'big')
    return np.random.default_rng(s).permutation(N*N)
P = load('camera', 256); N = 256
X = P.reshape(-1)[perm(N, KEY['seed'])].reshape(N, N).astype(float)
a, b, t = KEY['a1'], KEY['b1'], KEY['t1']; lam0 = a+1+2*b
x, V = golub_welsch(N, 2*a+1); u = phases(N, a, b)
R = rp_matrix(N, a, b, t)
print("delta | measured rel.err (beta) | bound 2*theta*delta/(sqrt(l0+2d)+sqrt(l0)) | measured rel.err (theta) | bound delta*||sqrt(T)X||/||X||")
for d in [1e-2, 1e-4, 1e-6]:
    Rb = rp_matrix(N, a, b+d, t); Rt = rp_matrix(N, a, b, t+d)
    eb = np.linalg.norm((Rb-R)@X)/np.linalg.norm(X)
    et = np.linalg.norm((Rt-R)@X)/np.linalg.norm(X)
    bb = 2*t*d/(np.sqrt(lam0+2*d)+np.sqrt(lam0))
    # discrete analogue of ||sqrt(T) X||: rotation angle rates u_{2k} applied to both members of each pair
    rate = np.repeat(u[0::2], 2)
    bt = d*np.linalg.norm(rate[:, None]*(V@X))/np.linalg.norm(X)
    print(f"{d:.0e} | {eb:.3e} | {bb:.3e} | {et:.3e} | {bt:.3e}")
print("operator norm ||R(beta+1e-2)-R(beta)|| =", np.linalg.norm(rp_matrix(N,a,b+1e-2,t)-R, 2))
print("operator norm ||R(theta+1e-2)-R(theta)|| =", np.linalg.norm(rp_matrix(N,a,b,t+1e-2)-R, 2), " (finite N, max angle change", 1e-2*u[-2], ")")
