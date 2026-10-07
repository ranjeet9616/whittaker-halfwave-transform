import numpy as np, time, json, hashlib
from skimage import data
from wlcore import rp_matrix
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

def load(name, N):
    im = getattr(data, name)()
    if im.ndim == 3: im = (0.299*im[...,0]+0.587*im[...,1]+0.114*im[...,2])
    f = im.shape[0]//N
    im = im[:f*N, :f*N].reshape(N, f, N, f).mean(axis=(1, 3))
    return np.round(im).astype(np.uint8)

KEY = dict(a1=0.3, b1=0.25, t1=2.7, a2=0.7, b2=0.6, t2=3.9, seed="NMIMS-2026")
def perm(N, seed):
    s = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], 'big')
    return np.random.default_rng(s).permutation(N*N)
def mats(N, k):
    return rp_matrix(N, k['a1'], k['b1'], k['t1']), rp_matrix(N, k['a2'], k['b2'], k['t2'])
def enc(P, k, R=None):
    N = P.shape[0]; R1, R2 = R if R else mats(N, k)
    X = P.reshape(-1)[perm(N, k['seed'])].reshape(N, N).astype(float)
    return R1 @ X @ R2.T
def dec(C, k, R=None):
    N = C.shape[0]; R1, R2 = R if R else mats(N, k)
    X = (R1.T @ C @ R2).reshape(-1)
    out = np.empty_like(X); out[perm(N, k['seed'])] = X
    return out.reshape(N, N)
def corr(I):
    I = I.astype(float)
    pairs = {'H': (I[:, :-1], I[:, 1:]), 'V': (I[:-1, :], I[1:, :]), 'D': (I[:-1, :-1], I[1:, 1:])}
    return {k: float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) for k, (a, b) in pairs.items()}
def quant(C, lo=None, hi=None):
    lo = C.min() if lo is None else lo; hi = C.max() if hi is None else hi
    return np.clip(np.round(255*(C-lo)/(hi-lo)), 0, 255).astype(np.uint8)
def entropy(Q):
    h = np.bincount(Q.ravel(), minlength=256)/Q.size; h = h[h > 0]
    return float(-(h*np.log2(h)).sum())
def psnr(A, B):
    mse = np.mean((A.astype(float)-B.astype(float))**2)
    return float('inf') if mse == 0 else float(10*np.log10(255**2/mse))




res = {}
for name in ['camera', 'astronaut', 'coffee']:
    P = load(name, 256)
    R = mats(256, KEY)
    C = enc(P, KEY, R); D = dec(C, KEY, R)
    Dr = np.clip(np.round(D), 0, 255).astype(np.uint8)
    Q = quant(C)
    # NPCR/UACI: one-pixel change, quantised with the range of the first cipher
    P2 = P.copy(); P2[128, 128] = (int(P2[128, 128]) + 1) % 256
    C2 = enc(P2, KEY, R)
    Q1, Q2 = quant(C, C.min(), C.max()), quant(C2, C.min(), C.max())
    npcr = float(np.mean(Q1 != Q2)*100); uaci = float(np.mean(np.abs(Q1.astype(float)-Q2)/255)*100)
    res[name] = dict(maxerr=float(np.abs(D-P).max()), wrong=int(np.sum(Dr != P)),
                     corr_plain=corr(P), corr_cipher=corr(C), H_plain=entropy(P), H_cipher=entropy(Q),
                     npcr=npcr, uaci=uaci, maxdiff=float(np.abs(C2-C).max()))
    if name == 'camera': Pcam, Ccam, Dcam, Qcam = P, C, Dr, Q
print(json.dumps(res, indent=1))

# key sensitivity (camera, correct permutation, one transform key perturbed)
P = Pcam; C = Ccam
sens = {}
for key in ['a1', 'b1', 't1', 'a2', 'b2', 't2']:
    row = []
    for d in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
        k2 = dict(KEY); k2[key] = KEY[key] + d
        row.append(round(psnr(np.clip(np.round(dec(C, k2)), 0, 255), P), 2))
    sens[key] = row
print("PSNR of wrong-key decryption (dB), delta = 1e-2,1e-4,1e-6,1e-8,1e-10:")
for k, v in sens.items(): print(k, v)
k2 = dict(KEY); k2['seed'] = "NMIMS-2027"
print("wrong permutation seed PSNR:", psnr(np.clip(np.round(dec(C, k2)), 0, 255), P))

# timing
for N in [256, 512]:
    P = load('camera', N)
    t0 = time.perf_counter(); R = mats(N, KEY); t1 = time.perf_counter()
    for _ in range(5): C = enc(P, KEY, R)
    t2 = time.perf_counter()
    print(f"N={N}: matrix build {t1-t0:.3f}s, encryption {(t2-t1)/5*1000:.1f} ms")

# figure
kb = dict(KEY); kb['b1'] = KEY['b1'] + 1e-2
Dwrong = np.clip(np.round(dec(Ccam, kb)), 0, 255)
fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.9), gridspec_kw={'height_ratios': [1.25, 1]})
for a, im, t in zip(ax[0], [Pcam, Qcam, Dcam, Dwrong],
                    ['(a) plain image', '(b) cipher (8-bit view)', '(c) decrypted, correct key', r'(d) decrypted, $\beta_1+10^{-2}$']):
    a.imshow(im, cmap='gray', vmin=0, vmax=255); a.set_title(t, fontsize=8); a.axis('off')
for a, im, t in zip(ax[1][:2], [Pcam, Qcam], ['(e) histogram of (a)', '(f) histogram of (b)']):
    a.hist(im.ravel(), bins=256, range=(0, 255), color='#2a78d6', lw=0)
    a.set_title(t, fontsize=8); a.tick_params(labelsize=6)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
# adjacent-pixel scatter
rng = np.random.default_rng(0); idx = rng.integers(0, 255, size=(3000, 2))
for a, I, t in zip(ax[1][2:], [Pcam.astype(float), Ccam], ['(g) H-neighbours, plain', '(h) H-neighbours, cipher']):
    a.plot(I[idx[:, 0], idx[:, 1]], I[idx[:, 0], idx[:, 1]+1], '.', ms=1, color='#eb6834')
    a.set_title(t, fontsize=8); a.tick_params(labelsize=6)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig('fig_encryption.pdf'); print("figure saved")
