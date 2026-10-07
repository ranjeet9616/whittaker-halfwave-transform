"""Figure 1 of the manuscript: orthonormal kernel functions and spectral lattice."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre, gammaln
al = 0.3; g = 2*al + 1
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']; LS = ['-', '--', '-.', ':']
x = np.linspace(1e-4, 14, 1200)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={'width_ratios': [1.6, 1]})
for n in range(4):
    lognorm = 0.5*(gammaln(n+g+1) - gammaln(n+1))
    e = np.sqrt(x)*eval_genlaguerre(n, g, x)*x**al*np.exp(-x/2)/np.exp(lognorm)
    a1.plot(x, e, color=C[n], lw=1.6, ls=LS[n], label=f'n = {n}')
a1.axhline(0, color='#c9c8c3', lw=0.8); a1.set_xlabel('x')
a1.set_ylabel(r'$e_n(x)\sqrt{w(x)}$'); a1.legend(frameon=False, fontsize=8, loc='lower right')
a1.set_title(r'Orthonormal kernel functions, $\alpha=0.3$', fontsize=9)
for be, c, m, j in [(0.0, C[0], 'o', 0), (0.5, C[1], 's', 1)]:
    u = np.sqrt(np.arange(10) + 1 + al + 2*be)
    a2.plot(u, [j]*10, m, color=c, ms=6, mec='white', ls='none')
a2.set_yticks([0, 1]); a2.set_yticklabels([r'$\beta=0$', r'$\beta=0.5$']); a2.set_ylim(-0.6, 1.6)
a2.set_xlabel(r'$u_n=\sqrt{n+1+\alpha+2\beta}$'); a2.set_title('Spectral lattice', fontsize=9)
fig.tight_layout(); fig.savefig('wl_kernels.pdf'); print('saved wl_kernels.pdf')
